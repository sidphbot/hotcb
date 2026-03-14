"""
hotcb.server.projections — XGBoost-based metric forecasting engine.

Provides univariate and what-if (multivariate) forecasting for training
metrics, with confidence bands via quantile regression or residual stddev.

XGBoost is the primary backend; falls back to numpy linear regression
when xgboost is not installed.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("hotcb.server.projections")

# ---------------------------------------------------------------------------
# Optional XGBoost import
# ---------------------------------------------------------------------------
try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ForecastResult:
    """Container for a forecast output."""

    steps: list[int]
    values: list[float]
    lower: list[float]
    upper: list[float]
    metric_name: str
    method: str  # "xgboost" or "fallback_linear"

    def to_dict(self) -> dict:
        from ..util import sanitize_floats
        return sanitize_floats({
            "steps": self.steps,
            "values": self.values,
            "forecast": self.values,  # alias for dashboard JS
            "lower": self.lower,
            "upper": self.upper,
            "metric_name": self.metric_name,
            "method": self.method,
        })


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

_LAG_OFFSETS = [1, 3, 5]
_ROLL_WINDOW = 5


def _build_features(
    values: np.ndarray,
    steps: np.ndarray,
    hp_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build feature matrix from a 1-D series of metric values.

    Features per row:
      step, lag_1, lag_3, lag_5, rolling_mean_5, rolling_std_5
      (+ optional HP columns)

    Returns features starting at index ``max(_LAG_OFFSETS)`` since earlier
    rows lack sufficient history.
    """
    n = len(values)
    start = max(_LAG_OFFSETS)
    rows: list[list[float]] = []
    for i in range(start, n):
        row: list[float] = [float(steps[i])]
        for lag in _LAG_OFFSETS:
            row.append(float(values[i - lag]))
        window = values[max(0, i - _ROLL_WINDOW + 1) : i + 1]
        row.append(float(np.mean(window)))
        row.append(float(np.std(window)))
        if hp_matrix is not None:
            row.extend(hp_matrix[i].tolist())
        rows.append(row)
    return np.array(rows, dtype=np.float64) if rows else np.empty((0, 6))


def _build_targets(values: np.ndarray) -> np.ndarray:
    start = max(_LAG_OFFSETS)
    return values[start:]


# ---------------------------------------------------------------------------
# Projection engine
# ---------------------------------------------------------------------------


class ProjectionEngine:
    """
    Metric forecasting engine.

    Maintains an internal buffer of metric records and provides
    univariate and what-if forecasting.
    """

    def __init__(self, min_history: int = 20, max_history: int = 500) -> None:
        self.min_history = min_history
        self.max_history = max_history
        # Keyed by metric name
        self._steps: list[int] = []
        self._metrics: Dict[str, list[float]] = {}
        # Optional HP columns for what-if (populated from records that carry them)
        self._hp_keys: list[str] = []
        self._hp_values: list[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update(self, records: list[dict]) -> None:
        """Ingest new metric records."""
        for rec in records:
            step = rec.get("step")
            metrics = rec.get("metrics")
            if step is None or not isinstance(metrics, dict):
                continue
            self._steps.append(int(step))
            for name, val in metrics.items():
                if name not in self._metrics:
                    self._metrics[name] = []
                # Pad with NaN if this metric appeared late
                while len(self._metrics[name]) < len(self._steps) - 1:
                    self._metrics[name].append(float("nan"))
                self._metrics[name].append(float(val))
            # Pad metrics that didn't appear in this record
            for name in self._metrics:
                if len(self._metrics[name]) < len(self._steps):
                    self._metrics[name].append(float("nan"))
            # HP columns (optional)
            hp = rec.get("hp", {})
            self._hp_values.append(hp)
            for k in hp:
                if k not in self._hp_keys:
                    self._hp_keys.append(k)

        # Trim to max_history
        if len(self._steps) > self.max_history:
            excess = len(self._steps) - self.max_history
            self._steps = self._steps[excess:]
            self._hp_values = self._hp_values[excess:]
            for name in self._metrics:
                self._metrics[name] = self._metrics[name][excess:]

    @property
    def record_count(self) -> int:
        return len(self._steps)

    # ------------------------------------------------------------------
    # Internal: prepare arrays for a given metric
    # ------------------------------------------------------------------

    def _prepare(self, metric_name: str) -> Optional[tuple]:
        """Return (values, steps) arrays for *metric_name*, or None."""
        raw = self._metrics.get(metric_name)
        if raw is None:
            return None
        vals = np.array(raw, dtype=np.float64)
        steps = np.array(self._steps, dtype=np.float64)
        # Drop NaN pairs
        mask = ~np.isnan(vals)
        vals = vals[mask]
        steps = steps[mask]
        if len(vals) < self.min_history:
            return None
        return vals, steps

    # ------------------------------------------------------------------
    # Forecast clamping
    # ------------------------------------------------------------------

    def _clamp_forecast(self, result: ForecastResult, vals: np.ndarray) -> ForecastResult:
        """Clamp forecast values to a reasonable range based on observed data."""
        if len(vals) == 0 or len(result.values) == 0:
            return result
        observed_min = float(np.min(vals))
        observed_max = float(np.max(vals))
        data_range = observed_max - observed_min
        if data_range < 1e-8:
            data_range = abs(observed_max) * 0.5 or 1.0
        margin = data_range * 2.0
        lo = observed_min - margin
        hi = observed_max + margin
        result.values = [max(lo, min(hi, v)) for v in result.values]
        result.lower = [max(lo, min(hi, v)) for v in result.lower]
        result.upper = [max(lo, min(hi, v)) for v in result.upper]
        return result

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast_univariate(
        self,
        metric_name: str,
        horizon: int = 50,
    ) -> ForecastResult:
        """Predict a single metric *horizon* steps ahead."""
        prep = self._prepare(metric_name)
        if prep is None:
            return ForecastResult(
                steps=[],
                values=[],
                lower=[],
                upper=[],
                metric_name=metric_name,
                method="xgboost" if HAS_XGB else "fallback_linear",
            )
        vals, steps = prep

        if HAS_XGB:
            result = self._forecast_xgb(metric_name, vals, steps, horizon)
        else:
            result = self._forecast_linear(metric_name, vals, steps, horizon)
        return self._clamp_forecast(result, vals)

    def forecast_whatif(
        self,
        metric_name: str,
        hp_changes: dict,
        horizon: int = 50,
    ) -> ForecastResult:
        """Predict metric under hypothetical HP change."""
        prep = self._prepare(metric_name)
        if prep is None:
            return ForecastResult(
                steps=[],
                values=[],
                lower=[],
                upper=[],
                metric_name=metric_name,
                method="xgboost" if HAS_XGB else "fallback_linear",
            )
        vals, steps = prep

        # Build HP matrix from history (fill missing with 0)
        hp_keys = sorted(
            set(self._hp_keys) | set(hp_changes.keys())
        )
        n = len(self._steps)
        hp_mat = np.zeros((n, len(hp_keys)), dtype=np.float64)
        for i, hp_rec in enumerate(self._hp_values):
            for j, k in enumerate(hp_keys):
                hp_mat[i, j] = float(hp_rec.get(k, 0.0))

        # Apply NaN mask from metric
        raw = self._metrics.get(metric_name, [])
        raw_arr = np.array(raw, dtype=np.float64)
        mask = ~np.isnan(raw_arr)
        hp_mat = hp_mat[mask]

        if HAS_XGB:
            result = self._forecast_xgb(
                metric_name, vals, steps, horizon,
                hp_matrix=hp_mat, hp_keys=hp_keys, hp_changes=hp_changes,
            )
        else:
            result = self._forecast_linear(
                metric_name, vals, steps, horizon,
                hp_matrix=hp_mat, hp_keys=hp_keys, hp_changes=hp_changes,
            )
        return self._clamp_forecast(result, vals)

    # ------------------------------------------------------------------
    # XGBoost backend
    # ------------------------------------------------------------------

    def _forecast_xgb(
        self,
        metric_name: str,
        vals: np.ndarray,
        steps: np.ndarray,
        horizon: int,
        hp_matrix: Optional[np.ndarray] = None,
        hp_keys: Optional[list[str]] = None,
        hp_changes: Optional[dict] = None,
    ) -> ForecastResult:
        X = _build_features(vals, steps, hp_matrix)
        y = _build_targets(vals)
        if len(y) < 5:
            return self._forecast_linear(
                metric_name, vals, steps, horizon,
                hp_matrix=hp_matrix, hp_keys=hp_keys, hp_changes=hp_changes,
            )

        # Fit median model
        model_med = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            objective="reg:squarederror", verbosity=0,
        )
        model_med.fit(X, y)

        # Fit quantile models for confidence bands
        model_lo = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            objective="reg:quantileerror", quantile_alpha=0.1, verbosity=0,
        )
        model_hi = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            objective="reg:quantileerror", quantile_alpha=0.9, verbosity=0,
        )
        model_lo.fit(X, y)
        model_hi.fit(X, y)

        # Autoregressive forecasting
        last_step = int(steps[-1])
        current_vals = list(vals)
        pred_steps: list[int] = []
        pred_vals: list[float] = []
        pred_lo: list[float] = []
        pred_hi: list[float] = []

        for h in range(1, horizon + 1):
            s = last_step + h
            pred_steps.append(s)
            arr = np.array(current_vals, dtype=np.float64)
            sarr = np.array(list(steps) + pred_steps[:h - 1] + [s], dtype=np.float64)

            # Build single-row feature
            hp_row = None
            if hp_matrix is not None and hp_keys:
                row_hp = np.zeros((len(sarr), len(hp_keys)), dtype=np.float64)
                # Copy historical HP values
                min_len = min(len(hp_matrix), len(row_hp))
                row_hp[:min_len] = hp_matrix[:min_len]
                # Fill future rows with hp_changes
                if hp_changes:
                    for j, k in enumerate(hp_keys):
                        row_hp[min_len:, j] = float(hp_changes.get(k, 0.0))
                hp_row = row_hp

            feat = _build_features(arr, sarr, hp_row)
            if len(feat) == 0:
                break
            row = feat[-1:, :]
            v = float(model_med.predict(row)[0])
            lo = float(model_lo.predict(row)[0])
            hi = float(model_hi.predict(row)[0])
            pred_vals.append(v)
            pred_lo.append(min(lo, v))
            pred_hi.append(max(hi, v))
            current_vals.append(v)

        return ForecastResult(
            steps=pred_steps,
            values=pred_vals,
            lower=pred_lo,
            upper=pred_hi,
            metric_name=metric_name,
            method="xgboost",
        )

    # ------------------------------------------------------------------
    # Linear fallback
    # ------------------------------------------------------------------

    def _forecast_linear(
        self,
        metric_name: str,
        vals: np.ndarray,
        steps: np.ndarray,
        horizon: int,
        hp_matrix: Optional[np.ndarray] = None,
        hp_keys: Optional[list[str]] = None,
        hp_changes: Optional[dict] = None,
    ) -> ForecastResult:
        X = _build_features(vals, steps, hp_matrix)
        y = _build_targets(vals)
        if len(y) < 2:
            return ForecastResult(
                steps=[], values=[], lower=[], upper=[],
                metric_name=metric_name, method="fallback_linear",
            )

        # OLS: y = X @ beta
        # Add intercept
        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        Xa = np.hstack([ones, X])
        # Solve via least squares
        beta, residuals, _, _ = np.linalg.lstsq(Xa, y, rcond=None)
        y_hat = Xa @ beta
        resid_std = float(np.std(y - y_hat)) if len(y) > 2 else 0.0

        # Autoregressive forecasting
        last_step = int(steps[-1])
        current_vals = list(vals)
        pred_steps: list[int] = []
        pred_vals: list[float] = []
        pred_lo: list[float] = []
        pred_hi: list[float] = []

        for h in range(1, horizon + 1):
            s = last_step + h
            pred_steps.append(s)
            arr = np.array(current_vals, dtype=np.float64)
            sarr = np.array(list(steps) + pred_steps[:h - 1] + [s], dtype=np.float64)

            hp_row = None
            if hp_matrix is not None and hp_keys:
                row_hp = np.zeros((len(sarr), len(hp_keys)), dtype=np.float64)
                min_len = min(len(hp_matrix), len(row_hp))
                row_hp[:min_len] = hp_matrix[:min_len]
                if hp_changes:
                    for j, k in enumerate(hp_keys):
                        row_hp[min_len:, j] = float(hp_changes.get(k, 0.0))
                hp_row = row_hp

            feat = _build_features(arr, sarr, hp_row)
            if len(feat) == 0:
                break
            row = feat[-1:, :]
            row_a = np.hstack([np.ones((1, 1)), row])
            v = float((row_a @ beta)[0])
            pred_vals.append(v)
            pred_lo.append(v - 1.645 * resid_std)
            pred_hi.append(v + 1.645 * resid_std)
            current_vals.append(v)

        return ForecastResult(
            steps=pred_steps,
            values=pred_vals,
            lower=pred_lo,
            upper=pred_hi,
            metric_name=metric_name,
            method="fallback_linear",
        )


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------


def create_projections_router(engine: ProjectionEngine) -> Any:
    """Build and return the projections APIRouter."""
    from fastapi import APIRouter
    from pydantic import BaseModel

    router = APIRouter(prefix="/api/projections", tags=["projections"])

    class WhatIfRequest(BaseModel):
        hp_changes: Dict[str, float]
        horizon: int = 50

    @router.get("/forecast/{metric_name}")
    async def forecast(metric_name: str, horizon: int = 50):
        result = engine.forecast_univariate(metric_name, horizon=horizon)
        return result.to_dict()

    @router.post("/whatif/{metric_name}")
    async def whatif(metric_name: str, body: WhatIfRequest):
        result = engine.forecast_whatif(
            metric_name, hp_changes=body.hp_changes, horizon=body.horizon,
        )
        return result.to_dict()

    return router
