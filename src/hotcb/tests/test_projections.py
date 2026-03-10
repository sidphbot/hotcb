"""Tests for hotcb.server.projections — metric forecasting engine."""
from __future__ import annotations

import math
import sys
from unittest import mock

import numpy as np
import pytest

from hotcb.server.projections import (
    ForecastResult,
    ProjectionEngine,
    _build_features,
    _build_targets,
    HAS_XGB,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic data
# ---------------------------------------------------------------------------

def _make_loss_curve(n: int = 100, noise: float = 0.02, seed: int = 42) -> list[dict]:
    """Generate an exponential-decay loss curve with Gaussian noise."""
    rng = np.random.RandomState(seed)
    records: list[dict] = []
    for i in range(n):
        loss = 1.0 * math.exp(-0.03 * i) + rng.normal(0, noise)
        acc = 1.0 - loss + rng.normal(0, noise * 0.5)
        records.append({
            "step": i,
            "epoch": i // 10,
            "wall_time": float(i) * 0.5,
            "metrics": {"loss": loss, "accuracy": acc},
        })
    return records


def _make_loss_curve_with_hp(
    n: int = 100, noise: float = 0.02, seed: int = 42,
) -> list[dict]:
    """Loss curve with HP metadata."""
    rng = np.random.RandomState(seed)
    records: list[dict] = []
    lr = 0.01
    for i in range(n):
        if i == 50:
            lr = 0.001
        loss = 1.0 * math.exp(-0.03 * i) * (lr / 0.01) + rng.normal(0, noise)
        records.append({
            "step": i,
            "epoch": i // 10,
            "wall_time": float(i) * 0.5,
            "metrics": {"loss": loss},
            "hp": {"lr": lr},
        })
    return records


# ---------------------------------------------------------------------------
# ForecastResult
# ---------------------------------------------------------------------------

class TestForecastResult:
    def test_to_dict(self):
        r = ForecastResult(
            steps=[1, 2], values=[0.5, 0.4], lower=[0.4, 0.3],
            upper=[0.6, 0.5], metric_name="loss", method="fallback_linear",
        )
        d = r.to_dict()
        assert d["metric_name"] == "loss"
        assert d["method"] == "fallback_linear"
        assert len(d["steps"]) == 2

    def test_empty_result(self):
        r = ForecastResult(
            steps=[], values=[], lower=[], upper=[],
            metric_name="loss", method="fallback_linear",
        )
        assert len(r.steps) == 0


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    def test_build_features_shape(self):
        vals = np.arange(20, dtype=np.float64)
        steps = np.arange(20, dtype=np.float64)
        X = _build_features(vals, steps)
        # Starts at index 5 (max lag offset), so 15 rows
        assert X.shape[0] == 15
        # 6 features: step, lag_1, lag_3, lag_5, rolling_mean_5, rolling_std_5
        assert X.shape[1] == 6

    def test_build_features_with_hp(self):
        vals = np.arange(20, dtype=np.float64)
        steps = np.arange(20, dtype=np.float64)
        hp = np.ones((20, 2), dtype=np.float64)
        X = _build_features(vals, steps, hp)
        assert X.shape[1] == 8  # 6 base + 2 HP

    def test_build_targets_alignment(self):
        vals = np.arange(20, dtype=np.float64)
        y = _build_targets(vals)
        assert len(y) == 15  # starts at index 5

    def test_empty_input(self):
        vals = np.array([], dtype=np.float64)
        steps = np.array([], dtype=np.float64)
        X = _build_features(vals, steps)
        assert X.shape[0] == 0


# ---------------------------------------------------------------------------
# ProjectionEngine — core
# ---------------------------------------------------------------------------

class TestProjectionEngineCore:
    def test_update_ingests_records(self):
        engine = ProjectionEngine(min_history=5)
        records = _make_loss_curve(30)
        engine.update(records)
        assert engine.record_count == 30

    def test_update_skips_bad_records(self):
        engine = ProjectionEngine(min_history=5)
        engine.update([
            {"step": 1},           # no metrics
            {"metrics": {"x": 1}}, # no step
            {},                     # empty
        ])
        assert engine.record_count == 0

    def test_max_history_trimming(self):
        engine = ProjectionEngine(min_history=5, max_history=50)
        records = _make_loss_curve(100)
        engine.update(records)
        assert engine.record_count == 50

    def test_incremental_update(self):
        engine = ProjectionEngine(min_history=5)
        records = _make_loss_curve(60)
        engine.update(records[:30])
        assert engine.record_count == 30
        engine.update(records[30:])
        assert engine.record_count == 60


# ---------------------------------------------------------------------------
# Univariate forecast
# ---------------------------------------------------------------------------

class TestForecastUnivariate:
    def test_insufficient_history_returns_empty(self):
        engine = ProjectionEngine(min_history=20)
        engine.update(_make_loss_curve(10))
        result = engine.forecast_univariate("loss", horizon=10)
        assert len(result.steps) == 0
        assert len(result.values) == 0

    def test_unknown_metric_returns_empty(self):
        engine = ProjectionEngine(min_history=5)
        engine.update(_make_loss_curve(30))
        result = engine.forecast_univariate("nonexistent", horizon=10)
        assert len(result.steps) == 0

    def test_valid_forecast(self):
        engine = ProjectionEngine(min_history=10)
        engine.update(_make_loss_curve(50))
        result = engine.forecast_univariate("loss", horizon=20)
        assert result.metric_name == "loss"
        assert len(result.steps) == 20
        assert len(result.values) == 20
        assert len(result.lower) == 20
        assert len(result.upper) == 20
        # Steps should be sequential starting after last training step (49)
        assert result.steps[0] == 50
        assert result.steps[-1] == 69

    def test_forecast_method_field(self):
        engine = ProjectionEngine(min_history=10)
        engine.update(_make_loss_curve(50))
        result = engine.forecast_univariate("loss", horizon=5)
        assert result.method in ("xgboost", "fallback_linear")

    def test_confidence_bands_ordered(self):
        engine = ProjectionEngine(min_history=10)
        engine.update(_make_loss_curve(80))
        result = engine.forecast_univariate("loss", horizon=20)
        for lo, val, hi in zip(result.lower, result.values, result.upper):
            assert lo <= val, f"lower {lo} > value {val}"
            assert val <= hi, f"value {val} > upper {hi}"

    def test_forecast_accuracy(self):
        engine = ProjectionEngine(min_history=10)
        engine.update(_make_loss_curve(100, noise=0.005))
        result = engine.forecast_univariate("accuracy", horizon=10)
        assert len(result.steps) == 10
        # Accuracy values should be reasonable (0 to 2 range)
        for v in result.values:
            assert -1.0 < v < 3.0


# ---------------------------------------------------------------------------
# What-if forecast
# ---------------------------------------------------------------------------

class TestForecastWhatIf:
    def test_whatif_returns_forecast(self):
        engine = ProjectionEngine(min_history=10)
        engine.update(_make_loss_curve_with_hp(60))
        result = engine.forecast_whatif(
            "loss", hp_changes={"lr": 0.001}, horizon=10,
        )
        assert result.metric_name == "loss"
        assert len(result.steps) == 10
        assert len(result.values) == 10

    def test_whatif_insufficient_history(self):
        engine = ProjectionEngine(min_history=20)
        engine.update(_make_loss_curve_with_hp(10))
        result = engine.forecast_whatif(
            "loss", hp_changes={"lr": 0.001}, horizon=10,
        )
        assert len(result.steps) == 0

    def test_whatif_confidence_bands(self):
        engine = ProjectionEngine(min_history=10)
        engine.update(_make_loss_curve_with_hp(80))
        result = engine.forecast_whatif(
            "loss", hp_changes={"lr": 0.0001}, horizon=15,
        )
        for lo, val, hi in zip(result.lower, result.values, result.upper):
            assert lo <= val
            assert val <= hi

    def test_whatif_with_new_hp_key(self):
        """HP key not seen in history should still work."""
        engine = ProjectionEngine(min_history=10)
        engine.update(_make_loss_curve(50))  # no HP in records
        result = engine.forecast_whatif(
            "loss", hp_changes={"momentum": 0.9}, horizon=10,
        )
        assert len(result.steps) == 10


# ---------------------------------------------------------------------------
# Fallback linear regression
# ---------------------------------------------------------------------------

class TestFallbackLinear:
    def test_linear_fallback_when_xgb_missing(self):
        """Force the fallback path by temporarily hiding xgboost."""
        engine = ProjectionEngine(min_history=10)
        engine.update(_make_loss_curve(50))
        # Directly call the linear fallback
        prep = engine._prepare("loss")
        assert prep is not None
        vals, steps = prep
        result = engine._forecast_linear("loss", vals, steps, horizon=10)
        assert result.method == "fallback_linear"
        assert len(result.steps) == 10
        assert len(result.values) == 10

    def test_linear_confidence_bands_ordered(self):
        engine = ProjectionEngine(min_history=10)
        engine.update(_make_loss_curve(60))
        prep = engine._prepare("loss")
        vals, steps = prep
        result = engine._forecast_linear("loss", vals, steps, horizon=15)
        for lo, val, hi in zip(result.lower, result.values, result.upper):
            assert lo <= val
            assert val <= hi

    def test_linear_whatif(self):
        engine = ProjectionEngine(min_history=10)
        records = _make_loss_curve_with_hp(60)
        engine.update(records)
        prep = engine._prepare("loss")
        vals, steps = prep
        # Build hp_matrix manually
        hp_keys = ["lr"]
        hp_mat = np.zeros((len(engine._steps), 1), dtype=np.float64)
        for i, hp_rec in enumerate(engine._hp_values):
            hp_mat[i, 0] = float(hp_rec.get("lr", 0.0))
        raw = np.array(engine._metrics["loss"], dtype=np.float64)
        mask = ~np.isnan(raw)
        hp_mat = hp_mat[mask]

        result = engine._forecast_linear(
            "loss", vals, steps, horizon=10,
            hp_matrix=hp_mat, hp_keys=hp_keys, hp_changes={"lr": 0.0001},
        )
        assert result.method == "fallback_linear"
        assert len(result.steps) == 10


# ---------------------------------------------------------------------------
# Engine used via module-level HAS_XGB=False patch
# ---------------------------------------------------------------------------

class TestForcedFallback:
    def test_forecast_uses_linear_when_xgb_disabled(self):
        """Patch HAS_XGB to False and verify fallback is used."""
        with mock.patch("hotcb.server.projections.HAS_XGB", False):
            engine = ProjectionEngine(min_history=10)
            engine.update(_make_loss_curve(50))
            result = engine.forecast_univariate("loss", horizon=10)
            assert result.method == "fallback_linear"
            assert len(result.steps) == 10

    def test_whatif_uses_linear_when_xgb_disabled(self):
        with mock.patch("hotcb.server.projections.HAS_XGB", False):
            engine = ProjectionEngine(min_history=10)
            engine.update(_make_loss_curve_with_hp(60))
            result = engine.forecast_whatif(
                "loss", hp_changes={"lr": 0.001}, horizon=10,
            )
            assert result.method == "fallback_linear"
            assert len(result.steps) == 10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_metric_sparse(self):
        """Metric appears only in some records."""
        engine = ProjectionEngine(min_history=5)
        records = []
        for i in range(30):
            m = {"loss": float(i) * 0.1}
            if i % 3 == 0:
                m["special"] = float(i) * 0.01
            records.append({"step": i, "metrics": m})
        engine.update(records)
        # "special" has fewer valid points — may or may not meet min_history
        result = engine.forecast_univariate("special", horizon=5)
        # Should either produce valid results or empty (not crash)
        assert isinstance(result, ForecastResult)

    def test_horizon_zero(self):
        engine = ProjectionEngine(min_history=10)
        engine.update(_make_loss_curve(50))
        result = engine.forecast_univariate("loss", horizon=0)
        assert len(result.steps) == 0

    def test_very_small_history(self):
        """Exactly min_history records."""
        engine = ProjectionEngine(min_history=20)
        engine.update(_make_loss_curve(20))
        result = engine.forecast_univariate("loss", horizon=5)
        # Should work — exactly at the threshold
        assert isinstance(result, ForecastResult)

    def test_constant_metric(self):
        """Metric that never changes."""
        engine = ProjectionEngine(min_history=10)
        records = [{"step": i, "metrics": {"flat": 1.0}} for i in range(30)]
        engine.update(records)
        result = engine.forecast_univariate("flat", horizon=5)
        assert len(result.steps) == 5
        # Predictions should be close to 1.0
        for v in result.values:
            assert abs(v - 1.0) < 1.0, f"Expected ~1.0, got {v}"


# ---------------------------------------------------------------------------
# to_dict serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_forecast_result_round_trip(self):
        engine = ProjectionEngine(min_history=10)
        engine.update(_make_loss_curve(50))
        result = engine.forecast_univariate("loss", horizon=5)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert set(d.keys()) == {
            "steps", "values", "forecast", "lower", "upper", "metric_name", "method",
        }
        assert d["forecast"] == d["values"]
        # All lists should be JSON-serializable (no numpy types)
        import json
        json.dumps(d)  # should not raise
