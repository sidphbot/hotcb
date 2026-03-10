"""
hotcb.server.manifolds — Metric manifold and feature-space visualization.

Provides:
- ``ManifoldEngine``: projects multi-metric histories into 2D/3D via PCA
  (with optional UMAP / t-SNE when extra packages are installed).
- ``TrajectoryStats``: velocity, curvature and intervention-impact analysis.
- FastAPI router mounted at ``/api/manifolds``.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("hotcb.server.manifolds")

# ---------------------------------------------------------------------------
# Optional backend probing
# ---------------------------------------------------------------------------

_HAS_UMAP = False
_HAS_TSNE = False

try:
    import umap as _umap  # noqa: F401
    _HAS_UMAP = True
except ImportError:
    pass

try:
    from sklearn.manifold import TSNE as _TSNE  # noqa: F401
    _HAS_TSNE = True
except ImportError:
    pass


def available_methods() -> List[str]:
    """Return list of dimensionality-reduction methods available at runtime."""
    methods = ["pca"]
    if _HAS_UMAP:
        methods.append("umap")
    if _HAS_TSNE:
        methods.append("tsne")
    return methods


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ManifoldResult:
    points: List[List[float]]  # Nx2 or Nx3 coordinates
    steps: List[int]
    is_intervention: List[bool]
    method: str
    metric_names: List[str]
    explained_variance: Optional[List[float]] = None  # PCA only


@dataclass
class TrajectoryStats:
    total_distance: float
    velocities: List[float]
    mean_velocity: float
    intervention_impacts: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# ManifoldEngine
# ---------------------------------------------------------------------------

class ManifoldEngine:
    """Incrementally accumulates metric records and computes projections."""

    def __init__(self, max_points: int = 2000) -> None:
        self.max_points = max_points
        self._metric_history: List[Dict[str, Any]] = []
        self._intervention_steps: set = set()

    def reset(self) -> None:
        """Clear all accumulated data for a fresh training run."""
        self._metric_history.clear()
        self._intervention_steps.clear()

    # -- ingest -------------------------------------------------------------

    def update_metrics(self, records: List[Dict[str, Any]]) -> None:
        """Ingest metric records from the tailer."""
        for rec in records:
            step = rec.get("step")
            metrics = rec.get("metrics")
            if step is None or not metrics:
                continue
            self._metric_history.append({"step": int(step), "metrics": metrics})
        # Trim to ring-buffer size
        if len(self._metric_history) > self.max_points:
            self._metric_history = self._metric_history[-self.max_points:]

    def update_interventions(self, records: List[Dict[str, Any]]) -> None:
        """Mark intervention steps from the applied ledger."""
        for rec in records:
            step = rec.get("step")
            if step is not None:
                self._intervention_steps.add(int(step))

    # -- projection ---------------------------------------------------------

    def compute_metric_manifold(
        self,
        method: str = "pca",
        n_components: int = 3,
    ) -> ManifoldResult:
        """Project multi-metric space to 2D/3D."""
        if not self._metric_history:
            return ManifoldResult(
                points=[], steps=[], is_intervention=[],
                method=method, metric_names=[], explained_variance=[],
            )

        # Discover all metric names (union across records)
        all_names: set = set()
        for rec in self._metric_history:
            all_names.update(rec["metrics"].keys())
        metric_names = sorted(all_names)

        if not metric_names:
            return ManifoldResult(
                points=[], steps=[], is_intervention=[],
                method=method, metric_names=[], explained_variance=[],
            )

        # Build numeric matrix with forward-fill for missing values
        steps: List[int] = []
        rows: List[List[float]] = []
        last_values: Dict[str, float] = {}

        for rec in self._metric_history:
            step = rec["step"]
            m = rec["metrics"]
            row: List[float] = []
            for name in metric_names:
                val = m.get(name)
                if val is not None:
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        val = last_values.get(name)
                else:
                    val = last_values.get(name)
                if val is None:
                    val = 0.0
                last_values[name] = val
                row.append(val)
            rows.append(row)
            steps.append(step)

        X = np.array(rows, dtype=np.float64)  # (N, D)

        # Need at least 2 points for projection
        if X.shape[0] < 2:
            return ManifoldResult(
                points=[rows[0][:n_components]] if rows else [],
                steps=steps,
                is_intervention=[s in self._intervention_steps for s in steps],
                method=method, metric_names=metric_names, explained_variance=[],
            )

        # Normalize: zero-mean, unit-variance
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        X_norm = (X - mean) / std

        n_comp = min(n_components, X_norm.shape[1], X_norm.shape[0])

        explained_variance: Optional[List[float]] = None

        if method == "pca" or method not in available_methods():
            projected, explained_variance = _pca(X_norm, n_comp)
        elif method == "umap":
            projected = _umap_project(X_norm, n_comp)
        elif method == "tsne":
            projected = _tsne_project(X_norm, n_comp)
        else:
            projected, explained_variance = _pca(X_norm, n_comp)

        is_intervention = [s in self._intervention_steps for s in steps]

        return ManifoldResult(
            points=projected.tolist(),
            steps=steps,
            is_intervention=is_intervention,
            method=method if method in available_methods() else "pca",
            metric_names=metric_names,
            explained_variance=explained_variance,
        )

    # -- trajectory stats ---------------------------------------------------

    def compute_trajectory_stats(self) -> TrajectoryStats:
        """Compute trajectory statistics in raw metric space."""
        if len(self._metric_history) < 2:
            return TrajectoryStats(
                total_distance=0.0, velocities=[], mean_velocity=0.0,
                intervention_impacts=[],
            )

        # Build matrix (same procedure as manifold, but no projection)
        all_names: set = set()
        for rec in self._metric_history:
            all_names.update(rec["metrics"].keys())
        metric_names = sorted(all_names)

        last_values: Dict[str, float] = {}
        rows: List[List[float]] = []
        steps: List[int] = []
        for rec in self._metric_history:
            m = rec["metrics"]
            row: List[float] = []
            for name in metric_names:
                val = m.get(name)
                if val is not None:
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        val = last_values.get(name, 0.0)
                else:
                    val = last_values.get(name, 0.0)
                last_values[name] = val
                row.append(val)
            rows.append(row)
            steps.append(rec["step"])

        X = np.array(rows, dtype=np.float64)
        diffs = np.diff(X, axis=0)  # (N-1, D)
        distances = np.linalg.norm(diffs, axis=1).tolist()

        total_distance = float(sum(distances))
        mean_velocity = total_distance / len(distances) if distances else 0.0

        # Intervention impact: compare velocity before vs after each intervention
        intervention_impacts: List[Dict[str, Any]] = []
        for idx, step in enumerate(steps):
            if step in self._intervention_steps and 0 < idx < len(distances):
                v_before = distances[idx - 1] if idx - 1 < len(distances) else 0.0
                v_after = distances[idx] if idx < len(distances) else 0.0
                intervention_impacts.append({
                    "step": step,
                    "velocity_before": v_before,
                    "velocity_after": v_after,
                })

        return TrajectoryStats(
            total_distance=total_distance,
            velocities=distances,
            mean_velocity=mean_velocity,
            intervention_impacts=intervention_impacts,
        )


# ---------------------------------------------------------------------------
# Projection backends
# ---------------------------------------------------------------------------

def _pca(X: np.ndarray, n_components: int) -> tuple:
    """Pure-numpy PCA via eigendecomposition of the covariance matrix."""
    cov = np.cov(X, rowvar=False)
    if cov.ndim == 0:
        # Single feature edge case
        cov = cov.reshape(1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    components = eigenvectors[:, :n_components]
    projected = X @ components

    total_var = eigenvalues.sum()
    if total_var > 0:
        explained = (eigenvalues[:n_components] / total_var).tolist()
    else:
        explained = [0.0] * n_components

    return projected, explained


def _umap_project(X: np.ndarray, n_components: int) -> np.ndarray:
    """UMAP projection (requires umap-learn)."""
    import umap
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(X)


def _tsne_project(X: np.ndarray, n_components: int) -> np.ndarray:
    """t-SNE projection (requires scikit-learn)."""
    from sklearn.manifold import TSNE
    perplexity = min(30, max(2, X.shape[0] - 1))
    reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    return reducer.fit_transform(X)


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

def _get_engine(request: Any) -> ManifoldEngine:
    from fastapi import Request
    return request.app.state.manifold_engine


def create_router(engine: Optional[ManifoldEngine] = None) -> Any:
    """Create and return the manifolds API router.

    If *engine* is ``None`` the endpoints look it up from ``app.state``.
    """
    from fastapi import APIRouter

    _engine = engine
    router = APIRouter(prefix="/api/manifolds", tags=["manifolds"])

    def _get(request_or_app: Any) -> ManifoldEngine:
        if _engine is not None:
            return _engine
        return request_or_app.app.state.manifold_engine

    @router.get("/metric")
    async def metric_manifold(method: str = "pca", n_components: int = 3):
        if _engine is None:
            raise RuntimeError("ManifoldEngine not initialised")
        result = _engine.compute_metric_manifold(method=method, n_components=n_components)
        return {
            "points": result.points,
            "steps": result.steps,
            "is_intervention": result.is_intervention,
            "method": result.method,
            "metric_names": result.metric_names,
            "explained_variance": result.explained_variance,
        }

    @router.get("/trajectory")
    async def trajectory_stats():
        if _engine is None:
            raise RuntimeError("ManifoldEngine not initialised")
        stats = _engine.compute_trajectory_stats()
        return {
            "total_distance": stats.total_distance,
            "velocities": stats.velocities,
            "mean_velocity": stats.mean_velocity,
            "intervention_impacts": stats.intervention_impacts,
        }

    @router.get("/available-methods")
    async def get_available_methods():
        return {"methods": available_methods()}

    return router
