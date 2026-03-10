"""
hotcb.metrics.features — Optional forward-hook manager that captures layer
activations for feature-space visualization.

Usage::

    capture = FeatureCapture(every_n_steps=50, max_samples=256)
    capture.register(model, layer_names=["encoder.layer.4"])
    # On each step:
    capture.step(current_step)
    # Get latest snapshot:
    snapshot = capture.latest_snapshot()  # returns PCA-reduced activations

Activations are pre-reduced to ``pre_reduce_dim`` dimensions via PCA
in-process and stored in a ring buffer.  Optionally, snapshots are persisted
to ``hotcb.features.jsonl`` for offline analysis.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("hotcb.metrics.features")


@dataclass(frozen=True)
class FeatureSnapshot:
    """A single captured activation snapshot, already PCA-reduced."""
    step: int
    layer_name: str
    activations: List[List[float]]  # (samples, pre_reduce_dim)


class FeatureCapture:
    """
    Opt-in forward-hook manager that captures layer activations for
    feature-space visualization.

    Parameters
    ----------
    every_n_steps : int
        Capture activations every N training steps.
    max_samples : int
        Maximum number of samples to keep per capture (truncated from batch).
    pre_reduce_dim : int
        PCA-reduce each captured tensor to this many dimensions before storing.
    ring_size : int
        Maximum number of snapshots kept in memory.
    output_path : str | None
        If set, append snapshots to this JSONL file.
    """

    def __init__(
        self,
        every_n_steps: int = 50,
        max_samples: int = 256,
        pre_reduce_dim: int = 64,
        ring_size: int = 100,
        output_path: Optional[str] = None,
    ) -> None:
        self.every_n_steps = max(1, every_n_steps)
        self.max_samples = max_samples
        self.pre_reduce_dim = pre_reduce_dim
        self.ring_size = ring_size
        self.output_path = output_path

        self._hooks: List[Any] = []
        self._layer_names: List[str] = []
        self._latest_activations: Dict[str, Any] = {}  # layer -> tensor/ndarray
        self._snapshots: Deque[FeatureSnapshot] = collections.deque(maxlen=ring_size)
        self._lock = threading.Lock()
        self._current_step = 0

    # -- registration -------------------------------------------------------

    def register(self, model: Any, layer_names: List[str]) -> None:
        """
        Register forward hooks on named layers of ``model``.

        Parameters
        ----------
        model : torch.nn.Module
            The model to hook into.
        layer_names : list[str]
            Dot-separated layer names (e.g. ``["encoder.layer.4"]``).
        """
        self.unregister()
        for name in layer_names:
            module = _resolve_module(model, name)
            if module is None:
                log.warning("Layer %r not found in model — skipping", name)
                continue

            def _make_hook(layer_name: str):
                def hook(mod: Any, inp: Any, out: Any) -> None:
                    self._latest_activations[layer_name] = out
                return hook

            handle = module.register_forward_hook(_make_hook(name))
            self._hooks.append(handle)
            self._layer_names.append(name)

        log.info("FeatureCapture registered %d hooks", len(self._hooks))

    def unregister(self) -> None:
        """Remove all forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._layer_names.clear()
        self._latest_activations.clear()

    # -- step ---------------------------------------------------------------

    def step(self, current_step: int) -> None:
        """
        Call once per training step.  If ``current_step`` is a capture step,
        snapshot the latest activations.
        """
        self._current_step = current_step
        if current_step % self.every_n_steps != 0:
            return

        for layer_name in self._layer_names:
            act = self._latest_activations.get(layer_name)
            if act is None:
                continue

            arr = _to_numpy(act)
            if arr is None or arr.ndim < 2:
                continue

            # Flatten to (batch, features)
            batch_size = arr.shape[0]
            arr = arr.reshape(batch_size, -1)

            # Truncate batch
            if arr.shape[0] > self.max_samples:
                arr = arr[: self.max_samples]

            # PCA reduce if needed
            if arr.shape[1] > self.pre_reduce_dim and arr.shape[0] >= 2:
                arr = _quick_pca(arr, self.pre_reduce_dim)

            snapshot = FeatureSnapshot(
                step=current_step,
                layer_name=layer_name,
                activations=arr.tolist(),
            )

            with self._lock:
                self._snapshots.append(snapshot)

            # Persist
            if self.output_path:
                _write_snapshot(self.output_path, snapshot)

    # -- access -------------------------------------------------------------

    def latest_snapshot(self) -> Optional[FeatureSnapshot]:
        """Return the most recent snapshot, or None."""
        with self._lock:
            return self._snapshots[-1] if self._snapshots else None

    def all_snapshots(self) -> List[FeatureSnapshot]:
        """Return all snapshots in the ring buffer."""
        with self._lock:
            return list(self._snapshots)

    @property
    def snapshot_count(self) -> int:
        with self._lock:
            return len(self._snapshots)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_module(model: Any, name: str) -> Any:
    """Resolve a dot-separated module name on a torch Module."""
    parts = name.split(".")
    mod = model
    for part in parts:
        mod = getattr(mod, part, None)
        if mod is None:
            return None
    return mod


def _to_numpy(tensor: Any) -> Optional[np.ndarray]:
    """Convert a torch tensor (or ndarray) to numpy."""
    if isinstance(tensor, np.ndarray):
        return tensor.astype(np.float64)
    # torch.Tensor
    try:
        return tensor.detach().cpu().numpy().astype(np.float64)
    except Exception:
        return None


def _quick_pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """Fast PCA via covariance eigendecomposition."""
    mean = X.mean(axis=0)
    X_c = X - mean
    cov = np.cov(X_c, rowvar=False)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    components = eigenvectors[:, :n_components]
    return X_c @ components


def _write_snapshot(path: str, snapshot: FeatureSnapshot) -> None:
    """Append a snapshot as a single JSON line."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        record = {
            "step": snapshot.step,
            "layer_name": snapshot.layer_name,
            "activations": snapshot.activations,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        log.warning("Failed to write feature snapshot: %s", e)
