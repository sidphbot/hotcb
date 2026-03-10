"""
MetricsCollector — captures env metrics at each kernel safe-point and writes
them to ``hotcb.metrics.jsonl``.

Design goals:
- Zero overhead when not attached (kernel checks ``if self._metrics_collector``)
- Configurable whitelist / blacklist of metric names
- Ring buffer of recent snapshots for downstream consumers (projections, server)
- Thread-safe append to JSONL
"""
from __future__ import annotations

import collections
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Set

from ..util import ensure_dir

log = logging.getLogger("hotcb.metrics")

# ---------------------------------------------------------------------------
# Snapshot dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSnapshot:
    """A single point-in-time capture of all whitelisted metrics."""
    step: int
    epoch: Optional[int]
    wall_time: float
    metrics: Dict[str, float]


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Collects metrics from the training ``env`` dict at every kernel safe-point
    and persists them as JSONL.

    Parameters
    ----------
    path : str
        Output JSONL file path.
    whitelist : set[str] | None
        If given, only record metrics whose name is in this set.
    blacklist : set[str] | None
        If given, skip metrics whose name is in this set.
    every_n_steps : int
        Record at most once every N steps (decimation). 1 = every step.
    ring_size : int
        Number of recent snapshots kept in-memory for live consumers.
    metric_fn_key : str
        Key in ``env`` for the callable metric accessor (default ``"metric"``).
        If the env contains a callable at this key, it is called with each
        metric name to retrieve values.  Otherwise we scan ``env["metrics"]``
        (dict) if present, and fall back to top-level ``env`` keys.
    extra_metric_names : list[str]
        Additional metric names to attempt to read from ``env["metric"](name)``.
    """

    def __init__(
        self,
        path: str,
        *,
        whitelist: Optional[Set[str]] = None,
        blacklist: Optional[Set[str]] = None,
        every_n_steps: int = 1,
        ring_size: int = 2000,
        metric_fn_key: str = "metric",
        extra_metric_names: Optional[List[str]] = None,
    ) -> None:
        self.path = path
        self._whitelist = set(whitelist) if whitelist else None
        self._blacklist = set(blacklist) if blacklist else set()
        self._every = max(1, int(every_n_steps))
        self._ring: Deque[MetricSnapshot] = collections.deque(maxlen=ring_size)
        self._metric_fn_key = metric_fn_key
        self._extra_names: List[str] = list(extra_metric_names or [])
        self._lock = threading.Lock()
        self._step_counter = 0
        self._discovered_names: Set[str] = set()

        ensure_dir(os.path.dirname(path) or ".")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self, env: Dict[str, Any]) -> Optional[MetricSnapshot]:
        """
        Extract metrics from *env*, apply filters, persist, and buffer.

        Returns the snapshot if one was recorded, else ``None``.
        """
        self._step_counter += 1
        if self._step_counter % self._every != 0:
            return None

        step = int(env.get("step", self._step_counter) or self._step_counter)
        epoch = env.get("epoch")
        if epoch is not None:
            epoch = int(epoch)

        raw = self._extract(env)
        if not raw:
            return None

        snap = MetricSnapshot(
            step=step,
            epoch=epoch,
            wall_time=time.time(),
            metrics=raw,
        )

        with self._lock:
            self._ring.append(snap)

        self._persist(snap)
        return snap

    @property
    def recent(self) -> List[MetricSnapshot]:
        """Return a copy of the ring buffer (oldest first)."""
        with self._lock:
            return list(self._ring)

    @property
    def discovered_names(self) -> Set[str]:
        """Metric names seen so far (useful for UI auto-discovery)."""
        return set(self._discovered_names)

    def tail(self, last_n: int = 100) -> List[MetricSnapshot]:
        """Return the most recent *last_n* snapshots."""
        with self._lock:
            items = list(self._ring)
        return items[-last_n:]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract(self, env: Dict[str, Any]) -> Dict[str, float]:
        """Pull numeric values from env, respecting whitelist/blacklist."""
        out: Dict[str, float] = {}

        # Strategy 1: callable metric accessor (Lightning/HF adapters set this)
        metric_fn: Optional[Callable] = env.get(self._metric_fn_key)
        if callable(metric_fn):
            # Try well-known names + discovered + extras
            names = self._well_known_names() | self._discovered_names | set(self._extra_names)
            for name in names:
                try:
                    v = metric_fn(name)
                    fv = self._to_float(v)
                    if fv is not None:
                        out[name] = fv
                except Exception:
                    pass

        # Strategy 2: env["metrics"] dict
        metrics_dict = env.get("metrics")
        if isinstance(metrics_dict, dict):
            for k, v in metrics_dict.items():
                fv = self._to_float(v)
                if fv is not None:
                    out[k] = fv

        # Strategy 3: well-known top-level keys
        for k in ("loss", "train_loss", "val_loss"):
            if k in env and k not in out:
                fv = self._to_float(env[k])
                if fv is not None:
                    out[k] = fv

        # Apply filters
        filtered: Dict[str, float] = {}
        for k, v in out.items():
            if self._blacklist and k in self._blacklist:
                continue
            if self._whitelist and k not in self._whitelist:
                continue
            filtered[k] = v

        self._discovered_names.update(filtered.keys())
        return filtered

    def _persist(self, snap: MetricSnapshot) -> None:
        """Append snapshot to JSONL file."""
        record = {
            "step": snap.step,
            "epoch": snap.epoch,
            "wall_time": snap.wall_time,
            "metrics": snap.metrics,
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning("Failed to write metrics: %s", e)

    @staticmethod
    def _to_float(v: Any) -> Optional[float]:
        """Best-effort conversion to float."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        # torch.Tensor
        if hasattr(v, "item"):
            try:
                return float(v.item())
            except Exception:
                pass
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _well_known_names() -> Set[str]:
        return {
            "loss", "train_loss", "val_loss",
            "train/loss", "val/loss",
            "lr", "learning_rate",
            "accuracy", "train_accuracy", "val_accuracy",
            "train/accuracy", "val/accuracy",
            "grad_norm",
        }
