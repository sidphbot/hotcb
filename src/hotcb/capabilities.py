"""
Training capabilities descriptor.

Adapters populate this at fit_start to describe what the training setup
supports.  Downstream consumers (modules, actuators, dashboard, autopilot)
read it from ``env["capabilities"]`` to adapt their behavior.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class TrainingCapabilities:
    """Immutable snapshot of what the current training setup supports."""

    framework: str = "unknown"  # "lightning" | "hf" | "vanilla"
    num_optimizers: int = 1
    optimizer_names: Tuple[str, ...] = ()
    num_param_groups: Tuple[int, ...] = ()  # per-optimizer param group counts
    has_scheduler: bool = False
    scheduler_types: Tuple[str, ...] = ()
    grad_accumulation_steps: int = 1  # 1 = no accumulation
    automatic_optimization: bool = True
    mutable_state_detected: bool = False
    mutable_state_keys: Tuple[str, ...] = ()
    grad_clip_value: Optional[float] = None  # None = not configured
    grad_clip_wired: bool = False  # True only if hotcb can actually modify it
    metric_names: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert tuples to lists for JSON
        for k in ("optimizer_names", "num_param_groups", "scheduler_types",
                   "mutable_state_keys", "metric_names"):
            if isinstance(d[k], tuple):
                d[k] = list(d[k])
        return d

    def save(self, run_dir: str) -> None:
        """Write capabilities to run_dir for the dashboard server to read."""
        path = os.path.join(run_dir, "hotcb.capabilities.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, run_dir: str) -> Optional["TrainingCapabilities"]:
        """Load capabilities from a run directory, or None if not found."""
        path = os.path.join(run_dir, "hotcb.capabilities.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            # Convert lists back to tuples
            for k in ("optimizer_names", "num_param_groups", "scheduler_types",
                       "mutable_state_keys", "metric_names"):
                if k in d and isinstance(d[k], list):
                    d[k] = tuple(d[k])
            return cls(**d)
        except Exception:
            return None


def validate_mutable_state(obj: Any) -> Tuple[bool, dict]:
    """Check if obj looks like a valid mutable_state dict.

    Returns (valid, normalized_dict).  Accepts both the standard
    ``{"weights": {...}, "terms": {...}}`` layout and a flat
    ``{"key": float, ...}`` dict (auto-wrapped as weights).
    """
    if not isinstance(obj, dict):
        return False, {}
    weights = obj.get("weights")
    if isinstance(weights, dict):
        return True, obj
    # Auto-wrap: flat dict of str->numeric → treat as weights
    if obj and all(
        isinstance(k, str) and isinstance(v, (int, float))
        for k, v in obj.items()
        if k not in ("terms", "ramps")
    ):
        return True, {"weights": dict(obj), "terms": {}, "ramps": {}}
    return False, {}
