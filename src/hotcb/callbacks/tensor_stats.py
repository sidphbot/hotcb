# src/hotcb/callbacks/tensor_stats.py
from __future__ import annotations

from typing import Any, Dict, List
from .utils import get_log, get_in, safe_int, tensor_basic_stats


class TensorStatsCallback:
    """
    Periodically compute and log summary statistics for selected env values.

    This callback is a general-purpose "what are my tensors doing" tool. You
    specify one or more "paths" that are resolved against `env` via dotted keys.

    Parameters
    ----------
    id:
        Callback identifier.

    every:
        Log stats every N steps.

    paths:
        List of dotted paths to inspect. Paths are resolved using:
          - dict lookup for dicts
          - getattr for objects

        Common examples:
          - ["loss"]
          - ["outputs.logits", "batch.images"]
          - ["metrics.eval_loss"]

        If `paths` is None, defaults to ["loss"].

    prefix:
        Optional string prefixed to the path in logs (useful for namespaces).

    Performance guidance
    --------------------
    - Computing full-tensor statistics can be expensive for large tensors.
      Prefer logging reduced outputs (e.g. logits) or sample every ~100 steps.

    Example (YAML)
    -------------
    callbacks:
      tstats:
        enabled: true
        target: { kind: module, path: hotcb.callbacks.tensor_stats, symbol: TensorStatsCallback }
        init:
          every: 100
          paths: ["loss", "outputs.logits"]

    Example (CLI)
    -------------
    $ hotcb --dir runs/exp1 enable tstats
    $ hotcb --dir runs/exp1 set tstats every=20 paths=loss,outputs.logits
    """

    def __init__(self, id: str, every: int = 100, paths: List[str] | None = None, prefix: str = "") -> None:
        self.id = id
        self.every = int(every)
        self.paths = list(paths) if paths else ["loss"]
        self.prefix = str(prefix)

    def set_params(self, **kwargs: Any) -> None:
        """
        Supported hot params
        --------------------
        every: int
            Logging frequency.

        paths: list[str] or comma-separated str
            Example: "loss,outputs.logits"

        prefix: str
            Prefix for log labels.
        """
        if "every" in kwargs:
            self.every = int(kwargs["every"])
        if "paths" in kwargs:
            v = kwargs["paths"]
            if isinstance(v, str):
                self.paths = [s.strip() for s in v.split(",") if s.strip()]
            else:
                self.paths = list(v)
        if "prefix" in kwargs:
            self.prefix = str(kwargs["prefix"])

    def handle(self, event: str, env: Dict[str, Any]) -> None:
        """
        Resolve paths and log tensor stats.

        Parameters
        ----------
        event:
            Event name string.

        env:
            Environment dict.
        """
        step = safe_int(env.get("step", 0))
        if self.every <= 0 or step % self.every != 0:
            return

        log = get_log(env)
        for p in self.paths:
            x = get_in(env, p)
            st = tensor_basic_stats(x)
            if st is None:
                # scalar fallback for common "loss" use if user didn't pass tensor
                if p == "loss":
                    log(f"[hotcb:{self.id}] {self.prefix}{p}: {env.get('loss')}")
                continue
            log(f"[hotcb:{self.id}] {self.prefix}{p}: {st}")