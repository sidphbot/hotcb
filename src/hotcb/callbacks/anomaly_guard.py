# src/hotcb/callbacks/anomaly_guard.py
from __future__ import annotations

from typing import Any, Dict, List
from .utils import get_log, get_in, safe_int, tensor_basic_stats


class AnomalyGuardCallback:
    """
    Detect NaN/Inf anomalies in selected tensors and respond.

    This callback is meant to be always-on (cheap), scanning a small set of
    tensors (e.g. loss, logits) for anomalies that often signal instability.

    Parameters
    ----------
    id:
        Callback identifier.

    every:
        Check every N steps. Use 1 for every step, or larger for cheaper checks.

    paths:
        Env paths to examine. Defaults to ["loss"].

    raise_on_trigger:
        If True, raises RuntimeError when anomaly detected. This will likely stop
        training unless your framework catches it.

    disable_after_trigger:
        If True, the callback becomes inert after first anomaly detection.
        Useful if you only want a single alert rather than repeated logs.

    Example (YAML)
    -------------
    callbacks:
      guard:
        enabled: true
        target: { kind: module, path: hotcb.callbacks.anomaly_guard, symbol: AnomalyGuardCallback }
        init:
          every: 1
          paths: ["loss", "outputs.logits"]
          raise_on_trigger: false
          disable_after_trigger: false

    Example (CLI)
    -------------
    $ hotcb --dir runs/exp1 set guard raise_on_trigger=true
    """

    def __init__(
        self,
        id: str,
        every: int = 1,
        paths: List[str] | None = None,
        raise_on_trigger: bool = False,
        disable_after_trigger: bool = False,
    ) -> None:
        self.id = id
        self.every = int(every)
        self.paths = list(paths) if paths else ["loss"]
        self.raise_on_trigger = bool(raise_on_trigger)
        self.disable_after_trigger = bool(disable_after_trigger)
        self._triggered = False

    def set_params(self, **kwargs: Any) -> None:
        """
        Supported hot params
        --------------------
        every: int
        paths: list[str] or comma-separated string
        raise_on_trigger: bool
        disable_after_trigger: bool
        """
        if "every" in kwargs:
            self.every = int(kwargs["every"])
        if "paths" in kwargs:
            v = kwargs["paths"]
            if isinstance(v, str):
                self.paths = [s.strip() for s in v.split(",") if s.strip()]
            else:
                self.paths = list(v)
        if "raise_on_trigger" in kwargs:
            self.raise_on_trigger = bool(kwargs["raise_on_trigger"])
        if "disable_after_trigger" in kwargs:
            self.disable_after_trigger = bool(kwargs["disable_after_trigger"])

    def handle(self, event: str, env: Dict[str, Any]) -> None:
        """
        Check selected paths for NaN/Inf and take action on trigger.

        Action on trigger
        -----------------
        - Always logs a line describing the anomaly.
        - If raise_on_trigger=True: raises RuntimeError.
        - If disable_after_trigger=True: stops doing work after first trigger.

        Parameters
        ----------
        event:
            Event name string.

        env:
            Environment dict.
        """
        if self._triggered and self.disable_after_trigger:
            return

        step = safe_int(env.get("step", 0))
        if self.every <= 0 or step % self.every != 0:
            return

        for p in self.paths:
            x = get_in(env, p)
            st = tensor_basic_stats(x)
            if not st:
                continue
            if st.get("nan", 0) > 0 or st.get("inf", 0) > 0:
                self._triggered = True
                log = get_log(env)
                log(f"[hotcb:{self.id}] ANOMALY DETECTED path={p} event={event} step={step} stats={st}")
                if self.raise_on_trigger:
                    raise RuntimeError(f"[hotcb:{self.id}] anomaly in {p}: {st}")
                return