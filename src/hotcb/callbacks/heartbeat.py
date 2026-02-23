# src/hotcb/callbacks/heartbeat.py
from __future__ import annotations

from typing import Any, Dict
from .utils import get_log, safe_int, safe_epoch


class HeartbeatCallback:
    """
    Emit a periodic "heartbeat" log message to confirm the run is alive.

    Use cases
    ---------
    - Quick sanity check that HotController is dispatching events.
    - Lightweight progress logging when you don't want full metric logging.

    Parameters
    ----------
    id:
        Callback identifier used by hotcb control plane (enable/disable/set/load).

    every:
        Emit a heartbeat every N steps (env["step"]).
        Values:
          - <= 0 : never emit
          - 1 : emit every step (not recommended)
          - 10/50/100 : typical choices

    message:
        Free-form message string included in heartbeat.

    Events
    ------
    The callback does not care which event triggered it; it logs `event` in the
    output. Typically you attach it to:
      - "train_step_end" or
      - "train_batch_end".

    Example (YAML)
    -------------
    callbacks:
      hb:
        enabled: true
        target: { kind: module, path: hotcb.callbacks.heartbeat, symbol: HeartbeatCallback }
        init: { every: 50, message: "training alive" }

    Example (CLI)
    -------------
    $ hotcb --dir runs/exp1 load hb --module hotcb.callbacks.heartbeat --symbol HeartbeatCallback --enabled --init every=25 message="alive"
    $ hotcb --dir runs/exp1 set hb every=5
    """

    def __init__(self, id: str, every: int = 50, message: str = "alive") -> None:
        self.id = id
        self.every = int(every)
        self.message = str(message)

    def set_params(self, **kwargs: Any) -> None:
        """
        Hot-update parameters at runtime.

        Supported keys
        --------------
        every: int
            New heartbeat frequency.

        message: str
            New message string.
        """
        if "every" in kwargs:
            self.every = int(kwargs["every"])
        if "message" in kwargs:
            self.message = str(kwargs["message"])

    def handle(self, event: str, env: Dict[str, Any]) -> None:
        """
        Emit heartbeat if step % every == 0.

        Parameters
        ----------
        event:
            Event name from adapter.

        env:
            Environment dict; expects:
              - step: int-like
              - epoch: optional
              - phase: optional
              - log: optional callable(str)
        """
        step = safe_int(env.get("step", 0))
        if self.every <= 0 or step % self.every != 0:
            return
        log = get_log(env)
        log(
            f"[hotcb:{self.id}] {self.message} "
            f"event={event} step={step} epoch={safe_epoch(env):.3f} phase={env.get('phase')}"
        )