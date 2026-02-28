# src/hotcb/callbacks/timing.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from .utils import get_log, now_s, safe_int


class TimingCallback:
    """
    Measure time between successive invocations and report rolling stats.

    This callback is most meaningful when:
      - it is attached to a consistent event (e.g., train_step_end),
      - the event happens once per step, and
      - `apply()` is called regularly.

    Parameters
    ----------
    id:
        Callback identifier.

    every:
        Log timing summary every N steps.

    window:
        Rolling window size for summary stats.

        Values:
          - 50-500 typical.
          - larger windows stabilize statistics but lag in reflecting changes.

    What it logs
    ------------
    - dt: time since previous handle() invocation (seconds)
    - mean: rolling mean over window
    - p50: median over window
    - p95: approx 95th percentile over window (simple sorted index)

    Limitations
    ----------
    - Percentiles are computed via sorting window values (O(window log window)).
      Keep window moderate if performance matters.
    - dt measures wall time between callback invocations; it includes any time
      spent outside the step, e.g., validation or logging overhead, if the event
      is not strictly step-aligned.

    Example (YAML)
    -------------
    callbacks:
      timing:
        enabled: true
        target: { kind: module, path: hotcb.callbacks.timing, symbol: TimingCallback }
        init: { every: 50, window: 200 }

    Example (CLI)
    -------------
    $ hotcb --dir runs/exp1 set timing every=10 window=100
    """

    def __init__(self, id: str, every: int = 50, window: int = 200) -> None:
        self.id = id
        self.every = int(every)
        self.window = int(window)
        self._t_last: Optional[float] = None
        self._dt: List[float] = []

    def set_params(self, **kwargs: Any) -> None:
        """
        Supported hot params
        --------------------
        every: int
            Summary frequency.

        window: int
            Rolling window size.
        """
        if "every" in kwargs:
            self.every = int(kwargs["every"])
        if "window" in kwargs:
            self.window = int(kwargs["window"])

    def handle(self, event: str, env: Dict[str, Any]) -> None:
        """
        Record dt and periodically log rolling stats.

        Parameters
        ----------
        event:
            Event name string (included in log line).

        env:
            Environment dict; expects:
              - step: int-like
              - log: optional callable
        """
        t = now_s()
        if self._t_last is None:
            self._t_last = t
            return

        dt = t - self._t_last
        self._t_last = t

        self._dt.append(dt)
        if len(self._dt) > max(1, self.window):
            self._dt = self._dt[-self.window :]

        step = safe_int(env.get("step", 0))
        if self.every <= 0 or step % self.every != 0:
            return

        s = sorted(self._dt)
        mean = sum(s) / len(s)
        p50 = s[len(s) // 2]
        p95 = s[int(0.95 * (len(s) - 1))] if len(s) > 1 else s[0]

        log = get_log(env)
        log(
            f"[hotcb:{self.id}] timing event={event} step={step} "
            f"dt={dt:.4f}s mean={mean:.4f}s p50={p50:.4f}s p95={p95:.4f}s (n={len(s)})"
        )