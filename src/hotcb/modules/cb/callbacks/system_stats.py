# src/hotcb/callbacks/system_stats.py
from __future__ import annotations

from typing import Any, Dict
from .utils import get_log, safe_int


class SystemStatsCallback:
    """
    Log basic system resource stats (CPU RSS and optionally GPU memory).

    Parameters
    ----------
    id:
        Callback identifier.

    every:
        Log every N steps.

    gpu:
        If True, attempt to log GPU memory via torch.cuda if available.

        Behavior:
          - If torch is not installed or CUDA is unavailable, GPU section is skipped.

    What it logs
    ------------
    CPU (Unix):
      - ru_maxrss from `resource.getrusage(resource.RUSAGE_SELF)`
        Note: units differ by OS:
          - Linux: kilobytes
          - macOS: bytes
        We log raw value to avoid lying.

    GPU (if enabled):
      - memory_allocated
      - memory_reserved
      in MiB for current device.

    Example (YAML)
    -------------
    callbacks:
      sys:
        enabled: true
        target: { kind: module, path: hotcb.callbacks.system_stats, symbol: SystemStatsCallback }
        init: { every: 200, gpu: true }
    """

    def __init__(self, id: str, every: int = 200, gpu: bool = True) -> None:
        self.id = id
        self.every = int(every)
        self.gpu = bool(gpu)

    def set_params(self, **kwargs: Any) -> None:
        """
        Supported hot params
        --------------------
        every: int
        gpu: bool
        """
        if "every" in kwargs:
            self.every = int(kwargs["every"])
        if "gpu" in kwargs:
            self.gpu = bool(kwargs["gpu"])

    def handle(self, event: str, env: Dict[str, Any]) -> None:
        """
        Emit system stats log line.

        Parameters
        ----------
        event:
            Event name string.

        env:
            Environment dict; expects:
              - step: int-like
              - log: optional callable
        """
        step = safe_int(env.get("step", 0))
        if self.every <= 0 or step % self.every != 0:
            return

        log = get_log(env)

        try:
            import resource

            ru = resource.getrusage(resource.RUSAGE_SELF)
            log(f"[hotcb:{self.id}] sys event={event} step={step} ru_maxrss={ru.ru_maxrss}")
        except Exception:
            log(f"[hotcb:{self.id}] sys event={event} step={step} (resource stats unavailable)")

        if not self.gpu:
            return

        try:
            import torch

            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / (1024**2)
                reserv = torch.cuda.memory_reserved() / (1024**2)
                log(f"[hotcb:{self.id}] gpu mem alloc={alloc:.1f}MiB reserved={reserv:.1f}MiB")
        except Exception:
            pass