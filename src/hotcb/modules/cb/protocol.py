# src/hotcb/modules/cb/protocol.py
from __future__ import annotations

from typing import Any, Dict, Protocol

# CallbackTarget is defined in hotcb.ops; re-export from here for internal use
from ...ops import CallbackTarget  # noqa: F401


class HotCallback(Protocol):
    """
    Framework-agnostic callback protocol used by `hotcb`.

    A "hot callback" is a small plugin object that can be:
      - enabled/disabled during a live run (without restart),
      - have its parameters updated at runtime, and
      - (optionally) be loaded from a Python module or a standalone .py file.

    The callback is invoked by the `HotController` through adapter-defined "events"
    at safe points (e.g., end of train step, end of eval step). `hotcb` itself
    does not enforce a fixed set of events; you define them in your adapter or
    your raw training loop.

    Required attributes / methods
    -----------------------------
    id : str
        Stable identifier for the callback instance. This is used for:
          - CLI operations (enable/disable/set/load),
          - config reconciliation, and
          - logging / status reporting.

    handle(event: str, env: Dict[str, Any]) -> None
        Called whenever an adapter emits an event and the callback is enabled.

    set_params(**kwargs: Any) -> None
        Applies hot-updated parameters.

    Optional methods (recommended)
    ------------------------------
    on_attach(env: Dict[str, Any]) -> None
        Called once when the callback is first instantiated/loaded by the
        controller.

    close() -> None
        Called on "unload" operations.
    """

    id: str

    def handle(self, event: str, env: Dict[str, Any]) -> None: ...
    def set_params(self, **kwargs: Any) -> None: ...
