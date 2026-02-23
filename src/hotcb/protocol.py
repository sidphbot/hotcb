# src/hotcb/protocol.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


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

        Parameters
        ----------
        event:
            A string naming the event. Examples:
              - "fit_start"
              - "train_step_end"
              - "train_batch_end"
              - "val_batch_end"
              - "eval_end"

            Event naming is intentionally flexible. You can standardize within
            your own project or rely on the adapters provided by hotcb.

        env:
            A dict containing runtime context. This is your stable integration
            surface across frameworks.

            Typical fields (adapters may supply a subset):
              - "framework": "lightning" | "hf" | "torch" | ...
              - "phase": "train" | "val" | "eval" | ...
              - "step": int global step
              - "epoch": int/float epoch
              - "model": model instance (LightningModule, nn.Module, HF model)
              - "trainer": framework trainer object (optional)
              - "batch": current batch (optional)
              - "outputs": model outputs (optional)
              - "metrics": evaluation metrics dict (optional)
              - "loss": scalar loss (optional, recommended for diagnostics)
              - "log": callable(str) -> None (optional logging sink)

            You should treat env values as best-effort / optional. Your callback
            should degrade gracefully if keys are missing.

    set_params(**kwargs: Any) -> None
        Applies hot-updated parameters, typically sent via:
          - `hotcb set <id> key=value ...` (CLI → commands.jsonl), or
          - a desired-state config file change (YAML reconcile).

        This method should:
          - validate/normalize values,
          - be idempotent, and
          - avoid heavy allocations when possible.

    Optional methods (recommended)
    ------------------------------
    on_attach(env: Dict[str, Any]) -> None
        Called once when the callback is first instantiated/loaded by the
        controller. Use it to set up resources (create dirs, open writers).

    close() -> None
        Called on "unload" operations (optional feature). Use it to clean up
        threads/files/processes.

    Design guidance
    ---------------
    - Keep callbacks fast. For expensive work, sample only every N steps or
      enqueue work to a background worker.
    - Do not keep GPU tensors alive longer than needed; `.detach()` and move to
      CPU if you store artifacts.
    - Avoid crashing the training loop. If exceptions happen, `HotController`
      can auto-disable the callback (configurable).

    Example
    -------
    >>> class Heartbeat:
    ...     def __init__(self, id: str, every: int = 10):
    ...         self.id = id
    ...         self.every = every
    ...     def set_params(self, **kwargs):
    ...         if "every" in kwargs: self.every = int(kwargs["every"])
    ...     def handle(self, event, env):
    ...         step = int(env.get("step", 0))
    ...         if step % self.every == 0:
    ...             (env.get("log") or print)(f"[{self.id}] {event} step={step}")
    """

    id: str

    def handle(self, event: str, env: Dict[str, Any]) -> None: ...
    def set_params(self, **kwargs: Any) -> None: ...


@dataclass
class CallbackTarget:
    """
    Declarative specification for locating a callback class for dynamic loading.

    Attributes
    ----------
    kind:
        Loader type. Supported values:
          - "python_file": load from a filesystem path to a .py file
          - "module": import via Python module path

    path:
        When kind == "python_file":
            Absolute or relative file path to a Python source file.
            Examples:
              - "/tmp/my_diag.py"
              - "callbacks/feat_viz.py"

        When kind == "module":
            Importable module path (must be resolvable in sys.path).
            Examples:
              - "hotcb.callbacks.timing"
              - "mypkg.callbacks.hotspot"

    symbol:
        The attribute name inside the module/file that points to the callback
        class (or a callable returning an instance, but class is recommended).

        Example:
          - "TimingCallback"
          - "HotspotMonitorCallback"

    Notes
    -----
    - `hotcb` instantiates the class with `init_kwargs` (see controller load op).
    - For python_file loading, `hotcb` creates a unique module name derived from
      the file path hash to avoid collisions.

    Example (YAML)
    -------------
    callbacks:
      timing:
        enabled: true
        target:
          kind: module
          path: hotcb.callbacks.timing
          symbol: TimingCallback
        init:
          every: 50
    """

    kind: str
    path: str
    symbol: str