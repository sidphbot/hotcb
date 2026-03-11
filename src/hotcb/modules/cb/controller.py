# src/hotcb/controller.py
from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import parse_yaml_config
from .loader import instantiate_callback
from .ops import Op
from .protocol import CallbackTarget
from .util import FileCursor, read_new_jsonl, safe_mtime
from dataclasses import dataclass, field
from .util import FileCursor, read_new_jsonl, safe_mtime


@dataclass
class CallbackHandle:
    """
    Internal registry entry for a callback ID.

    The controller tracks "handles" rather than raw callback instances so it can:
      - enable/disable without deleting objects,
      - remember where a callback was loaded from (target),
      - remember init kwargs for future reloads, and
      - store the last applied params for status visibility and deferred apply.

    Attributes
    ----------
    id:
        Callback identifier (unique within a controller).

    enabled:
        Whether the callback should receive events when dispatching.
        "Remove" semantics can be implemented as `enabled=False`.

    instance:
        The instantiated callback object, or None if not loaded/unloaded.

    target:
        CallbackTarget describing where to load the callback from.
        Used primarily for `load` operation.

    init:
        Init kwargs used for instantiation. Only applied at first load.

    last_params:
        Latest set of hot params received for the callback. Useful for:
          - deferred application if set_params arrives before load, and
          - status reporting / debugging.

    Notes
    -----
    - The controller will inject `id` into init kwargs if absent.
    - `enabled=False` does not necessarily free resources; to free, use unload.
    """
    id: str
    enabled: bool = True
    instance: Optional[Any] = None
    target: Optional[CallbackTarget] = None
    init: Dict[str, Any] = field(default_factory=dict)
    last_params: Dict[str, Any] = field(default_factory=dict)
    loaded_target_mtime: float = 0.0


class HotController:
    """
    Framework-agnostic hot callback controller.

    This is the core of the project. It is intentionally independent of:
      - PyTorch Lightning,
      - HuggingFace Trainer,
      - Accelerate,
      - and any particular training loop.

    You integrate it via:
      - an adapter (Lightning/HF), or
      - direct calls from your training loop (bare torch).

    Core Responsibilities
    ---------------------
    1) Load and instantiate callbacks dynamically ("load" op)
    2) Enable/disable callbacks ("enable"/"disable" ops)
    3) Apply live param updates ("set_params" op)
    4) Dispatch events to enabled callbacks at safe points
    5) Watch control-plane inputs:
         - desired-state config file (YAML), and/or
         - command stream file (JSONL) from CLI

    Control Plane Inputs
    --------------------
    config_path:
        A desired-state YAML file. When the file mtime changes, the controller
        parses it and applies ops to converge to the desired state.

        Requires PyYAML if you use YAML:
          pip install "hotcb[yaml]"

    commands_path:
        Append-only JSONL file containing imperative ops, typically written by
        the CLI (`hotcb enable ...`, `hotcb set ...`, `hotcb load ...`).

        This is stdlib-only (no YAML dependency).

    Safety and Performance
    ----------------------
    - You must call `apply()` only at safe points (e.g., end of step).
    - Polling is debounced by `debounce_steps` and optionally `poll_interval_sec`.
    - Exceptions inside callbacks can auto-disable the crashing callback to avoid
      killing the training run.

    Parameters
    ----------
    config_path:
        Path to desired-state YAML file (can exist or not). If it doesn't exist,
        YAML reconciliation produces no ops.

    commands_path:
        Path to command JSONL file. If None, command polling is disabled.

    poll_interval_sec:
        Minimum seconds between polls. Use this if your step rate is high and
        you want to cap filesystem checks by time rather than by steps.

        Typical values:
          - 0.0 (default): poll only when debounce_steps hits
          - 0.5 to 2.0: reasonable for very fast training loops

    debounce_steps:
        Poll every N calls to `apply()`. Use 1 for immediate updates, use 5/10
        to reduce overhead.

    auto_disable_on_error:
        If True, any callback exception (during set_params or handle) disables
        the callback. This prevents a noisy callback from repeatedly crashing
        and spamming logs.

    log_path:
        Optional file path to append controller logs (one line per message).
        Useful on remote servers.

    Example: Lightning
    ------------------
    >>> controller = HotController(
    ...     config_path="runs/exp1/hotcb.yaml",
    ...     commands_path="runs/exp1/hotcb.commands.jsonl",
    ...     debounce_steps=5,
    ...     log_path="runs/exp1/hotcb.log",
    ... )
    >>> hot = HotCallbackController(controller)
    >>> trainer = pl.Trainer(callbacks=[hot])

    Example: bare torch
    -------------------
    >>> controller = HotController("hotcb.yaml", "hotcb.commands.jsonl")
    >>> for step, batch in enumerate(loader):
    ...     loss = train_step(batch)
    ...     env = {"framework":"torch", "phase":"train", "step":step, "loss":loss, "log":print, "model":model}
    ...     controller.apply(env, events=["train_step_end"])
    """

    def __init__(
        self,
        config_path: str,
        commands_path: Optional[str] = None,
        poll_interval_sec: float = 0.0,
        debounce_steps: int = 1,
        auto_disable_on_error: bool = True,
        log_path: Optional[str] = None,
    ) -> None:
        self.config_path = config_path
        self.commands_path = commands_path
        self.poll_interval_sec = float(poll_interval_sec)
        self.debounce_steps = max(1, int(debounce_steps))
        self.auto_disable_on_error = bool(auto_disable_on_error)
        self.log_path = log_path

        self._handles: Dict[str, CallbackHandle] = {}
        self._last_cfg_mtime: float = 0.0
        self._last_poll_t: float = 0.0
        self._step_counter: int = 0

        self._cmd_cursor = FileCursor(path=commands_path or "", offset=0)

    def status(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a structured snapshot of current controller state.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Mapping callback_id -> status dict with keys:
              - enabled: bool
              - loaded: bool (instance exists)
              - target: dict(kind/path/symbol) or None
              - init: dict init kwargs (stored)
              - last_params: dict last applied/stored params

        Intended Use
        ------------
        - debugging via prints,
        - future `hotcb status` CLI command,
        - lightweight telemetry.

        Example
        -------
        >>> import pprint
        >>> pprint.pprint(controller.status())
        {'timing': {'enabled': True, 'loaded': True, ...}}
        """
        out: Dict[str, Dict[str, Any]] = {}
        for k, h in self._handles.items():
            out[k] = {
                "enabled": h.enabled,
                "loaded": h.instance is not None,
                "target": None
                if h.target is None
                else {"kind": h.target.kind, "path": h.target.path, "symbol": h.target.symbol},
                "init": dict(h.init),
                "last_params": dict(h.last_params),
            }
        return out

    def apply(self, env: Dict[str, Any], events: List[str]) -> None:
        """
        Apply pending updates (config/commands) and dispatch events to callbacks.

        This is the main entrypoint called by adapters or training loops.

        Parameters
        ----------
        env:
            Environment/context dict. See `HotCallback` docstring for typical keys.

            Minimum useful keys:
              - step: int
              - log: callable for logging (optional)

        events:
            List of event names to dispatch after applying updates.
            Common patterns:
              - events=["train_step_end"]
              - events=["train_batch_end"]
              - events=["val_batch_end"]

        Operational semantics
        ---------------------
        1) Increments internal step counter
        2) If debounce boundary is hit, polls control plane and applies ops
        3) Dispatches each event in order to each enabled callback

        Notes
        -----
        - You should call `apply()` at a "safe point" in your training loop.
          For example, at end of step, not during backward.

        - If your adapter calls apply frequently, use debounce_steps to reduce
          filesystem overhead.

        Example
        -------
        >>> controller.apply(env={"step": 100, "log": print}, events=["train_step_end"])
        """
        self._step_counter += 1

        if (self._step_counter % self.debounce_steps) == 0:
            self._poll_and_apply_updates(env)

        for event in events:
            self._dispatch(event, env)

    def _poll_and_apply_updates(self, env: Dict[str, Any]) -> None:
        """
        Poll config and command sources and apply resulting ops.

        Polling is limited by:
          - `poll_interval_sec` (time-based), and
          - `debounce_steps` (call-based, handled by apply()).

        Sources
        -------
        1) YAML desired-state config:
           - reload if file mtime increases
           - parse into ops and apply

        2) JSONL command stream:
           - read new appended lines using offset cursor
           - convert to ops and apply

        Error behavior
        --------------
        - YAML errors: logged and ignored (no partial apply).
        - command errors: logged; best-effort continues.

        This method is internal; external callers should use `apply()`.
        """
        now = time.time()
        if self.poll_interval_sec > 0 and (now - self._last_poll_t) < self.poll_interval_sec:
            return
        self._last_poll_t = now

        for cb_id, h in list(self._handles.items()):
            if h.instance is None or h.target is None:
                continue
            if h.target.kind != "python_file":
                continue

            cur_m = safe_mtime(h.target.path)
            if cur_m > 0 and cur_m > (h.loaded_target_mtime or 0.0):
                try:
                    # IMPORTANT: force_reload=True is what makes the loader drop cache + re-exec file.
                    new_instance = instantiate_callback(h.target, h.init, force_reload=True)

                    # swap only after successful instantiation
                    h.instance = new_instance
                    h.loaded_target_mtime = cur_m

                    try:
                        modname = getattr(getattr(h.instance, "__class__", None), "__module__", "")
                        self._log(env, f"[hotcb.cb] reload module={modname} file={h.target.path}")
                    except Exception:
                        pass

                    if hasattr(h.instance, "on_attach"):
                        h.instance.on_attach(env)

                    # re-apply params after reload
                    if h.last_params and hasattr(h.instance, "set_params"):
                        h.instance.set_params(**h.last_params)

                    self._log(env, f"[hotcb.cb] auto-reloaded {cb_id} from file (mtime={cur_m})")

                except Exception as e:
                    self._log(env, f"[hotcb.cb] auto-reload failed {cb_id}: {e}")
                    if self.auto_disable_on_error:
                        h.enabled = False
                        self._log(env, f"[hotcb.cb] auto-disabled {cb_id} after reload failure")

        ops: List[Op] = []

        cfg_mtime = safe_mtime(self.config_path)
        if cfg_mtime > self._last_cfg_mtime:
            self._last_cfg_mtime = cfg_mtime
            try:
                ops.extend(parse_yaml_config(self.config_path))
                self._log(env, f"[hotcb.cb] loaded config '{self.config_path}' (mtime={cfg_mtime})")
            except Exception as e:
                self._log(env, f"[hotcb.cb] config error: {e}")
                return

        if self.commands_path:
            try:
                raw_cmds, new_cursor = read_new_jsonl(self._cmd_cursor)
                self._cmd_cursor = new_cursor
                ops.extend(self._ops_from_commands(raw_cmds))
            except Exception as e:
                self._log(env, f"[hotcb.cb] commands error: {e}")

        if not ops:
            return

        for op in ops:
            self._apply_op(op, env)

    def _ops_from_commands(self, cmds: List[dict]) -> List[Op]:
        """
        Convert decoded JSON command dicts into internal `Op` objects.

        Parameters
        ----------
        cmds:
            List of dicts parsed from JSONL. Each dict should match:
              {"op": "...", "id": "...", ...}

        Returns
        -------
        List[Op]
            Parsed operations.

        Supported command schema
        ------------------------
        enable/disable:
          {"op":"enable", "id":"feat_viz"}
          {"op":"disable", "id":"feat_viz"}

        set_params:
          {"op":"set_params", "id":"feat_viz", "params":{"every":25}}

        load:
          {"op":"load","id":"my_diag",
           "target":{"kind":"python_file","path":"/tmp/x.py","symbol":"MyDiag"},
           "init":{"msg":"hello"},
           "enabled":true}

        unload (optional):
          {"op":"unload", "id":"my_diag"}

        Notes
        -----
        - Values are accepted as-is; callbacks should validate types in set_params.
        - Unknown keys are ignored by this parser.
        """
        out: List[Op] = []
        for c in cmds:
            op = str(c.get("op"))
            cb_id = str(c.get("id"))
            params = c.get("params")
            init = c.get("init")
            enabled = c.get("enabled")

            target = None
            if "target" in c and c["target"] is not None:
                t = c["target"]
                target = CallbackTarget(kind=str(t["kind"]), path=str(t["path"]), symbol=str(t["symbol"]))

            out.append(Op(op=op, id=cb_id, params=params, target=target, init=init, enabled=enabled))
        return out

    def _apply_op(self, op: Op, env: Dict[str, Any]) -> None:
        """
        Apply a single operation to the internal registry.

        This method is responsible for:
          - handle creation,
          - dynamic instantiation,
          - enable/disable,
          - parameter updates, and
          - optional unload.

        Parameters
        ----------
        op:
            Operation to apply.

        env:
            Current environment dict for logging and optional callback on_attach.

        Error behavior
        --------------
        - Load failures are logged; callback remains unloaded.
        - set_params failures are logged; callback may be auto-disabled.
        - Unknown ops are logged.

        Notes
        -----
        - `load` injects `id` into init kwargs if not present.
        - `set_params` stores params even if callback not loaded yet; they are
          applied later once the instance exists.
        """
        h = self._handles.get(op.id)
        if h is None:
            h = CallbackHandle(id=op.id)
            self._handles[op.id] = h

        if op.op == "load":
            if op.target is None:
                self._log(env, f"[hotcb.cb] load missing target for {op.id}")
                return
            h.target = op.target
            h.init = op.init or {}
            if "id" not in h.init:
                h.init["id"] = op.id

            if h.instance is None:
                try:
                    h.instance = instantiate_callback(h.target, h.init)
                    if h.target.kind == "python_file":
                        h.loaded_target_mtime = safe_mtime(h.target.path)
                    if hasattr(h.instance, "on_attach"):
                        h.instance.on_attach(env)
                    self._log(
                        env,
                        f"[hotcb.cb] loaded callback {op.id} from {h.target.kind}:{h.target.path}:{h.target.symbol}",
                    )
                except Exception as e:
                    self._log(env, f"[hotcb.cb] failed to load {op.id}: {e}")
                    return

            if op.enabled is not None:
                h.enabled = bool(op.enabled)
            return

        if op.op == "enable":
            h.enabled = True
            self._log(env, f"[hotcb.cb] enabled {op.id}")
            return

        if op.op == "disable":
            h.enabled = False
            self._log(env, f"[hotcb.cb] disabled {op.id}")
            return

        if op.op == "set_params":
            if not op.params:
                return

            for k, v in op.params.items():
                h.last_params[k] = v

            if h.instance is not None and hasattr(h.instance, "set_params"):
                try:
                    h.instance.set_params(**op.params)
                    self._log(env, f"[hotcb.cb] set_params {op.id}: {list(op.params.keys())}")
                except Exception as e:
                    self._log(env, f"[hotcb.cb] set_params failed {op.id}: {e}")
                    if self.auto_disable_on_error:
                        h.enabled = False
                        self._log(env, f"[hotcb.cb] auto-disabled {op.id} after set_params error")
            return

        if op.op == "unload":
            if h.instance is not None:
                try:
                    if hasattr(h.instance, "close"):
                        h.instance.close()
                finally:
                    h.instance = None
            h.enabled = False
            self._log(env, f"[hotcb.cb] unloaded {op.id}")
            return

        self._log(env, f"[hotcb.cb] unknown op {op.op} for {op.id}")

    # hotcb integration hooks
    def apply_op(self, op: Op, env: Dict[str, Any]) -> None:
        """Public wrapper around `_apply_op` for kernel routing."""
        self._apply_op(op, env)

    def _dispatch(self, event: str, env: Dict[str, Any]) -> None:
        """
        Dispatch a single event to all enabled, loaded callbacks.

        Parameters
        ----------
        event:
            Event name string.

        env:
            Environment dict passed to callbacks.

        Behavior
        --------
        - Skips disabled callbacks.
        - Skips callbacks that are not loaded.
        - Best-effort applies deferred params before dispatch (if any exist).

        Error handling
        --------------
        - If callback.handle raises, logs traceback.
        - If auto_disable_on_error is True, disables the callback after crash.

        Performance notes
        -----------------
        - Dispatch iterates handles in insertion order (Python dict order).
          If you need priority ordering, add priority fields and sort.
        """
        for cb_id, h in list(self._handles.items()):
            if not h.enabled:
                continue
            if h.instance is None:
                continue

            if h.last_params and hasattr(h.instance, "set_params"):
                try:
                    h.instance.set_params(**h.last_params)
                except Exception:
                    pass

            try:
                h.instance.handle(event=event, env=env)
            except Exception as e:
                self._log(
                    env,
                    f"[hotcb.cb] callback {cb_id} crashed on event '{event}': {e}\n{traceback.format_exc()}",
                )
                if self.auto_disable_on_error:
                    h.enabled = False
                    self._log(env, f"[hotcb.cb] auto-disabled {cb_id} after crash")

    def dispatch_events(self, events: List[str], env: Dict[str, Any]) -> None:
        """Dispatch multiple events in order."""
        for event in events:
            self._dispatch(event, env)

    def _log(self, env: Dict[str, Any], msg: str) -> None:
        """
        Log a controller message.

        Logging sinks (in order)
        ------------------------
        1) If env["log"] exists and is callable, call it with the message.
           This lets adapters integrate with framework-specific logging
           (e.g., trainer.print in Lightning).

        2) If `log_path` is set, append the message to that file.

        Parameters
        ----------
        env:
            Environment dict that may contain a "log" callable.

        msg:
            Log message string.

        Notes
        -----
        - Logging is best-effort; failures are silently ignored.
        - The file logger creates directories if needed.
        """
        log_fn = env.get("log")
        if callable(log_fn):
            try:
                log_fn(msg)
            except Exception:
                pass

        if self.log_path:
            try:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(msg.rstrip() + "\n")
            except Exception:
                pass
