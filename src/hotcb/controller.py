from __future__ import annotations
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import parse_yaml_config, ConfigError
from .loader import instantiate_callback, CallbackLoadError
from .ops import Op
from .protocol import CallbackTarget
from .util import FileCursor, read_new_jsonl, safe_mtime


@dataclass
class CallbackHandle:
    id: str
    enabled: bool = True
    instance: Optional[Any] = None
    target: Optional[CallbackTarget] = None
    init: Dict[str, Any] = field(default_factory=dict)
    last_params: Dict[str, Any] = field(default_factory=dict)


class HotController:
    """
    Framework-agnostic hot callback controller.

    Safe usage pattern:
      - adapters call controller.apply(env, events=[...]) at safe points.

    Inputs:
      - desired-state YAML (hotcb.yaml)
      - optional commands JSONL (hotcb.commands.jsonl)
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
        self.poll_interval_sec = poll_interval_sec
        self.debounce_steps = max(1, int(debounce_steps))
        self.auto_disable_on_error = auto_disable_on_error
        self.log_path = log_path

        self._handles: Dict[str, CallbackHandle] = {}
        self._last_cfg_mtime: float = 0.0
        self._last_poll_t: float = 0.0
        self._step_counter: int = 0

        self._cmd_cursor = FileCursor(path=commands_path or "", offset=0)

    # ---------------------------
    # Public API
    # ---------------------------
    def status(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for k, h in self._handles.items():
            out[k] = {
                "enabled": h.enabled,
                "loaded": h.instance is not None,
                "target": None if h.target is None else {"kind": h.target.kind, "path": h.target.path, "symbol": h.target.symbol},
                "init": dict(h.init),
                "last_params": dict(h.last_params),
            }
        return out

    def apply(self, env: Dict[str, Any], events: List[str]) -> None:
        """
        Apply any pending updates, then dispatch specified events to enabled callbacks.
        Call this from adapters at safe points.
        """
        self._step_counter += 1

        # Poll updates (debounced)
        if (self._step_counter % self.debounce_steps) == 0:
            self._poll_and_apply_updates(env)

        # Dispatch
        for event in events:
            self._dispatch(event, env)

    # ---------------------------
    # Internals: polling
    # ---------------------------
    def _poll_and_apply_updates(self, env: Dict[str, Any]) -> None:
        now = time.time()
        if self.poll_interval_sec > 0 and (now - self._last_poll_t) < self.poll_interval_sec:
            return
        self._last_poll_t = now

        ops: List[Op] = []

        # 1) desired-state YAML (mtime-based)
        cfg_mtime = safe_mtime(self.config_path)
        if cfg_mtime > self._last_cfg_mtime:
            self._last_cfg_mtime = cfg_mtime
            try:
                ops.extend(parse_yaml_config(self.config_path))
                self._log(env, f"[hotcb] loaded config '{self.config_path}' (mtime={cfg_mtime})")
            except Exception as e:
                self._log(env, f"[hotcb] config error: {e}")
                return  # don't apply partial junk

        # 2) command stream JSONL (offset-based)
        if self.commands_path:
            try:
                raw_cmds, new_cursor = read_new_jsonl(self._cmd_cursor)
                self._cmd_cursor = new_cursor
                ops.extend(self._ops_from_commands(raw_cmds))
            except Exception as e:
                self._log(env, f"[hotcb] commands error: {e}")

        if not ops:
            return

        for op in ops:
            self._apply_op(op, env)

    def _ops_from_commands(self, cmds: List[dict]) -> List[Op]:
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

    # ---------------------------
    # Internals: ops
    # ---------------------------
    def _apply_op(self, op: Op, env: Dict[str, Any]) -> None:
        h = self._handles.get(op.id)
        if h is None:
            h = CallbackHandle(id=op.id)
            self._handles[op.id] = h

        if op.op == "load":
            if op.target is None:
                self._log(env, f"[hotcb] load missing target for {op.id}")
                return
            h.target = op.target
            h.init = op.init or {}
            if "id" not in h.init:
                h.init["id"] = op.id  # ensure callback receives id
            # instantiate if not loaded
            if h.instance is None:
                try:
                    h.instance = instantiate_callback(h.target, h.init)
                    # optional on_attach
                    if hasattr(h.instance, "on_attach"):
                        h.instance.on_attach(env)
                    self._log(env, f"[hotcb] loaded callback {op.id} from {h.target.kind}:{h.target.path}:{h.target.symbol}")
                except Exception as e:
                    self._log(env, f"[hotcb] failed to load {op.id}: {e}")
                    return
            # enabled can be set in load op
            if op.enabled is not None:
                h.enabled = bool(op.enabled)
            return

        if op.op == "enable":
            h.enabled = True
            self._log(env, f"[hotcb] enabled {op.id}")
            return

        if op.op == "disable":
            h.enabled = False
            self._log(env, f"[hotcb] disabled {op.id}")
            return

        if op.op == "set_params":
            if not op.params:
                return
            # if callback isn't loaded yet, store params and apply later
            for k, v in op.params.items():
                h.last_params[k] = v
            if h.instance is not None and hasattr(h.instance, "set_params"):
                try:
                    h.instance.set_params(**op.params)
                    self._log(env, f"[hotcb] set_params {op.id}: {list(op.params.keys())}")
                except Exception as e:
                    self._log(env, f"[hotcb] set_params failed {op.id}: {e}")
                    if self.auto_disable_on_error:
                        h.enabled = False
                        self._log(env, f"[hotcb] auto-disabled {op.id} after set_params error")
            return

        if op.op == "unload":
            # Optional: not required by your spec; provided for convenience.
            if h.instance is not None:
                try:
                    if hasattr(h.instance, "close"):
                        h.instance.close()
                finally:
                    h.instance = None
            h.enabled = False
            self._log(env, f"[hotcb] unloaded {op.id}")
            return

        self._log(env, f"[hotcb] unknown op {op.op} for {op.id}")

    # ---------------------------
    # Internals: dispatch
    # ---------------------------
    def _dispatch(self, event: str, env: Dict[str, Any]) -> None:
        for cb_id, h in list(self._handles.items()):
            if not h.enabled:
                continue
            if h.instance is None:
                continue

            # Apply deferred params once callback exists
            if h.last_params and hasattr(h.instance, "set_params"):
                try:
                    h.instance.set_params(**h.last_params)
                    h.last_params = dict(h.last_params)  # keep record
                except Exception:
                    pass

            try:
                h.instance.handle(event=event, env=env)
            except Exception as e:
                self._log(env, f"[hotcb] callback {cb_id} crashed on event '{event}': {e}\n{traceback.format_exc()}")
                if self.auto_disable_on_error:
                    h.enabled = False
                    self._log(env, f"[hotcb] auto-disabled {cb_id} after crash")

    def _log(self, env: Dict[str, Any], msg: str) -> None:
        # 1) send to adapter-provided logger if exists
        log_fn = env.get("log")
        if callable(log_fn):
            try:
                log_fn(msg)
            except Exception:
                pass

        # 2) optionally append to file
        if self.log_path:
            try:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(msg.rstrip() + "\n")
            except Exception:
                pass