from __future__ import annotations

import traceback as tb_mod
from dataclasses import dataclass, field
from typing import Dict, Optional

from ..ops import HotOp
from .result import ModuleResult


@dataclass
class OptHandle:
    id: str
    enabled: bool = True
    last_params: Dict[str, float] = field(default_factory=dict)
    last_error: Optional[str] = None


class HotOptController:
    """
    Live optimizer control (lr, weight decay, scheduler nudges).
    """

    def __init__(self, auto_disable_on_error: bool = True) -> None:
        self.handles: Dict[str, OptHandle] = {}
        self.auto_disable_on_error = auto_disable_on_error

    def _resolve_optimizer(self, env: dict):
        if "optimizer" in env:
            return env.get("optimizer")
        resolver = env.get("resolve_optimizer")
        if callable(resolver):
            try:
                return resolver()
            except Exception:
                return None
        return None

    def apply_op(self, op: HotOp, env: dict) -> ModuleResult:
        hid = op.id or "main"
        handle = self.handles.get(hid)
        if handle is None:
            handle = OptHandle(id=hid)
            self.handles[hid] = handle

        if op.op == "enable":
            handle.enabled = True
            return ModuleResult(decision="applied")

        if op.op == "disable":
            handle.enabled = False
            return ModuleResult(decision="applied")

        if op.op == "set_params":
            if not handle.enabled:
                return ModuleResult(decision="skipped_noop", notes="handle_disabled")

            params = op.params or {}
            handle.last_params.update(params)
            opt = self._resolve_optimizer(env)
            if opt is None:
                return ModuleResult(decision="failed", error="missing_optimizer")

            try:
                self._apply_params(opt, params)
                return ModuleResult(decision="applied", payload=params)
            except Exception as e:
                handle.last_error = str(e)
                if self.auto_disable_on_error:
                    handle.enabled = False
                return ModuleResult(decision="failed", error=str(e), traceback=tb_mod.format_exc())

        return ModuleResult(decision="ignored", notes=f"unknown_op:{op.op}")

    def _apply_params(self, optimizer, params: dict) -> None:
        """Mutate optimizer param groups."""
        groups = optimizer.param_groups
        if "group" in params and params["group"] is not None:
            try:
                idx = int(params["group"])
                groups = [optimizer.param_groups[idx]]
            except Exception:
                raise ValueError(f"invalid group index {params['group']}")

        if "groups" in params and isinstance(params["groups"], dict):
            # explicit mapping group_idx -> params
            for k, v in params["groups"].items():
                idx = int(k)
                self._apply_params_to_group(optimizer.param_groups[idx], v)
            return

        for g in groups:
            self._apply_params_to_group(g, params)

    def _apply_params_to_group(self, group: dict, params: dict) -> None:
        if "lr" in params and params["lr"] is not None:
            group["lr"] = float(params["lr"])
        if "weight_decay" in params and params["weight_decay"] is not None:
            group["weight_decay"] = float(params["weight_decay"])
        if "scheduler_scale" in params and params["scheduler_scale"] is not None:
            # multiplicative bump
            group["lr"] = float(group.get("lr", 0.0)) * float(params["scheduler_scale"])
        if "scheduler_drop" in params and params["scheduler_drop"] is not None:
            group["lr"] = float(group.get("lr", 0.0)) * float(params["scheduler_drop"])
        # clip_norm is stored for visibility; user loop can read it if needed
        if "clip_norm" in params and params["clip_norm"] is not None:
            group["hotcb_clip_norm"] = float(params["clip_norm"])

    def status(self) -> Dict[str, dict]:
        return {
            hid: {"enabled": h.enabled, "last_params": dict(h.last_params), "last_error": h.last_error}
            for hid, h in self.handles.items()
        }
