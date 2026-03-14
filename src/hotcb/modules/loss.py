from __future__ import annotations

import traceback as tb_mod
from dataclasses import dataclass, field
from typing import Dict, Optional

from ..ops import HotOp
from .result import ModuleResult


@dataclass
class LossHandle:
    id: str
    enabled: bool = True
    last_params: Dict[str, float] = field(default_factory=dict)
    last_error: Optional[str] = None


class HotLossController:
    """
    Live loss-state mutations (weights/toggles/ramps).
    """

    def __init__(self, auto_disable_on_error: bool = True) -> None:
        self.handles: Dict[str, LossHandle] = {}
        self.auto_disable_on_error = auto_disable_on_error
        self._actuator = None

    def set_actuator(self, actuator) -> None:
        """Register a MutableStateActuator for pre-apply validation."""
        self._actuator = actuator

    def _resolve_mutable_state(self, env: dict):
        if "mutable_state" in env:
            return env.get("mutable_state")
        resolver = env.get("resolve_mutable_state")
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
            handle = LossHandle(id=hid)
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
            mutable_state = self._resolve_mutable_state(env)
            if mutable_state is None:
                return ModuleResult(decision="failed", error="missing_mutable_state")

            # Pre-validate via actuator if registered
            if self._actuator is not None:
                try:
                    for key, value in self._actuator_weight_patches(params):
                        vresult = self._actuator.validate(
                            {"op": "set", "key": key, "value": value}, env,
                        )
                        if not vresult.valid:
                            return ModuleResult(
                                decision="failed",
                                error=f"validation: {'; '.join(vresult.errors)}",
                                notes="actuator_validation_failed",
                            )
                except Exception:
                    pass  # validation is best-effort

            try:
                self._apply_params(mutable_state, params)
                return ModuleResult(decision="applied", payload=params)
            except Exception as e:
                handle.last_error = str(e)
                if self.auto_disable_on_error:
                    handle.enabled = False
                return ModuleResult(decision="failed", error=str(e), traceback=tb_mod.format_exc())

        return ModuleResult(decision="ignored", notes=f"unknown_op:{op.op}")

    @staticmethod
    def _actuator_weight_patches(params: dict):
        """Yield (key, value) pairs for loss weight actuator validation."""
        for k, v in params.items():
            if k.endswith("_w"):
                yield k[:-2], v
            elif k.startswith("terms.") or k == "terms":
                continue  # terms are toggles, not validated by actuator
            elif k.startswith("ramps.") or k == "ramps":
                continue  # ramps are configs, not validated by actuator
            elif isinstance(v, (int, float)):
                yield k, v

    def _apply_params(self, mutable_state: dict, params: dict) -> None:
        weights = mutable_state.setdefault("weights", {})
        terms = mutable_state.setdefault("terms", {})
        ramps = mutable_state.setdefault("ramps", {})

        for k, v in params.items():
            if k.endswith("_w"):
                weights[k[:-2]] = v
                continue
            if k.startswith("terms."):
                terms[k.split(".", 1)[1]] = bool(v)
                continue
            if k == "terms" and isinstance(v, dict):
                for tk, tv in v.items():
                    terms[tk] = bool(tv)
                continue
            if k.startswith("ramps.") and isinstance(v, dict):
                ramps[k.split(".", 1)[1]] = v
                continue
            if k == "ramps" and isinstance(v, dict):
                for rk, rv in v.items():
                    ramps[rk] = rv
                continue
            # Fallback to weights bucket
            weights[k] = v

    def status(self) -> Dict[str, dict]:
        return {
            hid: {"enabled": h.enabled, "last_params": dict(h.last_params), "last_error": h.last_error}
            for hid, h in self.handles.items()
        }
