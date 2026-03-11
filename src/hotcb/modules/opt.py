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

    Multi-optimizer support
    -----------------------
    Commands can target a specific optimizer via ``params.opt_idx``
    (0-based index into ``env["optimizers"]``).  Defaults to 0.

    Scheduler coordination
    ----------------------
    When a scheduler is present in ``env["scheduler"]`` or
    ``env["schedulers"]``, setting ``lr`` also updates the scheduler's
    ``base_lrs`` so the scheduler scales from the new value instead of
    fighting with it.

    Grad clip
    ---------
    ``clip_norm`` is applied to ``trainer.gradient_clip_val`` (Lightning)
    or ``args.max_grad_norm`` (HF) when available.  If neither is
    present, it's stored as ``hotcb_clip_norm`` in the param group as
    advisory and the result notes ``clip_norm_advisory_only``.

    Actuator validation
    -------------------
    When an ``OptimizerActuator`` is registered via ``set_actuator()``,
    mutations are pre-validated against the actuator's bounds before
    applying.  Validation failures are returned as ``failed`` results
    without mutating state.
    """

    def __init__(self, auto_disable_on_error: bool = True) -> None:
        self.handles: Dict[str, OptHandle] = {}
        self.auto_disable_on_error = auto_disable_on_error
        self._actuator = None

    def set_actuator(self, actuator) -> None:
        """Register an OptimizerActuator for pre-apply validation."""
        self._actuator = actuator

    def _resolve_optimizer(self, env: dict, opt_idx: int = 0):
        """Resolve a specific optimizer by index.

        Falls back to env["optimizer"] (the default) if the optimizers
        list is not available or the index is out of range.
        """
        optimizers = env.get("optimizers")
        if isinstance(optimizers, (list, tuple)) and 0 <= opt_idx < len(optimizers):
            return optimizers[opt_idx]
        # Fallback: single optimizer
        if opt_idx == 0:
            opt = env.get("optimizer")
            if opt is not None:
                return opt
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

            params = dict(op.params or {})
            # Extract opt_idx (don't pass it to the optimizer)
            opt_idx = int(params.pop("opt_idx", 0))
            handle.last_params.update(params)

            opt = self._resolve_optimizer(env, opt_idx)
            if opt is None:
                num_opts = len(env.get("optimizers", []))
                error = f"missing_optimizer (opt_idx={opt_idx}"
                if num_opts:
                    error += f", {num_opts} optimizer(s) available"
                error += ")"
                return ModuleResult(decision="failed", error=error)

            # Pre-validate via actuator if registered
            if self._actuator is not None:
                try:
                    for act_op, act_val in self._actuator_patches(params):
                        vresult = self._actuator.validate(
                            {"op": act_op, "value": act_val, "opt_idx": opt_idx}, env,
                        )
                        if not vresult.valid:
                            return ModuleResult(
                                decision="failed",
                                error=f"validation: {'; '.join(vresult.errors)}",
                                notes="actuator_validation_failed",
                            )
                except Exception:
                    pass  # validation is best-effort, don't block if actuator errors

            try:
                notes = self._apply_params(opt, params, env, opt_idx)
                return ModuleResult(decision="applied", payload=params, notes=notes)
            except Exception as e:
                handle.last_error = str(e)
                if self.auto_disable_on_error:
                    handle.enabled = False
                return ModuleResult(decision="failed", error=str(e), traceback=tb_mod.format_exc())

        return ModuleResult(decision="ignored", notes=f"unknown_op:{op.op}")

    @staticmethod
    def _actuator_patches(params: dict):
        """Yield (actuator_op, value) pairs for actuator validation."""
        if "lr" in params and params["lr"] is not None:
            yield "lr_set", float(params["lr"])
        if "weight_decay" in params and params["weight_decay"] is not None:
            yield "wd_set", float(params["weight_decay"])
        if "betas" in params and params["betas"] is not None:
            yield "betas_set", params["betas"]

    def _apply_params(self, optimizer, params: dict, env: dict, opt_idx: int = 0) -> Optional[str]:
        """Mutate optimizer param groups. Returns optional notes string."""
        notes_parts: list[str] = []
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
            return None

        for g in groups:
            self._apply_params_to_group(g, params)

        # -- scheduler coordination --
        if "lr" in params and params["lr"] is not None:
            sched = self._resolve_scheduler(env, opt_idx)
            if sched is not None and hasattr(sched, "base_lrs"):
                new_lr = float(params["lr"])
                try:
                    for i in range(len(sched.base_lrs)):
                        sched.base_lrs[i] = new_lr
                    notes_parts.append("scheduler_base_lrs_updated")
                except Exception:
                    notes_parts.append("scheduler_base_lrs_update_failed")

        # -- grad clip wiring --
        if "clip_norm" in params and params["clip_norm"] is not None:
            clip_applied = self._apply_clip_norm(float(params["clip_norm"]), env)
            if not clip_applied:
                notes_parts.append("clip_norm_advisory_only")

        return "; ".join(notes_parts) if notes_parts else None

    def _apply_params_to_group(self, group: dict, params: dict) -> None:
        if "lr" in params and params["lr"] is not None:
            group["lr"] = float(params["lr"])
        if "weight_decay" in params and params["weight_decay"] is not None:
            group["weight_decay"] = float(params["weight_decay"])
        if "betas" in params and params["betas"] is not None:
            group["betas"] = tuple(float(b) for b in params["betas"])
        if "eps" in params and params["eps"] is not None:
            group["eps"] = float(params["eps"])
        if "scheduler_scale" in params and params["scheduler_scale"] is not None:
            group["lr"] = float(group.get("lr", 0.0)) * float(params["scheduler_scale"])
        if "scheduler_drop" in params and params["scheduler_drop"] is not None:
            group["lr"] = float(group.get("lr", 0.0)) * float(params["scheduler_drop"])
        if "clip_norm" in params and params["clip_norm"] is not None:
            group["hotcb_clip_norm"] = float(params["clip_norm"])

    def _resolve_scheduler(self, env: dict, opt_idx: int = 0):
        """Resolve the scheduler for a given optimizer index."""
        schedulers = env.get("schedulers")
        if isinstance(schedulers, (list, tuple)) and 0 <= opt_idx < len(schedulers):
            return schedulers[opt_idx]
        if opt_idx == 0:
            return env.get("scheduler")
        return None

    def _apply_clip_norm(self, clip_val: float, env: dict) -> bool:
        """Try to wire clip_norm into the framework. Returns True if wired."""
        # Lightning: trainer.gradient_clip_val
        trainer = env.get("trainer")
        if trainer is not None and hasattr(trainer, "gradient_clip_val"):
            try:
                trainer.gradient_clip_val = clip_val
                return True
            except Exception:
                pass
        # HF: args.max_grad_norm
        args = env.get("args")
        if args is not None and hasattr(args, "max_grad_norm"):
            try:
                args.max_grad_norm = clip_val
                return True
            except Exception:
                pass
        return False

    def status(self) -> Dict[str, dict]:
        return {
            hid: {"enabled": h.enabled, "last_params": dict(h.last_params), "last_error": h.last_error}
            for hid, h in self.handles.items()
        }
