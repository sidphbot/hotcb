from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import ApplyResult, ValidationResult


class OptimizerActuator:
    """
    Actuator for live optimizer parameter mutations.

    Supports: lr_mult, lr_set, wd_mult, wd_set, betas_set.

    Multi-optimizer support
    -----------------------
    ``snapshot`` and ``restore`` capture/restore all optimizers in
    ``env["optimizers"]``.  ``apply`` targets a specific optimizer via
    ``patch["opt_idx"]`` (default 0).
    """

    name: str = "opt"

    def __init__(
        self,
        lr_bounds: tuple[float, float] = (1e-7, 1.0),
        wd_bounds: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.lr_bounds = lr_bounds
        self.wd_bounds = wd_bounds

    def _resolve_optimizer(self, env: dict, opt_idx: int = 0):
        optimizers = env.get("optimizers")
        if isinstance(optimizers, (list, tuple)) and 0 <= opt_idx < len(optimizers):
            return optimizers[opt_idx]
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

    def _resolve_all_optimizers(self, env: dict) -> list:
        optimizers = env.get("optimizers")
        if isinstance(optimizers, (list, tuple)) and optimizers:
            return list(optimizers)
        opt = env.get("optimizer")
        if opt is not None:
            return [opt]
        return []

    def snapshot(self, env: dict) -> dict:
        """Snapshot all optimizers for rollback."""
        all_opts = self._resolve_all_optimizers(env)
        if not all_opts:
            return {}
        all_groups = []
        for opt in all_opts:
            groups = []
            for g in opt.param_groups:
                snap = {"lr": g.get("lr")}
                if "weight_decay" in g:
                    snap["weight_decay"] = g["weight_decay"]
                if "betas" in g:
                    snap["betas"] = list(g["betas"])
                groups.append(snap)
            all_groups.append(groups)
        return {"all_groups": all_groups, "groups": all_groups[0] if all_groups else []}

    def validate(self, patch: dict, env: dict) -> ValidationResult:
        errors: List[str] = []
        op = patch.get("op")
        value = patch.get("value")

        valid_ops = {"lr_mult", "lr_set", "wd_mult", "wd_set", "betas_set"}
        if op not in valid_ops:
            errors.append(f"unknown op: {op}")
            return ValidationResult(valid=False, errors=errors)

        if value is None:
            errors.append("missing value")
            return ValidationResult(valid=False, errors=errors)

        # Validate opt_idx if specified
        opt_idx = patch.get("opt_idx", 0)
        all_opts = self._resolve_all_optimizers(env)
        if all_opts and opt_idx >= len(all_opts):
            errors.append(
                f"opt_idx={opt_idx} out of range "
                f"(only {len(all_opts)} optimizer(s) available)"
            )

        if op == "lr_mult":
            if not isinstance(value, (int, float)) or value <= 0:
                errors.append(f"lr_mult value must be positive number, got {value}")
        elif op == "lr_set":
            if not isinstance(value, (int, float)) or value <= 0:
                errors.append(f"lr_set value must be positive number, got {value}")
            elif not (self.lr_bounds[0] <= value <= self.lr_bounds[1]):
                errors.append(f"lr_set value {value} out of bounds {self.lr_bounds}")
        elif op == "wd_mult":
            if not isinstance(value, (int, float)) or value <= 0:
                errors.append(f"wd_mult value must be positive number, got {value}")
        elif op == "wd_set":
            if not isinstance(value, (int, float)) or value < 0:
                errors.append(f"wd_set value must be non-negative number, got {value}")
            elif not (self.wd_bounds[0] <= value <= self.wd_bounds[1]):
                errors.append(f"wd_set value {value} out of bounds {self.wd_bounds}")
        elif op == "betas_set":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                errors.append(f"betas_set expects [beta1, beta2], got {value}")
            else:
                for i, b in enumerate(value):
                    if not isinstance(b, (int, float)) or not (0.0 <= b < 1.0):
                        errors.append(f"beta{i+1} must be in [0, 1), got {b}")

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def apply(self, patch: dict, env: dict) -> ApplyResult:
        opt_idx = int(patch.get("opt_idx", 0))
        opt = self._resolve_optimizer(env, opt_idx)
        if opt is None:
            return ApplyResult(success=False, error=f"missing_optimizer (opt_idx={opt_idx})")

        op = patch.get("op")
        value = patch.get("value")

        try:
            for g in opt.param_groups:
                if op == "lr_mult":
                    new_lr = g["lr"] * float(value)
                    new_lr = max(self.lr_bounds[0], min(self.lr_bounds[1], new_lr))
                    g["lr"] = new_lr
                elif op == "lr_set":
                    g["lr"] = float(value)
                elif op == "wd_mult":
                    wd = g.get("weight_decay", 0.0)
                    new_wd = wd * float(value)
                    new_wd = max(self.wd_bounds[0], min(self.wd_bounds[1], new_wd))
                    g["weight_decay"] = new_wd
                elif op == "wd_set":
                    g["weight_decay"] = float(value)
                elif op == "betas_set":
                    g["betas"] = tuple(float(b) for b in value)
            return ApplyResult(success=True, detail=patch)
        except Exception as e:
            return ApplyResult(success=False, error=str(e))

    def restore(self, snapshot: dict, env: dict) -> ApplyResult:
        all_groups = snapshot.get("all_groups")
        if all_groups is None:
            # Backward compat: old snapshots have "groups" for optimizer 0
            groups = snapshot.get("groups", [])
            all_groups = [groups] if groups else []

        all_opts = self._resolve_all_optimizers(env)
        if not all_opts:
            return ApplyResult(success=False, error="missing_optimizer")

        try:
            for opt_i, groups_snap in enumerate(all_groups):
                if opt_i >= len(all_opts):
                    break
                opt = all_opts[opt_i]
                for i, snap in enumerate(groups_snap):
                    if i >= len(opt.param_groups):
                        break
                    g = opt.param_groups[i]
                    if "lr" in snap:
                        g["lr"] = snap["lr"]
                    if "weight_decay" in snap:
                        g["weight_decay"] = snap["weight_decay"]
                    if "betas" in snap:
                        g["betas"] = tuple(snap["betas"])
            return ApplyResult(success=True)
        except Exception as e:
            return ApplyResult(success=False, error=str(e))

    def describe_space(self) -> dict:
        return {
            "actuator": self.name,
            "mutations": {
                "lr_mult": {"type": "float", "description": "Multiplicative LR change"},
                "lr_set": {"type": "float", "bounds": list(self.lr_bounds)},
                "wd_mult": {"type": "float", "description": "Multiplicative weight decay change"},
                "wd_set": {"type": "float", "bounds": list(self.wd_bounds)},
                "betas_set": {"type": "list[float]", "length": 2, "element_bounds": [0.0, 1.0]},
            },
            "supports_opt_idx": True,
        }
