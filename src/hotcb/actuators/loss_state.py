from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from .base import ApplyResult, ValidationResult


class LossStateActuator:
    """
    Actuator for live loss-state mutations.

    Supports: set, mult, delta on named scalar keys within loss_state["weights"].
    """

    name: str = "loss"

    def __init__(
        self,
        global_bounds: tuple[float, float] = (0.0, 100.0),
        key_bounds: Optional[Dict[str, tuple[float, float]]] = None,
    ) -> None:
        self.global_bounds = global_bounds
        self.key_bounds: Dict[str, tuple[float, float]] = key_bounds or {}

    def _resolve_loss_state(self, env: dict) -> Optional[dict]:
        ls = env.get("loss_state")
        if ls is not None:
            return ls
        resolver = env.get("resolve_loss_state")
        if callable(resolver):
            try:
                return resolver()
            except Exception:
                return None
        return None

    def _get_bounds(self, key: str) -> tuple[float, float]:
        return self.key_bounds.get(key, self.global_bounds)

    def snapshot(self, env: dict) -> dict:
        ls = self._resolve_loss_state(env)
        if ls is None:
            return {}
        weights = ls.get("weights", {})
        return {"weights": copy.deepcopy(weights)}

    def validate(self, patch: dict, env: dict) -> ValidationResult:
        errors: List[str] = []
        op = patch.get("op")
        key = patch.get("key")
        value = patch.get("value")

        valid_ops = {"set", "mult", "delta"}
        if op not in valid_ops:
            errors.append(f"unknown op: {op}")
            return ValidationResult(valid=False, errors=errors)

        if key is None:
            errors.append("missing key")
        if value is None:
            errors.append("missing value")
        if not isinstance(value, (int, float)):
            errors.append(f"value must be numeric, got {type(value).__name__}")

        if errors:
            return ValidationResult(valid=False, errors=errors)

        ls = self._resolve_loss_state(env)
        if ls is not None:
            weights = ls.get("weights", {})
            bounds = self._get_bounds(key)
            if op == "set":
                if not (bounds[0] <= value <= bounds[1]):
                    errors.append(f"set value {value} out of bounds {bounds} for key {key}")
            elif op == "mult":
                if value <= 0:
                    errors.append(f"mult value must be positive, got {value}")
                current = weights.get(key, 1.0)
                result = current * value
                if not (bounds[0] <= result <= bounds[1]):
                    errors.append(f"mult would produce {result}, out of bounds {bounds} for key {key}")
            elif op == "delta":
                current = weights.get(key, 1.0)
                result = current + value
                if not (bounds[0] <= result <= bounds[1]):
                    errors.append(f"delta would produce {result}, out of bounds {bounds} for key {key}")

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def apply(self, patch: dict, env: dict) -> ApplyResult:
        ls = self._resolve_loss_state(env)
        if ls is None:
            return ApplyResult(success=False, error="missing_loss_state")

        op = patch.get("op")
        key = patch.get("key")
        value = patch.get("value")
        weights = ls.setdefault("weights", {})
        bounds = self._get_bounds(key)

        try:
            current = weights.get(key, 1.0)
            if op == "set":
                new_val = float(value)
            elif op == "mult":
                new_val = current * float(value)
            elif op == "delta":
                new_val = current + float(value)
            else:
                return ApplyResult(success=False, error=f"unknown op: {op}")

            new_val = max(bounds[0], min(bounds[1], new_val))
            weights[key] = new_val
            return ApplyResult(success=True, detail={"key": key, "old": current, "new": new_val})
        except Exception as e:
            return ApplyResult(success=False, error=str(e))

    def restore(self, snapshot: dict, env: dict) -> ApplyResult:
        ls = self._resolve_loss_state(env)
        if ls is None:
            return ApplyResult(success=False, error="missing_loss_state")

        saved_weights = snapshot.get("weights", {})
        try:
            ls["weights"] = copy.deepcopy(saved_weights)
            return ApplyResult(success=True)
        except Exception as e:
            return ApplyResult(success=False, error=str(e))

    def describe_space(self) -> dict:
        return {
            "actuator": self.name,
            "mutations": {
                "set": {"type": "float", "description": "Set weight to absolute value"},
                "mult": {"type": "float", "description": "Multiply weight by value"},
                "delta": {"type": "float", "description": "Add value to weight"},
            },
            "global_bounds": list(self.global_bounds),
            "key_bounds": {k: list(v) for k, v in self.key_bounds.items()},
        }
