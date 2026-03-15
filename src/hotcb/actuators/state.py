"""MutableState — container of HotcbActuator instances.

This is the user-facing runtime container.  The kernel holds one
``MutableState`` and routes all param-mutation ops through it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .actuator import ActuatorState, HotcbActuator, Mutation, _INIT_SENTINEL
from .base import ApplyResult


class MutableState:
    """Container of :class:`HotcbActuator` instances."""

    def __init__(self, actuators: List[HotcbActuator]) -> None:
        self._actuators: Dict[str, HotcbActuator] = {a.param_key: a for a in actuators}

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[HotcbActuator]:
        return self._actuators.get(key)

    def keys(self) -> List[str]:
        return list(self._actuators.keys())

    def __len__(self) -> int:
        return len(self._actuators)

    def __contains__(self, key: str) -> bool:
        return key in self._actuators

    # ------------------------------------------------------------------
    # Core mutation path
    # ------------------------------------------------------------------

    def apply(self, key: str, value: Any, env: dict, step: int) -> ApplyResult:
        """Validate -> apply_fn -> record mutation -> transition state."""
        act = self._actuators.get(key)
        if act is None:
            return ApplyResult(success=False, error=f"unknown_param:{key}")

        if act.state == ActuatorState.DISABLED:
            return ApplyResult(success=False, error=f"actuator_disabled:{key}")

        vr = act.validate(value)
        if not vr.valid:
            return ApplyResult(success=False, error="; ".join(vr.errors))

        old = act.current_value

        try:
            result = act.apply_fn(value, env)
        except Exception as exc:
            return ApplyResult(success=False, error=f"apply_fn_exception: {exc}")

        if result.success:
            act.current_value = value
            act.mutations.append(Mutation(step=step, old_value=old, new_value=value))
            act.last_changed_step = step
            act.state = ActuatorState.UNVERIFIED

        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, env: dict) -> None:
        """First-step initialization.

        For each actuator in INIT state, attempt to read its current value
        from the environment and transition to UNTOUCHED.
        """
        for act in self._actuators.values():
            if act.state != ActuatorState.INIT:
                continue
            # Best-effort: try to read via apply_fn's target
            # (the apply_fn closure usually captures the live object)
            # We don't call apply_fn here — just mark as ready.
            act.state = ActuatorState.UNTOUCHED

    def verify(self, key: str, metrics: dict) -> bool:
        """Check if ``metrics[act.metrics_dict_name]`` matches ``current_value``.

        Transitions UNVERIFIED -> VERIFIED if the metric matches.
        Returns True on successful verification.
        """
        act = self._actuators.get(key)
        if act is None:
            return False
        if act.state != ActuatorState.UNVERIFIED:
            return False
        if not act.metrics_dict_name:
            return False

        metric_val = metrics.get(act.metrics_dict_name)
        if metric_val is None:
            return False

        # For floats, use approximate comparison
        if isinstance(act.current_value, float) and isinstance(metric_val, (int, float)):
            if abs(act.current_value - metric_val) < 1e-9:
                act.state = ActuatorState.VERIFIED
                # Mark the latest mutation as verified
                if act.mutations:
                    act.mutations[-1].verified = True
                return True
        elif act.current_value == metric_val:
            act.state = ActuatorState.VERIFIED
            if act.mutations:
                act.mutations[-1].verified = True
            return True

        return False

    def disable(self, key: str) -> None:
        """Set actuator to DISABLED."""
        act = self._actuators.get(key)
        if act is not None:
            act.state = ActuatorState.DISABLED

    def enable(self, key: str) -> None:
        """Re-enable: DISABLED -> UNTOUCHED."""
        act = self._actuators.get(key)
        if act is not None and act.state == ActuatorState.DISABLED:
            act.state = ActuatorState.UNTOUCHED

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def snapshot_all(self) -> Dict[str, dict]:
        """Snapshot every actuator."""
        return {key: act.snapshot() for key, act in self._actuators.items()}

    def restore_all(self, snapshot: dict, env: dict) -> Dict[str, ApplyResult]:
        """Restore all actuators from snapshot."""
        results: Dict[str, ApplyResult] = {}
        for key, snap in snapshot.items():
            act = self._actuators.get(key)
            if act is None:
                results[key] = ApplyResult(success=False, error=f"unknown_param:{key}")
                continue
            value = snap.get("value")
            if value is _INIT_SENTINEL:
                # Nothing to restore — was never set
                results[key] = ApplyResult(success=True, detail={"skipped": "init_sentinel"})
                continue
            try:
                result = act.apply_fn(value, env)
                if result.success:
                    act.current_value = value
                    saved_state = snap.get("state")
                    if saved_state:
                        try:
                            act.state = ActuatorState(saved_state)
                        except ValueError:
                            pass
                results[key] = result
            except Exception as exc:
                results[key] = ApplyResult(success=False, error=f"restore_exception: {exc}")
        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def describe_all(self) -> List[dict]:
        """Return describe_space() for all non-DISABLED actuators."""
        return [
            act.describe_space()
            for act in self._actuators.values()
            if act.state != ActuatorState.DISABLED
        ]
