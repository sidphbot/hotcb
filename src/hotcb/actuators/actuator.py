"""Unified actuator types for hotcb.

HotcbActuator is a single controllable parameter — every tunable scalar,
toggle, or choice becomes one instance.  ActuatorType drives validation and
UI control generation; ActuatorState tracks the lifecycle from registration
through mutation to verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional

from .base import ApplyResult, ValidationResult


class ActuatorType(Enum):
    BOOL = "bool"
    FLOAT = "float"
    INT = "int"
    CHOICE = "choice"
    LOG_FLOAT = "log_float"  # float on log scale (lr, wd)
    TUPLE = "tuple"          # e.g. betas


class ActuatorState(Enum):
    INIT = "init"               # registered but not yet observed
    UNTOUCHED = "untouched"     # observed initial value, no mutations applied
    UNVERIFIED = "unverified"   # mutation applied, not yet confirmed via metrics
    VERIFIED = "verified"       # mutation confirmed via metrics_dict_name
    DISABLED = "disabled"       # user-disabled or auto-disabled on error


@dataclass
class Mutation:
    step: int
    old_value: Any
    new_value: Any
    verified: bool = False


# Sentinel for initial value — distinguishes "never set" from None
_INIT_SENTINEL = object()


@dataclass
class HotcbActuator:
    """Single controllable parameter."""

    param_key: str                                       # unique key, e.g. "lr", "recon_w"
    type: ActuatorType                                   # drives UI control type
    apply_fn: Callable[[Any, dict], ApplyResult]         # (value, env) -> result
    metrics_dict_name: str = ""                          # metric name for verification
    label: str = ""                                      # display label
    group: str = ""                                      # UI grouping hint

    # Bounds (FLOAT, LOG_FLOAT, INT)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step_size: Optional[float] = None
    log_base: float = 10.0                               # for LOG_FLOAT

    # CHOICE type
    choices: Optional[list] = None

    # Mutable runtime state
    current_value: Any = field(default_factory=lambda: _INIT_SENTINEL)
    state: ActuatorState = field(default=ActuatorState.INIT)
    last_changed_step: int = -1
    mutations: List[Mutation] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, value: Any) -> ValidationResult:
        """Type-check and bounds-check a proposed value."""
        errors: List[str] = []

        if self.type == ActuatorType.BOOL:
            if not isinstance(value, bool):
                errors.append(f"expected bool, got {type(value).__name__}")

        elif self.type in (ActuatorType.FLOAT, ActuatorType.LOG_FLOAT):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"expected numeric, got {type(value).__name__}")
            else:
                if self.min_value is not None and value < self.min_value:
                    errors.append(f"{value} below min {self.min_value}")
                elif self.max_value is not None and value > self.max_value:
                    errors.append(f"{value} above max {self.max_value}")
                if self.type == ActuatorType.LOG_FLOAT and value <= 0:
                    errors.append(f"log_float must be positive, got {value}")

        elif self.type == ActuatorType.INT:
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"expected int, got {type(value).__name__}")
            else:
                if self.min_value is not None and value < self.min_value:
                    errors.append(f"{value} below min {self.min_value}")
                elif self.max_value is not None and value > self.max_value:
                    errors.append(f"{value} above max {self.max_value}")

        elif self.type == ActuatorType.CHOICE:
            if self.choices is not None and value not in self.choices:
                errors.append(f"{value!r} not in choices {self.choices}")

        elif self.type == ActuatorType.TUPLE:
            if not isinstance(value, (tuple, list)):
                errors.append(f"expected tuple/list, got {type(value).__name__}")

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    # ------------------------------------------------------------------
    # Snapshot / describe
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return state for rollback."""
        return {"value": self.current_value, "state": self.state.value}

    def describe_space(self) -> dict:
        """Return schema for tune search + UI generation."""
        return {
            "param_key": self.param_key,
            "type": self.type.value,
            "label": self.label or self.param_key,
            "group": self.group,
            "min": self.min_value,
            "max": self.max_value,
            "step": self.step_size,
            "log_base": self.log_base if self.type == ActuatorType.LOG_FLOAT else None,
            "choices": self.choices,
            "current": None if self.current_value is _INIT_SENTINEL else self.current_value,
            "state": self.state.value,
        }
