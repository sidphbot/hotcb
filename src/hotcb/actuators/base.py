from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)


@dataclass
class ApplyResult:
    success: bool
    error: Optional[str] = None
    detail: Optional[Dict[str, Any]] = None


@runtime_checkable
class BaseActuator(Protocol):
    name: str

    def snapshot(self, env: dict) -> dict:
        """Return minimal state needed for rollback."""
        ...

    def validate(self, patch: dict, env: dict) -> ValidationResult:
        """Check types, bounds, allowed phase, cooldown, reversibility."""
        ...

    def apply(self, patch: dict, env: dict) -> ApplyResult:
        """Apply mutation to the live object."""
        ...

    def restore(self, snapshot: dict, env: dict) -> ApplyResult:
        """Best-effort rollback to prior state."""
        ...

    def describe_space(self) -> dict:
        """Return the legal mutation schema for search and documentation."""
        ...
