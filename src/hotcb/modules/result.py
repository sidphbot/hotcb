from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModuleResult:
    """Result of applying an op within a module controller."""

    decision: str
    payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    notes: Optional[str] = None
    traceback: Optional[str] = None
