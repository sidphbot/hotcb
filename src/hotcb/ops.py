from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
from .protocol import CallbackTarget


@dataclass
class Op:
    op: str  # enable|disable|set_params|load|unload
    id: str
    params: Optional[Dict[str, Any]] = None
    target: Optional[CallbackTarget] = None
    init: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None