from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import logging

from .util import safe_mtime

_log = logging.getLogger("hotcb.freeze")
_VALID_FREEZE_MODES = {"off", "prod", "replay", "replay_adjusted"}


@dataclass
class FreezeState:
    """Kernel-level freeze/replay configuration."""

    mode: str = "off"
    recipe_path: Optional[str] = None
    adjust_path: Optional[str] = None
    policy: str = "best_effort"
    step_offset: int = 0
    _mtime: float = 0.0

    @classmethod
    def load(cls, path: str) -> "FreezeState":
        data = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return cls(mode="off", _mtime=0.0)
        except Exception:
            # On parse error, default to off but keep mtime to avoid loops.
            return cls(mode="off", _mtime=safe_mtime(path))

        mode = str(data.get("mode", "off"))
        if mode not in _VALID_FREEZE_MODES:
            _log.warning("Invalid freeze mode %r, defaulting to 'off'", mode)
            mode = "off"

        return cls(
            mode=mode,
            recipe_path=data.get("recipe_path"),
            adjust_path=data.get("adjust_path"),
            policy=str(data.get("policy", "best_effort")),
            step_offset=int(data.get("step_offset", 0) or 0),
            _mtime=safe_mtime(path),
        )
