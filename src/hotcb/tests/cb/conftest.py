# tests/conftest.py
from __future__ import annotations

from typing import Any, Dict, List


def make_env(logs: List[str] | None = None, **extra: Any) -> Dict[str, Any]:
    logs = logs if logs is not None else []

    def _log(s: str) -> None:
        logs.append(s)

    env: Dict[str, Any] = {"step": int(extra.get("step", 0)), "log": _log}
    env.update(extra)
    return env