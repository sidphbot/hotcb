from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


class HotCallback(Protocol):
    """
    Framework-agnostic callback protocol.

    Required:
      - handle(event, env): called by controller at adapter-defined safe points.
      - set_params(**kwargs): live updates
    Optional:
      - on_attach(env)
      - close()
    """

    id: str

    def handle(self, event: str, env: Dict[str, Any]) -> None: ...
    def set_params(self, **kwargs: Any) -> None: ...


@dataclass
class CallbackTarget:
    kind: str  # "python_file" | "module"
    path: str  # file path or module path
    symbol: str  # class name