from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CallbackTarget:
    kind: str
    path: str
    symbol: str


@dataclass
class HotOp:
    """
    Normalized hotcb operation routed through HotKernel.
    """

    module: str
    op: str
    id: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    target: Optional[CallbackTarget] = None
    init: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None
    mode: Optional[str] = None
    recipe_path: Optional[str] = None
    adjust_path: Optional[str] = None
    source: str = "external"
    raw: Optional[dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict for debugging or ledger payload usage."""
        out: Dict[str, Any] = {
            "module": self.module,
            "op": self.op,
            "id": self.id,
            "params": self.params,
            "init": self.init,
            "enabled": self.enabled,
            "mode": self.mode,
            "recipe_path": self.recipe_path,
            "adjust_path": self.adjust_path,
            "source": self.source,
        }
        if self.target is not None:
            out["target"] = {
                "kind": self.target.kind,
                "path": self.target.path,
                "symbol": self.target.symbol,
            }
        return {k: v for k, v in out.items() if v is not None}


def command_to_hotop(cmd: dict, default_module: str = "cb") -> HotOp:
    """
    Convert an external command record into HotOp.
    """
    module = str(cmd.get("module") or default_module)
    op = str(cmd.get("op"))
    cb_id = cmd.get("id")
    params = cmd.get("params")
    init = cmd.get("init")
    enabled = cmd.get("enabled")
    mode = cmd.get("mode")
    target = None
    if "target" in cmd and cmd.get("target") is not None:
        t = cmd["target"]
        target = CallbackTarget(kind=str(t["kind"]), path=str(t["path"]), symbol=str(t["symbol"]))

    recipe_path = cmd.get("recipe_path")
    adjust_path = cmd.get("adjust_path")

    return HotOp(
        module=module,
        op=op,
        id=str(cb_id) if cb_id is not None else None,
        params=params,
        init=init,
        enabled=enabled,
        target=target,
        mode=mode,
        recipe_path=recipe_path,
        adjust_path=adjust_path,
        raw=cmd,
        source="external",
    )
