from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional

from .controller import HotController
from .ops import Op as CallbackOp

from ...ops import HotOp
from ...util import ensure_dir
from ..result import ModuleResult


def _capture_source(target_path: str, out_dir: str) -> Optional[Dict[str, str]]:
    """Capture python_file bytes for deterministic replay."""
    try:
        with open(target_path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return None
    sha = hashlib.sha256(data).hexdigest()
    ensure_dir(out_dir)
    captured_path = os.path.join(out_dir, f"{sha}.py")
    if not os.path.exists(captured_path):
        with open(captured_path, "wb") as f:
            f.write(data)
    return {"sha256": sha, "captured_path": captured_path}


class CallbackModule:
    """
    Wrapper around the existing HotController to make it kernel-routable.
    """

    def __init__(
        self,
        auto_disable_on_error: bool = True,
        log_path: Optional[str] = None,
        source_capture_dir: Optional[str] = None,
    ) -> None:
        # The underlying controller will not poll files; kernel drives ops.
        self._controller = HotController(
            config_path="",
            commands_path=None,
            auto_disable_on_error=auto_disable_on_error,
            log_path=log_path,
        )
        self._source_capture_dir = source_capture_dir

    def apply_op(self, op: HotOp, env: dict) -> ModuleResult:
        cb_op = CallbackOp(
            op=op.op,
            id=op.id or "",
            params=op.params,
            init=op.init,
            target=op.target,
            enabled=op.enabled,
        )

        extra_payload: Dict[str, str] = {}
        if op.op == "load" and op.target is not None and op.target.kind == "python_file":
            if op.source == "replay" and op.raw and isinstance(op.raw.get("source_capture"), dict):
                captured_path = op.raw["source_capture"].get("captured_path")
                if captured_path and os.path.exists(captured_path):
                    cb_op.target.path = captured_path
                elif captured_path:
                    extra_payload["capture_missing_fallback"] = True
            elif self._source_capture_dir:
                capture = _capture_source(op.target.path, self._source_capture_dir)
                if capture:
                    extra_payload["source_capture"] = {
                        "sha256": capture["sha256"],
                        "captured_path": capture["captured_path"],
                    }
        try:
            self._controller.apply_op(cb_op, env)
            payload = op.params or {}
            payload.update(extra_payload)
            return ModuleResult(decision="applied", payload=payload or None)
        except Exception as e:  # pragma: no cover - defensive
            return ModuleResult(decision="failed", error=str(e), payload=extra_payload or None)

    def dispatch_events(self, events: List[str], env: dict) -> None:
        self._controller.dispatch_events(events, env)

    def status(self) -> Dict[str, dict]:
        return self._controller.status()
