from __future__ import annotations

import os
from typing import List

from .ops import CallbackTarget, HotOp


def _load_yaml_file(path: str):
    import yaml  # type: ignore

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_yaml(path: str) -> List[HotOp]:
    """
    Parse hotcb.yaml desired-state and emit hotcb operations.

    This is intentionally permissive and only covers the v1 schema subset.
    """
    if not os.path.exists(path):
        return []

    try:
        data = _load_yaml_file(path)
    except ImportError:
        return []
    except Exception:
        return []

    ops: List[HotOp] = []

    core_cfg = data.get("core") or {}
    if core_cfg:
        mode = core_cfg.get("freeze_mode")
        if mode:
            ops.append(
                HotOp(
                    module="core",
                    op="freeze",
                    mode=str(mode),
                    recipe_path=core_cfg.get("replay", {}).get("recipe_path"),
                    adjust_path=core_cfg.get("replay", {}).get("adjust_path"),
                    source="yaml",
                )
            )

    cb_cfg = data.get("cb", {}).get("callbacks", {}) or {}
    for cb_id, spec in cb_cfg.items():
        target = None
        t = spec.get("target")
        if t:
            target = CallbackTarget(kind=str(t.get("kind")), path=str(t.get("path")), symbol=str(t.get("symbol")))
        init = spec.get("init") or {}
        enabled = spec.get("enabled")
        params = spec.get("params")
        ops.append(
            HotOp(
                module="cb",
                op="load",
                id=str(cb_id),
                target=target,
                init=init,
                enabled=enabled,
                source="yaml",
            )
        )
        if params:
            ops.append(HotOp(module="cb", op="set_params", id=str(cb_id), params=params, source="yaml"))

    opt_cfg = data.get("opt") or {}
    if opt_cfg:
        opt_enabled = opt_cfg.get("enabled", True)
        opt_id = str(opt_cfg.get("id", "main"))
        if not opt_enabled:
            ops.append(HotOp(module="opt", op="disable", id=opt_id, source="yaml"))
        else:
            params = opt_cfg.get("params") or {}
            ops.append(HotOp(module="opt", op="set_params", id=opt_id, params=params, source="yaml"))

    loss_cfg = data.get("loss") or {}
    if loss_cfg:
        loss_enabled = loss_cfg.get("enabled", True)
        loss_id = str(loss_cfg.get("id", "main"))
        if not loss_enabled:
            ops.append(HotOp(module="loss", op="disable", id=loss_id, source="yaml"))
        else:
            params = loss_cfg.get("params") or {}
            ops.append(HotOp(module="loss", op="set_params", id=loss_id, params=params, source="yaml"))

    tune_cfg = data.get("tune") or {}
    if tune_cfg:
        tune_enabled = tune_cfg.get("enabled", False)
        if tune_enabled:
            mode = tune_cfg.get("mode", "active")
            ops.append(HotOp(module="tune", op="enable", params={"mode": mode}, source="yaml"))
        else:
            ops.append(HotOp(module="tune", op="disable", source="yaml"))

    return ops
