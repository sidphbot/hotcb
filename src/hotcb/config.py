from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# YAML is optional. If you prefer strict stdlib-only, switch to JSON.
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from .ops import Op
from .protocol import CallbackTarget


class ConfigError(RuntimeError):
    pass


def _require_yaml():
    if yaml is None:
        raise ConfigError(
            "PyYAML not installed. Install with: pip install pyyaml "
            "or switch config file to JSON in your implementation."
        )


def parse_yaml_config(path: str) -> List[Op]:
    """
    Parses a desired-state YAML file into ops.
    This is a RECONCILIATION source (not append-only).
    """
    _require_yaml()

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    callbacks: Dict[str, Any] = (cfg.get("callbacks") or {})
    ops: List[Op] = []

    for cb_id, spec in callbacks.items():
        enabled = bool(spec.get("enabled", True))

        target_spec = spec.get("target")
        if not target_spec:
            raise ConfigError(f"Callback '{cb_id}' missing 'target'")

        target = CallbackTarget(
            kind=str(target_spec.get("kind")),
            path=str(target_spec.get("path")),
            symbol=str(target_spec.get("symbol")),
        )

        init_kwargs = spec.get("init") or {}
        params = spec.get("params") or {}

        # Ensure it's loaded (idempotent)
        ops.append(Op(op="load", id=cb_id, target=target, init=init_kwargs, enabled=enabled))
        # Ensure params applied (idempotent)
        if params:
            ops.append(Op(op="set_params", id=cb_id, params=params))
        # Ensure enabled state (idempotent)
        ops.append(Op(op="enable" if enabled else "disable", id=cb_id))

    return ops