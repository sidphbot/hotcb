# src/hotcb/config.py
from __future__ import annotations

from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from .ops import Op
from .protocol import CallbackTarget


class ConfigError(RuntimeError):
    """
    Raised when parsing or validating the desired-state configuration fails.

    Typical causes:
      - PyYAML not installed but YAML parsing requested,
      - malformed YAML syntax,
      - missing required keys (e.g., target.kind/path/symbol),
      - type mismatches (e.g., callbacks not a mapping).
    """
    pass


def _require_yaml() -> None:
    """
    Ensure PyYAML is installed, otherwise raise ConfigError with installation hint.

    This enables a clean packaging pattern:
      - core `hotcb` has zero dependencies,
      - YAML support is provided via extras: `pip install "hotcb[yaml]"`.

    Raises
    ------
    ConfigError
        If PyYAML is not installed.
    """
    if yaml is None:
        raise ConfigError(
            "PyYAML not installed. Install with: pip install 'hotcb[yaml]' "
            "or switch your config to JSON/commands-only control."
        )


def parse_yaml_config(path: str) -> List[Op]:
    """
    Parse a desired-state YAML config into a list of idempotent controller ops.

    This function implements "reconciliation": it describes the desired callback
    set and their parameters at a point in time. The controller may call this
    whenever the file changes (mtime increases) and then apply the returned ops.

    YAML schema (v1)
    ---------------
    version: 1
    callbacks:
      <callback_id>:
        enabled: true|false
        target:
          kind: "module" | "python_file"
          path: <module_path_or_file_path>
          symbol: <ClassName>
        init:   # only used when instantiating first time
          any_kw: any_value
        params: # applied via set_params repeatedly (hot updates)
          any_param: any_value

    Parameters
    ----------
    path:
        Filesystem path to YAML config file.

    Returns
    -------
    List[Op]
        A list of operations that, when applied in order, will:
          - ensure callbacks are loaded (load ops),
          - apply params (set_params ops),
          - enforce enabled state (enable/disable ops).

    Raises
    ------
    ConfigError
        If YAML support is unavailable or schema is invalid.

    Notes
    -----
    - Ops are generated in an idempotent manner; applying them repeatedly should
      converge to the same state.
    - This parser does not currently emit "disable callbacks not mentioned in YAML".
      If you want strict reconciliation, implement "prune" at controller level.

    Example
    -------
    >>> ops = parse_yaml_config("runs/exp1/hotcb.yaml")
    >>> for op in ops:
    ...     print(op.op, op.id)
    load timing
    set_params timing
    enable timing
    """
    _require_yaml()

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    callbacks: Dict[str, Any] = (cfg.get("callbacks") or {})
    if not isinstance(callbacks, dict):
        raise ConfigError("'callbacks' must be a mapping")

    ops: List[Op] = []

    for cb_id, spec in callbacks.items():
        if not isinstance(spec, dict):
            raise ConfigError(f"Callback '{cb_id}' spec must be a mapping")

        enabled = bool(spec.get("enabled", True))

        target_spec = spec.get("target")
        if not isinstance(target_spec, dict):
            raise ConfigError(f"Callback '{cb_id}' missing/invalid 'target' mapping")

        kind = target_spec.get("kind")
        path_ = target_spec.get("path")
        symbol = target_spec.get("symbol")
        if not (kind and path_ and symbol):
            raise ConfigError(f"Callback '{cb_id}' target requires kind/path/symbol")

        target = CallbackTarget(kind=str(kind), path=str(path_), symbol=str(symbol))

        init_kwargs = spec.get("init") or {}
        if not isinstance(init_kwargs, dict):
            raise ConfigError(f"Callback '{cb_id}' init must be a mapping")

        params = spec.get("params") or {}
        if not isinstance(params, dict):
            raise ConfigError(f"Callback '{cb_id}' params must be a mapping")

        ops.append(Op(op="load", id=cb_id, target=target, init=init_kwargs, enabled=enabled))
        if params:
            ops.append(Op(op="set_params", id=cb_id, params=params))
        ops.append(Op(op="enable" if enabled else "disable", id=cb_id))

    return ops