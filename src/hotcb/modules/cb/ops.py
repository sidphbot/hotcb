# src/hotcb/ops.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .protocol import CallbackTarget


@dataclass
class Op:
    """
    Internal operation type applied by `HotController`.

    Instances of this dataclass represent a single mutation or instruction that
    the controller applies to its callback registry. Ops may come from:
      - desired-state config reconciliation (YAML), and/or
      - append-only command stream (JSONL) written by the CLI.

    Fields
    ------
    op:
        Operation name. Supported values:
          - "load":
              Ensure a callback exists. If not loaded, instantiate it using
              `target` + `init`. Can also set enabled state via `enabled`.
          - "enable":
              Mark callback enabled (it will receive events).
          - "disable":
              Mark callback disabled (it will not receive events).
          - "set_params":
              Hot-update parameters via callback.set_params(**params).
          - "unload" (optional):
              Disable callback and drop the instance from memory; if callback
              implements close(), it will be called.

    id:
        Callback identifier. Must match callback's `id` argument during init.
        The controller enforces id injection into init kwargs if absent.

    params:
        Used for "set_params". Dict of param names to values.
        Values can be any JSON-serializable types when coming from CLI JSONL.
        For YAML, values can be native YAML types.

    target:
        Used for "load". Points to the class location (file/module + symbol).

    init:
        Used for "load". Init kwargs used only when instantiating the callback.
        If the callback already exists, init is stored but not re-applied.

    enabled:
        Used for "load". If provided, sets enabled state as part of load.

    Example (JSONL command)
    -----------------------
    {"op":"set_params","id":"timing","params":{"every":10,"window":100}}
    """

    op: str
    id: str
    params: Optional[Dict[str, Any]] = None
    target: Optional[CallbackTarget] = None
    init: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None