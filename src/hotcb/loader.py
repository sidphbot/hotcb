from __future__ import annotations
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from typing import Any, Dict, Type

from .protocol import CallbackTarget


class CallbackLoadError(RuntimeError):
    pass


def _load_class_from_module(module_path: str, symbol: str) -> Type:
    mod = importlib.import_module(module_path)
    try:
        return getattr(mod, symbol)
    except AttributeError as e:
        raise CallbackLoadError(f"Module '{module_path}' has no symbol '{symbol}'") from e


def _load_class_from_file(file_path: str, symbol: str) -> Type:
    # Create a unique module name per path to avoid collisions
    module_name = f"hotcb_dyn_{abs(hash(file_path))}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise CallbackLoadError(f"Cannot import file '{file_path}'")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    try:
        return getattr(mod, symbol)
    except AttributeError as e:
        raise CallbackLoadError(f"File '{file_path}' has no symbol '{symbol}'") from e


def load_callback_class(target: CallbackTarget) -> Type:
    if target.kind == "module":
        return _load_class_from_module(target.path, target.symbol)
    if target.kind == "python_file":
        return _load_class_from_file(target.path, target.symbol)
    raise CallbackLoadError(f"Unknown target.kind: {target.kind}")


def instantiate_callback(target: CallbackTarget, init_kwargs: Dict[str, Any]) -> Any:
    cls = load_callback_class(target)
    return cls(**init_kwargs)