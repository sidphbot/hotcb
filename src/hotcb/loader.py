# src/hotcb/loader.py
from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Any, Dict, Type

from .protocol import CallbackTarget


class CallbackLoadError(RuntimeError):
    """
    Raised when dynamic callback loading fails.

    Scenarios include:
      - module import failure,
      - file import failure,
      - missing symbol/class in module/file,
      - unsupported target.kind.

    This error is caught by `HotController` and typically results in:
      - a log message, and
      - the callback not being instantiated.
    """
    pass


def _dyn_module_name_for_file(file_path: str) -> str:
    """
    Deterministically derive the dynamic module name for a python_file target.
    """
    return f"hotcb_dyn_{abs(hash(file_path))}"


def _load_class_from_module(module_path: str, symbol: str) -> Type:
    """
    Import a class or callable from an importable Python module.

    Parameters
    ----------
    module_path:
        Importable module path (e.g., "hotcb.callbacks.timing").

    symbol:
        Attribute name to retrieve from imported module.

    Returns
    -------
    type
        The loaded class/type.

    Raises
    ------
    CallbackLoadError
        If the symbol does not exist on the module.

    Example
    -------
    >>> cls = _load_class_from_module("hotcb.callbacks.heartbeat", "HeartbeatCallback")
    >>> cb = cls(id="hb", every=10)
    """
    mod = importlib.import_module(module_path)
    try:
        return getattr(mod, symbol)
    except AttributeError as e:
        raise CallbackLoadError(f"Module '{module_path}' has no symbol '{symbol}'") from e


def _load_class_from_file(file_path: str, symbol: str, *, force_reload: bool = False) -> Type:
    """
    Load a class or callable from a Python source file (.py) at a filesystem path.
        This enables "hot loading" a brand-new callback file created during runtime.

    Parameters
    ----------
    file_path:
        Filesystem path to a Python file. May be absolute or relative.

    symbol:
        Attribute name to retrieve from the loaded module.

    Returns
    -------
    type
        The loaded class/type.
    If force_reload=True and the module has been loaded before, reload it to
    pick up code changes.
    """
    module_name = _dyn_module_name_for_file(file_path)

    if force_reload and module_name in sys.modules:
        # Reload the existing module (it must have a spec)
        mod = sys.modules[module_name]
        try:
            mod = importlib.reload(mod)
        except Exception as e:
            raise CallbackLoadError(f"Failed to reload file '{file_path}': {e}") from e
    else:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise CallbackLoadError(f"Cannot import file '{file_path}'")

        mod = importlib.util.module_from_spec(spec)
        # Critical for reload(): module must have __spec__ and be in sys.modules
        sys.modules[module_name] = mod
        try:
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        except Exception as e:
            raise CallbackLoadError(f"Cannot exec file '{file_path}': {e}") from e

    try:
        return getattr(mod, symbol)
    except AttributeError as e:
        raise CallbackLoadError(f"File '{file_path}' has no symbol '{symbol}'") from e


def load_callback_class(target: CallbackTarget, *, force_reload: bool = False) -> Type:
    """
    Load callback class specified by `CallbackTarget`.
    Parameters
    ----------
    target:
        CallbackTarget describing where to load from.

    Returns
    -------
    type
        Loaded callback class/type.

    Raises
    ------
    CallbackLoadError
        If target.kind is unsupported or loading fails.

    Supported target kinds
    ----------------------
    - "module": import from Python module path
    - "python_file": import from .py file path

    Example
    -------
    >>> target = CallbackTarget(kind="module", path="hotcb.callbacks.timing", symbol="TimingCallback")
    >>> cls = load_callback_class(target)
    force_reload:
      - module: uses importlib.reload(module) when True
      - python_file: reloads the derived dynamic module when True
    """
    if target.kind == "module":
        mod = importlib.import_module(target.path)
        if force_reload:
            mod = importlib.reload(mod)
        try:
            return getattr(mod, target.symbol)
        except AttributeError as e:
            raise CallbackLoadError(f"Module '{target.path}' has no symbol '{target.symbol}'") from e

    if target.kind == "python_file":
        return _load_class_from_file(target.path, target.symbol, force_reload=force_reload)

    raise CallbackLoadError(f"Unknown target.kind: {target.kind}")


def instantiate_callback(target: CallbackTarget, init_kwargs: Dict[str, Any], *, force_reload: bool = False) -> Any:
    """
    Instantiate a callback from a target + init kwargs.

    Parameters
    ----------
    target:
        Location of the callback class.

    init_kwargs:
        Keyword arguments passed to the callback constructor.
        Conventionally includes "id" (HotController injects it if absent).

    Returns
    -------
    Any
        Instantiated callback object.

    Raises
    ------
    CallbackLoadError
        If class loading fails.

    TypeError
        If constructor does not accept provided init_kwargs.

    Example
    -------
    >>> cb = instantiate_callback(
    ...   CallbackTarget("module", "hotcb.callbacks.heartbeat", "HeartbeatCallback"),
    ...   {"id": "hb", "every": 10}
    ... )
    """
    cls = load_callback_class(target, force_reload=force_reload)
    return cls(**init_kwargs)