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


def _load_class_from_file(file_path: str, symbol: str) -> Type:
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

    Raises
    ------
    CallbackLoadError
        If the file cannot be imported or the symbol is missing.

    Notes
    -----
    - A unique module name is derived from hash(file_path) to avoid collisions.
    - Re-loading the same path will reuse the same module name; Python's module
      caching will apply. If you want auto-reload-on-change, implement explicit
      reload semantics (not included in this baseline).

    Example
    -------
    Suppose /tmp/my_diag.py contains:
      class MyDiag: ...
    >>> cls = _load_class_from_file("/tmp/my_diag.py", "MyDiag")
    >>> cb = cls(id="my_diag")
    """
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
    """
    if target.kind == "module":
        return _load_class_from_module(target.path, target.symbol)
    if target.kind == "python_file":
        return _load_class_from_file(target.path, target.symbol)
    raise CallbackLoadError(f"Unknown target.kind: {target.kind}")


def instantiate_callback(target: CallbackTarget, init_kwargs: Dict[str, Any]) -> Any:
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
    cls = load_callback_class(target)
    return cls(**init_kwargs)