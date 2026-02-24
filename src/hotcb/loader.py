# src/hotcb/loader.py
from __future__ import annotations

import hashlib
import importlib
import os
import sys
from types import ModuleType
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

    Notes
    -----
    We use a stable digest of the absolute path (not Python's built-in `hash()`),
    because `hash()` is randomized per-process by default.

    Parameters
    ----------
    file_path:
        Filesystem path to a Python file.

    Returns
    -------
    str
        Stable module name to store in `sys.modules`.
    """
    norm = os.path.abspath(file_path)
    h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
    return f"hotcb_dyn_{h}"


def _exec_file_module(module_name: str, file_path: str) -> ModuleType:
    """
    Execute a Python source file into a fresh module object and register it.

    Critical guardrail
    ------------------
    This function intentionally executes from **source text** using `compile+exec`
    instead of `importlib` loaders. This avoids any possibility of stale execution
    from `.pyc` caches or loader-level reuse during rapid edits (common in hot-reload).

    Parameters
    ----------
    module_name:
        Name used for `sys.modules[module_name]`.

    file_path:
        Path to the `.py` file.

    Returns
    -------
    types.ModuleType
        The executed module.

    Raises
    ------
    CallbackLoadError
        If reading, compiling, or executing the file fails.
    """
    mod = ModuleType(module_name)
    mod.__file__ = file_path
    mod.__package__ = ""  # not part of a package
    mod.__loader__ = None
    mod.__spec__ = None

    # Publish before exec so the module is visible during execution if needed.
    sys.modules[module_name] = mod

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise CallbackLoadError(f"Cannot read file '{file_path}': {e}") from e

    try:
        code = compile(src, file_path, "exec")
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise CallbackLoadError(f"Cannot compile file '{file_path}': {e}") from e

    try:
        exec(code, mod.__dict__, mod.__dict__)
    except Exception as e:
        # Remove partially loaded module to prevent poisoning later loads.
        sys.modules.pop(module_name, None)
        raise CallbackLoadError(f"Cannot exec file '{file_path}': {e}") from e

    return mod


def _load_class_from_module(module_path: str, symbol: str, *, force_reload: bool = False) -> Type:
    """
    Load a class or callable from an importable Python module.

    Parameters
    ----------
    module_path:
        Importable module path (e.g., "hotcb.callbacks.timing").

    symbol:
        Attribute name to retrieve from imported module.

    force_reload:
        If True, reload the module with `importlib.reload()` before resolving `symbol`.

    Returns
    -------
    type
        The loaded class/type.

    Raises
    ------
    CallbackLoadError
        If import fails or the symbol is missing.
    """
    try:
        mod = importlib.import_module(module_path)
        if force_reload:
            mod = importlib.reload(mod)
    except Exception as e:
        raise CallbackLoadError(f"Cannot import module '{module_path}': {e}") from e

    try:
        return getattr(mod, symbol)
    except AttributeError as e:
        raise CallbackLoadError(f"Module '{module_path}' has no symbol '{symbol}'") from e


def _load_class_from_file(file_path: str, symbol: str, *, force_reload: bool = False) -> Type:
    """
    Load a class or callable from a Python source file (.py) at a filesystem path.

    Semantics
    ---------
    - If `force_reload=True`, this function **always** re-executes the file from
      source text into a fresh module object and replaces `sys.modules` entry.
    - We do not use `importlib.reload()` for file targets.

    Parameters
    ----------
    file_path:
        Filesystem path to a Python file.

    symbol:
        Attribute name to retrieve from the loaded module.

    force_reload:
        If True, discard any cached module and execute the latest file contents.

    Returns
    -------
    type
        The loaded class/type.

    Raises
    ------
    CallbackLoadError
        If execution fails or symbol is missing.
    """
    module_name = _dyn_module_name_for_file(file_path)

    if force_reload:
        # Critical: invalidate cache so code changes are reflected.
        sys.modules.pop(module_name, None)

    mod = sys.modules.get(module_name)
    if mod is None:
        mod = _exec_file_module(module_name, file_path)

    try:
        return getattr(mod, symbol)
    except AttributeError as e:
        raise CallbackLoadError(f"File '{file_path}' has no symbol '{symbol}'") from e


def load_callback_class(target: CallbackTarget, *, force_reload: bool = False) -> Type:
    """
    Load the callback class specified by a `CallbackTarget`.

    Parameters
    ----------
    target:
        CallbackTarget describing where to load from.

    force_reload:
        If True, attempt to reload the underlying code before resolving the symbol.

        - For `target.kind == "module"`: uses `importlib.reload(module)`
        - For `target.kind == "python_file"`: re-executes from source text

    Returns
    -------
    type
        Loaded callback class/type.

    Raises
    ------
    CallbackLoadError
        If target.kind is unsupported or loading fails.
    """
    if target.kind == "module":
        return _load_class_from_module(target.path, target.symbol, force_reload=force_reload)

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

    force_reload:
        If True, reload/re-exec code before instantiation.

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
    """
    cls = load_callback_class(target, force_reload=force_reload)
    return cls(**init_kwargs)