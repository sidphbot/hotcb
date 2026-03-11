# src/hotcb/callbacks/utils.py
from __future__ import annotations

from typing import Any, Dict, Optional
import time


def get_log(env: Dict[str, Any]):
    """
    Return a logging callable from env, or fallback to built-in print.

    Parameters
    ----------
    env:
        Environment dict passed to callback.handle(). May include:
          - "log": callable(str) -> None

    Returns
    -------
    callable
        Logging function. Guaranteed callable.

    Example
    -------
    >>> log = get_log(env)
    >>> log("hello")
    """
    lf = env.get("log")
    return lf if callable(lf) else print


def now_s() -> float:
    """
    High-resolution monotonic timestamp in seconds (perf_counter).

    Returns
    -------
    float
        Seconds from an arbitrary point, suitable for durations.

    Example
    -------
    >>> t0 = now_s()
    >>> do_work()
    >>> dt = now_s() - t0
    """
    return time.perf_counter()


def get_in(obj: Any, path: str) -> Any:
    """
    Resolve a dotted path into nested dict/object structures.

    This helper is used by diagnostics callbacks to access values inside `env`
    using strings such as:
      - "loss"
      - "outputs.logits"
      - "batch.images"
      - "metrics.eval_loss"

    Parameters
    ----------
    obj:
        Root object (often the env dict).

    path:
        Dotted path string. Each segment tries:
          - dict lookup if current is a dict
          - attribute access otherwise

    Returns
    -------
    Any
        Resolved value or None if any segment is missing.

    Examples
    --------
    >>> env = {"outputs": {"logits": 123}}
    >>> get_in(env, "outputs.logits")
    123
    """
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur


def is_torch_tensor(x: Any) -> bool:
    """
    Check whether an object is a torch.Tensor, without hard depending on torch.

    Returns
    -------
    bool
        True if torch is importable and x is a torch.Tensor.

    Notes
    -----
    This function intentionally catches import errors to keep hotcb lightweight.
    """
    try:
        import torch
        return isinstance(x, torch.Tensor)
    except Exception:
        return False


def to_float(x: Any) -> Optional[float]:
    """
    Best-effort conversion to float.

    Parameters
    ----------
    x:
        Any object that might represent a numeric scalar.

    Returns
    -------
    Optional[float]
        float(x) if possible, else None.

    Examples
    --------
    >>> to_float(3)
    3.0
    >>> to_float("x") is None
    True
    """
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x: Any, default: int = 0) -> int:
    """
    Convert a value to int safely.

    Parameters
    ----------
    x:
        Any value.

    default:
        Returned if conversion fails.

    Returns
    -------
    int
        int(x) if possible else default.
    """
    try:
        return int(x)
    except Exception:
        return default


def safe_epoch(env: Dict[str, Any]) -> float:
    """
    Read epoch from env as float.

    Parameters
    ----------
    env:
        Environment dict; expected to have an "epoch" key (int or float).

    Returns
    -------
    float
        Epoch as float, else 0.0.
    """
    e = env.get("epoch", 0)
    try:
        return float(e)
    except Exception:
        return 0.0


def tensor_basic_stats(x: Any) -> Optional[Dict[str, Any]]:
    """
    Compute basic statistics for torch.Tensor or numpy-like arrays.

    Supported input types
    ---------------------
    1) torch.Tensor (preferred):
       - uses detach() to avoid autograd retention
       - counts NaNs/Infs
       - computes min/max/mean/std/l2 over finite entries

    2) numpy-like:
       - if numpy is available, coerces via np.asarray
       - counts NaNs/Infs for floating dtypes
       - computes min/max/mean/std/l2

    Parameters
    ----------
    x:
        Tensor/array-like object.

    Returns
    -------
    Optional[Dict[str, Any]]
        Stats dict if supported, else None.

    Returned keys (torch)
    ---------------------
    - shape: list[int]
    - dtype: str
    - device: str
    - nan: int
    - inf: int
    - numel: int
    - min/max/mean/std/l2: floats (if any finite values)

    Notes
    -----
    - "std" uses population std (unbiased=False) for torch.
    - For large tensors, this can still be expensive; use sampling with `every`
      or compute on reduced tensors in your callback.

    Example
    -------
    >>> import torch
    >>> tensor_basic_stats(torch.randn(2,3))
    {'shape':[2,3], 'dtype':'torch.float32', ...}
    """
    if is_torch_tensor(x):
        import torch
        t = x.detach()
        is_nan = torch.isnan(t)
        is_inf = torch.isinf(t)
        finite = t[~(is_nan | is_inf)]
        out: Dict[str, Any] = {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "nan": int(is_nan.sum().item()),
            "inf": int(is_inf.sum().item()),
            "numel": int(t.numel()),
        }
        if finite.numel() > 0:
            out.update({
                "min": float(finite.min().item()),
                "max": float(finite.max().item()),
                "mean": float(finite.mean().item()),
                "std": float(finite.std(unbiased=False).item()) if finite.numel() > 1 else 0.0,
                "l2": float(torch.linalg.vector_norm(finite).item()),
            })
        return out

    try:
        import numpy as np  # optional
        a = np.asarray(x)
        out = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "nan": int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0,
            "inf": int(np.isinf(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0,
            "numel": int(a.size),
        }
        if a.size > 0:
            finite = a[np.isfinite(a)] if np.issubdtype(a.dtype, np.floating) else a.reshape(-1)
            if finite.size > 0:
                out.update({
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                    "mean": float(finite.mean()),
                    "std": float(finite.std()) if finite.size > 1 else 0.0,
                    "l2": float(np.linalg.norm(finite)),
                })
        return out
    except Exception:
        return None