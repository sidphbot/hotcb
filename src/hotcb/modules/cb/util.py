# src/hotcb/util.py
from __future__ import annotations
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple, Union


@dataclass
class FileCursor:
    """
    Tracks incremental read state for an append-only file (e.g., JSONL commands).

    This cursor makes it safe and efficient to "tail" a file and read only newly
    appended data, which is ideal for a live control plane.

    Attributes
    ----------
    path:
        The file path to read.

    offset:
        Byte offset from which to continue reading. After reading new lines,
        the cursor offset is updated to the file's current position.

    Notes
    -----
    - If the file is deleted and recreated, behavior depends on your usage.
      Most users keep the file stable; if you support rotation, consider
      resetting offset if file size shrinks below offset.
    """
    path: str
    offset: int = 0


def safe_mtime(path: str) -> float:
    """
    Return modification time for `path`, or 0.0 if the file does not exist.

    Parameters
    ----------
    path:
        Path to file.

    Returns
    -------
    float
        The POSIX mtime timestamp (seconds since epoch) if file exists, else 0.0.

    Typical usage
    -------------
    Used for "desired state" config file watching:
    - if mtime increases, reload and reconcile.

    Example
    -------
    >>> if safe_mtime("hotcb.yaml") > last_mtime:
    ...     reload_config()
    """
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return 0.0


def read_new_jsonl(cursor: FileCursor, max_lines: int = 10_000) -> Tuple[List[dict], FileCursor]:
    """
    Read newly appended JSON Lines from `cursor.path` starting at `cursor.offset`.

    Parameters
    ----------
    cursor:
        FileCursor containing:
          - path: JSONL file path
          - offset: where to start reading

    max_lines:
        Safety cap to prevent unbounded read if the producer floods the file.
        If more than max_lines are appended between polls, remaining lines will
        be read in subsequent polls.

    Returns
    -------
    (records, new_cursor):
        records:
            A list of decoded JSON objects (dicts).
        new_cursor:
            Updated cursor with new offset.

    Behavior
    --------
    - If the file doesn't exist, returns empty list and the same cursor.
    - Skips blank lines.
    - Raises JSON decode errors if a line is not valid JSON.

    Example
    -------
    >>> cursor = FileCursor("hotcb.commands.jsonl", 0)
    >>> cmds, cursor = read_new_jsonl(cursor)
    >>> for cmd in cmds:
    ...     print(cmd["op"], cmd["id"])
    """
    if not os.path.exists(cursor.path):
        return [], cursor

    out: List[dict] = []
    with open(cursor.path, "r", encoding="utf-8") as f:
        f.seek(cursor.offset)
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
        new_offset = f.tell()

    return out, FileCursor(path=cursor.path, offset=new_offset)


def _dedupe_keep_order(items: Iterable[Any]) -> List[Any]:
    """Deduplicate while preserving order, using object identity when possible."""
    out: List[Any] = []
    seen_ids: set[int] = set()
    for x in items:
        if x is None:
            continue
        try:
            k = id(x)
        except Exception:
            # Very defensive fallback (almost never hit)
            k = hash(repr(x))
        if k in seen_ids:
            continue
        seen_ids.add(k)
        out.append(x)
    return out


def iter_env_loggers(env: dict) -> List[Any]:
    """
    Best-effort extraction of logger-like objects from an `env` dict.

    This function does *not* impose an adapter contract. It simply tries common
    patterns seen across training frameworks and user code:
      - env["logger"], env["loggers"]
      - env["trainer"].logger / .loggers (Lightning-style)
      - env["callback"].trainer.logger / .loggers (rare, but seen)
      - env["run"], env["experiment"] (sometimes directly provided)

    Returns
    -------
    List[Any]
        A list of candidate logger objects. May be empty.
    """
    candidates: List[Any] = []

    # Direct env keys
    if "logger" in env:
        candidates.append(env.get("logger"))
    if "loggers" in env:
        ls = env.get("loggers")
        if isinstance(ls, (list, tuple)):
            candidates.extend(list(ls))
        elif ls is not None:
            candidates.append(ls)

    # Sometimes people pass these directly
    if "experiment" in env:
        candidates.append(env.get("experiment"))
    if "run" in env:
        candidates.append(env.get("run"))

    # Trainer-like object
    tr = env.get("trainer")
    if tr is not None:
        try:
            candidates.append(getattr(tr, "logger", None))
        except Exception:
            pass
        try:
            tloggers = getattr(tr, "loggers", None)
            if isinstance(tloggers, (list, tuple)):
                candidates.extend(list(tloggers))
            elif tloggers is not None:
                candidates.append(tloggers)
        except Exception:
            pass

    # Callback wrapper object (some frameworks keep trainer under callback)
    cb = env.get("callback")
    if cb is not None:
        try:
            tr2 = getattr(cb, "trainer", None)
            if tr2 is not None:
                candidates.append(getattr(tr2, "logger", None))
                tloggers2 = getattr(tr2, "loggers", None)
                if isinstance(tloggers2, (list, tuple)):
                    candidates.extend(list(tloggers2))
                elif tloggers2 is not None:
                    candidates.append(tloggers2)
        except Exception:
            pass

    # Flatten nested lists/tuples that snuck in
    flat: List[Any] = []
    for x in candidates:
        if x is None:
            continue
        if isinstance(x, (list, tuple)):
            flat.extend([y for y in x if y is not None])
        else:
            flat.append(x)

    return _dedupe_keep_order(flat)


def _looks_like_tb_writer(obj: Any) -> bool:
    # torch.utils.tensorboard.SummaryWriter, tensorboardX writer, etc.
    return hasattr(obj, "add_scalar") and callable(getattr(obj, "add_scalar", None))


def resolve_tensorboard_writer(env: dict) -> Optional[Any]:
    """
    Try to resolve a TensorBoard-like SummaryWriter from env/loggers.

    Known-class fast paths (when installed):
      - Lightning: lightning.pytorch.loggers.TensorBoardLogger -> .experiment
      - HF: transformers.integrations.TensorBoardCallback -> (tb_writer-like attr)
    Fallback: heuristic attribute checks.
    """
    for lg in iter_env_loggers(env):
        # ---- Lightning official logger
        if _isinstance_optional(lg, "lightning.pytorch.loggers.TensorBoardLogger"):
            try:
                exp = getattr(lg, "experiment", None)
                if exp is not None and _looks_like_tb_writer(exp):
                    return exp
            except Exception:
                pass

        # ---- HF integration callback object may appear in env (rare, but possible)
        if _isinstance_optional(lg, "transformers.integrations.TensorBoardCallback"):
            # HF TensorBoardCallback may hold a writer; we accept common names
            for attr in ("tb_writer", "_tb_writer", "writer", "_writer", "experiment"):
                try:
                    w = getattr(lg, attr, None)
                    if w is not None and _looks_like_tb_writer(w):
                        return w
                except Exception:
                    pass

        # ---- Heuristics (keep these!)
        if _looks_like_tb_writer(lg):
            return lg
        try:
            exp = getattr(lg, "experiment", None)
            if exp is not None and _looks_like_tb_writer(exp):
                return exp
        except Exception:
            pass
        try:
            w = getattr(lg, "writer", None)
            if w is not None and _looks_like_tb_writer(w):
                return w
        except Exception:
            pass

    return None


def _looks_like_mlflow_experiment(obj: Any) -> bool:
    # MLflow client / experiment handle: commonly has log_metric
    return hasattr(obj, "log_metric") and callable(getattr(obj, "log_metric", None))


def resolve_mlflow(env: dict) -> Optional[Tuple[Any, Optional[str]]]:
    """
    Try to resolve (mlflow_experiment_or_client, run_id).

    Known-class fast paths:
      - Lightning: lightning.pytorch.loggers.MLFlowLogger -> (.experiment, .run_id)
      - HF: transformers.integrations.MLflowCallback (best-effort)
    Fallback: heuristic shape checks.
    """
    v = env.get("mlflow")
    if isinstance(v, tuple) and len(v) == 2:
        client, run_id = v
        if _looks_like_mlflow_experiment(client):
            return client, (str(run_id) if run_id is not None else None)

    for lg in iter_env_loggers(env):
        # Lightning MLFlowLogger
        if _isinstance_optional(lg, "lightning.pytorch.loggers.MLFlowLogger"):
            try:
                exp = getattr(lg, "experiment", None)
                run_id = getattr(lg, "run_id", None)
                if exp is not None and _looks_like_mlflow_experiment(exp):
                    return exp, (str(run_id) if run_id is not None else None)
            except Exception:
                pass

        # HF MLflowCallback (if someone sticks callbacks into env)
        if _isinstance_optional(lg, "transformers.integrations.MLflowCallback"):
            # there isn't a universally stable attribute contract; try common ones
            for exp_attr in ("client", "mlflow_client", "experiment", "_mlflow_client"):
                try:
                    exp = getattr(lg, exp_attr, None)
                    if exp is not None and _looks_like_mlflow_experiment(exp):
                        run_id = getattr(lg, "run_id", None)
                        return exp, (str(run_id) if run_id is not None else None)
                except Exception:
                    pass

        # Heuristic fallback
        try:
            exp = getattr(lg, "experiment", None)
            run_id = getattr(lg, "run_id", None)
            if exp is not None and _looks_like_mlflow_experiment(exp):
                return exp, (str(run_id) if run_id is not None else None)
        except Exception:
            pass

    return None


def _looks_like_comet_experiment(obj: Any) -> bool:
    # Comet Experiment commonly has log_metric (and many others)
    return hasattr(obj, "log_metric") and callable(getattr(obj, "log_metric", None))


def resolve_comet_experiment(env: dict) -> Optional[Any]:
    """
    Try to resolve a Comet experiment object.

    Known-class fast paths:
      - Lightning: lightning.pytorch.loggers.CometLogger -> .experiment
      - HF: transformers.integrations.CometCallback (best-effort)
    Fallback: heuristic + comet-hints.
    """
    ce = env.get("comet_experiment")
    if ce is not None and _looks_like_comet_experiment(ce):
        return ce

    for lg in iter_env_loggers(env):
        # Lightning CometLogger
        if _isinstance_optional(lg, "lightning.pytorch.loggers.CometLogger"):
            try:
                exp = getattr(lg, "experiment", None)
                if exp is not None and _looks_like_comet_experiment(exp):
                    return exp
            except Exception:
                pass

        # HF CometCallback (if present)
        if _isinstance_optional(lg, "transformers.integrations.CometCallback"):
            for attr in ("experiment", "_experiment", "comet_experiment"):
                try:
                    exp = getattr(lg, attr, None)
                    if exp is not None and _looks_like_comet_experiment(exp):
                        return exp
                except Exception:
                    pass

        # Existing heuristic behavior (keep)
        if _looks_like_comet_experiment(lg):
            name = (lg.__class__.__name__ or "").lower()
            mod = (getattr(lg.__class__, "__module__", "") or "").lower()
            if "comet" in name or "comet" in mod:
                return lg

        try:
            exp = getattr(lg, "experiment", None)
            if exp is not None and _looks_like_comet_experiment(exp):
                name = (exp.__class__.__name__ or "").lower()
                mod = (getattr(exp.__class__, "__module__", "") or "").lower()
                if "comet" in name or "comet" in mod:
                    return exp
        except Exception:
            pass

    return None


def log_scalar(env: dict, key: str, value: Union[int, float], step: Optional[int] = None) -> bool:
    """
    Best-effort scalar logging to any available backend (TB, MLflow, Comet).

    Returns
    -------
    bool
        True if logging succeeded to at least one backend, else False.
    """
    ok = False

    # TensorBoard-like
    w = resolve_tensorboard_writer(env)
    if w is not None:
        try:
            if step is None:
                step = int(env.get("step", 0))
            w.add_scalar(key, value, step)
            ok = True
        except Exception:
            pass

    # MLflow-like
    ml = resolve_mlflow(env)
    if ml is not None:
        exp, run_id = ml
        try:
            if run_id is not None:
                # Typical mlflow client signature: log_metric(run_id, key, value, step=?)
                if step is None:
                    step = int(env.get("step", 0))
                try:
                    exp.log_metric(run_id, key, float(value), step=step)
                except TypeError:
                    # Some clients may not accept step kw
                    exp.log_metric(run_id, key, float(value), step)
                ok = True
        except Exception:
            pass

    # Comet-like
    ce = resolve_comet_experiment(env)
    if ce is not None:
        try:
            if step is None:
                step = int(env.get("step", 0))
            try:
                ce.log_metric(key, float(value), step=step)
            except TypeError:
                ce.log_metric(key, float(value), step)
            ok = True
        except Exception:
            pass

    return ok


def _optional_import(path: str):
    """
    Import `path` like 'a.b.c:Thing' or 'a.b.c.Thing' safely.
    Returns imported object or None.
    """
    try:
        if ":" in path:
            mod, sym = path.split(":", 1)
        else:
            mod, sym = path.rsplit(".", 1)
        m = __import__(mod, fromlist=[sym])
        return getattr(m, sym, None)
    except Exception:
        return None


def _isinstance_optional(obj: Any, type_path: str) -> bool:
    t = _optional_import(type_path)
    if t is None:
        return False
    try:
        return isinstance(obj, t)
    except Exception:
        return False
