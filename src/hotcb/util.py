from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple


@dataclass
class FileCursor:
    """
    Tracks incremental read state for an append-only file (JSONL).
    """

    path: str
    offset: int = 0


def ensure_dir(path: str) -> None:
    """mkdir -p helper."""
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def safe_mtime(path: str) -> float:
    """Return mtime or 0.0 if missing."""
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return 0.0


def read_new_jsonl(cursor: FileCursor, max_lines: int = 10_000) -> Tuple[List[dict], FileCursor]:
    """
    Read newly appended JSONL records starting from cursor.offset.
    """
    if not os.path.exists(cursor.path):
        return [], cursor

    # Detect file truncation (reset)
    file_size = os.path.getsize(cursor.path)
    effective_offset = cursor.offset
    if file_size < effective_offset:
        effective_offset = 0  # file was truncated, start from beginning

    out: List[dict] = []
    with open(cursor.path, "r", encoding="utf-8") as f:
        f.seek(effective_offset)
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError:
                continue
        new_offset = f.tell()
    return out, FileCursor(path=cursor.path, offset=new_offset)


def append_jsonl(path: str, obj: dict) -> None:
    """Append a single JSON object as a line to a JSONL file (with file locking)."""
    import fcntl
    ensure_dir(os.path.dirname(path))
    line = json.dumps(sanitize_floats(obj), ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def dedupe_keep_order(items: Iterable[Any]) -> List[Any]:
    """Deduplicate while preserving order."""
    out: List[Any] = []
    seen_ids: set[int] = set()
    for x in items:
        if x is None:
            continue
        k = id(x)
        if k in seen_ids:
            continue
        seen_ids.add(k)
        out.append(x)
    return out


def sanitize_floats(obj: Any) -> Any:
    """Replace NaN/inf/-inf with None recursively in dicts/lists for JSON safety."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_floats(v) for v in obj]
    return obj


def now() -> float:
    """Epoch seconds helper."""
    return time.time()
