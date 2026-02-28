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


def append_jsonl(path: str, obj: dict) -> None:
    """Append a single JSON object to a JSONL file (creates parent dirs)."""
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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


def now() -> float:
    """Epoch seconds helper."""
    return time.time()
