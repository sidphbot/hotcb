from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class FileCursor:
    path: str
    offset: int = 0


def safe_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return 0.0


def read_new_jsonl(cursor: FileCursor, max_lines: int = 10_000) -> Tuple[List[dict], FileCursor]:
    """
    Append-only command stream. Reads new lines from cursor.offset.
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