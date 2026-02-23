# src/hotcb/util.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple


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