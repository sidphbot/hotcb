from __future__ import annotations

import json
import os
from typing import Any, Dict

from .util import ensure_dir, now


def append_ledger(path: str, entry: Dict[str, Any]) -> None:
    """Append a ledger entry (JSONL)."""
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        if "wall_time" not in entry:
            entry["wall_time"] = now()
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
