# tests/test_util_jsonl_cursor.py
from __future__ import annotations

import json

from hotcb.util import FileCursor, read_new_jsonl


def test_read_new_jsonl_incremental(tmp_path):
    p = tmp_path / "cmds.jsonl"
    p.write_text("", encoding="utf-8")

    cur = FileCursor(path=str(p), offset=0)
    recs, cur = read_new_jsonl(cur)
    assert recs == []

    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"op": "enable", "id": "a"}) + "\n")
        f.write(json.dumps({"op": "disable", "id": "b"}) + "\n")

    recs, cur2 = read_new_jsonl(cur)
    assert [r["op"] for r in recs] == ["enable", "disable"]
    assert cur2.offset > cur.offset

    # nothing new
    recs, cur3 = read_new_jsonl(cur2)
    assert recs == []
    assert cur3.offset == cur2.offset