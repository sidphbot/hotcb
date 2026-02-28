# tests/test_cli_parse_kv.py
from __future__ import annotations

import pytest

from hotcb.modules.cb.cli import _parse_kv


def test_parse_kv_basic_types():
    out = _parse_kv(["a=1", "b=2.5", "c=true", "d=false", "e=hello"])
    assert out["a"] == 1
    assert out["b"] == 2.5
    assert out["c"] is True
    assert out["d"] is False
    assert out["e"] == "hello"


def test_parse_kv_json_objects_arrays():
    out = _parse_kv(['meta={"x":1}', 'arr=[1,2,3]'])
    assert out["meta"] == {"x": 1}
    assert out["arr"] == [1, 2, 3]


def test_parse_kv_requires_equals():
    with pytest.raises(SystemExit):
        _parse_kv(["nope"])