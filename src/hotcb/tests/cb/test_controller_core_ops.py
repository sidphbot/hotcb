# tests/test_controller_core_ops.py
from __future__ import annotations

import json

from hotcb.modules.cb import HotController

from .conftest import make_env


def test_enable_disable_unload_and_status(tmp_path):
    run_dir = tmp_path
    cfg = run_dir / "hotcb.yaml"
    cmds = run_dir / "hotcb.commands.jsonl"
    cfg.write_text("version: 1\ncallbacks: {}\n", encoding="utf-8")
    cmds.write_text("", encoding="utf-8")

    cbfile = run_dir / "cb.py"
    cbfile.write_text(
        """
class C:
    def __init__(self, id: str): self.id = id
    def set_params(self, **kwargs): pass
    def handle(self, event, env): (env.get("log") or print)(f"[{self.id}] {event}")
    def close(self): (print)(f"[{self.id}] closed")
""",
        encoding="utf-8",
    )

    def append_cmd(obj):
        with cmds.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    c = HotController(config_path=str(cfg), commands_path=str(cmds), debounce_steps=1)

    append_cmd({"op": "load", "id": "c1", "target": {"kind": "python_file", "path": str(cbfile), "symbol": "C"}, "init": {}, "enabled": True})
    logs = []
    c.apply(make_env(logs), events=["e1"])
    assert any("[c1] e1" in s for s in logs)

    append_cmd({"op": "disable", "id": "c1"})
    logs2 = []
    c.apply(make_env(logs2), events=["e2"])
    assert not any("[c1] e2" in s for s in logs2)

    append_cmd({"op": "enable", "id": "c1"})
    logs3 = []
    c.apply(make_env(logs3), events=["e3"])
    assert any("[c1] e3" in s for s in logs3)

    append_cmd({"op": "unload", "id": "c1"})
    logs4 = []
    c.apply(make_env(logs4), events=["e4"])

    st = c.status()
    assert st["c1"]["loaded"] is False
    assert st["c1"]["enabled"] is False