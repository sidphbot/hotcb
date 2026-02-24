# tests/test_controller_autoreload_file.py
from __future__ import annotations

import json
import os
import time

from hotcb.controller import HotController

from .conftest import make_env


def _write_cb(path, version: str):
    path.write_text(
        f"""
class Versioned:
    def __init__(self, id: str, tag: str = "x"):
        self.id = id
        self.tag = tag
        self.version = "{version}"
    def set_params(self, **kwargs):
        if "tag" in kwargs:
            self.tag = str(kwargs["tag"])
    def handle(self, event, env):
        (env.get("log") or print)(f"[{{self.id}}] v={{self.version}} tag={{self.tag}} event={{event}}")
""",
        encoding="utf-8",
    )


def test_python_file_auto_reload_on_modify(tmp_path):
    run_dir = tmp_path
    cfg = run_dir / "hotcb.yaml"
    cmds = run_dir / "hotcb.commands.jsonl"
    cfg.write_text("version: 1\ncallbacks: {}\n", encoding="utf-8")
    cmds.write_text("", encoding="utf-8")

    cbfile = run_dir / "cb.py"
    _write_cb(cbfile, "v1")

    def append_cmd(obj):
        with cmds.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    c = HotController(config_path=str(cfg), commands_path=str(cmds), debounce_steps=1)

    append_cmd({"op": "load", "id": "vv", "target": {"kind": "python_file", "path": str(cbfile), "symbol": "Versioned"}, "init": {"tag": "A"}, "enabled": True})

    logs = []
    env = make_env(logs, step=1)
    c.apply(env, events=["train_step_end"])
    assert any("v=v1" in s and "tag=A" in s for s in logs)

    # mutate params to verify they survive reload
    append_cmd({"op": "set_params", "id": "vv", "params": {"tag": "B"}})
    logs2 = []
    env2 = make_env(logs2, step=2)
    c.apply(env2, events=["train_step_end"])
    assert any("v=v1" in s and "tag=B" in s for s in logs2)

    # Modify the file to v2 and bump mtime deterministically
    # (some filesystems have coarse mtime; ensure it advances)
    time.sleep(0.02)
    _write_cb(cbfile, "v2")
    os.utime(cbfile, None)

    logs3 = []
    env3 = make_env(logs3, step=3)
    c.apply(env3, events=["train_step_end"])

    # After reload, we should see v2 and still tag=B
    assert any("v=v2" in s and "tag=B" in s for s in logs3), logs3