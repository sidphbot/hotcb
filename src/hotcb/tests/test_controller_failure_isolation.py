# tests/test_controller_failure_isolation.py
from __future__ import annotations

import json

from hotcb.controller import HotController

from .conftest import make_env


def test_failure_isolation_callback_crash_disables_only_it(tmp_path):
    run_dir = tmp_path
    cfg = run_dir / "hotcb.yaml"
    cmds = run_dir / "hotcb.commands.jsonl"
    cfg.write_text("version: 1\ncallbacks: {}\n", encoding="utf-8")
    cmds.write_text("", encoding="utf-8")

    # Create a callback file with two callbacks, one crashes
    cbfile = run_dir / "cbs.py"
    cbfile.write_text(
        """
class Good:
    def __init__(self, id: str):
        self.id = id
        self.count = 0
    def set_params(self, **kwargs): pass
    def handle(self, event, env):
        self.count += 1
        (env.get("log") or print)(f"[good] {self.count}")

class Bad:
    def __init__(self, id: str):
        self.id = id
    def set_params(self, **kwargs): pass
    def handle(self, event, env):
        raise RuntimeError("boom")
""",
        encoding="utf-8",
    )

    def append_cmd(obj):
        with cmds.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    append_cmd({"op": "load", "id": "good", "target": {"kind": "python_file", "path": str(cbfile), "symbol": "Good"}, "init": {}, "enabled": True})
    append_cmd({"op": "load", "id": "bad", "target": {"kind": "python_file", "path": str(cbfile), "symbol": "Bad"}, "init": {}, "enabled": True})

    logs = []
    env = make_env(logs)

    c = HotController(config_path=str(cfg), commands_path=str(cmds), debounce_steps=1, auto_disable_on_error=True)

    # first tick: loads + dispatch -> bad crashes, good still runs
    c.apply(env, events=["train_step_end"])

    st = c.status()
    assert st["good"]["enabled"] is True
    assert st["bad"]["enabled"] is False  # auto-disabled
    assert any("callback bad crashed" in s for s in logs)

    # second tick: good still runs, bad skipped
    logs2 = []
    env2 = make_env(logs2)
    c.apply(env2, events=["train_step_end"])
    assert any("[good] 2" in s for s in logs2)
    assert not any("bad crashed" in s for s in logs2)