# tests/test_controller_param_mutation.py
from __future__ import annotations

import json

from hotcb.controller import HotController

from hotcb.tests.conftest import make_env


def test_param_mutation_runtime_and_deferred_apply(tmp_path):
    run_dir = tmp_path
    cfg = run_dir / "hotcb.yaml"
    cmds = run_dir / "hotcb.commands.jsonl"
    cfg.write_text("version: 1\ncallbacks: {}\n", encoding="utf-8")
    cmds.write_text("", encoding="utf-8")

    cbfile = run_dir / "cb.py"
    cbfile.write_text(
        """
class Every:
    def __init__(self, id: str, every: int = 10):
        self.id = id
        self.every = int(every)
        self.hit = 0
    def set_params(self, **kwargs):
        if "every" in kwargs:
            self.every = int(kwargs["every"])
    def handle(self, event, env):
        step = int(env.get("step", 0))
        if self.every > 0 and step % self.every == 0:
            self.hit += 1
            (env.get("log") or print)(f"[{self.id}] hit step={step} every={self.every} hit={self.hit}")
""",
        encoding="utf-8",
    )

    def append_cmd(obj):
        with cmds.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    c = HotController(config_path=str(cfg), commands_path=str(cmds), debounce_steps=1)

    # 1) set_params BEFORE load -> should be deferred into last_params
    append_cmd({"op": "set_params", "id": "e", "params": {"every": 2}})

    logs = []
    env = make_env(logs, step=1)
    c.apply(env, events=["train_step_end"])
    st = c.status()
    assert st["e"]["loaded"] is False
    assert st["e"]["last_params"]["every"] == 2

    # 2) load -> should apply last_params before first dispatch (controller best-effort)
    append_cmd({"op": "load", "id": "e", "target": {"kind": "python_file", "path": str(cbfile), "symbol": "Every"}, "init": {"every": 99}, "enabled": True})

    logs2 = []
    # step=2 is divisible by every=2, so it should hit immediately if deferred params apply
    env2 = make_env(logs2, step=2)
    c.apply(env2, events=["train_step_end"])
    assert any("every=2" in s for s in logs2)

    # 3) runtime mutation: change every to 3
    append_cmd({"op": "set_params", "id": "e", "params": {"every": 3}})

    logs3 = []
    env3 = make_env(logs3, step=3)
    c.apply(env3, events=["train_step_end"])
    assert any("every=3" in s for s in logs3)