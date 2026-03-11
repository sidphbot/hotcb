"""Tests for hotcb.server.api — REST command endpoints."""
from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from starlette.testclient import TestClient

from hotcb.server.app import create_app


@pytest.fixture()
def run_dir(tmp_path):
    """Create a temporary run directory."""
    d = str(tmp_path / "run")
    os.makedirs(d)
    return d


@pytest.fixture()
def client(run_dir):
    """TestClient wired to a fresh app with a tmp run_dir."""
    app = create_app(run_dir)
    return TestClient(app, raise_server_exceptions=True)


def _read_commands(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "hotcb.commands.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _read_recipe(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "hotcb.recipe.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


# ---- /api/opt/set ----

class TestOptSet:
    def test_basic(self, client, run_dir):
        resp = client.post("/api/opt/set", json={"params": {"lr": 0.001}})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "queued"
        assert body["command"]["module"] == "opt"
        assert body["command"]["op"] == "set_params"
        assert body["command"]["params"]["lr"] == 0.001
        cmds = _read_commands(run_dir)
        assert len(cmds) == 1
        assert cmds[0]["params"]["lr"] == 0.001

    def test_custom_id(self, client, run_dir):
        resp = client.post("/api/opt/set", json={"id": "aux", "params": {"lr": 0.01}})
        assert resp.status_code == 200
        cmds = _read_commands(run_dir)
        assert cmds[0]["id"] == "aux"

    def test_multiple_params(self, client, run_dir):
        resp = client.post("/api/opt/set", json={"params": {"lr": 0.001, "weight_decay": 0.01}})
        assert resp.status_code == 200
        cmds = _read_commands(run_dir)
        assert cmds[0]["params"]["weight_decay"] == 0.01

    def test_empty_params_rejected(self, client):
        resp = client.post("/api/opt/set", json={"params": {}})
        assert resp.status_code == 422

    def test_missing_params_rejected(self, client):
        resp = client.post("/api/opt/set", json={})
        assert resp.status_code == 422


# ---- /api/loss/set ----

class TestLossSet:
    def test_basic(self, client, run_dir):
        resp = client.post("/api/loss/set", json={"params": {"recon_w": 0.5}})
        assert resp.status_code == 200
        body = resp.json()
        assert body["command"]["module"] == "loss"
        assert body["command"]["op"] == "set_params"
        cmds = _read_commands(run_dir)
        assert cmds[0]["params"]["recon_w"] == 0.5

    def test_empty_params_rejected(self, client):
        resp = client.post("/api/loss/set", json={"params": {}})
        assert resp.status_code == 422


# ---- /api/tune/mode ----

class TestTuneMode:
    def test_enable_active(self, client, run_dir):
        resp = client.post("/api/tune/mode", json={"mode": "active"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["command"]["op"] == "enable"
        assert body["command"]["params"]["mode"] == "active"
        cmds = _read_commands(run_dir)
        assert cmds[0]["module"] == "tune"

    def test_enable_observe(self, client, run_dir):
        resp = client.post("/api/tune/mode", json={"mode": "observe"})
        assert resp.status_code == 200
        assert resp.json()["command"]["params"]["mode"] == "observe"

    def test_disable(self, client, run_dir):
        resp = client.post("/api/tune/mode", json={"mode": "off"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["command"]["op"] == "disable"
        assert "params" not in body["command"]

    def test_invalid_mode(self, client):
        resp = client.post("/api/tune/mode", json={"mode": "bogus"})
        assert resp.status_code == 422


# ---- /api/cb/{cb_id}/enable & disable ----

class TestCbEnableDisable:
    def test_enable(self, client, run_dir):
        resp = client.post("/api/cb/my_cb/enable")
        assert resp.status_code == 200
        body = resp.json()
        assert body["command"] == {"module": "cb", "op": "enable", "id": "my_cb"}
        cmds = _read_commands(run_dir)
        assert cmds[0]["id"] == "my_cb"

    def test_disable(self, client, run_dir):
        resp = client.post("/api/cb/grad_clip/disable")
        assert resp.status_code == 200
        body = resp.json()
        assert body["command"]["op"] == "disable"
        assert body["command"]["id"] == "grad_clip"

    def test_both_written(self, client, run_dir):
        client.post("/api/cb/a/enable")
        client.post("/api/cb/b/disable")
        cmds = _read_commands(run_dir)
        assert len(cmds) == 2
        assert cmds[0]["op"] == "enable"
        assert cmds[1]["op"] == "disable"


# ---- /api/freeze ----

class TestFreeze:
    def test_prod_mode(self, client, run_dir):
        resp = client.post("/api/freeze", json={"mode": "prod"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "written"
        assert body["config"]["mode"] == "prod"
        # Check file was written
        freeze_path = os.path.join(run_dir, "hotcb.freeze.json")
        assert os.path.exists(freeze_path)
        with open(freeze_path) as f:
            cfg = json.load(f)
        assert cfg["mode"] == "prod"

    def test_off_mode(self, client, run_dir):
        resp = client.post("/api/freeze", json={"mode": "off"})
        assert resp.status_code == 200

    def test_replay_with_recipe(self, client, run_dir):
        resp = client.post("/api/freeze", json={
            "mode": "replay",
            "recipe_path": "/tmp/recipe.jsonl",
            "policy": "strict",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["config"]["recipe_path"] == "/tmp/recipe.jsonl"
        assert body["config"]["policy"] == "strict"

    def test_invalid_mode(self, client):
        resp = client.post("/api/freeze", json={"mode": "invalid"})
        assert resp.status_code == 422


# ---- /api/schedule ----

class TestSchedule:
    def test_basic(self, client, run_dir):
        resp = client.post("/api/schedule", json={
            "at_step": 500,
            "module": "opt",
            "op": "set_params",
            "params": {"lr": 0.0001},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "scheduled"
        assert body["command"]["at"]["step"] == 500
        recs = _read_recipe(run_dir)
        assert len(recs) == 1
        assert recs[0]["at"]["step"] == 500
        assert recs[0]["module"] == "opt"

    def test_schedule_cb(self, client, run_dir):
        resp = client.post("/api/schedule", json={
            "at_step": 1000,
            "module": "cb",
            "op": "enable",
            "id": "grad_clip",
        })
        assert resp.status_code == 200
        recs = _read_recipe(run_dir)
        assert recs[0]["id"] == "grad_clip"

    def test_zero_step_rejected(self, client):
        resp = client.post("/api/schedule", json={
            "at_step": 0,
            "module": "opt",
            "op": "set_params",
        })
        assert resp.status_code == 422

    def test_invalid_module_rejected(self, client):
        resp = client.post("/api/schedule", json={
            "at_step": 100,
            "module": "bogus",
            "op": "set_params",
        })
        assert resp.status_code == 422

    def test_multiple_scheduled(self, client, run_dir):
        client.post("/api/schedule", json={"at_step": 100, "module": "opt", "op": "set_params", "params": {"lr": 0.1}})
        client.post("/api/schedule", json={"at_step": 200, "module": "opt", "op": "set_params", "params": {"lr": 0.01}})
        recs = _read_recipe(run_dir)
        assert len(recs) == 2


# ---- /api/validate ----

class TestValidate:
    def test_valid_opt_set_params(self, client):
        resp = client.post("/api/validate", json={
            "module": "opt",
            "op": "set_params",
            "params": {"lr": 0.001},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is True
        assert body["errors"] == []

    def test_invalid_op(self, client):
        resp = client.post("/api/validate", json={
            "module": "opt",
            "op": "explode",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is False
        assert any("Unknown op" in e for e in body["errors"])

    def test_set_params_empty(self, client):
        resp = client.post("/api/validate", json={
            "module": "opt",
            "op": "set_params",
            "params": {},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is False
        assert any("non-empty" in e for e in body["errors"])

    def test_negative_lr(self, client):
        resp = client.post("/api/validate", json={
            "module": "opt",
            "op": "set_params",
            "params": {"lr": -0.1},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is False
        assert any("lr" in e for e in body["errors"])

    def test_negative_weight_decay(self, client):
        resp = client.post("/api/validate", json={
            "module": "opt",
            "op": "set_params",
            "params": {"weight_decay": -0.01},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is False

    def test_valid_cb_enable(self, client):
        resp = client.post("/api/validate", json={
            "module": "cb",
            "op": "enable",
            "id": "my_cb",
        })
        assert resp.status_code == 200
        assert resp.json()["valid"] is True

    def test_invalid_module_rejected(self, client):
        resp = client.post("/api/validate", json={
            "module": "bogus",
            "op": "enable",
        })
        assert resp.status_code == 422


# ---- /api/chat (NL command) ----

class TestChat:
    def test_learn_faster(self, client, run_dir):
        resp = client.post("/api/chat", json={"message": "learn faster"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "queued"
        assert body["command"]["module"] == "opt"
        assert "__current__" in str(body["command"]["params"]["lr"])
        assert "Doubling" in body["reply"]
        cmds = _read_commands(run_dir)
        assert len(cmds) == 1

    def test_slow_down(self, client, run_dir):
        resp = client.post("/api/chat", json={"message": "please slow down"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "queued"
        assert "Halving" in body["reply"]

    def test_reduce_overfitting(self, client, run_dir):
        resp = client.post("/api/chat", json={"message": "reduce overfitting"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["command"]["params"]["weight_decay"] == "__current__ * 2.0"

    def test_enable_tuning(self, client, run_dir):
        resp = client.post("/api/chat", json={"message": "start tuning"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["command"]["module"] == "tune"

    def test_unknown_message(self, client):
        resp = client.post("/api/chat", json={"message": "what is the meaning of life"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["command"] is None
        assert "don't understand" in body["reply"]

    def test_info_only_pattern(self, client):
        resp = client.post("/api/chat", json={"message": "freeze the model"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["command"] is None
        assert "freeze" in body["reply"].lower()

    def test_empty_message_rejected(self, client):
        resp = client.post("/api/chat", json={"message": ""})
        assert resp.status_code == 422


# ---- /api/features/snapshots ----

class TestFeaturesSnapshots:
    def test_empty(self, client):
        resp = client.get("/api/features/snapshots")
        assert resp.status_code == 200
        assert resp.json()["snapshots"] == []

    def test_with_data(self, client, run_dir):
        import json as _json
        feat_path = os.path.join(run_dir, "hotcb.features.jsonl")
        with open(feat_path, "w") as f:
            f.write(_json.dumps({"step": 50, "layer_name": "fc1", "activations": [[1.0, 2.0]]}) + "\n")
            f.write(_json.dumps({"step": 100, "layer_name": "fc1", "activations": [[3.0, 4.0]]}) + "\n")
        resp = client.get("/api/features/snapshots?last_n=1")
        assert resp.status_code == 200
        snaps = resp.json()["snapshots"]
        assert len(snaps) == 1
        assert snaps[0]["step"] == 100


# ---- Accumulation test ----

class TestAccumulation:
    def test_commands_accumulate(self, client, run_dir):
        """Multiple endpoint calls append to the same JSONL file."""
        client.post("/api/opt/set", json={"params": {"lr": 0.001}})
        client.post("/api/loss/set", json={"params": {"recon_w": 0.5}})
        client.post("/api/cb/my_cb/enable")
        cmds = _read_commands(run_dir)
        assert len(cmds) == 3
        modules = [c["module"] for c in cmds]
        assert modules == ["opt", "loss", "cb"]
