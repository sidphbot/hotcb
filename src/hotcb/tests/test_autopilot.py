"""Tests for hotcb.server.autopilot — AutopilotEngine."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from hotcb.server.autopilot import (
    AutopilotEngine,
    AutopilotRule,
    AutopilotAction,
    _eval_plateau,
    _eval_divergence,
    _eval_overfitting,
    _eval_custom,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestConditionEvaluators:
    def test_plateau_detected(self):
        history = [1.0, 1.0001, 0.9999, 1.0, 1.0002]
        result = _eval_plateau(history, {"window": 5, "epsilon": 0.001})
        assert result is not None
        assert "plateau" in result.lower()

    def test_plateau_not_detected(self):
        history = [1.0, 0.9, 0.8, 0.7, 0.6]
        result = _eval_plateau(history, {"window": 5, "epsilon": 0.001})
        assert result is None

    def test_plateau_insufficient_history(self):
        result = _eval_plateau([1.0, 1.0], {"window": 5, "epsilon": 0.001})
        assert result is None

    def test_divergence_detected(self):
        history = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
        result = _eval_divergence(history, {"window": 10, "threshold": 2.0})
        assert result is not None
        assert "diverged" in result.lower()

    def test_divergence_not_detected(self):
        history = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        result = _eval_divergence(history, {"window": 10, "threshold": 2.0})
        assert result is None

    def test_overfitting_detected(self):
        metrics = {"train_loss": 0.1, "val_loss": 0.5}
        result = _eval_overfitting(metrics, {"ratio_threshold": 0.5})
        assert result is not None
        assert "overfitting" in result.lower()

    def test_overfitting_not_detected(self):
        metrics = {"train_loss": 0.4, "val_loss": 0.5}
        result = _eval_overfitting(metrics, {"ratio_threshold": 0.5})
        assert result is None

    def test_overfitting_missing_metrics(self):
        result = _eval_overfitting({"train_loss": 0.1}, {"ratio_threshold": 0.5})
        assert result is None

    def test_custom_true(self):
        result = _eval_custom({"val_loss": 2.5}, "val_loss > 2.0")
        assert result is not None

    def test_custom_false(self):
        result = _eval_custom({"val_loss": 1.5}, "val_loss > 2.0")
        assert result is None

    def test_custom_with_normalized_names(self):
        # Metric names with / get normalized to _
        result = _eval_custom({"val/loss": 3.0}, "val_loss > 2.0")
        assert result is not None

    def test_custom_invalid_expression(self):
        result = _eval_custom({"val_loss": 1.0}, "import os")
        assert result is None


class TestAutopilotEngine:
    def test_init_valid_modes(self, tmp_dir):
        for mode in ("off", "suggest", "auto"):
            engine = AutopilotEngine(run_dir=tmp_dir, mode=mode)
            assert engine.mode == mode

    def test_init_invalid_mode(self, tmp_dir):
        with pytest.raises(ValueError):
            AutopilotEngine(run_dir=tmp_dir, mode="invalid")

    def test_set_mode(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="off")
        engine.set_mode("auto")
        assert engine.mode == "auto"

    def test_set_mode_invalid(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="off")
        with pytest.raises(ValueError):
            engine.set_mode("bogus")

    def test_add_remove_rules(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir)
        rule = AutopilotRule(rule_id="r1", condition="plateau", metric_name="loss")
        engine.add_rule(rule)
        assert len(engine.get_rules()) == 1
        assert engine.remove_rule("r1")
        assert len(engine.get_rules()) == 0

    def test_remove_nonexistent(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir)
        assert not engine.remove_rule("nope")

    def test_off_mode_does_not_evaluate(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="off")
        engine.add_rule(AutopilotRule(
            rule_id="r1", condition="plateau", metric_name="loss",
            params={"window": 3, "epsilon": 0.001},
        ))
        for i in range(10):
            engine.evaluate(i, {"loss": 1.0})
        assert len(engine.history) == 0

    def test_suggest_mode_proposes(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="suggest")
        engine.add_rule(AutopilotRule(
            rule_id="r1", condition="plateau", metric_name="loss",
            params={"window": 3, "epsilon": 0.01, "cooldown": 1},
            action={"module": "opt", "op": "set_params", "params": {"lr": 1e-4}},
            confidence="high",
        ))
        for i in range(5):
            engine.evaluate(i, {"loss": 1.0})
        # Should have proposed (not applied) once cooldown allows
        proposals = [a for a in engine.history if a.status == "proposed"]
        assert len(proposals) >= 1

    def test_auto_mode_high_confidence_applies(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="auto")
        engine.add_rule(AutopilotRule(
            rule_id="r1", condition="plateau", metric_name="loss",
            params={"window": 3, "epsilon": 0.01, "cooldown": 1},
            action={"module": "opt", "op": "set_params", "params": {"lr": 1e-4}},
            confidence="high",
        ))
        for i in range(5):
            engine.evaluate(i, {"loss": 1.0})
        applied = [a for a in engine.history if a.status == "applied"]
        assert len(applied) >= 1
        # Verify command was written
        cmd_path = os.path.join(tmp_dir, "hotcb.commands.jsonl")
        assert os.path.exists(cmd_path)

    def test_auto_mode_low_confidence_proposes(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="auto")
        engine.add_rule(AutopilotRule(
            rule_id="r1", condition="plateau", metric_name="loss",
            params={"window": 3, "epsilon": 0.01, "cooldown": 1},
            action={"module": "opt", "op": "set_params"},
            confidence="low",
        ))
        for i in range(5):
            engine.evaluate(i, {"loss": 1.0})
        proposed = [a for a in engine.history if a.status == "proposed"]
        assert len(proposed) >= 1
        applied = [a for a in engine.history if a.status == "applied"]
        assert len(applied) == 0

    def test_cooldown_prevents_rapid_firing(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="suggest")
        engine.add_rule(AutopilotRule(
            rule_id="r1", condition="plateau", metric_name="loss",
            params={"window": 3, "epsilon": 0.01, "cooldown": 100},
            confidence="high",
        ))
        for i in range(20):
            engine.evaluate(i, {"loss": 1.0})
        # Should fire only once (cooldown=100 prevents re-firing)
        assert len(engine.history) == 1

    def test_disabled_rule_skipped(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="suggest")
        engine.add_rule(AutopilotRule(
            rule_id="r1", condition="plateau", metric_name="loss",
            params={"window": 3, "epsilon": 0.01, "cooldown": 1},
            enabled=False,
        ))
        for i in range(10):
            engine.evaluate(i, {"loss": 1.0})
        assert len(engine.history) == 0

    def test_divergence_rule(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="suggest")
        engine.add_rule(AutopilotRule(
            rule_id="r1", condition="divergence", metric_name="loss",
            params={"window": 5, "threshold": 1.0, "cooldown": 1},
        ))
        for i in range(10):
            engine.evaluate(i, {"loss": float(i)})
        assert len(engine.history) >= 1

    def test_overfitting_rule(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="suggest")
        engine.add_rule(AutopilotRule(
            rule_id="r1", condition="overfitting", metric_name="",
            params={"ratio_threshold": 0.5, "cooldown": 1},
        ))
        engine.evaluate(1, {"train_loss": 0.1, "val_loss": 0.5})
        assert len(engine.history) == 1

    def test_custom_rule(self, tmp_dir):
        engine = AutopilotEngine(run_dir=tmp_dir, mode="suggest")
        engine.add_rule(AutopilotRule(
            rule_id="r1", condition="custom", metric_name="loss",
            params={"expression": "loss > 5.0", "cooldown": 1},
        ))
        engine.evaluate(1, {"loss": 10.0})
        assert len(engine.history) == 1


class TestAutopilotGuidelines:
    def test_load_yaml_guidelines(self, tmp_dir):
        yaml_path = os.path.join(tmp_dir, "guidelines.yaml")
        import yaml
        data = {
            "version": 1,
            "rules": [
                {
                    "id": "plateau_lr",
                    "condition": "plateau",
                    "metric": "val_loss",
                    "params": {"window": 5, "epsilon": 0.001},
                    "action": {"module": "opt", "op": "set_params", "params": {"lr_mult": 0.5}},
                    "confidence": "high",
                    "description": "Reduce LR on plateau",
                },
                {
                    "id": "diverge_stop",
                    "condition": "divergence",
                    "metric": "train_loss",
                    "params": {"window": 10, "threshold": 2.0},
                    "action": {"module": "tune", "op": "disable"},
                    "confidence": "medium",
                },
            ],
        }
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)

        engine = AutopilotEngine(run_dir=tmp_dir)
        count = engine.load_guidelines(yaml_path)
        assert count == 2
        assert len(engine.get_rules()) == 2

    def test_load_invalid_yaml(self, tmp_dir):
        yaml_path = os.path.join(tmp_dir, "bad.yaml")
        import yaml
        with open(yaml_path, "w") as f:
            yaml.dump({"not_rules": []}, f)
        engine = AutopilotEngine(run_dir=tmp_dir)
        with pytest.raises(ValueError):
            engine.load_guidelines(yaml_path)


class TestAutopilotRESTEndpoints:
    @pytest.fixture
    def client(self, tmp_dir):
        fastapi = pytest.importorskip("fastapi")
        from hotcb.server.app import create_app
        from starlette.testclient import TestClient

        open(os.path.join(tmp_dir, "hotcb.commands.jsonl"), "w").close()
        app = create_app(tmp_dir, poll_interval=60)
        return TestClient(app)

    def test_status(self, client):
        r = client.get("/api/autopilot/status")
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "off"
        # Default guidelines are pre-loaded, so rules_count >= 0
        assert data["rules_count"] >= 0
        assert "recent_actions" in data

    def test_set_mode(self, client):
        r = client.post("/api/autopilot/mode", json={"mode": "suggest"})
        assert r.status_code == 200
        assert r.json()["mode"] == "suggest"

    def test_add_rule(self, client):
        r = client.post("/api/autopilot/rules", json={
            "rule_id": "test_rule",
            "condition": "plateau",
            "metric_name": "loss",
            "params": {"window": 5},
        })
        assert r.status_code == 200
        assert r.json()["rule_id"] == "test_rule"

    def test_list_rules(self, client):
        # Get baseline count (default guidelines may be pre-loaded)
        r0 = client.get("/api/autopilot/rules")
        baseline = len(r0.json()["rules"])
        client.post("/api/autopilot/rules", json={
            "rule_id": "r1", "condition": "plateau", "metric_name": "loss",
        })
        r = client.get("/api/autopilot/rules")
        assert r.status_code == 200
        assert len(r.json()["rules"]) == baseline + 1

    def test_delete_rule(self, client):
        client.post("/api/autopilot/rules", json={
            "rule_id": "r1", "condition": "plateau", "metric_name": "loss",
        })
        r = client.delete("/api/autopilot/rules/r1")
        assert r.status_code == 200

    def test_delete_nonexistent_rule(self, client):
        r = client.delete("/api/autopilot/rules/nope")
        assert r.status_code == 404

    def test_history(self, client):
        r = client.get("/api/autopilot/history")
        assert r.status_code == 200
        assert r.json()["actions"] == []

    def test_status_includes_recent_actions(self, client):
        r = client.get("/api/autopilot/status")
        assert r.status_code == 200
        data = r.json()
        assert "recent_actions" in data
        assert isinstance(data["recent_actions"], list)


class TestAutopilotWithDefaultGuidelines:
    """Tests verifying autopilot works with default guidelines in demo-like scenarios."""

    def test_default_guidelines_loaded(self, tmp_dir):
        engine = AutopilotEngine.with_default_guidelines(run_dir=tmp_dir, mode="suggest")
        rules = engine.get_rules()
        assert len(rules) >= 4  # At least plateau, divergence, overfitting, grad_spike
        rule_ids = {r.rule_id for r in rules}
        assert "plateau_lr_reduce" in rule_ids or "val_plateau_lr_reduce" in rule_ids

    def test_plateau_triggers_in_demo_scenario(self, tmp_dir):
        """Simulate a plateau like the demo produces after loss converges."""
        engine = AutopilotEngine.with_default_guidelines(run_dir=tmp_dir, mode="suggest")
        # Simulate 50 steps of plateaued loss around 0.15
        for i in range(50):
            engine.evaluate(i, {
                "train_loss": 0.15 + (i % 3) * 0.001,  # tiny variation
                "val_loss": 0.20 + (i % 3) * 0.001,
                "grad_norm": 0.5,
            })
        # Should have detected plateau
        assert len(engine.history) >= 1
        plateau_actions = [a for a in engine.history if "plateau" in a.condition_met.lower()]
        assert len(plateau_actions) >= 1

    def test_spike_triggers_divergence(self, tmp_dir):
        """Simulate a loss spike."""
        engine = AutopilotEngine.with_default_guidelines(run_dir=tmp_dir, mode="suggest")
        # Normal training for a few steps
        for i in range(10):
            engine.evaluate(i, {"train_loss": 0.5 - i * 0.01, "val_loss": 0.6, "grad_norm": 1.0})
        # Sudden spike
        for i in range(10, 16):
            engine.evaluate(i, {"train_loss": 0.5 + (i - 10) * 0.1, "val_loss": 0.6, "grad_norm": 1.0})
        diverge_actions = [a for a in engine.history if "diverge" in a.condition_met.lower()]
        assert len(diverge_actions) >= 1

    def test_auto_mode_writes_commands(self, tmp_dir):
        """Verify auto mode writes to commands JSONL."""
        engine = AutopilotEngine.with_default_guidelines(run_dir=tmp_dir, mode="auto")
        # Simulate plateau
        for i in range(50):
            engine.evaluate(i, {
                "train_loss": 0.15,
                "val_loss": 0.20,
                "grad_norm": 0.5,
            })
        applied = [a for a in engine.history if a.status == "applied"]
        assert len(applied) >= 1
        cmd_path = os.path.join(tmp_dir, "hotcb.commands.jsonl")
        assert os.path.exists(cmd_path)
        with open(cmd_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) >= 1
        cmd = json.loads(lines[0])
        assert cmd.get("source") == "autopilot"

    def test_suggest_mode_does_not_write_commands(self, tmp_dir):
        """Verify suggest mode does NOT write to commands JSONL."""
        engine = AutopilotEngine.with_default_guidelines(run_dir=tmp_dir, mode="suggest")
        for i in range(50):
            engine.evaluate(i, {
                "train_loss": 0.15,
                "val_loss": 0.20,
                "grad_norm": 0.5,
            })
        proposed = [a for a in engine.history if a.status == "proposed"]
        assert len(proposed) >= 1
        cmd_path = os.path.join(tmp_dir, "hotcb.commands.jsonl")
        # File should not exist or be empty
        if os.path.exists(cmd_path):
            with open(cmd_path) as f:
                assert f.read().strip() == ""
