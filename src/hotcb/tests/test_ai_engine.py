"""Tests for hotcb.server.ai_engine and hotcb.server.ai_prompts."""
import json
import os
import tempfile

import pytest

from hotcb.server.ai_engine import AIConfig, AIState, LLMAutopilotEngine
from hotcb.server.ai_prompts import (
    TrendCompressor,
    build_context,
    parse_ai_response,
    ACTION_SCHEMA,
)
from hotcb.server.autopilot import AutopilotEngine


# ---------------------------------------------------------------------------
# TrendCompressor
# ---------------------------------------------------------------------------


class TestTrendCompressor:
    def test_flat_trend(self):
        comp = TrendCompressor()
        vals = [1.0] * 50
        s = comp.compress(vals, "loss")
        assert s.trend == "flat"
        assert s.volatility == "none"

    def test_decreasing_trend(self):
        comp = TrendCompressor()
        vals = [1.0 - i * 0.01 for i in range(50)]
        s = comp.compress(vals, "loss")
        assert "down" in s.trend
        assert s.slope < 0

    def test_increasing_trend(self):
        comp = TrendCompressor()
        vals = [0.5 + i * 0.02 for i in range(50)]
        s = comp.compress(vals, "loss")
        assert "up" in s.trend or s.trend == "spike"
        assert s.slope > 0

    def test_high_volatility(self):
        comp = TrendCompressor()
        import math
        vals = [1.0 + 0.5 * math.sin(i) for i in range(50)]
        s = comp.compress(vals, "loss")
        assert s.volatility in ("medium", "high")

    def test_single_value(self):
        comp = TrendCompressor()
        s = comp.compress([0.5], "loss")
        assert s.trend == "flat"
        assert s.last_value == 0.5

    def test_empty(self):
        comp = TrendCompressor()
        s = comp.compress([], "loss")
        assert s.trend == "flat"

    def test_format_trend_table(self):
        comp = TrendCompressor()
        s1 = comp.compress([1.0 - i * 0.01 for i in range(30)], "train_loss")
        s2 = comp.compress([1.0] * 30, "lr")
        table = comp.format_trend_table([s1, s2])
        assert "train_loss" in table
        assert "lr" in table
        assert "|" in table

    def test_format_raw_metrics(self):
        comp = TrendCompressor()
        history = {"loss": [0.5, 0.4, 0.3], "lr": [0.001, 0.001, 0.001]}
        out = comp.format_raw_metrics(history, ["loss"], last_n=2)
        assert "loss" in out
        assert "0.3" in out

    def test_notable_new_min(self):
        comp = TrendCompressor()
        vals = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        s = comp.compress(vals, "loss")
        assert "new min" in s.notable


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------


class TestBuildContext:
    def test_basic_context(self):
        messages = build_context(
            step=100,
            metric_history={"loss": [0.5, 0.4, 0.3], "lr": [0.001, 0.001, 0.001]},
            alerts=[],
            action_history=[],
            current_state={"lr": 0.001},
            ai_state={"key_metric": "loss", "watch_metrics": [], "run_number": 1, "max_runs": 3, "carried_context": "", "run_history": []},
            mode="trend",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "100" in messages[1]["content"]
        assert "loss" in messages[0]["content"]

    def test_alert_mode_includes_raw(self):
        messages = build_context(
            step=200,
            metric_history={"val_loss": [0.5, 0.4, 0.3, 0.5, 0.6]},
            alerts=[{"rule_id": "div_1", "condition_met": "divergence detected"}],
            action_history=[],
            current_state={},
            ai_state={"key_metric": "val_loss", "watch_metrics": [], "run_number": 1, "max_runs": 3, "carried_context": "", "run_history": []},
            mode="alert",
        )
        user_content = messages[1]["content"]
        assert "Active Alerts" in user_content
        assert "div_1" in user_content

    def test_with_run_history(self):
        messages = build_context(
            step=50,
            metric_history={"loss": [0.5]},
            alerts=[],
            action_history=[],
            current_state={},
            ai_state={
                "key_metric": "loss",
                "watch_metrics": [],
                "run_number": 2,
                "max_runs": 3,
                "carried_context": "Previous run diverged",
                "run_history": [
                    {"run_id": "run_001", "final_key_metric": 0.8, "ai_verdict": "degenerate", "carried_learnings": ["lr too high"]}
                ],
            },
            mode="trend",
        )
        system = messages[0]["content"]
        assert "Run 2" in system
        assert "Previous run diverged" in system


# ---------------------------------------------------------------------------
# parse_ai_response
# ---------------------------------------------------------------------------


class TestParseAIResponse:
    def test_valid_response(self):
        raw = json.dumps({
            "reasoning": "Loss is plateauing, reduce lr",
            "actions": [{"action": "reduce_lr_factor", "params": {"factor": 0.5}}],
            "next_check": {"mode": "in_n_steps", "n": 30},
            "watch_metrics_raw": ["grad_norm"],
        })
        result = parse_ai_response(raw)
        assert result is not None
        assert result["reasoning"] == "Loss is plateauing, reduce lr"
        assert len(result["actions"]) == 1
        assert result["actions"][0]["action"] == "reduce_lr_factor"

    def test_noop_response(self):
        raw = json.dumps({
            "reasoning": "Training is healthy",
            "actions": [{"action": "noop", "params": {}}],
            "next_check": {"mode": "periodic", "interval": 50},
        })
        result = parse_ai_response(raw)
        assert result is not None
        assert len(result["actions"]) == 1

    def test_invalid_json(self):
        result = parse_ai_response("not json at all")
        assert result is None

    def test_markdown_fences(self):
        raw = "```json\n" + json.dumps({
            "reasoning": "test",
            "actions": [{"action": "noop", "params": {}}],
        }) + "\n```"
        result = parse_ai_response(raw)
        assert result is not None

    def test_out_of_bounds_param(self):
        raw = json.dumps({
            "reasoning": "test",
            "actions": [{"action": "set_lr", "params": {"lr": 100.0}}],  # way too high
        })
        result = parse_ai_response(raw)
        assert result is not None
        assert len(result["actions"]) == 0  # rejected

    def test_unknown_action(self):
        raw = json.dumps({
            "reasoning": "test",
            "actions": [{"action": "launch_missiles", "params": {}}],
        })
        result = parse_ai_response(raw)
        assert result is not None
        assert len(result["actions"]) == 0


# ---------------------------------------------------------------------------
# AIConfig
# ---------------------------------------------------------------------------


class TestAIConfig:
    def test_defaults(self):
        cfg = AIConfig()
        assert cfg.provider == "openai"
        assert cfg.temperature == 0.3
        assert cfg.budget_cap == 5.0

    def test_env_key(self):
        os.environ["HOTCB_AI_KEY"] = "test-key-123"
        try:
            cfg = AIConfig()
            assert cfg.api_key == "test-key-123"
        finally:
            del os.environ["HOTCB_AI_KEY"]

    def test_safe_dict(self):
        cfg = AIConfig(api_key="sk-1234567890abcdef")
        d = cfg.to_safe_dict()
        assert "1234567890" not in d["api_key"]
        assert d["api_key"].startswith("sk-1")


# ---------------------------------------------------------------------------
# AIState
# ---------------------------------------------------------------------------


class TestAIState:
    def test_roundtrip(self):
        state = AIState(key_metric="val_acc", watch_metrics=["grad_norm"], run_number=2)
        d = state.to_dict()
        restored = AIState.from_dict(d)
        assert restored.key_metric == "val_acc"
        assert restored.run_number == 2


# ---------------------------------------------------------------------------
# LLMAutopilotEngine
# ---------------------------------------------------------------------------


class TestLLMAutopilotEngine:
    def test_init_no_state_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LLMAutopilotEngine(run_dir=tmpdir)
            assert engine.state.run_number == 1
            assert engine.state.key_metric == "val_loss"

    def test_state_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LLMAutopilotEngine(run_dir=tmpdir)
            engine.state.key_metric = "val_acc"
            engine.state.run_number = 2
            engine.save_state()

            engine2 = LLMAutopilotEngine(run_dir=tmpdir)
            assert engine2.state.key_metric == "val_acc"
            assert engine2.state.run_number == 2

    def test_should_invoke_no_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LLMAutopilotEngine(run_dir=tmpdir)
            # No API key = never invoke
            assert not engine.should_invoke(100, [])

    def test_should_invoke_with_alert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = AIConfig(api_key="test-key")
            engine = LLMAutopilotEngine(run_dir=tmpdir, config=cfg)
            engine._last_invoked_step = 0
            assert engine.should_invoke(50, [{"rule_id": "test"}])

    def test_should_invoke_cooldown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = AIConfig(api_key="test-key")
            engine = LLMAutopilotEngine(run_dir=tmpdir, config=cfg)
            engine._last_invoked_step = 95
            # Within cooldown (10 steps)
            assert not engine.should_invoke(100, [])

    def test_should_invoke_periodic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = AIConfig(api_key="test-key", cadence=50)
            engine = LLMAutopilotEngine(run_dir=tmpdir, config=cfg)
            engine._last_invoked_step = 50
            assert engine.should_invoke(100, [])

    def test_budget_exhaustion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = AIConfig(api_key="test-key", budget_cap=0.01)
            engine = LLMAutopilotEngine(run_dir=tmpdir, config=cfg)
            engine._total_cost = 0.02
            assert not engine.should_invoke(100, [{"rule_id": "x"}])
            assert not engine.enabled

    def test_handle_set_key_metric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LLMAutopilotEngine(run_dir=tmpdir)
            assert engine.handle_set_key_metric("val_acc", ["val_acc", "loss"])
            assert engine.state.key_metric == "val_acc"
            assert not engine.handle_set_key_metric("nonexistent", ["val_acc"])

    def test_handle_watch_metric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LLMAutopilotEngine(run_dir=tmpdir)
            engine.handle_watch_metric("grad_norm", add=True)
            assert "grad_norm" in engine.state.watch_metrics
            engine.handle_watch_metric("grad_norm", add=False)
            assert "grad_norm" not in engine.state.watch_metrics

    def test_handle_declare_rerun(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LLMAutopilotEngine(run_dir=tmpdir)
            result = engine.handle_declare_rerun("lr too high", ["reduce lr"])
            assert result is not None
            assert len(engine.state.run_history) == 1

    def test_handle_declare_rerun_max_reached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = AIConfig(max_runs=2)
            engine = LLMAutopilotEngine(run_dir=tmpdir, config=cfg)
            engine.state.run_number = 2
            result = engine.handle_declare_rerun("bad", [])
            assert result is None

    def test_disable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LLMAutopilotEngine(run_dir=tmpdir)
            engine.disable("test reason")
            assert not engine.enabled

    def test_get_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LLMAutopilotEngine(run_dir=tmpdir)
            status = engine.get_status()
            assert "enabled" in status
            assert "key_metric" in status
            assert "total_cost_usd" in status

    def test_update_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LLMAutopilotEngine(run_dir=tmpdir)
            engine.update_config({"model": "gpt-4o", "cadence": 100})
            assert engine.config.model == "gpt-4o"
            assert engine.config.cadence == 100


# ---------------------------------------------------------------------------
# AutopilotEngine AI mode integration
# ---------------------------------------------------------------------------


class TestAutopilotAIModes:
    def test_valid_ai_modes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = AutopilotEngine(run_dir=tmpdir)
            ai = LLMAutopilotEngine(run_dir=tmpdir)
            engine.set_ai_engine(ai)
            engine.set_mode("ai_suggest")
            assert engine.mode == "ai_suggest"
            assert engine.is_ai_mode
            engine.set_mode("ai_auto")
            assert engine.mode == "ai_auto"

    def test_ai_mode_without_engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = AutopilotEngine(run_dir=tmpdir)
            with pytest.raises(ValueError, match="AI engine not configured"):
                engine.set_mode("ai_suggest")

    def test_evaluate_rules_for_alerts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from hotcb.server.autopilot import AutopilotRule
            engine = AutopilotEngine(run_dir=tmpdir)
            engine.add_rule(AutopilotRule(
                rule_id="test_plateau",
                condition="plateau",
                metric_name="loss",
                params={"window": 3, "epsilon": 0.01, "cooldown": 1},
                action={"module": "opt", "op": "set_params", "params": {"lr": 0.0001}},
            ))
            # Feed flat loss values to trigger plateau
            for i in range(5):
                engine.evaluate_rules_for_alerts(i, {"loss": 1.0})
            alerts = engine.evaluate_rules_for_alerts(5, {"loss": 1.0})
            assert len(alerts) > 0
            assert alerts[0].status == "alert"

    def test_ai_action_to_command_set_lr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = AutopilotEngine(run_dir=tmpdir)
            cmd = engine._ai_action_to_command("set_lr", {"lr": 0.0005})
            assert cmd is not None
            assert cmd["module"] == "opt"
            assert cmd["params"]["lr"] == 0.0005

    def test_ai_action_to_command_reduce_lr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = AutopilotEngine(run_dir=tmpdir)
            # Seed metric history with lr
            engine._metric_history["lr"].append(0.001)
            cmd = engine._ai_action_to_command("reduce_lr_factor", {"factor": 0.5})
            assert cmd is not None
            assert cmd["params"]["lr"] == pytest.approx(0.0005)

    def test_ai_action_to_command_loss_weight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = AutopilotEngine(run_dir=tmpdir)
            cmd = engine._ai_action_to_command("set_loss_weight", {"term": "weight_a", "weight": 0.8})
            assert cmd is not None
            assert cmd["module"] == "loss"
            assert cmd["params"]["weight_a"] == 0.8

    def test_ai_action_unknown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = AutopilotEngine(run_dir=tmpdir)
            cmd = engine._ai_action_to_command("unknown_action", {})
            assert cmd is None


# ---------------------------------------------------------------------------
# ACTION_SCHEMA
# ---------------------------------------------------------------------------


class TestActionSchema:
    def test_all_actions_have_descriptions(self):
        for name, schema in ACTION_SCHEMA.items():
            assert "description" in schema
            assert "params" in schema

    def test_bounds_present(self):
        assert ACTION_SCHEMA["set_lr"]["params"]["lr"]["min"] == 1e-7
        assert ACTION_SCHEMA["set_lr"]["params"]["lr"]["max"] == 1.0
