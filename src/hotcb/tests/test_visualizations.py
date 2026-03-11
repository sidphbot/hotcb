"""Tests for hotcb dashboard visualization and forecasting systems.

Covers ProjectionEngine, ManifoldEngine, NotificationEngine, and AutopilotEngine.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
from unittest import mock

import numpy as np
import pytest

from hotcb.server.projections import (
    ForecastResult,
    ProjectionEngine,
    HAS_XGB,
)
from hotcb.server.manifolds import (
    ManifoldEngine,
    ManifoldResult,
    TrajectoryStats,
    available_methods,
)
from hotcb.server.notifications import (
    Alert,
    AlertRule,
    LogChannel,
    NotificationEngine,
)
from hotcb.server.autopilot import (
    AutopilotAction,
    AutopilotEngine,
    AutopilotRule,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic data
# ---------------------------------------------------------------------------


def _make_records(n=50, start_loss=2.0, decay=0.02, seed=42):
    """Generate synthetic metric records."""
    rng = random.Random(seed)
    records = []
    loss = start_loss
    for i in range(1, n + 1):
        loss = max(0.1, loss - decay + rng.gauss(0, 0.005))
        records.append({
            "step": i,
            "metrics": {
                "train_loss": loss,
                "val_loss": loss + 0.1 + rng.gauss(0, 0.01),
            },
            "hp": {"lr": 0.001},
        })
    return records


def _make_plateau_records(n=30, value=1.0, noise=0.0001, seed=42):
    """Generate records where the metric plateaus."""
    rng = random.Random(seed)
    records = []
    for i in range(1, n + 1):
        records.append({
            "step": i,
            "metrics": {
                "train_loss": value + rng.gauss(0, noise),
                "val_loss": value + 0.05 + rng.gauss(0, noise),
            },
        })
    return records


def _make_diverging_records(n=30, seed=42):
    """Generate records where the loss diverges sharply."""
    rng = random.Random(seed)
    records = []
    for i in range(1, n + 1):
        # Loss increases quadratically
        loss = 0.5 + 0.1 * i + rng.gauss(0, 0.01)
        records.append({
            "step": i,
            "metrics": {"train_loss": loss, "val_loss": loss + 0.2},
        })
    return records


def _make_overfitting_records(n=30, seed=42):
    """Generate records where train_loss << val_loss (overfitting)."""
    rng = random.Random(seed)
    records = []
    for i in range(1, n + 1):
        train = max(0.01, 0.5 - 0.015 * i + rng.gauss(0, 0.005))
        val = 0.8 + 0.01 * i + rng.gauss(0, 0.005)
        records.append({
            "step": i,
            "metrics": {"train_loss": train, "val_loss": val},
        })
    return records


# ===========================================================================
# ProjectionEngine tests
# ===========================================================================


def test_projection_update_ingests_records():
    engine = ProjectionEngine(min_history=5)
    records = _make_records(30)
    engine.update(records)
    assert engine.record_count == 30


def test_projection_update_stores_multiple_metrics():
    engine = ProjectionEngine(min_history=5)
    engine.update(_make_records(20))
    assert "train_loss" in engine._metrics
    assert "val_loss" in engine._metrics
    assert len(engine._metrics["train_loss"]) == 20


def test_projection_update_skips_invalid():
    engine = ProjectionEngine(min_history=5)
    engine.update([
        {"step": 1},
        {"metrics": {"x": 1.0}},
        {},
        {"step": 2, "metrics": "bad"},
    ])
    assert engine.record_count == 0


def test_projection_forecast_univariate_with_history():
    engine = ProjectionEngine(min_history=10)
    engine.update(_make_records(50))
    result = engine.forecast_univariate("train_loss", horizon=20)
    assert len(result.steps) == 20
    assert len(result.values) == 20
    assert result.metric_name == "train_loss"
    assert result.method in ("xgboost", "fallback_linear")


def test_projection_forecast_univariate_too_little_history():
    engine = ProjectionEngine(min_history=20)
    engine.update(_make_records(10))
    result = engine.forecast_univariate("train_loss", horizon=10)
    assert result.steps == []
    assert result.values == []


def test_projection_forecast_univariate_unknown_metric():
    engine = ProjectionEngine(min_history=5)
    engine.update(_make_records(30))
    result = engine.forecast_univariate("nonexistent")
    assert result.steps == []


def test_projection_forecast_whatif_with_hp():
    engine = ProjectionEngine(min_history=10)
    engine.update(_make_records(50))
    result = engine.forecast_whatif("train_loss", hp_changes={"lr": 0.0001}, horizon=15)
    assert len(result.steps) == 15
    assert len(result.values) == 15
    assert result.metric_name == "train_loss"


def test_projection_forecast_whatif_empty_history():
    engine = ProjectionEngine(min_history=20)
    engine.update(_make_records(5))
    result = engine.forecast_whatif("train_loss", hp_changes={"lr": 0.01}, horizon=10)
    assert result.steps == []


def test_forecast_result_to_dict_format():
    fr = ForecastResult(
        steps=[51, 52, 53],
        values=[0.5, 0.4, 0.3],
        lower=[0.4, 0.3, 0.2],
        upper=[0.6, 0.5, 0.4],
        metric_name="train_loss",
        method="fallback_linear",
    )
    d = fr.to_dict()
    assert set(d.keys()) == {
        "steps", "values", "forecast", "lower", "upper", "metric_name", "method",
    }
    assert d["forecast"] == d["values"]
    assert d["metric_name"] == "train_loss"
    # Must be JSON-serializable
    json.dumps(d)


def test_forecast_confidence_bands_present():
    engine = ProjectionEngine(min_history=10)
    engine.update(_make_records(60))
    result = engine.forecast_univariate("train_loss", horizon=10)
    assert len(result.lower) == len(result.values)
    assert len(result.upper) == len(result.values)
    for lo, val, hi in zip(result.lower, result.values, result.upper):
        assert lo <= val, f"lower {lo} > value {val}"
        assert val <= hi, f"value {val} > upper {hi}"


def test_forecast_steps_continue_from_last():
    engine = ProjectionEngine(min_history=10)
    records = _make_records(40)
    engine.update(records)
    last_data_step = records[-1]["step"]
    result = engine.forecast_univariate("train_loss", horizon=10)
    assert result.steps[0] == last_data_step + 1
    assert result.steps[-1] == last_data_step + 10


def test_projection_forced_linear_fallback():
    with mock.patch("hotcb.server.projections.HAS_XGB", False):
        engine = ProjectionEngine(min_history=10)
        engine.update(_make_records(50))
        result = engine.forecast_univariate("train_loss", horizon=10)
        assert result.method == "fallback_linear"
        assert len(result.steps) == 10


# ===========================================================================
# ManifoldEngine tests
# ===========================================================================


def test_manifold_update_metrics():
    engine = ManifoldEngine()
    records = _make_records(20)
    engine.update_metrics(records)
    assert len(engine._metric_history) == 20


def test_manifold_update_metrics_skips_invalid():
    engine = ManifoldEngine()
    engine.update_metrics([
        {"step": 1},
        {"metrics": {"x": 1}},
        {},
    ])
    assert len(engine._metric_history) == 0


def test_manifold_update_interventions():
    engine = ManifoldEngine()
    engine.update_interventions([
        {"step": 10},
        {"step": 25},
        {"step": 40},
    ])
    assert engine._intervention_steps == {10, 25, 40}


def test_manifold_compute_pca():
    engine = ManifoldEngine()
    engine.update_metrics(_make_records(30))
    result = engine.compute_metric_manifold(method="pca", n_components=2)
    assert isinstance(result, ManifoldResult)
    assert result.method == "pca"
    assert len(result.points) == 30
    assert len(result.steps) == 30
    assert len(result.is_intervention) == 30
    # Each point should have 2 components
    assert len(result.points[0]) == 2
    assert result.explained_variance is not None
    assert len(result.explained_variance) == 2


def test_manifold_intervention_indices_correct():
    engine = ManifoldEngine()
    records = _make_records(20)
    engine.update_metrics(records)
    engine.update_interventions([{"step": 5}, {"step": 10}])
    result = engine.compute_metric_manifold(method="pca")
    # Steps 5 and 10 should be marked as interventions
    for i, step in enumerate(result.steps):
        if step in (5, 10):
            assert result.is_intervention[i] is True
        else:
            assert result.is_intervention[i] is False


def test_manifold_insufficient_data():
    engine = ManifoldEngine()
    # No data at all
    result = engine.compute_metric_manifold()
    assert result.points == []
    assert result.steps == []


def test_manifold_single_record():
    engine = ManifoldEngine()
    engine.update_metrics([{"step": 1, "metrics": {"loss": 0.5, "acc": 0.8}}])
    result = engine.compute_metric_manifold(method="pca", n_components=2)
    # Should handle gracefully with 1 point
    assert len(result.steps) == 1


def test_manifold_trajectory_stats():
    engine = ManifoldEngine()
    engine.update_metrics(_make_records(20))
    engine.update_interventions([{"step": 10}])
    stats = engine.compute_trajectory_stats()
    assert isinstance(stats, TrajectoryStats)
    assert stats.total_distance > 0
    assert len(stats.velocities) == 19  # N-1 differences
    assert stats.mean_velocity > 0


def test_manifold_trajectory_stats_empty():
    engine = ManifoldEngine()
    stats = engine.compute_trajectory_stats()
    assert stats.total_distance == 0.0
    assert stats.velocities == []


def test_manifold_max_points_trimming():
    engine = ManifoldEngine(max_points=20)
    engine.update_metrics(_make_records(50))
    assert len(engine._metric_history) == 20


def test_manifold_available_methods():
    methods = available_methods()
    assert "pca" in methods


# ===========================================================================
# NotificationEngine tests
# ===========================================================================


def test_notification_evaluate_gt():
    engine = NotificationEngine()
    rule = AlertRule(
        rule_id="r1",
        metric_name="train_loss",
        condition="gt",
        threshold=1.0,
        cooldown_steps=0,
    )
    engine.add_rule(rule)
    alerts = engine.evaluate(step=1, metrics={"train_loss": 1.5})
    assert len(alerts) == 1
    assert alerts[0].condition == "gt"
    assert alerts[0].current_value == 1.5


def test_notification_evaluate_lt():
    engine = NotificationEngine()
    rule = AlertRule(
        rule_id="r1",
        metric_name="train_loss",
        condition="lt",
        threshold=0.5,
        cooldown_steps=0,
    )
    engine.add_rule(rule)
    alerts = engine.evaluate(step=1, metrics={"train_loss": 0.3})
    assert len(alerts) == 1
    assert alerts[0].condition == "lt"


def test_notification_evaluate_no_trigger():
    engine = NotificationEngine()
    rule = AlertRule(
        rule_id="r1",
        metric_name="train_loss",
        condition="gt",
        threshold=2.0,
        cooldown_steps=0,
    )
    engine.add_rule(rule)
    alerts = engine.evaluate(step=1, metrics={"train_loss": 1.0})
    assert len(alerts) == 0


def test_notification_plateau_detection():
    engine = NotificationEngine()
    rule = AlertRule(
        rule_id="plateau1",
        metric_name="train_loss",
        condition="plateau",
        threshold=0.01,
        window=5,
        cooldown_steps=0,
    )
    engine.add_rule(rule)
    # Feed identical values to trigger plateau
    for i in range(1, 7):
        alerts = engine.evaluate(step=i, metrics={"train_loss": 1.0})
    # After 5+ identical values, plateau should fire
    assert len(alerts) >= 1
    assert alerts[0].condition == "plateau"


def test_notification_plateau_no_trigger_with_variation():
    engine = NotificationEngine()
    rule = AlertRule(
        rule_id="plateau1",
        metric_name="train_loss",
        condition="plateau",
        threshold=0.001,
        window=5,
        cooldown_steps=0,
    )
    engine.add_rule(rule)
    # Feed values with variation
    for i in range(1, 7):
        alerts = engine.evaluate(step=i, metrics={"train_loss": float(i)})
    assert len(alerts) == 0


def test_notification_spike_detection():
    engine = NotificationEngine()
    rule = AlertRule(
        rule_id="spike1",
        metric_name="train_loss",
        condition="spike",
        threshold=0.5,
        window=5,
        cooldown_steps=0,
    )
    engine.add_rule(rule)
    # Feed stable values
    for i in range(1, 6):
        engine.evaluate(step=i, metrics={"train_loss": 1.0})
    # Then a spike
    alerts = engine.evaluate(step=6, metrics={"train_loss": 2.0})
    assert len(alerts) == 1
    assert alerts[0].condition == "spike"


def test_notification_cooldown_enforcement():
    engine = NotificationEngine()
    rule = AlertRule(
        rule_id="r1",
        metric_name="train_loss",
        condition="gt",
        threshold=1.0,
        cooldown_steps=10,
    )
    engine.add_rule(rule)
    # First trigger
    alerts1 = engine.evaluate(step=1, metrics={"train_loss": 1.5})
    assert len(alerts1) == 1
    # Within cooldown -- should not fire
    alerts2 = engine.evaluate(step=5, metrics={"train_loss": 1.5})
    assert len(alerts2) == 0
    # After cooldown
    alerts3 = engine.evaluate(step=12, metrics={"train_loss": 1.5})
    assert len(alerts3) == 1


def test_notification_dispatch_log_channel():
    engine = NotificationEngine()
    logger = logging.getLogger("test_dispatch")
    channel = LogChannel(logger=logger)
    engine.register_channel("log", channel)

    rule = AlertRule(
        rule_id="r1",
        metric_name="train_loss",
        condition="gt",
        threshold=1.0,
        cooldown_steps=0,
        channels=["log"],
    )
    engine.add_rule(rule)
    alerts = engine.evaluate(step=1, metrics={"train_loss": 2.0})
    assert len(alerts) == 1

    # Dispatch should not raise
    asyncio.get_event_loop().run_until_complete(engine.dispatch(alerts))


def test_notification_add_and_remove_rule():
    engine = NotificationEngine()
    rule = AlertRule(
        rule_id="r1",
        metric_name="loss",
        condition="gt",
        threshold=1.0,
    )
    engine.add_rule(rule)
    assert "r1" in engine.rules
    engine.remove_rule("r1")
    assert "r1" not in engine.rules


def test_notification_remove_nonexistent_rule():
    engine = NotificationEngine()
    # Should not raise
    engine.remove_rule("does_not_exist")


def test_notification_disabled_rule_skipped():
    engine = NotificationEngine()
    rule = AlertRule(
        rule_id="r1",
        metric_name="train_loss",
        condition="gt",
        threshold=0.0,
        cooldown_steps=0,
        enabled=False,
    )
    engine.add_rule(rule)
    alerts = engine.evaluate(step=1, metrics={"train_loss": 999.0})
    assert len(alerts) == 0


def test_notification_history_accumulated():
    engine = NotificationEngine()
    rule = AlertRule(
        rule_id="r1",
        metric_name="loss",
        condition="gt",
        threshold=0.5,
        cooldown_steps=0,
    )
    engine.add_rule(rule)
    engine.evaluate(step=1, metrics={"loss": 1.0})
    engine.evaluate(step=2, metrics={"loss": 1.5})
    assert len(engine.history) == 2


def test_notification_gte_lte_conditions():
    engine = NotificationEngine()
    rule_gte = AlertRule(
        rule_id="gte", metric_name="x", condition="gte",
        threshold=1.0, cooldown_steps=0,
    )
    rule_lte = AlertRule(
        rule_id="lte", metric_name="x", condition="lte",
        threshold=1.0, cooldown_steps=0,
    )
    engine.add_rule(rule_gte)
    engine.add_rule(rule_lte)
    alerts = engine.evaluate(step=1, metrics={"x": 1.0})
    rule_ids = {a.rule_id for a in alerts}
    assert "gte" in rule_ids
    assert "lte" in rule_ids


# ===========================================================================
# AutopilotEngine tests
# ===========================================================================


def test_autopilot_off_returns_empty(tmp_path):
    engine = AutopilotEngine(run_dir=str(tmp_path), mode="off")
    actions = engine.evaluate(step=1, metrics={"val_loss": 1.0})
    assert actions == []


def test_autopilot_suggest_returns_proposed(tmp_path):
    engine = AutopilotEngine(run_dir=str(tmp_path), mode="suggest")
    rule = AutopilotRule(
        rule_id="plat",
        condition="plateau",
        metric_name="val_loss",
        params={"window": 3, "epsilon": 0.01, "cooldown": 0},
        action={"op": "set", "target": "lr", "value": 0.0001},
        confidence="high",
    )
    engine.add_rule(rule)
    # Feed identical values to trigger plateau
    all_actions = []
    for i in range(1, 6):
        actions = engine.evaluate(step=i, metrics={"val_loss": 1.0})
        all_actions.extend(actions)
    # Should trigger and be "proposed" in suggest mode
    assert len(all_actions) >= 1
    assert all(a.status == "proposed" for a in all_actions)


def test_autopilot_auto_returns_applied(tmp_path):
    engine = AutopilotEngine(run_dir=str(tmp_path), mode="auto")
    rule = AutopilotRule(
        rule_id="plat",
        condition="plateau",
        metric_name="val_loss",
        params={"window": 3, "epsilon": 0.01, "cooldown": 0},
        action={"op": "set", "target": "lr", "value": 0.0001},
        confidence="high",
    )
    engine.add_rule(rule)
    # Patch _apply_action to avoid filesystem writes
    all_actions = []
    with mock.patch.object(engine, "_apply_action"):
        for i in range(1, 6):
            actions = engine.evaluate(step=i, metrics={"val_loss": 1.0})
            all_actions.extend(actions)
    assert len(all_actions) >= 1
    assert any(a.status == "applied" for a in all_actions)


def test_autopilot_auto_low_confidence_proposed(tmp_path):
    engine = AutopilotEngine(run_dir=str(tmp_path), mode="auto")
    rule = AutopilotRule(
        rule_id="plat",
        condition="plateau",
        metric_name="val_loss",
        params={"window": 3, "epsilon": 0.01, "cooldown": 0},
        action={"op": "set", "target": "lr", "value": 0.0001},
        confidence="low",
    )
    engine.add_rule(rule)
    all_actions = []
    for i in range(1, 6):
        actions = engine.evaluate(step=i, metrics={"val_loss": 1.0})
        all_actions.extend(actions)
    assert len(all_actions) >= 1
    assert all(a.status == "proposed" for a in all_actions)


def test_autopilot_plateau_evaluation(tmp_path):
    engine = AutopilotEngine(run_dir=str(tmp_path), mode="suggest")
    rule = AutopilotRule(
        rule_id="plat",
        condition="plateau",
        metric_name="val_loss",
        params={"window": 5, "epsilon": 0.001, "cooldown": 0},
        action={"op": "set", "target": "lr", "value": 0.0001},
    )
    engine.add_rule(rule)
    all_actions = []
    for i in range(1, 8):
        actions = engine.evaluate(step=i, metrics={"val_loss": 1.0})
        all_actions.extend(actions)
    assert len(all_actions) >= 1
    assert "plateau" in all_actions[0].condition_met.lower()


def test_autopilot_divergence_evaluation(tmp_path):
    engine = AutopilotEngine(run_dir=str(tmp_path), mode="suggest")
    rule = AutopilotRule(
        rule_id="div",
        condition="divergence",
        metric_name="val_loss",
        params={"window": 5, "threshold": 1.0, "cooldown": 0},
        action={"op": "set", "target": "lr", "value": 0.0001},
    )
    engine.add_rule(rule)
    # Feed increasing values
    all_actions = []
    for i in range(1, 8):
        actions = engine.evaluate(step=i, metrics={"val_loss": float(i)})
        all_actions.extend(actions)
    assert len(all_actions) >= 1
    assert "diverge" in all_actions[0].condition_met.lower()


def test_autopilot_overfitting_evaluation(tmp_path):
    engine = AutopilotEngine(run_dir=str(tmp_path), mode="suggest")
    rule = AutopilotRule(
        rule_id="overfit",
        condition="overfitting",
        metric_name="val_loss",
        params={"ratio_threshold": 0.5, "train_metric": "train_loss", "val_metric": "val_loss"},
        action={"op": "enable", "target": "dropout"},
    )
    engine.add_rule(rule)
    # train_loss / val_loss = 0.1 / 1.0 = 0.1 < 0.5
    actions = engine.evaluate(
        step=1, metrics={"train_loss": 0.1, "val_loss": 1.0}
    )
    assert len(actions) == 1
    assert "overfitting" in actions[0].condition_met.lower()


def test_autopilot_cooldown(tmp_path):
    engine = AutopilotEngine(run_dir=str(tmp_path), mode="suggest")
    rule = AutopilotRule(
        rule_id="plat",
        condition="plateau",
        metric_name="val_loss",
        params={"window": 3, "epsilon": 0.01, "cooldown": 20},
        action={"op": "set", "target": "lr", "value": 0.0001},
    )
    engine.add_rule(rule)
    # Trigger plateau
    for i in range(1, 5):
        actions = engine.evaluate(step=i, metrics={"val_loss": 1.0})
    first_fire_step = None
    for a in engine.history:
        if a.rule_id == "plat":
            first_fire_step = a.step
            break
    assert first_fire_step is not None

    # Within cooldown: should not fire again
    actions = engine.evaluate(step=first_fire_step + 1, metrics={"val_loss": 1.0})
    assert len(actions) == 0


def test_autopilot_load_guidelines(tmp_path):
    guidelines = tmp_path / "guidelines.yaml"
    guidelines.write_text(
        "rules:\n"
        "  - id: g1\n"
        "    condition: plateau\n"
        "    metric: val_loss\n"
        "    params:\n"
        "      window: 5\n"
        "      epsilon: 0.001\n"
        "    action:\n"
        "      op: set\n"
        "      target: lr\n"
        "      value: 0.0001\n"
        "    confidence: high\n"
        "    description: Reduce LR on plateau\n"
    )
    engine = AutopilotEngine(run_dir=str(tmp_path), mode="suggest")
    count = engine.load_guidelines(str(guidelines))
    assert count == 1
    rules = engine.get_rules()
    assert len(rules) == 1
    assert rules[0].rule_id == "g1"
    assert rules[0].condition == "plateau"


def test_autopilot_invalid_mode(tmp_path):
    with pytest.raises(ValueError, match="Invalid mode"):
        AutopilotEngine(run_dir=str(tmp_path), mode="bad")


def test_autopilot_set_mode(tmp_path):
    engine = AutopilotEngine(run_dir=str(tmp_path), mode="off")
    assert engine.mode == "off"
    engine.set_mode("suggest")
    assert engine.mode == "suggest"
    engine.set_mode("auto")
    assert engine.mode == "auto"
    with pytest.raises(ValueError):
        engine.set_mode("invalid")


def test_autopilot_add_remove_rule(tmp_path):
    engine = AutopilotEngine(run_dir=str(tmp_path), mode="off")
    rule = AutopilotRule(rule_id="r1", condition="plateau", metric_name="loss")
    engine.add_rule(rule)
    assert len(engine.get_rules()) == 1
    removed = engine.remove_rule("r1")
    assert removed is True
    assert len(engine.get_rules()) == 0
    removed2 = engine.remove_rule("r1")
    assert removed2 is False


def test_autopilot_disabled_rule_skipped(tmp_path):
    engine = AutopilotEngine(run_dir=str(tmp_path), mode="suggest")
    rule = AutopilotRule(
        rule_id="r1",
        condition="plateau",
        metric_name="val_loss",
        params={"window": 3, "epsilon": 100},
        enabled=False,
    )
    engine.add_rule(rule)
    for i in range(1, 5):
        actions = engine.evaluate(step=i, metrics={"val_loss": 1.0})
    assert len(actions) == 0


def test_autopilot_history_accumulates(tmp_path):
    engine = AutopilotEngine(run_dir=str(tmp_path), mode="suggest")
    rule = AutopilotRule(
        rule_id="plat",
        condition="plateau",
        metric_name="val_loss",
        params={"window": 3, "epsilon": 0.01, "cooldown": 0},
        action={"op": "set", "target": "lr", "value": 0.0001},
    )
    engine.add_rule(rule)
    total_actions = 0
    for i in range(1, 20):
        actions = engine.evaluate(step=i, metrics={"val_loss": 1.0})
        total_actions += len(actions)
    assert len(engine.history) == total_actions
    assert total_actions > 0
