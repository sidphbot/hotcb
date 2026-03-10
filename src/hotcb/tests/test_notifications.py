"""Tests for hotcb.server.notifications — alert and notification engine."""
from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import AsyncMock

import pytest

from hotcb.server.notifications import (
    Alert,
    AlertRule,
    LogChannel,
    NotificationEngine,
    WebSocketChannel,
    _alert_to_dict,
    _rule_to_dict,
    make_metrics_subscriber,
    router,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _engine() -> NotificationEngine:
    return NotificationEngine()


def _rule(
    rule_id: str = "r1",
    metric_name: str = "loss",
    condition: str = "gt",
    threshold: float = 1.0,
    window: int = 5,
    cooldown_steps: int = 50,
    enabled: bool = True,
    channels: list[str] | None = None,
) -> AlertRule:
    return AlertRule(
        rule_id=rule_id,
        metric_name=metric_name,
        condition=condition,
        threshold=threshold,
        window=window,
        cooldown_steps=cooldown_steps,
        channels=channels or ["websocket"],
        enabled=enabled,
    )


# ---------------------------------------------------------------------------
# Condition tests
# ---------------------------------------------------------------------------


class TestGtCondition:
    def test_fires_when_above(self):
        eng = _engine()
        eng.add_rule(_rule(condition="gt", threshold=1.0))
        alerts = eng.evaluate(1, {"loss": 1.5})
        assert len(alerts) == 1
        assert alerts[0].condition == "gt"

    def test_does_not_fire_at_threshold(self):
        eng = _engine()
        eng.add_rule(_rule(condition="gt", threshold=1.0))
        alerts = eng.evaluate(1, {"loss": 1.0})
        assert len(alerts) == 0

    def test_does_not_fire_below(self):
        eng = _engine()
        eng.add_rule(_rule(condition="gt", threshold=1.0))
        alerts = eng.evaluate(1, {"loss": 0.5})
        assert len(alerts) == 0


class TestLtCondition:
    def test_fires_when_below(self):
        eng = _engine()
        eng.add_rule(_rule(condition="lt", threshold=0.5))
        alerts = eng.evaluate(1, {"loss": 0.3})
        assert len(alerts) == 1

    def test_does_not_fire_at_threshold(self):
        eng = _engine()
        eng.add_rule(_rule(condition="lt", threshold=0.5))
        alerts = eng.evaluate(1, {"loss": 0.5})
        assert len(alerts) == 0


class TestGteCondition:
    def test_fires_at_threshold(self):
        eng = _engine()
        eng.add_rule(_rule(condition="gte", threshold=1.0))
        alerts = eng.evaluate(1, {"loss": 1.0})
        assert len(alerts) == 1

    def test_fires_above(self):
        eng = _engine()
        eng.add_rule(_rule(condition="gte", threshold=1.0))
        alerts = eng.evaluate(1, {"loss": 1.1})
        assert len(alerts) == 1

    def test_does_not_fire_below(self):
        eng = _engine()
        eng.add_rule(_rule(condition="gte", threshold=1.0))
        alerts = eng.evaluate(1, {"loss": 0.9})
        assert len(alerts) == 0


class TestLteCondition:
    def test_fires_at_threshold(self):
        eng = _engine()
        eng.add_rule(_rule(condition="lte", threshold=0.5))
        alerts = eng.evaluate(1, {"loss": 0.5})
        assert len(alerts) == 1

    def test_fires_below(self):
        eng = _engine()
        eng.add_rule(_rule(condition="lte", threshold=0.5))
        alerts = eng.evaluate(1, {"loss": 0.3})
        assert len(alerts) == 1

    def test_does_not_fire_above(self):
        eng = _engine()
        eng.add_rule(_rule(condition="lte", threshold=0.5))
        alerts = eng.evaluate(1, {"loss": 0.6})
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Plateau detection
# ---------------------------------------------------------------------------


class TestPlateauDetection:
    def test_plateau_fires_when_flat(self):
        eng = _engine()
        eng.add_rule(_rule(condition="plateau", threshold=0.01, window=5, cooldown_steps=0))
        # Feed 5 nearly-identical values
        for step in range(5):
            alerts = eng.evaluate(step, {"loss": 0.500})
        assert len(alerts) == 1
        assert alerts[0].condition == "plateau"

    def test_plateau_does_not_fire_with_variance(self):
        eng = _engine()
        eng.add_rule(_rule(condition="plateau", threshold=0.01, window=5, cooldown_steps=0))
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
        for step, v in enumerate(values):
            alerts = eng.evaluate(step, {"loss": v})
        assert len(alerts) == 0

    def test_plateau_needs_enough_data(self):
        eng = _engine()
        eng.add_rule(_rule(condition="plateau", threshold=0.01, window=5, cooldown_steps=0))
        # Only 3 data points — not enough for window=5
        for step in range(3):
            alerts = eng.evaluate(step, {"loss": 0.5})
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------


class TestSpikeDetection:
    def test_spike_fires_on_sudden_change(self):
        eng = _engine()
        eng.add_rule(_rule(condition="spike", threshold=0.5, window=5, cooldown_steps=0))
        # Feed stable values then a spike
        for step in range(4):
            eng.evaluate(step, {"loss": 1.0})
        alerts = eng.evaluate(4, {"loss": 3.0})
        assert len(alerts) == 1
        assert alerts[0].condition == "spike"

    def test_spike_does_not_fire_on_gradual_change(self):
        eng = _engine()
        eng.add_rule(_rule(condition="spike", threshold=0.5, window=5, cooldown_steps=0))
        values = [1.0, 1.1, 1.2, 1.3, 1.4]
        alerts_all = []
        for step, v in enumerate(values):
            alerts_all.extend(eng.evaluate(step, {"loss": v}))
        assert len(alerts_all) == 0

    def test_spike_needs_at_least_two_points(self):
        eng = _engine()
        eng.add_rule(_rule(condition="spike", threshold=0.01, window=5, cooldown_steps=0))
        alerts = eng.evaluate(0, {"loss": 100.0})
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_cooldown_prevents_rapid_refiring(self):
        eng = _engine()
        eng.add_rule(_rule(condition="gt", threshold=1.0, cooldown_steps=10))
        # Fires at step 1
        alerts = eng.evaluate(1, {"loss": 2.0})
        assert len(alerts) == 1
        # Does NOT fire at step 5 (within cooldown)
        alerts = eng.evaluate(5, {"loss": 2.0})
        assert len(alerts) == 0
        # Fires again at step 11 (cooldown elapsed)
        alerts = eng.evaluate(11, {"loss": 2.0})
        assert len(alerts) == 1

    def test_zero_cooldown_fires_every_time(self):
        eng = _engine()
        eng.add_rule(_rule(condition="gt", threshold=1.0, cooldown_steps=0))
        for step in range(5):
            alerts = eng.evaluate(step, {"loss": 2.0})
            assert len(alerts) == 1


# ---------------------------------------------------------------------------
# Rule management
# ---------------------------------------------------------------------------


class TestRuleManagement:
    def test_add_and_remove_rule(self):
        eng = _engine()
        r = _rule(rule_id="test1")
        eng.add_rule(r)
        assert "test1" in eng.rules
        eng.remove_rule("test1")
        assert "test1" not in eng.rules

    def test_remove_nonexistent_rule_is_noop(self):
        eng = _engine()
        eng.remove_rule("nonexistent")  # should not raise

    def test_disabled_rules_dont_fire(self):
        eng = _engine()
        eng.add_rule(_rule(enabled=False, condition="gt", threshold=0.0))
        alerts = eng.evaluate(1, {"loss": 100.0})
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Alert history
# ---------------------------------------------------------------------------


class TestAlertHistory:
    def test_history_accumulates(self):
        eng = _engine()
        eng.add_rule(_rule(condition="gt", threshold=0.0, cooldown_steps=0))
        eng.evaluate(1, {"loss": 1.0})
        eng.evaluate(2, {"loss": 2.0})
        eng.evaluate(3, {"loss": 3.0})
        assert len(eng.history) == 3
        assert eng.history[0].step == 1
        assert eng.history[2].step == 3


# ---------------------------------------------------------------------------
# LogChannel
# ---------------------------------------------------------------------------


class TestLogChannel:
    def test_log_channel_sends(self):
        logger = logging.getLogger("test_notifications_log")
        ch = LogChannel(logger=logger)
        alert = Alert(
            alert_id="a1",
            rule_id="r1",
            metric_name="loss",
            condition="gt",
            current_value=2.0,
            threshold=1.0,
            step=10,
            wall_time=time.time(),
            message="test message",
        )
        result = asyncio.get_event_loop().run_until_complete(ch.send(alert))
        assert result is True


# ---------------------------------------------------------------------------
# WebSocketChannel
# ---------------------------------------------------------------------------


class TestWebSocketChannel:
    def test_websocket_channel_calls_broadcast(self):
        broadcast = AsyncMock()
        ch = WebSocketChannel(broadcast)
        alert = Alert(
            alert_id="a1",
            rule_id="r1",
            metric_name="loss",
            condition="gt",
            current_value=2.0,
            threshold=1.0,
            step=10,
            wall_time=time.time(),
            message="test",
        )
        result = asyncio.get_event_loop().run_until_complete(ch.send(alert))
        assert result is True
        broadcast.assert_awaited_once()
        args = broadcast.call_args
        assert args[0][0] == "alerts"

    def test_websocket_channel_handles_error(self):
        broadcast = AsyncMock(side_effect=RuntimeError("boom"))
        ch = WebSocketChannel(broadcast)
        alert = Alert(
            alert_id="a1",
            rule_id="r1",
            metric_name="loss",
            condition="gt",
            current_value=2.0,
            threshold=1.0,
            step=10,
            wall_time=time.time(),
            message="test",
        )
        result = asyncio.get_event_loop().run_until_complete(ch.send(alert))
        assert result is False


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_dispatch_sends_to_channels(self):
        eng = _engine()
        mock_ch = AsyncMock()
        mock_ch.send = AsyncMock(return_value=True)
        eng.register_channel("websocket", mock_ch)
        eng.add_rule(_rule(condition="gt", threshold=0.0, channels=["websocket"]))
        alerts = eng.evaluate(1, {"loss": 1.0})
        asyncio.get_event_loop().run_until_complete(eng.dispatch(alerts))
        mock_ch.send.assert_awaited_once()


# ---------------------------------------------------------------------------
# Metrics subscriber (tailer integration)
# ---------------------------------------------------------------------------


class TestMetricsSubscriber:
    def test_subscriber_evaluates_records(self):
        eng = _engine()
        eng.add_rule(_rule(condition="gt", threshold=0.5, cooldown_steps=0))
        sub = make_metrics_subscriber(eng)
        records = [
            {"step": 1, "epoch": 0, "wall_time": 1.0, "metrics": {"loss": 0.8}},
            {"step": 2, "epoch": 0, "wall_time": 2.0, "metrics": {"loss": 0.3}},
        ]
        asyncio.get_event_loop().run_until_complete(sub("metrics", records))
        # Only step 1 should fire (0.8 > 0.5), step 2 does not (0.3 < 0.5)
        assert len(eng.history) == 1
        assert eng.history[0].step == 1


# ---------------------------------------------------------------------------
# Metric not present in rule
# ---------------------------------------------------------------------------


class TestMetricMismatch:
    def test_rule_ignores_missing_metric(self):
        eng = _engine()
        eng.add_rule(_rule(metric_name="accuracy", condition="gt", threshold=0.9))
        alerts = eng.evaluate(1, {"loss": 5.0})
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_alert_to_dict(self):
        a = Alert(
            alert_id="x",
            rule_id="r",
            metric_name="loss",
            condition="gt",
            current_value=1.5,
            threshold=1.0,
            step=10,
            wall_time=123.0,
            message="msg",
        )
        d = _alert_to_dict(a)
        assert d["alert_id"] == "x"
        assert d["current_value"] == 1.5

    def test_rule_to_dict(self):
        r = _rule()
        d = _rule_to_dict(r)
        assert d["rule_id"] == "r1"
        assert d["condition"] == "gt"


# ---------------------------------------------------------------------------
# REST endpoint tests via TestClient
# ---------------------------------------------------------------------------


class TestRESTEndpoints:
    """Test the FastAPI notification router endpoints."""

    @pytest.fixture()
    def client(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        engine = NotificationEngine()
        app.state.notification_engine = engine
        app.include_router(router)
        return TestClient(app)

    def test_list_rules_empty(self, client):
        resp = client.get("/api/notifications/rules")
        assert resp.status_code == 200
        assert resp.json()["rules"] == []

    def test_add_rule(self, client):
        body = {
            "metric_name": "loss",
            "condition": "gt",
            "threshold": 1.0,
        }
        resp = client.post("/api/notifications/rules", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "added"
        assert data["rule"]["metric_name"] == "loss"

        # Verify it appears in listing
        resp2 = client.get("/api/notifications/rules")
        assert len(resp2.json()["rules"]) == 1

    def test_add_rule_with_custom_id(self, client):
        body = {
            "rule_id": "my_rule",
            "metric_name": "acc",
            "condition": "lte",
            "threshold": 0.5,
        }
        resp = client.post("/api/notifications/rules", json=body)
        assert resp.json()["rule"]["rule_id"] == "my_rule"

    def test_delete_rule(self, client):
        # Add then delete
        body = {"rule_id": "del_me", "metric_name": "loss", "condition": "gt", "threshold": 1.0}
        client.post("/api/notifications/rules", json=body)
        resp = client.delete("/api/notifications/rules/del_me")
        assert resp.status_code == 200
        assert resp.json()["status"] == "removed"

        # Confirm gone
        resp2 = client.get("/api/notifications/rules")
        assert len(resp2.json()["rules"]) == 0

    def test_delete_nonexistent_rule(self, client):
        resp = client.delete("/api/notifications/rules/nope")
        assert resp.status_code == 404

    def test_history_empty(self, client):
        resp = client.get("/api/notifications/history")
        assert resp.status_code == 200
        assert resp.json()["alerts"] == []

    def test_test_fire_rule(self, client):
        # Add a rule first
        body = {"rule_id": "test_r", "metric_name": "loss", "condition": "gt", "threshold": 1.0}
        client.post("/api/notifications/rules", json=body)

        # Test-fire it
        resp = client.post("/api/notifications/rules/test_r/test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "test_fired"
        assert "[TEST]" in data["alert"]["message"]

        # Confirm alert in history
        resp2 = client.get("/api/notifications/history")
        assert len(resp2.json()["alerts"]) == 1

    def test_test_fire_nonexistent_rule(self, client):
        resp = client.post("/api/notifications/rules/nope/test")
        assert resp.status_code == 404

    def test_invalid_condition_rejected(self, client):
        body = {
            "metric_name": "loss",
            "condition": "invalid_cond",
            "threshold": 1.0,
        }
        resp = client.post("/api/notifications/rules", json=body)
        assert resp.status_code == 422  # validation error


# ---------------------------------------------------------------------------
# Multiple rules
# ---------------------------------------------------------------------------


class TestMultipleRules:
    def test_multiple_rules_on_different_metrics(self):
        eng = _engine()
        eng.add_rule(_rule(rule_id="r_loss", metric_name="loss", condition="gt", threshold=1.0))
        eng.add_rule(_rule(rule_id="r_acc", metric_name="accuracy", condition="lt", threshold=0.5))
        alerts = eng.evaluate(1, {"loss": 2.0, "accuracy": 0.3})
        assert len(alerts) == 2
        rule_ids = {a.rule_id for a in alerts}
        assert rule_ids == {"r_loss", "r_acc"}

    def test_only_matching_rule_fires(self):
        eng = _engine()
        eng.add_rule(_rule(rule_id="r_loss", metric_name="loss", condition="gt", threshold=1.0))
        eng.add_rule(_rule(rule_id="r_acc", metric_name="accuracy", condition="lt", threshold=0.5))
        alerts = eng.evaluate(1, {"loss": 0.5, "accuracy": 0.3})
        assert len(alerts) == 1
        assert alerts[0].rule_id == "r_acc"
