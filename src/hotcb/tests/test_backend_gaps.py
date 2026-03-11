"""Tests for backend gap fixes: SmtpChannel, guidelines, UI mode, autopilot defaults."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from hotcb.server.notifications import Alert, SmtpChannel
from hotcb.server.autopilot import AutopilotEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_alert(**overrides) -> Alert:
    defaults = dict(
        alert_id="a1",
        rule_id="r1",
        metric_name="loss",
        condition="gt",
        current_value=2.0,
        threshold=1.0,
        step=100,
        wall_time=1000.0,
        message="loss = 2.0 (gt 1.0) at step 100",
    )
    defaults.update(overrides)
    return Alert(**defaults)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# SmtpChannel tests
# ---------------------------------------------------------------------------


class TestSmtpChannel:
    def test_smtp_channel_send_success(self):
        ch = SmtpChannel(
            smtp_host="smtp.example.com",
            smtp_port=587,
            from_addr="noreply@example.com",
            to_addrs=["user@example.com"],
            username="user",
            password="pass",
            use_tls=True,
        )
        alert = _make_alert()

        mock_smtp_instance = MagicMock()
        with patch("smtplib.SMTP", return_value=mock_smtp_instance) as mock_smtp_cls:
            result = asyncio.get_event_loop().run_until_complete(ch.send(alert))

        assert result is True
        mock_smtp_cls.assert_called_once_with("smtp.example.com", 587)
        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once_with("user", "pass")
        mock_smtp_instance.sendmail.assert_called_once()
        mock_smtp_instance.quit.assert_called_once()

    def test_smtp_channel_send_no_tls(self):
        ch = SmtpChannel(
            smtp_host="smtp.example.com",
            smtp_port=25,
            from_addr="noreply@example.com",
            to_addrs=["user@example.com"],
            use_tls=False,
        )
        alert = _make_alert()

        mock_smtp_instance = MagicMock()
        with patch("smtplib.SMTP", return_value=mock_smtp_instance):
            result = asyncio.get_event_loop().run_until_complete(ch.send(alert))

        assert result is True
        mock_smtp_instance.starttls.assert_not_called()
        mock_smtp_instance.login.assert_not_called()

    def test_smtp_channel_send_failure(self):
        ch = SmtpChannel(
            smtp_host="smtp.example.com",
            smtp_port=587,
            from_addr="noreply@example.com",
            to_addrs=["user@example.com"],
            use_tls=True,
        )
        alert = _make_alert()

        with patch("smtplib.SMTP", side_effect=ConnectionRefusedError("refused")):
            result = asyncio.get_event_loop().run_until_complete(ch.send(alert))

        assert result is False

    def test_smtp_channel_email_content(self):
        ch = SmtpChannel(
            smtp_host="smtp.example.com",
            smtp_port=587,
            from_addr="noreply@example.com",
            to_addrs=["a@b.com", "c@d.com"],
            use_tls=False,
        )
        alert = _make_alert(metric_name="val_loss", step=42)

        mock_smtp_instance = MagicMock()
        with patch("smtplib.SMTP", return_value=mock_smtp_instance):
            asyncio.get_event_loop().run_until_complete(ch.send(alert))

        call_args = mock_smtp_instance.sendmail.call_args
        assert call_args[0][0] == "noreply@example.com"
        assert call_args[0][1] == ["a@b.com", "c@d.com"]
        msg_text = call_args[0][2]
        assert "val_loss" in msg_text
        assert "step" in msg_text.lower()


# ---------------------------------------------------------------------------
# Default guidelines tests
# ---------------------------------------------------------------------------


class TestDefaultGuidelines:
    def test_default_guidelines_file_exists(self):
        from hotcb.server.guidelines import DEFAULT_GUIDELINES_PATH
        assert os.path.exists(DEFAULT_GUIDELINES_PATH)

    def test_default_guidelines_valid_yaml(self):
        from hotcb.server.guidelines import DEFAULT_GUIDELINES_PATH
        import yaml

        with open(DEFAULT_GUIDELINES_PATH, "r") as f:
            data = yaml.safe_load(f)
        assert "rules" in data
        assert isinstance(data["rules"], list)
        assert len(data["rules"]) >= 4

    def test_default_guidelines_rule_ids(self):
        from hotcb.server.guidelines import DEFAULT_GUIDELINES_PATH
        import yaml

        with open(DEFAULT_GUIDELINES_PATH, "r") as f:
            data = yaml.safe_load(f)
        ids = {r["id"] for r in data["rules"]}
        # Must contain at least these core rules
        assert {"plateau_lr_reduce", "divergence_emergency_lr",
                "overfitting_wd_increase", "grad_spike_lr_reduce"}.issubset(ids)

    def test_default_guidelines_all_have_required_fields(self):
        from hotcb.server.guidelines import DEFAULT_GUIDELINES_PATH
        import yaml

        with open(DEFAULT_GUIDELINES_PATH, "r") as f:
            data = yaml.safe_load(f)
        for rule in data["rules"]:
            assert "id" in rule
            assert "condition" in rule
            assert "metric" in rule
            assert "action" in rule
            assert "confidence" in rule


# ---------------------------------------------------------------------------
# AutopilotEngine.with_default_guidelines tests
# ---------------------------------------------------------------------------


class TestAutopilotWithDefaultGuidelines:
    def test_creates_engine_with_rules(self, tmp_dir):
        engine = AutopilotEngine.with_default_guidelines(tmp_dir, mode="off")
        assert engine.mode == "off"
        rules = engine.get_rules()
        assert len(rules) >= 4

    def test_rule_ids_match_defaults(self, tmp_dir):
        engine = AutopilotEngine.with_default_guidelines(tmp_dir, mode="suggest")
        rule_ids = {r.rule_id for r in engine.get_rules()}
        assert "plateau_lr_reduce" in rule_ids
        assert "divergence_emergency_lr" in rule_ids
        assert "overfitting_wd_increase" in rule_ids
        assert "grad_spike_lr_reduce" in rule_ids

    def test_with_default_guidelines_mode(self, tmp_dir):
        engine = AutopilotEngine.with_default_guidelines(tmp_dir, mode="auto")
        assert engine.mode == "auto"

    def test_with_default_guidelines_missing_file(self, tmp_dir):
        """If the default guidelines file is missing, engine should still create."""
        with patch("hotcb.server.autopilot.os.path.exists", return_value=False):
            engine = AutopilotEngine.with_default_guidelines(tmp_dir, mode="off")
        assert len(engine.get_rules()) == 0


# ---------------------------------------------------------------------------
# UI mode endpoint tests
# ---------------------------------------------------------------------------


class TestUIModeEndpoints:
    @pytest.fixture
    def client(self, tmp_dir):
        pytest.importorskip("fastapi")
        from hotcb.server.app import create_app
        from starlette.testclient import TestClient

        app = create_app(tmp_dir, poll_interval=60)
        return TestClient(app)

    def test_get_default_mode(self, client):
        r = client.get("/api/ui/mode")
        assert r.status_code == 200
        assert r.json()["mode"] == "engineer"

    def test_set_mode_engineer(self, client):
        r = client.post("/api/ui/mode", json={"mode": "engineer"})
        assert r.status_code == 200, f"Response: {r.json()}"
        assert r.json()["mode"] == "engineer"

    def test_set_mode_education(self, client):
        r = client.post("/api/ui/mode", json={"mode": "education"})
        assert r.status_code == 200
        assert r.json()["mode"] == "education"

    def test_set_mode_vibe_coder(self, client):
        r = client.post("/api/ui/mode", json={"mode": "vibe_coder"})
        assert r.status_code == 200
        assert r.json()["mode"] == "vibe_coder"

    def test_set_invalid_mode(self, client):
        r = client.post("/api/ui/mode", json={"mode": "invalid"})
        assert r.status_code == 400

    def test_mode_roundtrip(self, client):
        client.post("/api/ui/mode", json={"mode": "education"})
        r = client.get("/api/ui/mode")
        assert r.json()["mode"] == "education"

    def test_mode_persists_to_file(self, client, tmp_dir):
        client.post("/api/ui/mode", json={"mode": "vibe_coder"})
        ui_path = os.path.join(tmp_dir, "hotcb.ui.json")
        assert os.path.exists(ui_path)
        with open(ui_path, "r") as f:
            data = json.load(f)
        assert data["mode"] == "vibe_coder"

    def test_get_mode_from_persisted_file(self, tmp_dir):
        """Write hotcb.ui.json manually, then verify GET reads it."""
        pytest.importorskip("fastapi")
        from hotcb.server.app import create_app
        from starlette.testclient import TestClient

        ui_path = os.path.join(tmp_dir, "hotcb.ui.json")
        with open(ui_path, "w") as f:
            json.dump({"mode": "education"}, f)

        app = create_app(tmp_dir, poll_interval=60)
        client = TestClient(app)
        r = client.get("/api/ui/mode")
        assert r.json()["mode"] == "education"
