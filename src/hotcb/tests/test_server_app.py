"""Tests for hotcb.server.app — FastAPI application."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from hotcb.util import append_jsonl

# Skip all tests if fastapi not installed
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from hotcb.server.app import (
    ConnectionManager,
    _build_status,
    _discover_metric_names,
    _read_tail,
    create_app,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def populated_dir(tmp_dir):
    """A run dir with some JSONL data."""
    # Metrics
    metrics_path = os.path.join(tmp_dir, "hotcb.metrics.jsonl")
    for i in range(10):
        append_jsonl(metrics_path, {
            "step": i,
            "epoch": 0,
            "wall_time": 1000.0 + i,
            "metrics": {"loss": 1.0 - i * 0.1, "val_loss": 1.0 - i * 0.08},
        })

    # Applied ledger
    applied_path = os.path.join(tmp_dir, "hotcb.applied.jsonl")
    append_jsonl(applied_path, {
        "seq": 1, "step": 5, "module": "opt", "op": "set_params",
        "decision": "applied", "payload": {"lr": 1e-3},
    })

    # Commands (empty)
    open(os.path.join(tmp_dir, "hotcb.commands.jsonl"), "w").close()

    return tmp_dir


class TestHelpers:
    """Test standalone helper functions."""

    def test_build_status_empty_dir(self, tmp_dir):
        status = _build_status(tmp_dir)
        assert status["run_dir"] == tmp_dir
        assert status["freeze"]["mode"] == "off"
        assert not status["files"]["metrics"]

    def test_build_status_with_freeze(self, tmp_dir):
        with open(os.path.join(tmp_dir, "hotcb.freeze.json"), "w") as f:
            json.dump({"mode": "prod"}, f)
        status = _build_status(tmp_dir)
        assert status["freeze"]["mode"] == "prod"

    def test_discover_metric_names(self, populated_dir):
        names = _discover_metric_names(populated_dir)
        assert "loss" in names
        assert "val_loss" in names

    def test_discover_metric_names_empty(self, tmp_dir):
        names = _discover_metric_names(tmp_dir)
        assert names == []

    def test_read_tail(self, populated_dir):
        path = os.path.join(populated_dir, "hotcb.metrics.jsonl")
        records = _read_tail(path, 3)
        assert len(records) == 3
        assert records[0]["step"] == 7

    def test_read_tail_missing_file(self, tmp_dir):
        records = _read_tail(os.path.join(tmp_dir, "nope.jsonl"), 10)
        assert records == []


class TestConnectionManager:
    """Test WebSocket connection manager."""

    def test_connect_disconnect(self):
        mgr = ConnectionManager()
        ws = object()
        mgr.connect(ws, {"metrics"})
        assert mgr.connection_count == 1
        mgr.disconnect(ws)
        assert mgr.connection_count == 0

    def test_connect_default_channels(self):
        mgr = ConnectionManager()
        ws = object()
        mgr.connect(ws)
        # Should be in all default channels
        assert mgr.connection_count == 1

    @pytest.mark.asyncio
    async def test_broadcast(self):
        mgr = ConnectionManager()

        class MockWS:
            def __init__(self):
                self.sent = []
            async def send_json(self, data):
                self.sent.append(data)

        ws1 = MockWS()
        ws2 = MockWS()
        mgr.connect(ws1, {"metrics"})
        mgr.connect(ws2, {"applied"})

        await mgr.broadcast("metrics", [{"step": 1}])
        assert len(ws1.sent) == 1
        assert ws1.sent[0]["channel"] == "metrics"
        assert len(ws2.sent) == 0  # not subscribed to metrics

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_connections(self):
        mgr = ConnectionManager()

        class DeadWS:
            async def send_json(self, data):
                raise ConnectionError("dead")

        ws = DeadWS()
        mgr.connect(ws, {"metrics"})
        assert mgr.connection_count == 1

        await mgr.broadcast("metrics", [{"step": 1}])
        assert mgr.connection_count == 0


class TestRESTEndpoints:
    """Test REST API via TestClient."""

    @pytest.fixture
    def client(self, populated_dir):
        from starlette.testclient import TestClient
        app = create_app(populated_dir, poll_interval=60)  # long interval, won't poll
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"

    def test_status(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert "freeze" in data
        assert "files" in data

    def test_metric_names(self, client):
        r = client.get("/api/metrics/names")
        assert r.status_code == 200
        names = r.json()["names"]
        assert "loss" in names

    def test_metrics_history(self, client):
        r = client.get("/api/metrics/history?last_n=5")
        assert r.status_code == 200
        records = r.json()["records"]
        assert len(records) == 5
        assert records[0]["step"] == 5

    def test_applied_history(self, client):
        r = client.get("/api/applied/history?last_n=10")
        assert r.status_code == 200
        records = r.json()["records"]
        assert len(records) == 1
        assert records[0]["module"] == "opt"
