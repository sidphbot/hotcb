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


# ---------------------------------------------------------------------------
# Phase 5: Immutable run_dir tests
# ---------------------------------------------------------------------------


class TestImmutableRunDir:
    """Phase 5: run_dir is set once at create_app() and never changes."""

    def test_no_app_state_run_dir_attribute(self, populated_dir):
        """app.state should not have a mutable run_dir; use config.run_dir."""
        app = create_app(populated_dir, poll_interval=60)
        # app.state.run_dir should NOT be set (removed in Phase 5)
        assert not hasattr(app.state, "run_dir"), (
            "app.state.run_dir should not exist; use app.state.config.run_dir"
        )

    def test_config_run_dir_is_set(self, populated_dir):
        """app.state.config.run_dir should be set to the resolved run_dir."""
        app = create_app(populated_dir, poll_interval=60)
        assert hasattr(app.state, "config")
        assert app.state.config.run_dir == populated_dir

    def test_endpoints_use_config_run_dir(self, populated_dir):
        """All endpoints should read from config.run_dir."""
        from starlette.testclient import TestClient
        app = create_app(populated_dir, poll_interval=60)
        client = TestClient(app)

        # /api/health returns the config run_dir
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["run_dir"] == populated_dir

        # /api/status returns data from config run_dir
        r = client.get("/api/status")
        assert r.status_code == 200
        assert r.json()["run_dir"] == populated_dir

        # /api/metrics/history returns data from config run_dir
        r = client.get("/api/metrics/history?last_n=3")
        assert r.status_code == 200
        assert len(r.json()["records"]) == 3

    def test_config_endpoint_returns_run_dir(self, populated_dir):
        """GET /api/config should include the immutable run_dir."""
        from starlette.testclient import TestClient
        app = create_app(populated_dir, poll_interval=60)
        client = TestClient(app)

        r = client.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        assert data["run_dir"] == populated_dir

    def test_discover_runs_reads_only(self, populated_dir):
        """/api/runs/discover should scan read-only, never mutate run_dir."""
        from starlette.testclient import TestClient
        app = create_app(populated_dir, poll_interval=60)
        client = TestClient(app)

        r = client.get("/api/runs/discover")
        assert r.status_code == 200
        runs = r.json()["runs"]
        # populated_dir has metrics, so it should be discovered
        assert len(runs) >= 1
        # config.run_dir unchanged after discover
        assert app.state.config.run_dir == populated_dir


class TestTailerNoRewire:
    """Phase 5: JsonlTailer no longer has a rewire method."""

    def test_tailer_no_rewire_method(self):
        """JsonlTailer should not have a rewire() method."""
        from hotcb.server.tailer import JsonlTailer
        tailer = JsonlTailer()
        assert not hasattr(tailer, "rewire"), (
            "JsonlTailer.rewire() should be removed in Phase 5"
        )

    def test_tailer_still_has_diagnostics(self):
        """get_cursor_offsets should still be available for diagnostics."""
        from hotcb.server.tailer import JsonlTailer
        tailer = JsonlTailer()
        path = "/tmp/nonexistent.jsonl"
        tailer.watch("test", path)
        offsets = tailer.get_cursor_offsets()
        assert "test" in offsets
        assert offsets["test"] == 0


class TestLauncherImmutableRunDir:
    """Phase 5: Launcher writes directly to run_dir, no subdirs."""

    def test_launcher_writes_to_run_dir_directly(self):
        """Launcher.start() should write JSONL files to run_dir, not subdirs."""
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            from hotcb.server.launcher import TrainingLauncher, TrainingConfig

            # Register a minimal training config
            def _noop_train(run_dir, max_steps, step_delay, stop_event):
                # Write one metric record and exit
                with open(os.path.join(run_dir, "hotcb.metrics.jsonl"), "a") as f:
                    f.write(json.dumps({"step": 0, "metrics": {"loss": 0.5}}) + "\n")

            launcher = TrainingLauncher(tmpdir)
            launcher.register_config(TrainingConfig(
                config_id="test",
                name="Test",
                description="test",
                train_fn=_noop_train,
                defaults={"max_steps": 1, "step_delay": 0.0},
            ))

            result = launcher.start(config_id="test", max_steps=1, step_delay=0.0)
            assert result.get("started") is True
            assert result["run_dir"] == tmpdir  # writes to run_dir directly

            # Wait for training to finish
            import time
            for _ in range(50):
                if not launcher.running:
                    break
                time.sleep(0.1)

            # JSONL files should be in tmpdir, not in subdirs
            assert os.path.exists(os.path.join(tmpdir, "hotcb.metrics.jsonl"))
            assert os.path.exists(os.path.join(tmpdir, "hotcb.run.json"))

            # No subdirs should have been created
            entries = os.listdir(tmpdir)
            subdirs = [e for e in entries if os.path.isdir(os.path.join(tmpdir, e))]
            assert subdirs == [], f"No subdirs expected, found: {subdirs}"

    def test_launcher_truncates_on_restart(self):
        """Starting training again should truncate JSONL files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from hotcb.server.launcher import TrainingLauncher, TrainingConfig

            step_counter = [0]

            def _counting_train(run_dir, max_steps, step_delay, stop_event):
                step_counter[0] += 1
                with open(os.path.join(run_dir, "hotcb.metrics.jsonl"), "a") as f:
                    f.write(json.dumps({
                        "step": 0, "metrics": {"loss": 0.5, "run": step_counter[0]}
                    }) + "\n")

            launcher = TrainingLauncher(tmpdir)
            launcher.register_config(TrainingConfig(
                config_id="test",
                name="Test",
                description="test",
                train_fn=_counting_train,
                defaults={"max_steps": 1, "step_delay": 0.0},
            ))

            # First run: write some data
            metrics_path = os.path.join(tmpdir, "hotcb.metrics.jsonl")
            with open(metrics_path, "w") as f:
                f.write(json.dumps({"step": 99, "metrics": {"old": True}}) + "\n")

            # Start should truncate existing data
            result = launcher.start(config_id="test", max_steps=1, step_delay=0.0)
            assert result.get("started") is True

            import time
            for _ in range(50):
                if not launcher.running:
                    break
                time.sleep(0.1)

            # Read the metrics file — old data (step 99) should be gone
            records = []
            with open(metrics_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

            # Should only have the new record, not the old step=99
            assert all(r["step"] != 99 for r in records), (
                "Old data should be truncated on restart"
            )

    def test_launcher_no_active_run_dir(self):
        """TrainingLauncher should not have _active_run_dir attribute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from hotcb.server.launcher import TrainingLauncher
            launcher = TrainingLauncher(tmpdir)
            assert not hasattr(launcher, "_active_run_dir"), (
                "_active_run_dir removed in Phase 5; launcher uses _run_dir only"
            )
