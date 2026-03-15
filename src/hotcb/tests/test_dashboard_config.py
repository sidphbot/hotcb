"""Tests for hotcb.server.config — DashboardConfig dataclasses + loader."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from hotcb.server.config import (
    AutopilotConfig,
    ChartConfig,
    DashboardConfig,
    ServerConfig,
    UIConfig,
)

# Skip all endpoint tests if fastapi not installed
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestConfigDefaults:
    """test_config_defaults — all sub-configs have documented defaults."""

    def test_server_defaults(self):
        s = ServerConfig()
        assert s.host == "0.0.0.0"
        assert s.port == 8421
        assert s.poll_interval == 0.5
        assert s.history_limit_metrics == 500
        assert s.history_limit_applied == 200
        assert s.ws_initial_burst == 200
        assert s.ws_max_retries == 20
        assert s.ws_retry_base == 3.0
        assert s.ws_retry_cap == 30.0

    def test_chart_defaults(self):
        c = ChartConfig()
        assert c.max_render_points == 2000
        assert c.line_tension == 0.15
        assert c.forecast_dash == (6, 3)
        assert c.mutation_dash == (3, 4)
        assert c.annotation_stagger_rows == 10
        assert c.annotation_min_distance == 70

    def test_autopilot_defaults(self):
        a = AutopilotConfig()
        assert a.divergence_threshold == 2.0
        assert a.ratio_threshold == 0.5
        assert a.ai_min_interval == 10
        assert a.ai_max_wait == 200
        assert a.ai_default_cadence == 50

    def test_ui_defaults(self):
        u = UIConfig()
        assert u.state_save_interval == 5000
        assert u.alert_poll_interval == 15000
        assert u.manifold_refresh_interval == 10000
        assert u.recipe_refresh_interval == 5000
        assert u.forecast_poll_interval == 5000
        assert u.forecast_step_cadence == 10
        assert u.forecast_batch_size == 8
        assert u.staged_change_threshold == 0.005
        assert u.health_ema_alpha == 0.1

    def test_dashboard_config_all_defaults(self):
        cfg = DashboardConfig()
        assert isinstance(cfg.server, ServerConfig)
        assert isinstance(cfg.chart, ChartConfig)
        assert isinstance(cfg.autopilot, AutopilotConfig)
        assert isinstance(cfg.ui, UIConfig)
        assert cfg.run_dir == ""
        assert cfg.controls == []


class TestConfigFromYaml:
    """test_config_from_yaml — YAML overrides applied, others preserved."""

    def test_config_from_yaml(self, tmp_dir):
        yaml_path = os.path.join(tmp_dir, "dashboard.yaml")
        with open(yaml_path, "w") as f:
            f.write("server:\n  port: 9000\nchart:\n  line_tension: 0.3\n")

        cfg = DashboardConfig.load(tmp_dir, yaml_path=yaml_path)
        # Overrides applied
        assert cfg.server.port == 9000
        assert cfg.chart.line_tension == 0.3
        # Other defaults preserved
        assert cfg.server.host == "0.0.0.0"
        assert cfg.server.poll_interval == 0.5
        assert cfg.chart.max_render_points == 2000
        assert cfg.autopilot.divergence_threshold == 2.0

    def test_config_from_yaml_missing_file(self, tmp_dir):
        """Nonexistent YAML -> all defaults, no error."""
        cfg = DashboardConfig.load(
            tmp_dir, yaml_path=os.path.join(tmp_dir, "nonexistent.yaml")
        )
        assert cfg.server.port == 8421
        assert cfg.chart.line_tension == 0.15
        assert cfg.run_dir == tmp_dir


class TestConfigFromEnv:
    """test_config_from_env — HOTCB_PORT=9000 etc. override defaults."""

    def test_config_from_env(self, tmp_dir, monkeypatch):
        monkeypatch.setenv("HOTCB_PORT", "9000")
        monkeypatch.setenv("HOTCB_POLL_INTERVAL", "1.0")
        cfg = DashboardConfig.load(tmp_dir)
        assert cfg.server.port == 9000
        assert cfg.server.poll_interval == 1.0

    def test_config_env_overrides_yaml(self, tmp_dir, monkeypatch):
        """Env beats YAML."""
        yaml_path = os.path.join(tmp_dir, "dashboard.yaml")
        with open(yaml_path, "w") as f:
            f.write("server:\n  port: 9000\n")

        monkeypatch.setenv("HOTCB_PORT", "8000")
        cfg = DashboardConfig.load(tmp_dir, yaml_path=yaml_path)
        assert cfg.server.port == 8000


class TestConfigCliOverrides:
    """test_config_cli_overrides_all — CLI overrides beat both."""

    def test_config_cli_overrides_all(self, tmp_dir, monkeypatch):
        yaml_path = os.path.join(tmp_dir, "dashboard.yaml")
        with open(yaml_path, "w") as f:
            f.write("server:\n  port: 9000\n")

        monkeypatch.setenv("HOTCB_PORT", "8000")
        cfg = DashboardConfig.load(tmp_dir, yaml_path=yaml_path, port=7000)
        assert cfg.server.port == 7000


class TestConfigToDict:
    """test_config_to_dict_roundtrip — to_dict() is JSON serializable, has all sections."""

    def test_config_to_dict_roundtrip(self):
        cfg = DashboardConfig()
        d = cfg.to_dict()
        # Must be JSON-serializable
        serialized = json.dumps(d)
        roundtripped = json.loads(serialized)
        # All sections present
        assert "server" in roundtripped
        assert "chart" in roundtripped
        assert "autopilot" in roundtripped
        assert "ui" in roundtripped
        assert "run_dir" in roundtripped
        assert "controls" in roundtripped

    def test_config_run_dir_in_dict(self):
        cfg = DashboardConfig(run_dir="/tmp/x")
        d = cfg.to_dict()
        assert d["run_dir"] == "/tmp/x"

    def test_chart_tuples_serialize_as_arrays(self):
        """forecast_dash and mutation_dash are tuples in Python, arrays in JSON."""
        cfg = DashboardConfig()
        d = cfg.to_dict()
        # In the dict they should be lists (JSON arrays)
        assert isinstance(d["chart"]["forecast_dash"], list)
        assert isinstance(d["chart"]["mutation_dash"], list)
        assert d["chart"]["forecast_dash"] == [6, 3]
        assert d["chart"]["mutation_dash"] == [3, 4]


class TestConfigEndpoint:
    """test_config_endpoint_returns_full — GET /api/config returns all sections."""

    @pytest.fixture
    def tmp_run_dir(self):
        with tempfile.TemporaryDirectory() as d:
            # Create metrics file so _resolve_active_run_dir returns this dir
            with open(os.path.join(d, "hotcb.metrics.jsonl"), "w") as f:
                pass
            yield d

    @pytest.fixture
    def client(self, tmp_run_dir):
        from starlette.testclient import TestClient
        from hotcb.server.app import create_app

        app = create_app(tmp_run_dir, poll_interval=60)
        return TestClient(app)

    @pytest.fixture
    def client_with_yaml(self, tmp_run_dir):
        from starlette.testclient import TestClient
        from hotcb.server.app import create_app

        yaml_path = os.path.join(tmp_run_dir, "hotcb.dashboard.yaml")
        with open(yaml_path, "w") as f:
            f.write("server:\n  port: 9999\n")

        app = create_app(tmp_run_dir, poll_interval=60, config_yaml=yaml_path)
        return TestClient(app)

    def test_config_endpoint_returns_full(self, client):
        r = client.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        assert "server" in data
        assert "chart" in data
        assert "autopilot" in data
        assert "ui" in data
        assert "run_dir" in data
        assert "controls" in data

    def test_config_endpoint_reflects_overrides(self, client_with_yaml):
        r = client_with_yaml.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        assert data["server"]["port"] == 9999


# ====================================================================
# Phase 4: Dynamic Controls from Actuators
# ====================================================================


class TestControlsFromMutableState:
    """Phase 4 — /api/config controls populated from MutableState."""

    @pytest.fixture
    def tmp_run_dir(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "hotcb.metrics.jsonl"), "w") as f:
                pass
            yield d

    @pytest.fixture
    def _make_mutable_state(self):
        """Helper to create a MutableState with common actuators."""
        from hotcb.actuators import (
            ActuatorType,
            ApplyResult,
            HotcbActuator,
            MutableState,
        )

        def _factory(specs=None):
            if specs is None:
                specs = [
                    dict(
                        param_key="lr",
                        type=ActuatorType.LOG_FLOAT,
                        apply_fn=lambda v, e: ApplyResult(success=True),
                        label="Learning Rate",
                        group="optimizer",
                        min_value=1e-7,
                        max_value=1.0,
                        current_value=1e-3,
                    ),
                    dict(
                        param_key="weight_decay",
                        type=ActuatorType.LOG_FLOAT,
                        apply_fn=lambda v, e: ApplyResult(success=True),
                        label="Weight Decay",
                        group="optimizer",
                        min_value=0.0,
                        max_value=1.0,
                        current_value=1e-4,
                    ),
                    dict(
                        param_key="recon_w",
                        type=ActuatorType.FLOAT,
                        apply_fn=lambda v, e: ApplyResult(success=True),
                        label="Reconstruction Weight",
                        group="loss",
                        min_value=0.0,
                        max_value=100.0,
                        current_value=1.0,
                    ),
                ]
            actuators = [HotcbActuator(**s) for s in specs]
            return MutableState(actuators)

        return _factory

    @pytest.fixture
    def client_with_ms(self, tmp_run_dir, _make_mutable_state):
        from starlette.testclient import TestClient
        from hotcb.server.app import create_app

        ms = _make_mutable_state()
        app = create_app(tmp_run_dir, poll_interval=60)
        app.state.mutable_state = ms
        return TestClient(app)

    @pytest.fixture
    def client_no_ms(self, tmp_run_dir):
        from starlette.testclient import TestClient
        from hotcb.server.app import create_app

        app = create_app(tmp_run_dir, poll_interval=60)
        return TestClient(app)

    def test_config_controls_from_mutable_state(self, client_with_ms):
        """Controls populated from MutableState.describe_all()."""
        r = client_with_ms.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        controls = data["controls"]
        assert len(controls) == 3
        keys = {c["param_key"] for c in controls}
        assert keys == {"lr", "weight_decay", "recon_w"}
        # Each entry has required fields
        for c in controls:
            assert "param_key" in c
            assert "type" in c
            assert "label" in c
            assert "group" in c
            assert "current" in c

    def test_config_controls_defaults_when_no_mutable_state(self, client_no_ms):
        """No MutableState -> default optimizer controls are returned."""
        r = client_no_ms.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        # Should have default lr and weight_decay controls
        assert len(data["controls"]) >= 2
        keys = [c["param_key"] for c in data["controls"]]
        assert "lr" in keys
        assert "weight_decay" in keys

    def test_config_controls_types_match_actuators(
        self, tmp_run_dir, _make_mutable_state
    ):
        """Control types match actuator types."""
        from starlette.testclient import TestClient
        from hotcb.server.app import create_app
        from hotcb.actuators import ActuatorType, ApplyResult, HotcbActuator, MutableState

        ms = _make_mutable_state(
            [
                dict(
                    param_key="lr",
                    type=ActuatorType.LOG_FLOAT,
                    apply_fn=lambda v, e: ApplyResult(success=True),
                    group="optimizer",
                    min_value=1e-7,
                    max_value=1.0,
                    current_value=1e-3,
                ),
                dict(
                    param_key="recon_w",
                    type=ActuatorType.FLOAT,
                    apply_fn=lambda v, e: ApplyResult(success=True),
                    group="loss",
                    min_value=0.0,
                    max_value=10.0,
                    current_value=1.0,
                ),
                dict(
                    param_key="use_augment",
                    type=ActuatorType.BOOL,
                    apply_fn=lambda v, e: ApplyResult(success=True),
                    group="custom",
                    current_value=True,
                ),
            ]
        )

        app = create_app(tmp_run_dir, poll_interval=60)
        app.state.mutable_state = ms
        client = TestClient(app)

        r = client.get("/api/config")
        controls = r.json()["controls"]
        type_map = {c["param_key"]: c["type"] for c in controls}
        assert type_map["lr"] == "log_float"
        assert type_map["recon_w"] == "float"
        assert type_map["use_augment"] == "bool"

    def test_config_controls_groups_present(self, client_with_ms):
        """Controls have correct group field."""
        r = client_with_ms.get("/api/config")
        controls = r.json()["controls"]
        group_map = {c["param_key"]: c["group"] for c in controls}
        assert group_map["lr"] == "optimizer"
        assert group_map["weight_decay"] == "optimizer"
        assert group_map["recon_w"] == "loss"

    def test_control_state_endpoint_uses_mutable_state(self, client_with_ms):
        """GET /api/state/controls returns live values from MutableState."""
        r = client_with_ms.get("/api/state/controls")
        assert r.status_code == 200
        data = r.json()
        assert "controls" in data
        controls = data["controls"]
        assert len(controls) == 3
        # Verify current values are present
        val_map = {c["param_key"]: c["current"] for c in controls}
        assert val_map["lr"] == pytest.approx(1e-3)
        assert val_map["recon_w"] == pytest.approx(1.0)

    def test_control_state_endpoint_defaults_when_no_ms(self, client_no_ms):
        """GET /api/state/controls returns default controls when no MutableState."""
        r = client_no_ms.get("/api/state/controls")
        assert r.status_code == 200
        data = r.json()
        # Should have default lr and weight_decay controls
        assert len(data["controls"]) >= 2
        keys = [c["param_key"] for c in data["controls"]]
        assert "lr" in keys


class TestControlsFromMutableStateFunction:
    """Unit tests for controls_from_mutable_state()."""

    def test_none_returns_empty(self):
        from hotcb.server.config import controls_from_mutable_state
        assert controls_from_mutable_state(None) == []

    def test_with_mutable_state(self):
        from hotcb.server.config import controls_from_mutable_state
        from hotcb.actuators import (
            ActuatorType,
            ApplyResult,
            HotcbActuator,
            MutableState,
        )

        ms = MutableState([
            HotcbActuator(
                param_key="lr",
                type=ActuatorType.LOG_FLOAT,
                apply_fn=lambda v, e: ApplyResult(success=True),
                group="optimizer",
                min_value=1e-7,
                max_value=1.0,
                current_value=0.001,
            ),
        ])
        result = controls_from_mutable_state(ms)
        assert len(result) == 1
        assert result[0]["param_key"] == "lr"
        assert result[0]["type"] == "log_float"
        assert result[0]["current"] == pytest.approx(0.001)


# ====================================================================
# Phase 6: Magic Number Extraction
# ====================================================================


class TestTailerUsesConfigPollInterval:
    """Phase 6 — tailer created with config.server.poll_interval."""

    @pytest.fixture
    def tmp_run_dir(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "hotcb.metrics.jsonl"), "w") as f:
                pass
            yield d

    def test_tailer_uses_config_poll_interval(self, tmp_run_dir):
        """Tailer created with config's poll_interval, not hardcoded default."""
        from hotcb.server.app import create_app

        app = create_app(tmp_run_dir, poll_interval=2.5)
        tailer = app.state.tailer
        assert tailer._poll_interval == 2.5

    def test_tailer_uses_default_poll_interval(self, tmp_run_dir):
        """When no poll_interval override, tailer uses default 0.5."""
        from hotcb.server.app import create_app

        app = create_app(tmp_run_dir)
        tailer = app.state.tailer
        assert tailer._poll_interval == 0.5


class TestHistoryLimitsFromConfig:
    """Phase 6 — metrics/applied history endpoints respect config limits."""

    @pytest.fixture
    def tmp_run_dir(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "hotcb.metrics.jsonl"), "w") as f:
                pass
            yield d

    @pytest.fixture
    def populated_run_dir(self):
        """Run dir with 50 metrics records and 50 applied records."""
        with tempfile.TemporaryDirectory() as d:
            metrics_path = os.path.join(d, "hotcb.metrics.jsonl")
            with open(metrics_path, "w") as f:
                for i in range(50):
                    f.write(json.dumps({"step": i, "metrics": {"loss": 1.0 - i * 0.01}}) + "\n")
            applied_path = os.path.join(d, "hotcb.applied.jsonl")
            with open(applied_path, "w") as f:
                for i in range(50):
                    f.write(json.dumps({"step": i, "module": "opt", "decision": "applied", "params": {"lr": 0.001}}) + "\n")
            yield d

    def test_metrics_history_uses_config_limit(self, populated_run_dir):
        """GET /api/metrics/history with no last_n uses config limit."""
        from starlette.testclient import TestClient
        from hotcb.server.app import create_app

        yaml_path = os.path.join(populated_run_dir, "cfg.yaml")
        with open(yaml_path, "w") as f:
            f.write("server:\n  history_limit_metrics: 10\n")

        app = create_app(populated_run_dir, poll_interval=60, config_yaml=yaml_path)
        client = TestClient(app)
        r = client.get("/api/metrics/history")
        assert r.status_code == 200
        records = r.json()["records"]
        assert len(records) == 10

    def test_metrics_history_explicit_last_n_overrides_config(self, populated_run_dir):
        """GET /api/metrics/history?last_n=5 overrides config limit."""
        from starlette.testclient import TestClient
        from hotcb.server.app import create_app

        yaml_path = os.path.join(populated_run_dir, "cfg.yaml")
        with open(yaml_path, "w") as f:
            f.write("server:\n  history_limit_metrics: 10\n")

        app = create_app(populated_run_dir, poll_interval=60, config_yaml=yaml_path)
        client = TestClient(app)
        r = client.get("/api/metrics/history?last_n=5")
        assert r.status_code == 200
        records = r.json()["records"]
        assert len(records) == 5

    def test_applied_history_uses_config_limit(self, populated_run_dir):
        """GET /api/applied/history with no last_n uses config limit."""
        from starlette.testclient import TestClient
        from hotcb.server.app import create_app

        yaml_path = os.path.join(populated_run_dir, "cfg.yaml")
        with open(yaml_path, "w") as f:
            f.write("server:\n  history_limit_applied: 7\n")

        app = create_app(populated_run_dir, poll_interval=60, config_yaml=yaml_path)
        client = TestClient(app)
        r = client.get("/api/applied/history")
        assert r.status_code == 200
        records = r.json()["records"]
        assert len(records) == 7


class TestAutopilotThresholdsFromConfig:
    """Phase 6 — autopilot engine uses config thresholds as defaults."""

    def test_divergence_uses_config_threshold(self):
        """Divergence rule uses config threshold when rule doesn't specify one."""
        from hotcb.server.autopilot import AutopilotEngine, AutopilotRule

        config = AutopilotConfig(divergence_threshold=5.0)
        engine = AutopilotEngine(run_dir="/tmp/test", mode="suggest", config=config)
        engine.add_rule(AutopilotRule(
            rule_id="div1",
            condition="divergence",
            metric_name="val_loss",
            params={"window": 3},  # no threshold specified — should use config's 5.0
            action={"module": "opt", "op": "set_params", "params": {"lr": 0.0001}},
            confidence="high",
        ))

        # Feed metric history that diverges by 4.0 (below config threshold of 5.0)
        for i in range(3):
            engine.evaluate(i, {"val_loss": 1.0})
        actions = engine.evaluate(3, {"val_loss": 5.0})
        # 5.0 - 1.0 = 4.0, which is < 5.0 threshold, so no divergence
        assert len(actions) == 0

    def test_divergence_rule_threshold_overrides_config(self):
        """Rule-specified threshold takes precedence over config."""
        from hotcb.server.autopilot import AutopilotEngine, AutopilotRule

        config = AutopilotConfig(divergence_threshold=100.0)  # very high config threshold
        engine = AutopilotEngine(run_dir="/tmp/test", mode="suggest", config=config)
        engine.add_rule(AutopilotRule(
            rule_id="div2",
            condition="divergence",
            metric_name="val_loss",
            params={"window": 3, "threshold": 1.0},  # rule specifies threshold=1.0
            action={"module": "opt", "op": "set_params", "params": {"lr": 0.0001}},
            confidence="high",
        ))

        for i in range(3):
            engine.evaluate(i, {"val_loss": 1.0})
        actions = engine.evaluate(3, {"val_loss": 5.0})
        # 5.0 - 1.0 = 4.0, which is > rule threshold 1.0, so divergence detected
        assert len(actions) == 1
        assert actions[0].condition_met.startswith("Metric diverged")

    def test_overfitting_uses_config_ratio_threshold(self):
        """Overfitting rule uses config ratio_threshold when rule doesn't specify one."""
        from hotcb.server.autopilot import AutopilotEngine, AutopilotRule

        config = AutopilotConfig(ratio_threshold=0.01)  # very low threshold
        engine = AutopilotEngine(run_dir="/tmp/test", mode="suggest", config=config)
        engine.add_rule(AutopilotRule(
            rule_id="over1",
            condition="overfitting",
            metric_name="",
            params={"train_metric": "train_loss", "val_metric": "val_loss"},
            action={"module": "opt", "op": "set_params", "params": {"lr": 0.0001}},
            confidence="high",
        ))

        # train/val ratio = 0.1/1.0 = 0.1, which is > 0.01 threshold (no overfitting)
        actions = engine.evaluate(0, {"train_loss": 0.1, "val_loss": 1.0})
        assert len(actions) == 0

    def test_autopilot_no_config_uses_hardcoded_defaults(self):
        """Without config, autopilot uses hardcoded defaults (2.0, 0.5)."""
        from hotcb.server.autopilot import AutopilotEngine, AutopilotRule

        engine = AutopilotEngine(run_dir="/tmp/test", mode="suggest")  # no config
        engine.add_rule(AutopilotRule(
            rule_id="div_default",
            condition="divergence",
            metric_name="val_loss",
            params={"window": 3},
            action={"module": "opt", "op": "set_params", "params": {"lr": 0.0001}},
            confidence="high",
        ))

        # Feed metric history that diverges by 3.0 (above default 2.0)
        for i in range(3):
            engine.evaluate(i, {"val_loss": 1.0})
        actions = engine.evaluate(3, {"val_loss": 4.0})
        # 4.0 - 1.0 = 3.0, > default threshold 2.0
        assert len(actions) == 1


class TestAICadenceFromConfig:
    """Phase 6 — AI engine uses config cadence params."""

    def test_ai_engine_uses_config_min_interval(self):
        """AI engine min_interval from AutopilotConfig."""
        from hotcb.server.ai_engine import LLMAutopilotEngine, AIConfig

        ap_config = AutopilotConfig(ai_min_interval=25)
        engine = LLMAutopilotEngine(
            run_dir="/tmp/test",
            config=AIConfig(api_key="test-key"),
            autopilot_config=ap_config,
        )
        assert engine._min_interval == 25

    def test_ai_engine_uses_config_max_wait(self):
        """AI engine max_wait from AutopilotConfig."""
        from hotcb.server.ai_engine import LLMAutopilotEngine, AIConfig

        ap_config = AutopilotConfig(ai_max_wait=500)
        engine = LLMAutopilotEngine(
            run_dir="/tmp/test",
            config=AIConfig(api_key="test-key"),
            autopilot_config=ap_config,
        )
        assert engine._max_wait == 500

    def test_ai_engine_uses_config_default_cadence(self):
        """AI engine cadence from AutopilotConfig when AIConfig has default."""
        from hotcb.server.ai_engine import LLMAutopilotEngine, AIConfig

        ap_config = AutopilotConfig(ai_default_cadence=100)
        engine = LLMAutopilotEngine(
            run_dir="/tmp/test",
            config=AIConfig(api_key="test-key"),
            autopilot_config=ap_config,
        )
        assert engine.config.cadence == 100

    def test_ai_engine_min_interval_governs_should_invoke(self):
        """should_invoke respects config min_interval."""
        from hotcb.server.ai_engine import LLMAutopilotEngine, AIConfig

        ap_config = AutopilotConfig(ai_min_interval=30)
        engine = LLMAutopilotEngine(
            run_dir="/tmp/test",
            config=AIConfig(api_key="test-key"),
            autopilot_config=ap_config,
        )
        engine._last_invoked_step = 0

        # Step 15 is within min_interval=30, should not invoke
        assert engine.should_invoke(15, []) is False
        # Step 31 is past min_interval=30, periodic cadence check applies
        assert engine.should_invoke(31, [{"alert": "test"}]) is True

    def test_ai_engine_defaults_without_autopilot_config(self):
        """AI engine uses hardcoded defaults when no autopilot_config."""
        from hotcb.server.ai_engine import LLMAutopilotEngine, AIConfig

        engine = LLMAutopilotEngine(
            run_dir="/tmp/test",
            config=AIConfig(),
        )
        assert engine._min_interval == 10
        assert engine._max_wait == 200
        assert engine.config.cadence == 50
