"""
hotcb.server.config — Centralized dashboard configuration.

Resolution order: defaults -> YAML -> env vars -> CLI overrides.
Env vars use the HOTCB_ prefix (e.g. HOTCB_PORT, HOTCB_POLL_INTERVAL).
CLI overrides have the highest priority.
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

log = logging.getLogger("hotcb.server.config")

# ---------------------------------------------------------------------------
# Frozen sub-config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8421
    poll_interval: float = 0.5
    history_limit_metrics: int = 500
    history_limit_applied: int = 200
    ws_initial_burst: int = 200
    ws_max_retries: int = 20
    ws_retry_base: float = 3.0
    ws_retry_cap: float = 30.0


@dataclass(frozen=True)
class ChartConfig:
    max_render_points: int = 2000
    line_tension: float = 0.15
    forecast_dash: tuple = (6, 3)
    mutation_dash: tuple = (3, 4)
    annotation_stagger_rows: int = 10
    annotation_min_distance: int = 70


@dataclass(frozen=True)
class AutopilotConfig:
    divergence_threshold: float = 2.0
    ratio_threshold: float = 0.5
    ai_min_interval: int = 10
    ai_max_wait: int = 200
    ai_default_cadence: int = 50


@dataclass(frozen=True)
class UIConfig:
    state_save_interval: int = 5000
    alert_poll_interval: int = 15000
    manifold_refresh_interval: int = 10000
    recipe_refresh_interval: int = 5000
    forecast_poll_interval: int = 5000
    forecast_step_cadence: int = 10
    forecast_batch_size: int = 8
    staged_change_threshold: float = 0.005
    health_ema_alpha: float = 0.1


# ---------------------------------------------------------------------------
# Env var mapping: HOTCB_<KEY> -> (section, field, type_cast)
# ---------------------------------------------------------------------------

_ENV_MAP: Dict[str, tuple] = {
    "HOTCB_HOST": ("server", "host", str),
    "HOTCB_PORT": ("server", "port", int),
    "HOTCB_POLL_INTERVAL": ("server", "poll_interval", float),
    "HOTCB_HISTORY_LIMIT_METRICS": ("server", "history_limit_metrics", int),
    "HOTCB_HISTORY_LIMIT_APPLIED": ("server", "history_limit_applied", int),
    "HOTCB_WS_INITIAL_BURST": ("server", "ws_initial_burst", int),
    "HOTCB_WS_MAX_RETRIES": ("server", "ws_max_retries", int),
    "HOTCB_WS_RETRY_BASE": ("server", "ws_retry_base", float),
    "HOTCB_WS_RETRY_CAP": ("server", "ws_retry_cap", float),
    "HOTCB_MAX_RENDER_POINTS": ("chart", "max_render_points", int),
    "HOTCB_LINE_TENSION": ("chart", "line_tension", float),
    "HOTCB_ANNOTATION_STAGGER_ROWS": ("chart", "annotation_stagger_rows", int),
    "HOTCB_ANNOTATION_MIN_DISTANCE": ("chart", "annotation_min_distance", int),
    "HOTCB_DIVERGENCE_THRESHOLD": ("autopilot", "divergence_threshold", float),
    "HOTCB_RATIO_THRESHOLD": ("autopilot", "ratio_threshold", float),
    "HOTCB_AI_MIN_INTERVAL": ("autopilot", "ai_min_interval", int),
    "HOTCB_AI_MAX_WAIT": ("autopilot", "ai_max_wait", int),
    "HOTCB_AI_DEFAULT_CADENCE": ("autopilot", "ai_default_cadence", int),
    "HOTCB_STATE_SAVE_INTERVAL": ("ui", "state_save_interval", int),
    "HOTCB_ALERT_POLL_INTERVAL": ("ui", "alert_poll_interval", int),
    "HOTCB_MANIFOLD_REFRESH_INTERVAL": ("ui", "manifold_refresh_interval", int),
    "HOTCB_RECIPE_REFRESH_INTERVAL": ("ui", "recipe_refresh_interval", int),
    "HOTCB_FORECAST_POLL_INTERVAL": ("ui", "forecast_poll_interval", int),
    "HOTCB_FORECAST_STEP_CADENCE": ("ui", "forecast_step_cadence", int),
    "HOTCB_FORECAST_BATCH_SIZE": ("ui", "forecast_batch_size", int),
    "HOTCB_STAGED_CHANGE_THRESHOLD": ("ui", "staged_change_threshold", float),
    "HOTCB_HEALTH_EMA_ALPHA": ("ui", "health_ema_alpha", float),
}

# CLI override key -> (section, field)
_CLI_MAP: Dict[str, tuple] = {
    "host": ("server", "host"),
    "port": ("server", "port"),
    "poll_interval": ("server", "poll_interval"),
    "history_limit_metrics": ("server", "history_limit_metrics"),
    "history_limit_applied": ("server", "history_limit_applied"),
    "ws_initial_burst": ("server", "ws_initial_burst"),
    "ws_max_retries": ("server", "ws_max_retries"),
    "ws_retry_base": ("server", "ws_retry_base"),
    "ws_retry_cap": ("server", "ws_retry_cap"),
    "max_render_points": ("chart", "max_render_points"),
    "line_tension": ("chart", "line_tension"),
    "forecast_dash": ("chart", "forecast_dash"),
    "mutation_dash": ("chart", "mutation_dash"),
    "annotation_stagger_rows": ("chart", "annotation_stagger_rows"),
    "annotation_min_distance": ("chart", "annotation_min_distance"),
    "divergence_threshold": ("autopilot", "divergence_threshold"),
    "ratio_threshold": ("autopilot", "ratio_threshold"),
    "ai_min_interval": ("autopilot", "ai_min_interval"),
    "ai_max_wait": ("autopilot", "ai_max_wait"),
    "ai_default_cadence": ("autopilot", "ai_default_cadence"),
    "state_save_interval": ("ui", "state_save_interval"),
    "alert_poll_interval": ("ui", "alert_poll_interval"),
    "manifold_refresh_interval": ("ui", "manifold_refresh_interval"),
    "recipe_refresh_interval": ("ui", "recipe_refresh_interval"),
    "forecast_poll_interval": ("ui", "forecast_poll_interval"),
    "forecast_step_cadence": ("ui", "forecast_step_cadence"),
    "forecast_batch_size": ("ui", "forecast_batch_size"),
    "staged_change_threshold": ("ui", "staged_change_threshold"),
    "health_ema_alpha": ("ui", "health_ema_alpha"),
}

# Section name -> frozen dataclass type
_SECTION_CLASSES: Dict[str, type] = {
    "server": ServerConfig,
    "chart": ChartConfig,
    "autopilot": AutopilotConfig,
    "ui": UIConfig,
}


# ---------------------------------------------------------------------------
# DashboardConfig — mutable aggregate
# ---------------------------------------------------------------------------


@dataclass
class DashboardConfig:
    """Centralized dashboard configuration.

    Aggregates frozen sub-configs for server, chart, autopilot, and UI settings.
    The ``controls`` list is populated from ``MutableState.describe_all()`` when
    actuators are available (Phase 4); defaults to empty for Phase 1.
    """

    server: ServerConfig = field(default_factory=ServerConfig)
    chart: ChartConfig = field(default_factory=ChartConfig)
    autopilot: AutopilotConfig = field(default_factory=AutopilotConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    run_dir: str = ""
    controls: List[dict] = field(default_factory=list)

    @classmethod
    def load(
        cls,
        run_dir: str,
        yaml_path: Optional[str] = None,
        **cli_overrides: Any,
    ) -> "DashboardConfig":
        """Load config with resolution: defaults -> YAML -> env vars -> CLI.

        Parameters
        ----------
        run_dir : str
            Run directory path (set once, immutable after startup).
        yaml_path : str | None
            Path to a YAML config file. Missing file is silently ignored.
        **cli_overrides
            Keyword arguments that override any setting. Keys are flat field
            names (e.g. ``port=9000``, ``poll_interval=1.0``).
        """
        # Start with per-section override dicts (empty = all defaults)
        overrides: Dict[str, Dict[str, Any]] = {
            "server": {},
            "chart": {},
            "autopilot": {},
            "ui": {},
        }

        # --- Layer 1: YAML ---
        if yaml_path:
            yaml_data = _load_yaml(yaml_path)
            for section, values in yaml_data.items():
                if section in overrides and isinstance(values, dict):
                    overrides[section].update(values)

        # --- Layer 2: Env vars ---
        for env_key, (section, field_name, cast_fn) in _ENV_MAP.items():
            raw = os.environ.get(env_key)
            if raw is not None:
                try:
                    overrides[section][field_name] = cast_fn(raw)
                except (ValueError, TypeError) as exc:
                    log.warning(
                        "Ignoring invalid env var %s=%r: %s", env_key, raw, exc
                    )

        # --- Layer 3: CLI overrides (highest priority) ---
        for key, value in cli_overrides.items():
            if key in _CLI_MAP:
                section, field_name = _CLI_MAP[key]
                overrides[section][field_name] = value

        # --- Build frozen sub-configs ---
        server = ServerConfig(**overrides["server"]) if overrides["server"] else ServerConfig()
        chart = ChartConfig(**overrides["chart"]) if overrides["chart"] else ChartConfig()
        autopilot = AutopilotConfig(**overrides["autopilot"]) if overrides["autopilot"] else AutopilotConfig()
        ui = UIConfig(**overrides["ui"]) if overrides["ui"] else UIConfig()

        return cls(
            server=server,
            chart=chart,
            autopilot=autopilot,
            ui=ui,
            run_dir=run_dir,
            controls=[],
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict.

        Tuples (e.g. ``forecast_dash``, ``mutation_dash``) are converted to
        lists so that ``json.dumps()`` works without a custom encoder.
        """
        def _section_dict(obj: Any) -> dict:
            d = asdict(obj)
            # Convert any tuple values to lists for JSON serialization
            for k, v in d.items():
                if isinstance(v, tuple):
                    d[k] = list(v)
            return d

        return {
            "server": _section_dict(self.server),
            "chart": _section_dict(self.chart),
            "autopilot": _section_dict(self.autopilot),
            "ui": _section_dict(self.ui),
            "run_dir": self.run_dir,
            "controls": list(self.controls),
        }


# ---------------------------------------------------------------------------
# Controls from MutableState
# ---------------------------------------------------------------------------


def controls_from_mutable_state(ms: Any) -> list:
    """Build controls list from MutableState.describe_all() or empty if None."""
    if ms is None:
        return []
    try:
        return ms.describe_all()
    except Exception:
        return []


def controls_from_actuator_file(run_dir: str) -> list:
    """Read actuator descriptions from hotcb.actuators.json.

    The kernel writes this file on startup and after mutations so the
    dashboard server can discover controls without a live MutableState
    reference (filesystem IPC).
    """
    import json
    import os

    path = os.path.join(run_dir, "hotcb.actuators.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("controls", [])
    except Exception:
        return []


def controls_from_applied_ledger(run_dir: str) -> list:
    """Reconstruct control specs from applied JSONL when MutableState is unavailable.

    Used in serve mode (no live training) to populate slider controls from
    historical data.  Scans the applied ledger for ``set_params`` operations
    and infers actuator-like specs.
    """
    import json
    import os

    applied_path = os.path.join(run_dir, "hotcb.applied.jsonl")
    if not os.path.exists(applied_path):
        return []

    # Gather all param values seen, grouped by module
    param_values: dict = {}  # param_key -> {module, values: [float]}
    try:
        with open(applied_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("decision") != "applied":
                    continue
                module = rec.get("module", "opt")
                params = rec.get("params") or rec.get("payload")
                if not params or not isinstance(params, dict):
                    continue
                for k, v in params.items():
                    if not isinstance(v, (int, float)):
                        continue
                    if k not in param_values:
                        param_values[k] = {"module": module, "values": []}
                    param_values[k]["values"].append(v)
    except Exception:
        return []

    # Also scan last metrics for current values
    metrics_path = os.path.join(run_dir, "hotcb.metrics.jsonl")
    latest_metrics: dict = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "rb") as f:
                # Read last 2KB for efficiency
                f.seek(max(0, os.path.getsize(metrics_path) - 2048))
                tail = f.read().decode("utf-8", errors="replace")
            for line in tail.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    latest_metrics = rec.get("metrics", {})
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass

    controls = []
    for key, info in param_values.items():
        values = info["values"]
        module = info["module"]
        last_val = values[-1]
        all_vals = values

        # Check for current value in latest metrics
        current = latest_metrics.get(key, last_val)

        # Determine type from value characteristics
        is_log = (key in ("lr", "learning_rate", "weight_decay", "wd")
                  or (min(all_vals) > 0 and max(all_vals) / max(min(all_vals), 1e-15) > 100))

        if is_log and all(v > 0 for v in all_vals):
            min_val = min(all_vals) * 0.1
            max_val = max(all_vals) * 10
            controls.append({
                "param_key": key,
                "type": "log_float",
                "label": key,
                "group": "optimizer" if module == "opt" else module,
                "min": min_val,
                "max": max_val,
                "step": 0.01,
                "log_base": 10,
                "choices": None,
                "current": current,
                "state": "untouched",
            })
        else:
            min_val = min(0, min(all_vals) * 0.5) if all_vals else 0
            max_val = max(1, max(all_vals) * 2) if all_vals else 1
            controls.append({
                "param_key": key,
                "type": "float",
                "label": key,
                "group": "loss" if module == "loss" else ("optimizer" if module == "opt" else module),
                "min": min_val,
                "max": max_val,
                "step": 0.01,
                "log_base": None,
                "choices": None,
                "current": current,
                "state": "untouched",
            })

    return controls


def controls_from_capabilities(run_dir: str) -> list:
    """Generate controls from hotcb.capabilities.json mutable_state_keys.

    Used when an external training (Lightning/HF) registered mutable state
    but didn't write hotcb.actuators.json (e.g., older kernel or adapter).
    Infers control types from key names (lambda/weight → float, ramp/lr → log_float).
    Also reads latest metrics for current values.
    """
    import json

    caps_path = os.path.join(run_dir, "hotcb.capabilities.json")
    if not os.path.exists(caps_path):
        return []
    try:
        with open(caps_path, "r", encoding="utf-8") as f:
            caps = json.load(f)
    except Exception:
        return []

    keys = caps.get("mutable_state_keys", [])
    if not keys:
        return []

    # Read latest metrics for current values
    latest_metrics: dict = {}
    metrics_path = os.path.join(run_dir, "hotcb.metrics.jsonl")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "rb") as f:
                f.seek(max(0, os.path.getsize(metrics_path) - 4096))
                tail = f.read().decode("utf-8", errors="replace")
            for line in tail.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    latest_metrics.update(rec.get("metrics", {}))
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass

    # Detect optimizer names for lr controls
    opt_names = caps.get("optimizer_names", [])

    controls = []

    # Add lr controls per optimizer
    for i, opt_name in enumerate(opt_names):
        key = "lr" if i == 0 else f"lr_{i}"
        current = latest_metrics.get(key, latest_metrics.get("lr", 1e-3))
        controls.append({
            "param_key": key,
            "type": "log_float",
            "label": f"lr ({opt_name})" if len(opt_names) > 1 else "lr",
            "group": "optimizer",
            "min": 1e-7,
            "max": 1.0,
            "step": 0.01,
            "log_base": 10,
            "choices": None,
            "current": current,
            "state": "untouched",
        })

    # Add mutable state keys
    for key in keys:
        current = latest_metrics.get(key, 1.0)
        # Infer type from name
        is_log = key.startswith("lr") or key == "weight_decay" or key == "wd"
        is_ramp = key.startswith("ramp_") or key.endswith("_end") or key.endswith("_start")

        if is_log:
            controls.append({
                "param_key": key,
                "type": "log_float",
                "label": key,
                "group": "optimizer",
                "min": 1e-7,
                "max": 1.0,
                "step": 0.01,
                "log_base": 10,
                "choices": None,
                "current": current,
                "state": "untouched",
            })
        elif is_ramp:
            controls.append({
                "param_key": key,
                "type": "float",
                "label": key,
                "group": "scheduler",
                "min": 0,
                "max": max(5000, current * 2) if isinstance(current, (int, float)) else 5000,
                "step": 1,
                "log_base": None,
                "choices": None,
                "current": current,
                "state": "untouched",
            })
        else:
            # lambda_*, aug_*, or other float params
            controls.append({
                "param_key": key,
                "type": "float",
                "label": key,
                "group": "loss" if key.startswith("lambda") else "training",
                "min": 0.0,
                "max": max(10.0, current * 3) if isinstance(current, (int, float)) else 10.0,
                "step": 0.01,
                "log_base": None,
                "choices": None,
                "current": current,
                "state": "untouched",
            })

    return controls


def default_optimizer_controls() -> list:
    """Return default optimizer controls as a last-resort fallback."""
    return [
        {
            "param_key": "lr",
            "type": "log_float",
            "label": "lr",
            "group": "optimizer",
            "min": 1e-7,
            "max": 1.0,
            "step": 0.01,
            "log_base": 10,
            "choices": None,
            "current": 1e-3,
            "state": "untouched",
        },
        {
            "param_key": "weight_decay",
            "type": "log_float",
            "label": "weight_decay",
            "group": "optimizer",
            "min": 1e-7,
            "max": 1.0,
            "step": 0.01,
            "log_base": 10,
            "choices": None,
            "current": 1e-4,
            "state": "untouched",
        },
    ]


# ---------------------------------------------------------------------------
# YAML loader — optional pyyaml dependency
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> dict:
    """Load a YAML file, returning {} on missing file or import error."""
    if not os.path.exists(path):
        return {}
    try:
        import yaml
    except ImportError:
        log.warning("pyyaml not installed — ignoring config file %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log.warning("Failed to load config YAML %s: %s", path, exc)
        return {}
