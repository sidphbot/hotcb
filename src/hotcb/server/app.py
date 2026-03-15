"""
hotcb-server — FastAPI application for the live training dashboard.

Provides:
- WebSocket endpoint for streaming metrics / applied / tune data
- REST endpoints for status, config, and commands
- Static file serving for the dashboard frontend

Start via CLI::

    hotcb serve --dir runs/exp1 --port 8421
"""
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger("hotcb.server")

try:
    from pydantic import BaseModel as _PydanticBase

    class UIModeRequest(_PydanticBase):
        mode: str
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Lazy FastAPI import — dashboard is an optional dependency
# ---------------------------------------------------------------------------

def _check_deps() -> None:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "hotcb dashboard requires: pip install 'hotcb[dashboard]'\n"
            f"Missing: {e.name}"
        ) from e


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

@dataclass
class ConnectionManager:
    """Manages active WebSocket connections with channel subscriptions."""

    _connections: Dict[str, set] = field(default_factory=dict)

    def connect(self, ws: Any, channels: Optional[Set[str]] = None) -> None:
        channels = channels or {"metrics", "applied", "mutations", "segments"}
        for ch in channels:
            if ch not in self._connections:
                self._connections[ch] = set()
            self._connections[ch].add(ws)

    def disconnect(self, ws: Any) -> None:
        for conns in self._connections.values():
            conns.discard(ws)

    async def broadcast(self, channel: str, data: Any) -> None:
        from ..util import sanitize_floats
        safe_data = sanitize_floats(data)
        conns = self._connections.get(channel, set())
        dead: List[Any] = []
        for ws in conns:
            try:
                await ws.send_json({"channel": channel, "data": safe_data})
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def connection_count(self) -> int:
        all_ws: set = set()
        for conns in self._connections.values():
            all_ws.update(conns)
        return len(all_ws)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    run_dir: str,
    *,
    poll_interval: float = 0.5,
    multi_dirs: Optional[List[str]] = None,
    demo_mode: bool = False,
    config_yaml: Optional[str] = None,
) -> Any:
    """
    Build the FastAPI application wired to *run_dir*.

    Parameters
    ----------
    run_dir : str
        Primary run directory to monitor.
    poll_interval : float
        How often (seconds) to poll JSONL files for changes.
    multi_dirs : list[str] | None
        Additional run directories for multi-run comparison.
    config_yaml : str | None
        Path to a YAML config file for DashboardConfig overrides.
    """
    _check_deps()

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from .config import DashboardConfig
    from .tailer import JsonlTailer
    from .api import router as api_router
    from .recipe_editor import router as recipe_router, RecipeEditor
    from .projections import ProjectionEngine, create_projections_router
    from .notifications import (
        NotificationEngine,
        WebSocketChannel,
        LogChannel,
        router as notifications_router,
        make_metrics_subscriber,
    )
    from .manifolds import ManifoldEngine, create_router as manifolds_router_factory
    from .autopilot import AutopilotEngine, create_router as autopilot_router_factory
    from .launcher import TrainingLauncher, create_router as launcher_router_factory
    from .ai_engine import LLMAutopilotEngine, AIConfig

    # run_dir is immutable — set once at startup, never changed.
    # Keep the original CLI path for discovery; resolve for monitoring.
    cli_run_dir = run_dir
    run_dir = _resolve_active_run_dir(run_dir)

    # Build centralized config early — needed for tailer poll_interval
    dashboard_config = DashboardConfig.load(
        run_dir,
        yaml_path=config_yaml,
        poll_interval=poll_interval,
    )

    manager = ConnectionManager()
    tailer = JsonlTailer(poll_interval=dashboard_config.server.poll_interval)
    projection_engine = ProjectionEngine()
    notification_engine = NotificationEngine()
    manifold_engine = ManifoldEngine()

    autopilot_engine = AutopilotEngine.with_default_guidelines(
        run_dir=run_dir, mode="off", config=dashboard_config.autopilot,
    )
    training_launcher = TrainingLauncher(run_dir=cli_run_dir)

    # Initialize AI autopilot engine
    ai_engine = LLMAutopilotEngine(
        run_dir=run_dir, config=AIConfig(),
        autopilot_config=dashboard_config.autopilot,
    )
    autopilot_engine.set_ai_engine(ai_engine)
    _tailer_task: Optional[asyncio.Task] = None

    # Resolve JSONL paths — tailer watches run_dir once, forever
    all_dirs = [run_dir] + (multi_dirs or [])

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal _tailer_task
        # Register JSONL files to tail for each run dir
        for i, d in enumerate(all_dirs):
            prefix = "" if i == 0 else f"run{i}_"
            _watch_dir(tailer, d, prefix)

        # Wire tailer to WebSocket broadcast
        for name in list(tailer._targets.keys()):
            channel = name  # channel name = target name
            async def _broadcast(ch: str, records: list, _channel: str = channel) -> None:
                await manager.broadcast(_channel, records)
            tailer.subscribe(name, _broadcast)

        # Wire notification engine channels
        notification_engine.register_channel(
            "websocket", WebSocketChannel(manager.broadcast)
        )
        notification_engine.register_channel("log", LogChannel())

        # Subscribe notification engine to metrics stream
        if "metrics" in tailer._targets:
            tailer.subscribe("metrics", make_metrics_subscriber(notification_engine))

        # Feed metrics channel into projection engine
        if "metrics" in tailer._targets:
            async def _projection_feed(ch: str, records: list) -> None:
                projection_engine.update(records)
            tailer.subscribe("metrics", _projection_feed)

        # Feed metrics into manifold engine
        if "metrics" in tailer._targets:
            async def _manifold_metrics_feed(ch: str, records: list) -> None:
                manifold_engine.update_metrics(records)
            tailer.subscribe("metrics", _manifold_metrics_feed)

        # Feed applied ledger into manifold engine (intervention markers)
        if "applied" in tailer._targets:
            async def _manifold_applied_feed(ch: str, records: list) -> None:
                manifold_engine.update_interventions(records)
            tailer.subscribe("applied", _manifold_applied_feed)

        # Feed metrics into autopilot engine (async for AI mode support)
        if "metrics" in tailer._targets:
            async def _autopilot_feed(ch: str, records: list) -> None:
                for rec in records:
                    step = rec.get("step")
                    metrics = rec.get("metrics")
                    if step is not None and metrics:
                        if autopilot_engine.is_ai_mode:
                            actions = await autopilot_engine.evaluate_async(
                                int(step), metrics
                            )
                        else:
                            actions = autopilot_engine.evaluate(int(step), metrics)
                        if actions:
                            from dataclasses import asdict
                            action_data = [asdict(a) for a in actions]
                            await manager.broadcast("autopilot", action_data)
            tailer.subscribe("metrics", _autopilot_feed)

        _tailer_task = asyncio.create_task(tailer.run())
        log.info("Tailer started for %d directories", len(all_dirs))
        yield
        tailer.stop()
        if _tailer_task:
            _tailer_task.cancel()
            try:
                await _tailer_task
            except asyncio.CancelledError:
                pass
        log.info("Tailer stopped")

    app = FastAPI(
        title="hotcb Dashboard",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store references for use by route modules
    app.state.demo_mode = demo_mode
    app.state.all_dirs = all_dirs
    app.state.manager = manager
    app.state.tailer = tailer
    app.state.notification_engine = notification_engine
    app.state.projection_engine = projection_engine
    app.state.manifold_engine = manifold_engine
    app.state.autopilot_engine = autopilot_engine
    app.state.ai_engine = ai_engine
    app.state.cb_registry = {}

    app.state.config = dashboard_config
    app.state.mutable_state = None  # populated externally when training starts

    # Initialize recipe editor with default recipe path
    default_recipe = os.path.join(run_dir, "hotcb.recipe.jsonl")
    if os.path.exists(default_recipe):
        app.state.recipe_editor = RecipeEditor(default_recipe)
    else:
        app.state.recipe_editor = None

    # Include the command API router
    app.include_router(api_router)
    app.include_router(notifications_router)
    app.include_router(create_projections_router(projection_engine))
    app.include_router(recipe_router)
    app.include_router(manifolds_router_factory(manifold_engine))
    app.include_router(autopilot_router_factory(autopilot_engine, ai_engine=ai_engine))
    app.include_router(launcher_router_factory(training_launcher))

    # --- Feature snapshots endpoint ---
    @app.get("/api/features/snapshots")
    async def get_feature_snapshots(last_n: int = 50):
        """Return recent feature capture snapshots."""
        feat_path = os.path.join(app.state.config.run_dir, "hotcb.features.jsonl")
        records = _read_tail(feat_path, last_n)
        return {"snapshots": records}

    # --- Static file serving (dashboard) ---
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/")
        async def dashboard():
            return FileResponse(os.path.join(static_dir, "index.html"))

    # --- WebSocket endpoint ---
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        try:
            await ws.accept()
        except Exception as exc:
            log.warning("WebSocket accept failed: %s", exc)
            return
        # Client can send initial subscription message
        try:
            init_msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
            channels = set(init_msg.get("channels", ["metrics", "applied"]))
        except (asyncio.TimeoutError, Exception):
            channels = {"metrics", "applied", "mutations", "segments"}

        manager.connect(ws, channels)
        try:
            # Send initial burst of recent data
            await _send_initial_data(
                ws, app.state.config.run_dir, channels,
                metrics_limit=app.state.config.server.history_limit_metrics,
                applied_limit=app.state.config.server.ws_initial_burst,
            )
            # Keep alive — client sends pings, we relay new data via broadcast
            while True:
                # Wait for client messages (ping/subscribe changes)
                msg = await ws.receive_text()
                try:
                    parsed = json.loads(msg)
                    if parsed.get("type") == "subscribe":
                        new_channels = set(parsed.get("channels", []))
                        manager.disconnect(ws)
                        manager.connect(ws, new_channels)
                except (json.JSONDecodeError, Exception):
                    pass
        except WebSocketDisconnect:
            log.debug("WebSocket client disconnected normally")
            manager.disconnect(ws)
        except Exception as exc:
            log.warning("WebSocket connection error: %s", exc)
            manager.disconnect(ws)

    # --- REST endpoints ---
    @app.get("/api/config")
    async def get_config():
        """Return the full dashboard configuration.

        Controls are populated live from MutableState when available,
        falling back to reconstructed controls from the applied ledger.
        """
        from .config import controls_from_mutable_state, controls_from_applied_ledger

        d = app.state.config.to_dict()
        ms = getattr(app.state, "mutable_state", None)
        if ms is not None:
            d["controls"] = controls_from_mutable_state(ms)
        elif not d.get("controls"):
            d["controls"] = controls_from_applied_ledger(app.state.config.run_dir)
        return d

    @app.get("/api/status")
    async def get_status():
        return _build_status(app.state.config.run_dir)

    @app.get("/api/metrics/names")
    async def get_metric_names():
        """Return discovered metric names from the metrics JSONL."""
        return {"names": _discover_metric_names(app.state.config.run_dir)}

    @app.get("/api/metrics/history")
    async def get_metrics_history(last_n: Optional[int] = None):
        """Return recent metric records."""
        limit = last_n if last_n is not None else app.state.config.server.history_limit_metrics
        return {"records": _read_tail(
            os.path.join(app.state.config.run_dir, "hotcb.metrics.jsonl"), limit
        )}

    @app.get("/api/applied/history")
    async def get_applied_history(last_n: Optional[int] = None):
        """Return recent applied ledger entries."""
        limit = last_n if last_n is not None else app.state.config.server.history_limit_applied
        return {"records": _read_tail(
            os.path.join(app.state.config.run_dir, "hotcb.applied.jsonl"), limit
        )}

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "run_dir": app.state.config.run_dir,
            "connections": manager.connection_count,
            "tailer_running": tailer.is_running,
        }

    @app.get("/api/runs/discover")
    async def discover_runs():
        """Scan CLI-provided dir for run subdirs with metrics data (TensorBoard-style)."""
        parent = cli_run_dir
        discovered = []
        # Check if parent itself is a run dir
        parent_metrics = os.path.join(parent, "hotcb.metrics.jsonl")
        if os.path.exists(parent_metrics):
            discovered.append(_build_run_info(parent))
        # Scan subdirs (skip backups and hidden dirs)
        try:
            for name in sorted(os.listdir(parent)):
                if name.startswith("hotcb_bkp") or name.startswith(".") or name == "__pycache__":
                    continue
                subpath = os.path.join(parent, name)
                if os.path.isdir(subpath) and os.path.exists(
                    os.path.join(subpath, "hotcb.metrics.jsonl")
                ):
                    discovered.append(_build_run_info(subpath))
        except OSError:
            pass
        return {"runs": discovered}

    class _ExternalDirRequest(_PydanticBase):
        dir: str

    @app.post("/api/runs/load-external")
    async def load_external_runs(request_body: _ExternalDirRequest):
        """Load runs from an external directory for comparison."""
        from fastapi.responses import JSONResponse
        dir_path = request_body.dir
        if not dir_path or not os.path.isdir(dir_path):
            return JSONResponse(
                status_code=400,
                content={"detail": f"Invalid directory: {dir_path!r}"},
            )
        discovered = []
        # Check if dir itself is a run
        if os.path.exists(os.path.join(dir_path, "hotcb.metrics.jsonl")):
            discovered.append(_build_run_info(dir_path))
        # Scan subdirs
        try:
            for name in sorted(os.listdir(dir_path)):
                if name.startswith(".") or name.startswith("hotcb_bkp") or name == "__pycache__":
                    continue
                subpath = os.path.join(dir_path, name)
                if os.path.isdir(subpath) and os.path.exists(
                    os.path.join(subpath, "hotcb.metrics.jsonl")
                ):
                    discovered.append(_build_run_info(subpath))
        except OSError:
            pass
        return {"runs": discovered, "source_dir": dir_path}

    @app.get("/api/runs/external/metrics")
    async def external_run_metrics(dir: str, last_n: int = 50000):
        """Read metrics from an external run directory."""
        metrics_path = os.path.join(dir, "hotcb.metrics.jsonl")
        return {"records": _read_tail(metrics_path, last_n)}

    @app.get("/api/runs/external/applied")
    async def external_run_applied(dir: str, last_n: int = 200):
        """Read applied ledger from an external run directory."""
        applied_path = os.path.join(dir, "hotcb.applied.jsonl")
        return {"records": _read_tail(applied_path, last_n)}

    # --- UI mode endpoints ---
    _VALID_UI_MODES = {"engineer", "education", "vibe_coder"}

    def _ui_mode_path() -> str:
        return os.path.join(app.state.config.run_dir, "hotcb.ui.json")

    def _read_ui_mode() -> str:
        path = _ui_mode_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.loads(f.read().strip())
                    mode = data.get("mode", "engineer")
                    if mode in _VALID_UI_MODES:
                        return mode
            except Exception:
                pass
        return "engineer"

    def _write_ui_mode(mode: str) -> None:
        path = _ui_mode_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"mode": mode}, f)

    @app.get("/api/ui/mode")
    async def get_ui_mode():
        return {"mode": _read_ui_mode()}

    @app.post("/api/ui/mode")
    async def set_ui_mode(body: UIModeRequest):
        mode = body.mode
        if mode not in _VALID_UI_MODES:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=400,
                content={"detail": f"Invalid mode: {mode!r}. Must be one of {sorted(_VALID_UI_MODES)}"},
            )
        _write_ui_mode(mode)
        return {"mode": mode}

    @app.get("/api/capabilities")
    async def get_capabilities():
        """Return training capabilities detected by the adapter."""
        from ..capabilities import TrainingCapabilities
        caps = TrainingCapabilities.load(app.state.config.run_dir)
        if caps is None:
            return {"detected": False}
        return {"detected": True, **caps.to_dict()}

    @app.get("/api/state/controls")
    async def get_control_state():
        """Return current control state for UI restoration on refresh.

        Reads latest metrics for slider values, active config, autopilot mode,
        and freeze state so the frontend can restore its controls.
        """
        rd = app.state.config.run_dir
        state: dict = {}

        # Latest metric values (for sliders)
        metrics_path = os.path.join(rd, "hotcb.metrics.jsonl")
        last_metrics = _read_tail(metrics_path, 1)
        if last_metrics:
            state["latest_metrics"] = last_metrics[0].get("metrics", {})
            state["latest_step"] = last_metrics[0].get("step", 0)
        else:
            state["latest_metrics"] = {}
            state["latest_step"] = 0

        # Active training config
        run_json = os.path.join(rd, "hotcb.run.json")
        if os.path.exists(run_json):
            try:
                with open(run_json, "r") as f:
                    state["run_config"] = json.load(f)
            except Exception:
                state["run_config"] = {}
        else:
            state["run_config"] = {}

        # Autopilot mode
        state["autopilot_mode"] = autopilot_engine.mode

        # Freeze state
        state["freeze"] = _build_status(rd).get("freeze", {"mode": "off"})

        # Training status
        state["training_running"] = training_launcher.running
        state["training_config"] = training_launcher._config if training_launcher.running else {}

        # Last applied opt/loss params — use cached scan to avoid re-reading
        # the entire file on every poll cycle
        applied_path = os.path.join(rd, "hotcb.applied.jsonl")
        last_opt_params, last_loss_params, modules_seen = _get_applied_summary(applied_path)
        state["last_opt_params"] = last_opt_params
        state["last_loss_params"] = last_loss_params

        # Module availability detection
        loss_active = "loss" in modules_seen
        if not loss_active:
            try:
                from ..capabilities import TrainingCapabilities
                caps = TrainingCapabilities.load(rd)
                if caps and getattr(caps, 'mutable_state_detected', False):
                    loss_active = True
            except Exception:
                pass
        # Also detect loss from MutableState actuators
        ms = getattr(app.state, "mutable_state", None)
        if not loss_active and ms is not None:
            try:
                for desc in ms.describe_all():
                    if desc.get("group") == "loss":
                        loss_active = True
                        break
            except Exception:
                pass
        state["modules_active"] = {
            "opt": True,
            "loss": loss_active,
            "cb": "cb" in modules_seen,
        }

        # Live controls from MutableState, or fallback to applied ledger
        from .config import controls_from_mutable_state, controls_from_applied_ledger
        live_controls = controls_from_mutable_state(ms)
        state["controls"] = live_controls if live_controls else controls_from_applied_ledger(rd)

        # External training detection
        is_launcher = training_launcher.running or bool(
            training_launcher._active_config_id
        )
        state["is_external"] = not is_launcher and bool(last_metrics)

        # AI autopilot state
        if ai_engine:
            state["ai_key_metric"] = ai_engine.state.key_metric

        # Demo mode flag — frontend hides Training card when false
        state["demo_mode"] = demo_mode

        return state

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _watch_dir(tailer: Any, run_dir: str, prefix: str = "") -> None:
    """Register standard JSONL files from a run directory."""
    files = {
        "metrics": "hotcb.metrics.jsonl",
        "applied": "hotcb.applied.jsonl",
        "mutations": "hotcb.tune.mutations.jsonl",
        "segments": "hotcb.tune.segments.jsonl",
    }
    for name, filename in files.items():
        path = os.path.join(run_dir, filename)
        tailer.watch(f"{prefix}{name}", path)


def _build_status(run_dir: str) -> dict:
    """Build a status summary from filesystem state."""
    freeze_path = os.path.join(run_dir, "hotcb.freeze.json")
    freeze = {"mode": "off"}
    if os.path.exists(freeze_path):
        try:
            with open(freeze_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    freeze = json.loads(content)
        except Exception:
            pass

    tune_summary_path = os.path.join(run_dir, "hotcb.tune.summary.json")
    tune = {}
    if os.path.exists(tune_summary_path):
        try:
            with open(tune_summary_path, "r", encoding="utf-8") as f:
                tune = json.load(f)
        except Exception:
            pass

    return {
        "run_dir": run_dir,
        "freeze": freeze,
        "tune": tune,
        "files": {
            name: os.path.exists(os.path.join(run_dir, f))
            for name, f in [
                ("commands", "hotcb.commands.jsonl"),
                ("applied", "hotcb.applied.jsonl"),
                ("metrics", "hotcb.metrics.jsonl"),
                ("recipe", "hotcb.recipe.jsonl"),
                ("tune_recipe", "hotcb.tune.recipe.yaml"),
            ]
        },
    }


def _discover_metric_names(run_dir: str) -> List[str]:
    """Scan metrics JSONL for unique metric names."""
    path = os.path.join(run_dir, "hotcb.metrics.jsonl")
    names: set = set()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    names.update(rec.get("metrics", {}).keys())
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return sorted(names)


def _read_tail(path: str, last_n: int) -> List[dict]:
    """Read the last N records from a JSONL file efficiently.

    For small files or small last_n, reads from the end using seek
    to avoid scanning the entire file.
    """
    if not os.path.exists(path):
        return []
    try:
        file_size = os.path.getsize(path)
    except OSError:
        return []
    if file_size == 0:
        return []

    # For small files (< 512KB) or requesting everything, just read all
    if file_size < 512 * 1024 or last_n > 50000:
        records: List[dict] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            return []
        return records[-last_n:]

    # For large files, read from the end — estimate ~200 bytes per record
    chunk_size = min(file_size, last_n * 300 + 4096)
    try:
        with open(path, "rb") as f:
            f.seek(max(0, file_size - chunk_size))
            if f.tell() > 0:
                f.readline()  # skip partial first line
            tail_bytes = f.read()
        lines = tail_bytes.decode("utf-8", errors="replace").splitlines()
        records = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if len(records) >= last_n:
            return records[-last_n:]
        # If we didn't get enough, fall back to full scan
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records[-last_n:]
    except Exception:
        return []


# Cache for applied ledger scan — avoids re-reading entire file every 5s
_applied_cache: Dict[str, Any] = {"mtime": 0, "size": 0, "opt": {}, "loss": {}, "modules": set()}


def _get_applied_summary(applied_path: str) -> tuple:
    """Return (last_opt_params, last_loss_params, modules_seen) with mtime caching."""
    try:
        st = os.stat(applied_path)
        mtime, size = st.st_mtime, st.st_size
    except OSError:
        return {}, {}, set()

    if mtime == _applied_cache["mtime"] and size == _applied_cache["size"]:
        return dict(_applied_cache["opt"]), dict(_applied_cache["loss"]), set(_applied_cache["modules"])

    last_opt: dict = {}
    last_loss: dict = {}
    modules: set = set()
    try:
        with open(applied_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mod = rec.get("module")
                if mod:
                    modules.add(mod)
                if rec.get("decision") != "applied":
                    continue
                p = rec.get("params") or rec.get("payload")
                if mod == "opt" and p:
                    last_opt.update(p)
                elif mod == "loss" and p:
                    last_loss.update(p)
    except Exception:
        pass

    _applied_cache.update({"mtime": mtime, "size": size, "opt": dict(last_opt), "loss": dict(last_loss), "modules": set(modules)})
    return last_opt, last_loss, modules


def _resolve_active_run_dir(run_dir: str) -> str:
    """Resolve run_dir to the active run subdir if it's a parent directory.

    If run_dir contains hotcb.metrics.jsonl directly, it's a single run dir.
    Otherwise, scan for subdirs with metrics and return the latest.
    Skips backup dirs (hotcb_bkp_*) and hidden dirs (.*).
    """
    metrics_direct = os.path.join(run_dir, "hotcb.metrics.jsonl")
    if os.path.exists(metrics_direct):
        return run_dir
    subdirs = []
    try:
        for name in os.listdir(run_dir):
            # Skip backup dirs, hidden dirs, and __pycache__
            if name.startswith("hotcb_bkp") or name.startswith(".") or name == "__pycache__":
                continue
            subpath = os.path.join(run_dir, name)
            if os.path.isdir(subpath) and os.path.exists(
                os.path.join(subpath, "hotcb.metrics.jsonl")
            ):
                subdirs.append(subpath)
    except OSError:
        pass
    if subdirs:
        subdirs.sort(key=os.path.getmtime, reverse=True)
        return subdirs[0]
    return run_dir


def _build_run_info(run_path: str) -> dict:
    """Build run discovery info from a run directory."""
    run_id = os.path.basename(run_path)
    label = run_id
    step_count = 0
    metric_names: set = set()

    run_json = os.path.join(run_path, "hotcb.run.json")
    if os.path.exists(run_json):
        try:
            with open(run_json) as f:
                meta = json.load(f)
            label = meta.get("config_name", meta.get("config_id", run_id))
        except Exception:
            pass

    metrics_path = os.path.join(run_path, "hotcb.metrics.jsonl")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                first_rec = None
                last_rec = None
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    step_count += 1
                    try:
                        rec = json.loads(line)
                        if first_rec is None:
                            first_rec = rec
                        last_rec = rec
                    except json.JSONDecodeError:
                        continue
                if first_rec:
                    metric_names.update(first_rec.get("metrics", {}).keys())
                if last_rec:
                    metric_names.update(last_rec.get("metrics", {}).keys())
        except Exception:
            pass

    return {
        "run_id": run_id,
        "dir": run_path,
        "label": label,
        "step_count": step_count,
        "metric_names": sorted(metric_names),
    }


async def _send_initial_data(
    ws: Any,
    run_dir: str,
    channels: Set[str],
    *,
    metrics_limit: int = 500,
    applied_limit: int = 200,
) -> None:
    """Send a burst of recent data when a client first connects."""
    from ..util import sanitize_floats
    channel_files = {
        "metrics": "hotcb.metrics.jsonl",
        "applied": "hotcb.applied.jsonl",
        "mutations": "hotcb.tune.mutations.jsonl",
        "segments": "hotcb.tune.segments.jsonl",
    }
    for ch in channels:
        filename = channel_files.get(ch)
        if not filename:
            continue
        limit = metrics_limit if ch == "metrics" else applied_limit
        records = _read_tail(os.path.join(run_dir, filename), limit)
        if records:
            try:
                await ws.send_json({"channel": ch, "data": sanitize_floats(records), "initial": True})
            except Exception:
                break


# ---------------------------------------------------------------------------
# Server runner
# ---------------------------------------------------------------------------

def run_server(
    run_dir: str,
    *,
    host: str = "0.0.0.0",
    port: int = 8421,
    poll_interval: float = 0.5,
    multi_dirs: Optional[List[str]] = None,
    demo_mode: bool = False,
) -> None:
    """Start the dashboard server (blocking)."""
    _check_deps()
    import uvicorn

    app = create_app(run_dir, poll_interval=poll_interval, multi_dirs=multi_dirs, demo_mode=demo_mode)
    log.info("Starting hotcb dashboard at http://%s:%d", host, port)
    log.info("Monitoring: %s", run_dir)
    uvicorn.run(app, host=host, port=port, log_level="info")
