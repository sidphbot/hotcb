"""Training launch platform — configurable training configs with start/stop/reset."""
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("hotcb.server.launcher")

# JSONL files that belong to a training run
_JSONL_FILES = [
    "hotcb.metrics.jsonl",
    "hotcb.applied.jsonl",
    "hotcb.commands.jsonl",
    "hotcb.features.jsonl",
    "hotcb.recipe.jsonl",
    "hotcb.tune.mutations.jsonl",
    "hotcb.tune.segments.jsonl",
]


# ---------------------------------------------------------------------------
# Training configurations registry
# ---------------------------------------------------------------------------

class TrainingConfig:
    """A named training configuration that the launcher can run."""

    def __init__(
        self,
        config_id: str,
        name: str,
        description: str,
        train_fn: Callable,
        defaults: Dict[str, Any],
    ) -> None:
        self.config_id = config_id
        self.name = name
        self.description = description
        self.train_fn = train_fn
        self.defaults = defaults

    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "name": self.name,
            "description": self.description,
            "defaults": self.defaults,
        }


def _get_builtin_configs() -> List[TrainingConfig]:
    """Return the 3 built-in training configurations."""
    return [
        TrainingConfig(
            config_id="simple",
            name="Simple (Quadratic)",
            description=(
                "Single-task synthetic training on a quadratic loss surface. "
                "Good for testing basic dashboard controls (lr, wd sliders)."
            ),
            train_fn=_run_simple,
            defaults={"max_steps": 500, "step_delay": 0.15},
        ),
        TrainingConfig(
            config_id="multitask",
            name="Multi-Objective (Golden Demo)",
            description=(
                "Two-task training (classification + reconstruction) with "
                "recipe-driven loss weight shifts at steps 200, 400, 500. "
                "Demonstrates multi-objective control and lambda tuning."
            ),
            train_fn=_run_multitask,
            defaults={"max_steps": 800, "step_delay": 0.12},
        ),
        TrainingConfig(
            config_id="finetune",
            name="Finetune (Pretrained Backbone)",
            description=(
                "Transfer learning simulation: pretrained backbone finetuned "
                "on a small dataset with recipe-driven LR scheduling. "
                "Shows overfitting dynamics and mutation impact on convergence."
            ),
            train_fn=_run_finetune,
            defaults={"max_steps": 600, "step_delay": 0.12},
        ),
    ]


def _seed_run(run_dir: str) -> None:
    """Seed random from run metadata so runs are reproducible by config+seed."""
    import random as _rng
    run_json = os.path.join(run_dir, "hotcb.run.json")
    seed = int(time.time() * 1000) & 0xFFFFFFFF
    try:
        with open(run_json) as f:
            meta = json.load(f)
        seed = meta.get("seed", seed)
    except Exception:
        pass
    _rng.seed(seed)


def _run_simple(
    run_dir: str,
    max_steps: int,
    step_delay: float,
    stop_event: threading.Event,
) -> None:
    _seed_run(run_dir)
    from ..demo import _demo_training
    _demo_training(run_dir, max_steps=max_steps, step_delay=step_delay,
                   _stop_event=stop_event)


def _run_multitask(
    run_dir: str,
    max_steps: int,
    step_delay: float,
    stop_event: threading.Event,
) -> None:
    _seed_run(run_dir)
    from ..golden_demo import _golden_training
    _golden_training(run_dir, max_steps=max_steps, step_delay=step_delay,
                     _stop_event=stop_event)


def _run_finetune(
    run_dir: str,
    max_steps: int,
    step_delay: float,
    stop_event: threading.Event,
) -> None:
    _seed_run(run_dir)
    from ..finetune_demo import _finetune_training
    _finetune_training(run_dir, max_steps=max_steps, step_delay=step_delay,
                       _stop_event=stop_event)


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

class TrainingLauncher:
    """Manages training runs with configurable training types."""

    def __init__(self, run_dir: str) -> None:
        self._run_dir = run_dir
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started_at: Optional[str] = None
        self._active_run_dir: Optional[str] = None
        self._active_config_id: Optional[str] = None
        self._config: dict = {}
        self._configs: Dict[str, TrainingConfig] = {}
        for cfg in _get_builtin_configs():
            self._configs[cfg.config_id] = cfg

    def register_config(self, config: TrainingConfig) -> None:
        """Register a custom training configuration."""
        self._configs[config.config_id] = config

    def get_configs(self) -> List[dict]:
        return [c.to_dict() for c in self._configs.values()]

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        *,
        config_id: str = "multitask",
        max_steps: Optional[int] = None,
        step_delay: Optional[float] = None,
        run_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> dict:
        if self.running:
            # Try to wait a moment for a winding-down thread
            if self._stop_event.is_set() and self._thread is not None:
                self._thread.join(timeout=3)
                if not self._thread.is_alive():
                    self._thread = None
            if self.running:
                return {"error": "Training already running"}

        cfg = self._configs.get(config_id)
        if cfg is None:
            return {"error": f"Unknown config: {config_id!r}. "
                    f"Available: {list(self._configs.keys())}"}

        effective_steps = max_steps if max_steps is not None else cfg.defaults["max_steps"]
        effective_delay = step_delay if step_delay is not None else cfg.defaults["step_delay"]
        # Generate seed if not provided — makes runs reproducible
        import random as _rng
        effective_seed = seed if seed is not None else _rng.randint(0, 2**31 - 1)

        # Write flat to run_dir — no subdirs, no rewiring.
        # Truncate existing JSONL files for a clean start.
        target_dir = run_dir or self._run_dir
        self._active_run_dir = target_dir
        self._active_config_id = config_id
        self._config = {
            "config_id": config_id,
            "config_name": cfg.name,
            "max_steps": effective_steps,
            "step_delay": effective_delay,
            "seed": effective_seed,
        }
        self._stop_event.clear()

        os.makedirs(target_dir, exist_ok=True)
        for fname in _JSONL_FILES:
            path = os.path.join(target_dir, fname)
            open(path, "w").close()  # truncate for clean start

        freeze_path = os.path.join(target_dir, "hotcb.freeze.json")
        if not os.path.exists(freeze_path):
            with open(freeze_path, "w") as f:
                json.dump({"mode": "off"}, f)

        # Write run metadata
        import time as _time
        run_meta = {
            "run_id": _time.strftime("%Y%m%d_%H%M%S"),
            "config_id": config_id,
            "config_name": cfg.name,
            "max_steps": effective_steps,
            "step_delay": effective_delay,
            "seed": effective_seed,
            "started_at": self._started_at,
            "run_dir": target_dir,
        }
        self._run_meta = run_meta
        with open(os.path.join(target_dir, "hotcb.run.json"), "w") as f:
            json.dump(run_meta, f, indent=2)

        log.info(
            "[hotcb.launcher] starting %s (run_dir=%s, steps=%d, delay=%.3f)",
            cfg.name, target_dir, effective_steps, effective_delay,
        )

        self._thread = threading.Thread(
            target=self._run_training,
            kwargs={
                "train_fn": cfg.train_fn,
                "run_dir": target_dir,
                "max_steps": effective_steps,
                "step_delay": effective_delay,
            },
            daemon=True,
            name=f"hotcb-training-{config_id}",
        )
        self._started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._thread.start()

        return {
            "started": True,
            "run_dir": target_dir,
            "config": self._config,
        }

    def _run_training(
        self,
        train_fn: Callable,
        run_dir: str,
        max_steps: int,
        step_delay: float,
    ) -> None:
        try:
            train_fn(run_dir, max_steps, step_delay, self._stop_event)
        except Exception:
            log.exception("[hotcb.launcher] training thread crashed")
        finally:
            self._save_run_summary(run_dir)

    def _save_run_summary(self, run_dir: str) -> None:
        """Append completed run summary to runs history."""
        import time as _time
        # Read final metrics from metrics JSONL
        metrics_path = os.path.join(run_dir, "hotcb.metrics.jsonl")
        final_metrics: dict = {}
        if os.path.exists(metrics_path):
            last_line = None
            with open(metrics_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last_line = line
            if last_line:
                try:
                    rec = json.loads(last_line)
                    final_metrics = rec.get("metrics", {})
                except Exception:
                    pass

        summary = dict(getattr(self, '_run_meta', {}))
        summary["completed_at"] = _time.strftime("%Y-%m-%dT%H:%M:%S")
        summary["final_metrics"] = final_metrics
        summary["stopped_early"] = self._stop_event.is_set()

        # Update the run.json
        run_json = os.path.join(run_dir, "hotcb.run.json")
        try:
            with open(run_json, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass

        # Run summary is now in the run's own hotcb.run.json (no global index)

    def stop(self) -> dict:
        if not self.running:
            return {"stopped": True, "was_running": False}
        log.info("[hotcb.launcher] stopping training thread")
        self._stop_event.set()
        self._thread.join(timeout=5)
        still_alive = self._thread.is_alive()
        if not still_alive:
            self._thread = None
        return {"stopped": not still_alive}

    def reset(self) -> dict:
        """Stop training and wipe all JSONL files in run_dir."""
        if self.running:
            self.stop()
        # Ensure thread reference is cleared even if join timed out
        if self._thread is not None and not self._thread.is_alive():
            self._thread = None

        rd = self._active_run_dir or self._run_dir
        cleared = []
        for fname in _JSONL_FILES:
            path = os.path.join(rd, fname)
            if os.path.exists(path):
                open(path, "w").close()  # truncate
                cleared.append(fname)

        log.info("[hotcb.launcher] reset: cleared %d files in %s", len(cleared), rd)
        return {"reset": True, "run_dir": rd, "cleared": cleared}

    def status(self) -> dict:
        return {
            "running": self.running,
            "run_dir": self._active_run_dir if self.running else None,
            "started_at": self._started_at if self.running else None,
            "config": self._config if self.running else {},
        }


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------


def create_router(launcher: Optional[TrainingLauncher] = None) -> Any:
    """Build the training launcher API router. Requires FastAPI."""
    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    _launcher = launcher
    router = APIRouter(prefix="/api/train", tags=["train"])

    class StartRequest(BaseModel):
        config_id: str = "multitask"
        max_steps: Optional[int] = None
        step_delay: Optional[float] = None
        run_dir: Optional[str] = None
        seed: Optional[int] = None

    @router.get("/configs")
    async def list_configs():
        return {"configs": _launcher.get_configs()}

    @router.post("/start")
    async def start_training(body: StartRequest, request: Request):
        result = _launcher.start(
            config_id=body.config_id,
            max_steps=body.max_steps,
            step_delay=body.step_delay,
            run_dir=body.run_dir,
            seed=body.seed,
        )
        if "error" in result:
            return JSONResponse(status_code=409, content=result)
        # run_dir is immutable — launcher writes flat to it, tailer already watches it.
        # Clear stale engine state from previous runs.
        projection = getattr(request.app.state, 'projection_engine', None)
        if projection:
            projection._steps.clear()
            projection._metrics.clear()
            projection._hp_keys.clear()
            projection._hp_values.clear()
        autopilot = getattr(request.app.state, 'autopilot_engine', None)
        if autopilot and hasattr(autopilot, 'reset'):
            autopilot.reset()
        return result

    @router.get("/status")
    async def get_status():
        return _launcher.status()

    @router.post("/stop")
    async def stop_training():
        result = _launcher.stop()
        if "error" in result:
            return JSONResponse(status_code=404, content=result)
        return result

    class RegisterRequest(BaseModel):
        config_id: str
        name: str
        description: str = ""
        train_fn_path: str  # "module.path:function_name"
        defaults: dict = {"max_steps": 1000, "step_delay": 0.1}
        recipe_path: Optional[str] = None

    @router.post("/configs/register")
    async def register_config(body: RegisterRequest):
        """Dynamically register a custom training config."""
        import importlib
        try:
            module_path, fn_name = body.train_fn_path.rsplit(":", 1)
            mod = importlib.import_module(module_path)
            train_fn = getattr(mod, fn_name)
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content={"error": f"Cannot import '{body.train_fn_path}': {exc}"},
            )
        cfg = TrainingConfig(
            config_id=body.config_id,
            name=body.name,
            description=body.description,
            train_fn=train_fn,
            defaults=body.defaults,
        )
        _launcher.register_config(cfg)
        return {"status": "registered", "config_id": body.config_id}

    @router.get("/runs/history")
    async def list_runs():
        """List completed training runs from the runs history."""
        run_dir = _launcher._run_dir
        parent_dir = os.path.dirname(run_dir.rstrip("/"))
        history_path = os.path.join(parent_dir, "hotcb.runs.jsonl") if parent_dir else None

        runs = []
        # Check for history file
        if history_path and os.path.exists(history_path):
            with open(history_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            runs.append(json.loads(line))
                        except Exception:
                            pass

        # Also check for run.json in the current run_dir
        run_json = os.path.join(run_dir, "hotcb.run.json")
        if os.path.exists(run_json):
            try:
                with open(run_json) as f:
                    current = json.load(f)
                # Check if it's already in history
                if not any(r.get("run_id") == current.get("run_id") for r in runs):
                    runs.append(current)
            except Exception:
                pass

        return {"runs": runs}

    def _find_run_dir(run_id: str) -> Optional[str]:
        """Find a run directory by run_id (subdir name or run.json match)."""
        rd = _launcher._run_dir
        # Check as subdir of run_dir
        subpath = os.path.join(rd, run_id)
        if os.path.isdir(subpath):
            return subpath
        # Check current dir's run.json
        rj = os.path.join(rd, "hotcb.run.json")
        if os.path.exists(rj):
            try:
                with open(rj) as f:
                    meta = json.load(f)
                if meta.get("run_id") == run_id:
                    return rd
            except Exception:
                pass
        # Check active run dir
        if _launcher._active_run_dir:
            if os.path.basename(_launcher._active_run_dir.rstrip("/")) == run_id:
                return _launcher._active_run_dir
        return None

    @router.get("/runs/{run_id}/metrics")
    async def get_run_metrics(run_id: str, last_n: int = 2000):
        """Get metrics from a specific run by its ID."""
        target_dir = _find_run_dir(run_id)

        if not target_dir:
            return JSONResponse(status_code=404, content={"error": f"Run {run_id} not found"})

        metrics_path = os.path.join(target_dir, "hotcb.metrics.jsonl")
        records = []
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            pass

        return {"run_id": run_id, "records": records[-last_n:]}

    @router.get("/runs/{run_id}/applied")
    async def get_run_applied(run_id: str, last_n: int = 200):
        """Get applied mutations from a specific run by its ID."""
        target_dir = _find_run_dir(run_id)

        if not target_dir:
            return JSONResponse(status_code=404, content={"error": f"Run {run_id} not found"})

        applied_path = os.path.join(target_dir, "hotcb.applied.jsonl")
        records = []
        if os.path.exists(applied_path):
            with open(applied_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            pass

        return {"run_id": run_id, "records": records[-last_n:]}

    @router.post("/reset")
    async def reset_training(request: Request):
        result = _launcher.reset()
        # Reset server-side engine state so stale data doesn't carry over
        autopilot = getattr(request.app.state, 'autopilot_engine', None)
        if autopilot and hasattr(autopilot, 'reset'):
            autopilot.reset()
        projection = getattr(request.app.state, 'projection_engine', None)
        if projection:
            projection._steps.clear()
            projection._metrics.clear()
            projection._hp_keys.clear()
            projection._hp_values.clear()
        manifold = getattr(request.app.state, 'manifold_engine', None)
        if manifold and hasattr(manifold, 'reset'):
            manifold.reset()
        cb_registry = getattr(request.app.state, 'cb_registry', None)
        if cb_registry is not None:
            cb_registry.clear()
        return result

    return router
