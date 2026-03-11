"""
hotcb.launch — Programmatic API for launching training with autopilot.

Provides a single entry point to start training + dashboard server +
autopilot mode in one call.  Designed for notebook/script use::

    from hotcb.launch import launch

    handle = launch(
        train_fn="my_module:train",
        autopilot="ai_suggest",
        key_metric="val_loss",
        max_steps=1000,
    )
    # Training is running, dashboard is live
    handle.wait()           # block until training finishes
    handle.metrics()        # latest metrics dict
    handle.stop()           # stop training early

For adapter-based training (Lightning/HF), pass the train function that
uses the adapter internally.  The function must conform to the hotcb
training contract::

    def train_fn(run_dir, max_steps, step_delay, stop_event):
        # Write metrics to {run_dir}/hotcb.metrics.jsonl
        # Read commands from {run_dir}/hotcb.commands.jsonl
        ...
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

log = logging.getLogger("hotcb.launch")


@dataclass
class LaunchHandle:
    """Handle returned by ``launch()`` for programmatic control."""

    run_dir: str
    server_url: str
    autopilot_mode: str
    key_metric: str
    _stop_event: threading.Event = field(repr=False)
    _train_thread: Optional[threading.Thread] = field(default=None, repr=False)
    _server_thread: Optional[threading.Thread] = field(default=None, repr=False)

    def wait(self, timeout: Optional[float] = None) -> None:
        """Block until training finishes."""
        if self._train_thread and self._train_thread.is_alive():
            self._train_thread.join(timeout=timeout)

    def stop(self) -> None:
        """Stop training (server keeps running)."""
        self._stop_event.set()
        if self._train_thread:
            self._train_thread.join(timeout=5)

    def stop_all(self) -> None:
        """Stop both training and server."""
        self.stop()
        # Server thread is daemon, will die with process

    @property
    def running(self) -> bool:
        return self._train_thread is not None and self._train_thread.is_alive()

    def metrics(self, last_n: int = 1) -> List[dict]:
        """Read latest metrics from the JSONL file."""
        path = os.path.join(self.run_dir, "hotcb.metrics.jsonl")
        if not os.path.exists(path):
            return []
        records = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception:
            pass
        return records[-last_n:]

    def latest_metrics(self) -> dict:
        """Return the most recent metrics dict (flat)."""
        recs = self.metrics(last_n=1)
        if recs:
            return recs[0].get("metrics", {})
        return {}

    def applied(self, last_n: int = 50) -> List[dict]:
        """Read recent applied mutations."""
        path = os.path.join(self.run_dir, "hotcb.applied.jsonl")
        if not os.path.exists(path):
            return []
        records = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception:
            pass
        return records[-last_n:]

    def ai_status(self) -> dict:
        """Read AI autopilot state from state file."""
        path = os.path.join(self.run_dir, "hotcb.ai.state.json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def send_command(self, cmd: dict) -> None:
        """Write a command to the commands JSONL file."""
        path = os.path.join(self.run_dir, "hotcb.commands.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(cmd) + "\n")

    def set_param(self, **kwargs: Any) -> None:
        """Convenience: set optimizer params (lr, weight_decay, etc.)."""
        self.send_command({
            "module": "opt",
            "op": "set_params",
            "params": kwargs,
            "source": "programmatic",
            "ts": time.time(),
        })

    def set_loss(self, **kwargs: Any) -> None:
        """Convenience: set loss params."""
        self.send_command({
            "module": "loss",
            "op": "set_params",
            "params": kwargs,
            "source": "programmatic",
            "ts": time.time(),
        })

    def metric_history(self, name: str, last_n: int = 500) -> List[float]:
        """Return values for a single metric over recent steps."""
        recs = self.metrics(last_n=last_n)
        return [
            r["metrics"][name]
            for r in recs
            if name in r.get("metrics", {})
        ]


def _resolve_train_fn(train_fn: Union[str, Callable]) -> Callable:
    """Resolve a train function from a string like 'module:fn' or a callable."""
    if callable(train_fn):
        return train_fn
    if isinstance(train_fn, str) and ":" in train_fn:
        import importlib
        module_path, fn_name = train_fn.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, fn_name)
    raise ValueError(
        f"train_fn must be a callable or 'module.path:fn_name', got {train_fn!r}"
    )


def _seed_run_dir(run_dir: str) -> None:
    """Ensure run_dir exists with all required files."""
    os.makedirs(run_dir, exist_ok=True)
    for fname in [
        "hotcb.metrics.jsonl",
        "hotcb.applied.jsonl",
        "hotcb.commands.jsonl",
        "hotcb.features.jsonl",
        "hotcb.recipe.jsonl",
    ]:
        path = os.path.join(run_dir, fname)
        if not os.path.exists(path):
            open(path, "w").close()

    freeze_path = os.path.join(run_dir, "hotcb.freeze.json")
    if not os.path.exists(freeze_path):
        with open(freeze_path, "w") as f:
            json.dump({"mode": "off"}, f)


def _load_launch_config(path: str) -> dict:
    """Load a hotcb.launch.json config file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def launch(
    train_fn: Union[str, Callable, None] = None,
    *,
    config: str = "multitask",
    config_file: Optional[str] = None,
    run_dir: Optional[str] = None,
    autopilot: str = "off",
    key_metric: str = "val_loss",
    watch_metrics: Optional[List[str]] = None,
    ai_key: Optional[str] = None,
    ai_model: str = "gpt-4o-mini",
    ai_base_url: str = "https://api.openai.com/v1",
    ai_budget: float = 5.0,
    ai_cadence: int = 50,
    max_steps: Optional[int] = None,
    max_time: Optional[float] = None,
    step_delay: Optional[float] = None,
    host: str = "127.0.0.1",
    port: int = 8421,
    seed: Optional[int] = None,
    serve: bool = True,
    block: bool = False,
) -> LaunchHandle:
    """
    Launch training with optional autopilot and dashboard server.

    Parameters
    ----------
    train_fn : str | callable | None
        Training function. Either a callable matching the hotcb contract
        ``(run_dir, max_steps, step_delay, stop_event)`` or a string
        like ``"module.path:fn_name"``. If None, uses a built-in config.
    config : str
        Built-in config to use if train_fn is None: "simple", "multitask",
        "finetune".
    config_file : str | None
        Path to a ``hotcb.launch.json`` file. Values from the file are used
        as defaults — explicit keyword arguments override them.
    run_dir : str | None
        Directory for run artifacts. Created if needed. Uses tempdir if None.
    autopilot : str
        Autopilot mode: "off", "suggest", "auto", "ai_suggest", "ai_auto".
    key_metric : str
        Primary metric the AI optimizes toward.
    watch_metrics : list[str] | None
        Additional metrics to monitor closely.
    ai_key : str | None
        API key for LLM provider. Falls back to HOTCB_AI_KEY env.
    ai_model : str
        LLM model name (default: gpt-4o-mini).
    ai_base_url : str
        LLM API base URL (OpenAI-compatible).
    ai_budget : float
        Max USD to spend on LLM calls.
    ai_cadence : int
        Steps between periodic AI check-ins.
    max_steps : int | None
        Override max training steps.
    max_time : float | None
        Wall-clock time limit in seconds. Training stops after this
        duration regardless of step count. Useful when step duration
        varies across hardware. Can be combined with max_steps (whichever
        limit is hit first wins).
    step_delay : float | None
        Seconds between steps (for demos/simulation).
    host : str
        Dashboard server bind host.
    port : int
        Dashboard server bind port.
    seed : int | None
        Random seed for reproducibility.
    serve : bool
        Whether to start the dashboard server.
    block : bool
        If True, block until training finishes.

    Returns
    -------
    LaunchHandle
        Handle for monitoring, metrics access, and control.
    """
    # Apply config file defaults (explicit kwargs override)
    if config_file is not None:
        _cfg = _load_launch_config(config_file)
        if train_fn is None and "train_fn" in _cfg:
            train_fn = _cfg["train_fn"]
        if run_dir is None and "run_dir" in _cfg:
            run_dir = _cfg["run_dir"]
        if autopilot == "off" and "autopilot" in _cfg:
            autopilot = _cfg["autopilot"]
        if key_metric == "val_loss" and "key_metric" in _cfg:
            key_metric = _cfg["key_metric"]
        if max_steps is None and "max_steps" in _cfg:
            max_steps = _cfg["max_steps"]
        if max_time is None and "max_time" in _cfg:
            max_time = _cfg["max_time"]
        if step_delay is None and "step_delay" in _cfg:
            step_delay = _cfg["step_delay"]
        if seed is None and "seed" in _cfg:
            seed = _cfg["seed"]
        if port == 8421 and "port" in _cfg:
            port = _cfg["port"]

    # Resolve run_dir
    if run_dir is None:
        run_dir = tempfile.mkdtemp(prefix="hotcb_run_")
    _seed_run_dir(run_dir)

    # Write AI state if AI mode
    if autopilot in ("ai_suggest", "ai_auto"):
        ai_state = {
            "key_metric": key_metric,
            "watch_metrics": watch_metrics or [],
            "run_number": 1,
            "max_runs": 3,
            "run_history": [],
            "carried_context": "",
            "next_check_step": None,
            "cadence_override": None,
        }
        state_path = os.path.join(run_dir, "hotcb.ai.state.json")
        with open(state_path, "w") as f:
            json.dump(ai_state, f, indent=2)

        # Set AI key in env if provided
        if ai_key:
            os.environ["HOTCB_AI_KEY"] = ai_key

    stop_event = threading.Event()

    # Resolve training function
    if train_fn is not None:
        resolved_fn = _resolve_train_fn(train_fn)
        effective_steps = max_steps or 1000
        effective_delay = step_delay if step_delay is not None else 0.1
    else:
        # Use built-in config
        _builtin_defaults = {
            "simple": (500, 0.15),
            "multitask": (800, 0.12),
            "finetune": (600, 0.12),
        }
        if config not in _builtin_defaults:
            raise ValueError(f"Unknown config: {config!r}. Use: {list(_builtin_defaults)}")
        default_steps, default_delay = _builtin_defaults[config]
        effective_steps = max_steps or default_steps
        effective_delay = step_delay if step_delay is not None else default_delay

        # Wrap built-in demo functions to match the external train_fn contract
        # (built-in demos use keyword-only args with _stop_event)
        def _make_wrapper(fn):
            def wrapper(run_dir, max_steps, step_delay, stop_event):
                fn(run_dir, max_steps=max_steps, step_delay=step_delay, _stop_event=stop_event)
            return wrapper

        if config == "simple":
            from .demo import _demo_training
            resolved_fn = _make_wrapper(_demo_training)
        elif config == "multitask":
            from .golden_demo import _golden_training
            resolved_fn = _make_wrapper(_golden_training)
        elif config == "finetune":
            from .finetune_demo import _finetune_training
            resolved_fn = _make_wrapper(_finetune_training)

    # Write run metadata
    run_meta = {
        "run_id": time.strftime("%Y%m%d_%H%M%S"),
        "config": config if train_fn is None else str(train_fn),
        "max_steps": effective_steps,
        "max_time": max_time,
        "step_delay": effective_delay,
        "seed": seed,
        "autopilot": autopilot,
        "key_metric": key_metric,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "hotcb.run.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    if seed is not None:
        import random
        random.seed(seed)

    # Start wall-clock timer if max_time is set
    if max_time is not None and max_time > 0:
        def _time_limit():
            stop_event.wait(timeout=max_time)
            if not stop_event.is_set():
                log.info("[hotcb.launch] max_time=%.1fs reached, stopping training", max_time)
                stop_event.set()
        timer_thread = threading.Thread(
            target=_time_limit, daemon=True, name="hotcb-timer",
        )
        timer_thread.start()

    # Start training thread
    def _train_wrapper():
        try:
            resolved_fn(run_dir, effective_steps, effective_delay, stop_event)
        except Exception:
            log.exception("[hotcb.launch] training crashed")

    train_thread = threading.Thread(
        target=_train_wrapper,
        daemon=True,
        name="hotcb-training",
    )
    train_thread.start()

    # Start server
    server_thread = None
    server_url = f"http://{host}:{port}"
    if serve:
        def _server_wrapper():
            try:
                from .server.app import create_app
                import uvicorn
                app = create_app(run_dir)

                # Configure autopilot mode after app is created
                ap_engine = app.state.autopilot_engine
                ai_engine = getattr(app.state, "ai_engine", None)

                if ai_engine and autopilot in ("ai_suggest", "ai_auto"):
                    ai_engine.update_config({
                        "model": ai_model,
                        "base_url": ai_base_url,
                        "budget_cap": ai_budget,
                        "cadence": ai_cadence,
                    })
                    if ai_key:
                        ai_engine.update_config({"api_key": ai_key})
                    ai_engine.state.key_metric = key_metric
                    ai_engine.state.watch_metrics = watch_metrics or []
                    ai_engine.save_state()

                try:
                    ap_engine.set_mode(autopilot)
                except ValueError as e:
                    log.warning("[hotcb.launch] Failed to set autopilot mode: %s", e)

                uvicorn.run(app, host=host, port=port, log_level="warning")
            except Exception:
                log.exception("[hotcb.launch] server crashed")

        server_thread = threading.Thread(
            target=_server_wrapper,
            daemon=True,
            name="hotcb-server",
        )
        server_thread.start()
        # Give server a moment to start
        time.sleep(0.5)

    handle = LaunchHandle(
        run_dir=run_dir,
        server_url=server_url,
        autopilot_mode=autopilot,
        key_metric=key_metric,
        _stop_event=stop_event,
        _train_thread=train_thread,
        _server_thread=server_thread,
    )

    log.info(
        "[hotcb.launch] started: run_dir=%s, autopilot=%s, key_metric=%s, dashboard=%s",
        run_dir, autopilot, key_metric, server_url if serve else "off",
    )

    if block:
        handle.wait()

    return handle
