"""
hotcb demo — launch a synthetic training with a live dashboard.

Runs a small neural network training loop in a background thread,
writing metrics to JSONL, while serving the dashboard on localhost.
Open ``http://localhost:8421`` to interact with knobs, see live charts,
forecasts, and tinker with the control plane.
"""
from __future__ import annotations

import json
import math
import os
import random
import tempfile
import threading
import time
from typing import Optional


def _write_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _read_commands(path: str, offset: int) -> tuple[list[dict], int]:
    """Read new commands from offset, return (commands, new_offset)."""
    if not os.path.exists(path):
        return [], offset
    cmds = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                cmds.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return cmds, offset + len(cmds)


def _demo_training(
    run_dir: str,
    *,
    max_steps: int = 500,
    step_delay: float = 0.15,
    _stop_event: Optional[threading.Event] = None,
) -> None:
    """Synthetic training loop that writes hotcb.metrics.jsonl.

    Simulates a 2-layer network training on a quadratic task.
    Responds to lr/wd changes from hotcb.commands.jsonl.
    """
    metrics_path = os.path.join(run_dir, "hotcb.metrics.jsonl")
    commands_path = os.path.join(run_dir, "hotcb.commands.jsonl")
    applied_path = os.path.join(run_dir, "hotcb.applied.jsonl")

    # Simulated hyperparameters
    lr = 1e-3
    wd = 1e-4

    # Simulated training state
    loss = 2.5 + random.uniform(-0.1, 0.1)
    val_loss = 2.8 + random.uniform(-0.1, 0.1)
    grad_norm = 5.0
    cmd_offset = 0

    for step in range(1, max_steps + 1):
        if _stop_event is not None and _stop_event.is_set():
            break
        # Check for commands
        cmds, cmd_offset = _read_commands(commands_path, cmd_offset)
        for cmd in cmds:
            module = cmd.get("module", "")
            op = cmd.get("op", "")
            params = cmd.get("params", {})
            if module == "opt" and op == "set_params":
                if "lr" in params:
                    lr = float(params["lr"])
                if "weight_decay" in params:
                    wd = float(params["weight_decay"])
                _write_jsonl(applied_path, {
                    "step": step, "module": "opt", "op": "set_params",
                    "params": {"lr": lr, "weight_decay": wd},
                    "decision": "applied", "status": "applied",
                    "source": cmd.get("source", "interactive"),
                })
            elif module == "loss" and op == "set_params":
                _write_jsonl(applied_path, {
                    "step": step, "module": "loss", "op": "set_params",
                    "params": params,
                    "decision": "applied", "status": "applied",
                    "source": cmd.get("source", "interactive"),
                })

        # Simulate loss decay with noise
        # lr affects convergence speed; wd adds regularization effect
        lr_factor = min(lr * 300, 1.0)  # higher lr = faster convergence up to a point
        decay = 0.005 * lr_factor * (1 + 0.3 * math.sin(step * 0.05))
        noise = random.gauss(0, 0.02 * max(loss, 0.1))
        loss = max(0.05, loss - decay + noise + wd * 0.5)

        # Grad norm decreases with training
        grad_norm = max(0.1, grad_norm * 0.997 + random.gauss(0, 0.05))

        # Val loss with slight overfitting tendency
        overfit_gap = 0.1 + step * 0.0003  # slowly grows
        val_noise = random.gauss(0, 0.03 * max(val_loss, 0.1))
        val_loss = max(0.08, loss + overfit_gap * (1 - wd * 200) + val_noise)

        # Accuracy (classification proxy)
        accuracy = min(0.99, max(0.1, 1.0 - loss * 0.35 + random.gauss(0, 0.01)))
        val_accuracy = min(0.99, max(0.1, 1.0 - val_loss * 0.3 + random.gauss(0, 0.015)))

        record = {
            "step": step,
            "metrics": {
                "train_loss": round(loss, 6),
                "val_loss": round(val_loss, 6),
                "grad_norm": round(grad_norm, 4),
                "lr": lr,
                "accuracy": round(accuracy, 4),
                "val_accuracy": round(val_accuracy, 4),
            },
        }
        _write_jsonl(metrics_path, record)
        if _stop_event is not None and _stop_event.is_set():
            break
        time.sleep(step_delay)

    # Write a final summary (only numeric metrics to avoid parse errors)
    _write_jsonl(metrics_path, {
        "step": max_steps,
        "metrics": {
            "train_loss": round(loss, 6),
            "val_loss": round(val_loss, 6),
        },
    })


def run_demo(
    *,
    host: str = "0.0.0.0",
    port: int = 8421,
    max_steps: int = 500,
    step_delay: float = 0.15,
    run_dir: Optional[str] = None,
) -> None:
    """Launch demo: synthetic training + dashboard server."""
    if run_dir is None:
        run_dir = tempfile.mkdtemp(prefix="hotcb_demo_")

    # Bootstrap run directory
    os.makedirs(run_dir, exist_ok=True)
    for fname in [
        "hotcb.commands.jsonl",
        "hotcb.applied.jsonl",
        "hotcb.metrics.jsonl",
        "hotcb.recipe.jsonl",
    ]:
        path = os.path.join(run_dir, fname)
        if not os.path.exists(path):
            open(path, "w").close()

    # Write freeze state
    with open(os.path.join(run_dir, "hotcb.freeze.json"), "w") as f:
        json.dump({"mode": "off"}, f)

    import sys
    w = sys.stderr.write
    w(f"\n  hotcb demo\n")
    w(f"  Run dir:   {run_dir}\n")
    w(f"  Dashboard: http://localhost:{port}\n\n")
    w("  Open the dashboard URL in your browser.\n")
    w("  Use the Training panel to start a run (Simple, Multi-Objective, or Finetune).\n")
    w("  Then use knobs, recipes, and autopilot to control training live.\n")
    w("  Press Ctrl+C to stop the server.\n\n")

    # Run dashboard server (blocking) — training is started from the UI
    from .server.app import run_server

    run_server(run_dir=run_dir, host=host, port=port, poll_interval=0.3)
