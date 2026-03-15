"""
hotcb demo — launch a synthetic training with a live dashboard.

Runs a synthetic training loop in a background thread using HotKernel
for control-plane integration, writing metrics via MetricsCollector,
while serving the dashboard on localhost.
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


class _OptProxy:
    """Minimal optimizer-like object for synthetic demos.

    Provides ``param_groups`` so optimizer actuators can mutate lr/wd
    exactly as they would with a real ``torch.optim.Optimizer``.
    """

    def __init__(self, **kwargs):
        self.param_groups = [kwargs]


def _demo_training(
    run_dir: str,
    *,
    max_steps: int = 500,
    step_delay: float = 0.15,
    _stop_event: Optional[threading.Event] = None,
) -> None:
    """Synthetic training loop using HotKernel + MetricsCollector.

    Simulates a 2-layer network training on a quadratic task.
    Responds to lr/wd changes via the kernel's opt module — dashboard
    commands are picked up automatically by HotKernel each step.
    """
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from hotcb.actuators import optimizer_actuators, mutable_state

    # --- Optimizer proxy (kernel mutates param_groups in-place) ---
    opt = _OptProxy(lr=1e-3, weight_decay=1e-4)

    # --- Wire up kernel + metrics ---
    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    ms = mutable_state(optimizer_actuators(opt))
    kernel = HotKernel(run_dir=run_dir, debounce_steps=1, metrics_collector=mc, mutable_state=ms)

    # --- Simulated training state ---
    loss = 2.5 + random.uniform(-0.1, 0.1)
    val_loss = 2.8 + random.uniform(-0.1, 0.1)
    grad_norm = 5.0

    for step in range(1, max_steps + 1):
        if _stop_event is not None and _stop_event.is_set():
            break

        # Read lr and wd FROM the proxy — kernel's opt module mutates these
        # when dashboard commands arrive via hotcb.commands.jsonl
        lr = opt.param_groups[0]["lr"]
        wd = opt.param_groups[0].get("weight_decay", 1e-4)

        # --- Simulate loss decay with noise ---
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

        # --- Build env dict for the kernel ---
        env = {
            "framework": "synthetic",
            "phase": "train",
            "step": step,
            "optimizer": opt,
            "metrics": {
                "train_loss": round(loss, 6),
                "val_loss": round(val_loss, 6),
                "grad_norm": round(grad_norm, 4),
                "lr": lr,
                "weight_decay": wd,
                "accuracy": round(accuracy, 4),
                "val_accuracy": round(val_accuracy, 4),
            },
            "log": lambda s: None,
        }

        # --- Kernel safe-point: poll commands, apply ops, collect metrics ---
        kernel.apply(env, events=["train_step_end"])

        if _stop_event is not None and _stop_event.is_set():
            break
        time.sleep(step_delay)

    # --- Finalize ---
    final_env = {
        "framework": "synthetic",
        "phase": "train",
        "step": max_steps,
        "optimizer": opt,
        "metrics": {
            "train_loss": round(loss, 6),
            "val_loss": round(val_loss, 6),
        },
        "log": lambda s: None,
    }
    kernel.close(final_env)


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

    # Bootstrap run directory — truncate any existing files for a clean start
    os.makedirs(run_dir, exist_ok=True)
    for fname in [
        "hotcb.commands.jsonl",
        "hotcb.applied.jsonl",
        "hotcb.metrics.jsonl",
        "hotcb.recipe.jsonl",
    ]:
        open(os.path.join(run_dir, fname), "w").close()

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

    run_server(run_dir=run_dir, host=host, port=port, poll_interval=0.3, demo_mode=True)
