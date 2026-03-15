"""
hotcb golden demo — Multi-task training with recipe-driven loss weight shifts.

Demonstrates:
- Multi-task training (classification + reconstruction) with guaranteed convergence
- Live loss weight manipulation via recipe
- Feature capture data for activation visualization
- Realistic training dynamics (warmup, plateau, intervention response)
- Proper HotKernel integration (same path as real training loops)

Usage:
    python -m hotcb.golden_demo
    # or: hotcb demo --golden
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
    """Minimal optimizer-like object for synthetic demos."""

    def __init__(self, **kwargs):
        self.param_groups = [kwargs]


def _golden_training(
    run_dir: str,
    *,
    max_steps: int = 800,
    step_delay: float = 0.12,
    _stop_event: Optional[threading.Event] = None,
) -> None:
    """Multi-task synthetic training loop with guaranteed convergence.

    Simulates two tasks:
    - Task A: Classification (cross-entropy loss -> accuracy)
    - Task B: Reconstruction (MSE loss -> reconstruction error)

    Both tasks share a backbone and have controllable loss weights.
    A recipe automatically shifts loss weights at steps 200, 400, 500.

    Uses HotKernel + MetricsCollector + actuators, exactly like a real
    training integration.
    """
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector
    from hotcb.actuators import optimizer_actuators, loss_actuators, mutable_state

    # --- File paths ---
    commands_path = os.path.join(run_dir, "hotcb.commands.jsonl")
    features_path = os.path.join(run_dir, "hotcb.features.jsonl")
    recipe_path = os.path.join(run_dir, "hotcb.recipe.jsonl")

    # --- Write recipe display entries (for dashboard UI) ---
    recipe_entries = [
        {"at": {"step": 200}, "module": "loss", "op": "set_params",
         "params": {"weight_a": 0.3, "weight_b": 0.7},
         "description": "Shift focus to reconstruction task"},
        {"at": {"step": 400}, "module": "opt", "op": "set_params",
         "params": {"lr": 5e-4},
         "description": "Reduce LR for fine-tuning phase"},
        {"at": {"step": 500}, "module": "loss", "op": "set_params",
         "params": {"weight_a": 0.5, "weight_b": 0.5},
         "description": "Rebalance tasks for final convergence"},
    ]
    with open(recipe_path, "w", encoding="utf-8") as f:
        for entry in recipe_entries:
            f.write(json.dumps(entry) + "\n")

    # --- Scheduled recipe commands (written to commands.jsonl at the right step) ---
    recipe_schedule = {
        200: {"module": "loss", "op": "set_params",
              "params": {"weight_a": 0.3, "weight_b": 0.7}, "source": "recipe"},
        400: {"module": "opt", "op": "set_params",
              "params": {"lr": 5e-4}, "source": "recipe"},
        500: {"module": "loss", "op": "set_params",
              "params": {"weight_a": 0.5, "weight_b": 0.5}, "source": "recipe"},
    }

    # --- Optimizer proxy ---
    opt = _OptProxy(lr=1e-3, weight_decay=1e-4)

    # --- Mutable state for multi-task loss weights ---
    loss_weights = {"weight_a": 0.7, "weight_b": 0.3}

    # --- Wire HotKernel + MetricsCollector + actuators ---
    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    ms = mutable_state(optimizer_actuators(opt) + loss_actuators(loss_weights))
    kernel = HotKernel(run_dir=run_dir, debounce_steps=1, metrics_collector=mc, mutable_state=ms)

    # --- Training state ---
    loss_a = 2.3 + random.uniform(-0.05, 0.05)   # CE loss
    loss_b = 1.5 + random.uniform(-0.05, 0.05)   # MSE loss
    val_loss_a = 2.5
    val_loss_b = 1.7
    grad_norm = 4.0

    # Convergence targets
    target_loss_a = 0.15
    target_loss_b = 0.08

    for step in range(1, max_steps + 1):
        # --- Check stop event ---
        if _stop_event is not None and _stop_event.is_set():
            break

        # --- Inject scheduled recipe commands ---
        if step in recipe_schedule:
            with open(commands_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(recipe_schedule[step]) + "\n")

        # --- Read current state from kernel-managed objects ---
        lr = opt.param_groups[0]["lr"]
        wd = opt.param_groups[0]["weight_decay"]
        weight_a = loss_weights["weight_a"]
        weight_b = loss_weights["weight_b"]

        # --- Simulate training dynamics ---
        # Warmup phase (steps 1-50)
        warmup_factor = min(1.0, step / 50.0)
        effective_lr = lr * warmup_factor

        # Learning rate factor -- higher LR = faster convergence to a point
        lr_factor = min(effective_lr * 500, 1.0)

        # Task A: Classification (exponential decay toward target)
        noise_a = random.gauss(0, 0.015 * max(loss_a, 0.05))
        loss_a = max(target_loss_a * 0.9,
                     loss_a * (1 - 0.008 * lr_factor * weight_a) + noise_a)

        # Task B: Reconstruction (slower convergence, responds to weight)
        noise_b = random.gauss(0, 0.01 * max(loss_b, 0.03))
        loss_b = max(target_loss_b * 0.9,
                     loss_b * (1 - 0.006 * lr_factor * weight_b) + noise_b)

        # Combined loss
        total_loss = weight_a * loss_a + weight_b * loss_b

        # Grad norm (decreases over training, spikes on weight changes)
        grad_decay = 0.995 if step > 50 else 0.99
        grad_norm = max(0.1, grad_norm * grad_decay + random.gauss(0, 0.03))

        # Validation losses (slightly worse, with overfitting tendency)
        overfit_a = 0.05 + step * 0.00015 * (1 - wd * 300)
        overfit_b = 0.03 + step * 0.0001 * (1 - wd * 300)
        val_loss_a = max(target_loss_a, loss_a + overfit_a + random.gauss(0, 0.02))
        val_loss_b = max(target_loss_b, loss_b + overfit_b + random.gauss(0, 0.015))

        # Accuracy metrics
        accuracy_a = min(0.98, max(0.1, 1.0 - loss_a * 0.4 + random.gauss(0, 0.008)))
        val_accuracy_a = min(0.98, max(0.1, 1.0 - val_loss_a * 0.35 + random.gauss(0, 0.01)))
        recon_score = min(0.99, max(0.1, 1.0 - loss_b * 0.5 + random.gauss(0, 0.01)))

        # --- Build env dict ---
        env = {
            "step": step,
            "epoch": step // 50,
            "optimizer": opt,
            "metrics": {
                "train_loss": round(total_loss, 6),
                "val_loss": round(weight_a * val_loss_a + weight_b * val_loss_b, 6),
                "task_a_loss": round(loss_a, 6),
                "task_b_loss": round(loss_b, 6),
                "val_task_a_loss": round(val_loss_a, 6),
                "val_task_b_loss": round(val_loss_b, 6),
                "accuracy": round(accuracy_a, 4),
                "val_accuracy": round(val_accuracy_a, 4),
                "recon_score": round(recon_score, 4),
                "grad_norm": round(grad_norm, 4),
                "lr": lr,
                "weight_decay": wd,
                "weight_a": weight_a,
                "weight_b": weight_b,
            },
            "hp": {
                "lr": lr,
                "weight_decay": wd,
            },
        }

        # --- Let kernel process commands + collect metrics ---
        kernel.apply(env, events=["train_step_end"])

        # --- Feature capture (every 20 steps) ---
        if step % 20 == 0:
            # Simulate PCA-reduced activations (3 components)
            # Features cluster tighter as training progresses
            spread = max(0.3, 2.0 - step * 0.002)
            activations = []
            for _ in range(5):
                activations.append([
                    round(math.sin(step * 0.02) * (1 + random.gauss(0, spread)), 4),
                    round(math.cos(step * 0.015) * (1 + random.gauss(0, spread)), 4),
                    round(math.sin(step * 0.01 + 1.5) * (1 + random.gauss(0, spread)), 4),
                ])
            with open(features_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step": step,
                    "layer": "backbone.fc2",
                    "activations": activations,
                }) + "\n")

        # Check stop event before sleeping
        if _stop_event is not None and _stop_event.is_set():
            break
        time.sleep(step_delay)

    # --- Finalize ---
    kernel.close(env)


def run_golden_demo(
    *,
    host: str = "0.0.0.0",
    port: int = 8421,
    max_steps: int = 800,
    step_delay: float = 0.12,
    run_dir: Optional[str] = None,
) -> None:
    """Launch golden demo: multi-task training + dashboard server."""
    if run_dir is None:
        run_dir = tempfile.mkdtemp(prefix="hotcb_golden_")

    # Bootstrap run directory — truncate any existing files for a clean start
    os.makedirs(run_dir, exist_ok=True)
    for fname in [
        "hotcb.commands.jsonl",
        "hotcb.applied.jsonl",
        "hotcb.metrics.jsonl",
        "hotcb.recipe.jsonl",
        "hotcb.features.jsonl",
    ]:
        open(os.path.join(run_dir, fname), "w").close()

    with open(os.path.join(run_dir, "hotcb.freeze.json"), "w") as f:
        json.dump({"mode": "off"}, f)

    import sys
    w = sys.stderr.write
    w("\n")
    w("  ╔══════════════════════════════════════════════╗\n")
    w("  ║     hotcb Golden Demo                       ║\n")
    w("  ║     Multi-task Training Control Plane        ║\n")
    w("  ╚══════════════════════════════════════════════╝\n")
    w(f"\n  Run dir:   {run_dir}\n")
    w(f"  Dashboard: http://localhost:{port}\n")
    w(f"  Training:  {max_steps} steps @ {step_delay}s/step\n\n")
    w("  This demo runs a multi-task training simulation with:\n")
    w("  * Task A: Classification (cross-entropy -> accuracy)\n")
    w("  * Task B: Reconstruction (MSE -> reconstruction score)\n\n")
    w("  A recipe automatically shifts loss weights at steps 200, 400, 500.\n")
    w("  Try adjusting knobs to see how it affects both tasks!\n\n")
    w("  Press Ctrl+C to stop.\n\n")

    train_thread = threading.Thread(
        target=_golden_training,
        kwargs={
            "run_dir": run_dir,
            "max_steps": max_steps,
            "step_delay": step_delay,
        },
        daemon=True,
    )
    train_thread.start()

    from .server.app import run_server
    run_server(run_dir=run_dir, host=host, port=port, poll_interval=0.3, demo_mode=True)


if __name__ == "__main__":
    run_golden_demo()
