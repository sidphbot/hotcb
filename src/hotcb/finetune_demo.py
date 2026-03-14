"""
hotcb finetune demo — Transfer learning with recipe-driven LR control.

Demonstrates:
- Recipe-driven LR changes at scheduled steps
- Interactive optimizer overrides from the dashboard
- Observable metric impact from mutations (loss, accuracy, grad_norm)
- Feature drift monitoring over training progression

Usage:
    python -m hotcb.finetune_demo
    # or: hotcb demo --finetune
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

from hotcb.kernel import HotKernel
from hotcb.metrics import MetricsCollector
from hotcb.actuators import OptimizerActuator, MutableStateActuator


class _OptProxy:
    """Minimal optimizer-like object for synthetic demos."""

    def __init__(self, **kwargs):
        self.param_groups = [kwargs]


def _write_jsonl(path: str, record: dict) -> None:
    """Append a single JSON record to a JSONL file (features/recipe only)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _finetune_training(
    run_dir: str,
    *,
    max_steps: int = 600,
    step_delay: float = 0.12,
    _stop_event: Optional[threading.Event] = None,
) -> None:
    """Transfer learning simulation proving mutation impact via HotKernel.

    Three phases driven by a recipe that reduces LR:
    - Phase 1 (steps 1-200):  Fast head convergence with default LR (5e-4)
    - Phase 2 (steps 200-400): Recipe reduces LR to 2e-4 for fine-tuning
    - Phase 3 (steps 400-600): Recipe reduces LR to 5e-5 to mitigate overfitting

    Interactive control: the dashboard can adjust LR and weight_decay at any
    time; the kernel routes those commands through the opt module which
    mutates the optimizer proxy in-place.
    """
    commands_path = os.path.join(run_dir, "hotcb.commands.jsonl")
    features_path = os.path.join(run_dir, "hotcb.features.jsonl")
    recipe_path = os.path.join(run_dir, "hotcb.recipe.jsonl")

    # --- Optimizer proxy (kernel mutates param_groups via opt module) ---
    opt = _OptProxy(lr=5e-4, weight_decay=5e-4)

    # --- Mutable state for loss weight (kernel mutates via loss module) ---
    mutable_state: dict = {
        "weights": {"main": 1.0},
        "terms": {},
        "ramps": {},
    }

    # --- Wire HotKernel + MetricsCollector + actuators ---
    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))
    kernel = HotKernel(run_dir=run_dir, debounce_steps=1, metrics_collector=mc)
    kernel.register_actuator("opt", OptimizerActuator())
    kernel.register_actuator("loss", MutableStateActuator())

    # --- Recipe: scheduled commands written to commands.jsonl at trigger steps ---
    recipe_schedule: dict = {
        200: {"module": "opt", "op": "set_params", "params": {"lr": 2e-4},
              "source": "recipe"},
        400: {"module": "opt", "op": "set_params", "params": {"lr": 5e-5},
              "source": "recipe"},
    }

    # Write recipe display entries so the dashboard shows the schedule
    recipe_display = [
        {
            "at": {"step": 200},
            "module": "opt",
            "op": "set_params",
            "params": {"lr": 2e-4},
            "description": "Reduce LR for fine-tuning phase",
        },
        {
            "at": {"step": 400},
            "module": "opt",
            "op": "set_params",
            "params": {"lr": 5e-5},
            "description": "Reduce LR further to mitigate overfitting",
        },
    ]
    with open(recipe_path, "w", encoding="utf-8") as f:
        for entry in recipe_display:
            f.write(json.dumps(entry) + "\n")

    # --- Training state ---
    train_loss = 2.5 + random.uniform(-0.05, 0.05)
    val_loss = 2.7 + random.uniform(-0.05, 0.05)
    head_loss = 2.5 + random.uniform(-0.03, 0.03)
    grad_norm = 2.0 + random.uniform(-0.1, 0.1)
    feature_drift = 0.0

    # Track which recipe steps have been injected
    recipe_injected: set = set()

    for step in range(1, max_steps + 1):
        if _stop_event is not None and _stop_event.is_set():
            break

        # --- Inject scheduled recipe commands ---
        for trigger_step, cmd in recipe_schedule.items():
            if step == trigger_step and trigger_step not in recipe_injected:
                _write_jsonl(commands_path, cmd)
                recipe_injected.add(trigger_step)

        # --- Read current optimizer state (kernel may have mutated it) ---
        lr = opt.param_groups[0]["lr"]
        wd = opt.param_groups[0].get("weight_decay", 5e-4)
        loss_weight = mutable_state["weights"].get("main", 1.0)

        # --- Simulate training dynamics ---

        # Effective LR with warmup (first 30 steps)
        warmup_factor = min(1.0, step / 30.0)
        effective_lr = lr * warmup_factor

        # Convergence rate scales with LR — demonstrates mutation impact
        convergence_rate = 0.015 * min(effective_lr * 2000, 1.5)

        if step <= 200:
            # Phase 1: Fast head convergence (pretrained features help)
            target = 0.70
            noise = random.gauss(0, 0.012 * max(train_loss, 0.05))
            train_loss = max(
                target * 0.95,
                train_loss * (1 - convergence_rate) + noise,
            )
            # Head loss tracks train loss closely
            head_loss = train_loss + random.gauss(0, 0.005)
            # Grad norm decreases as head converges
            grad_target = 1.2 + 1.0 * (train_loss / 2.5)
            grad_norm = grad_norm * 0.97 + grad_target * 0.03 + random.gauss(0, 0.03)
        elif step <= 400:
            # Phase 2: Fine-tuning with reduced LR
            target = 0.25
            noise = random.gauss(0, 0.008 * max(train_loss, 0.03))
            train_loss = max(
                target * 0.9,
                train_loss * (1 - convergence_rate) + noise,
            )
            # Small grad norm spike when LR changes, then settle
            steps_in_phase = step - 200
            if steps_in_phase <= 10:
                spike = 0.3 * math.exp(-steps_in_phase / 4.0)
                grad_target = 0.8 + spike + 0.5 * (train_loss / 0.7)
            else:
                grad_target = 0.6 + 0.5 * (train_loss / 0.7)
            grad_norm = grad_norm * 0.95 + grad_target * 0.05 + random.gauss(0, 0.02)
            head_loss = max(0.08, train_loss * 0.65 + random.gauss(0, 0.01))
        else:
            # Phase 3: Very low LR, overfitting tendency on small dataset
            target = 0.12
            noise = random.gauss(0, 0.005 * max(train_loss, 0.02))
            train_loss = max(
                target * 0.85,
                train_loss * (1 - convergence_rate) + noise,
            )
            # Another small spike at LR change
            steps_in_phase = step - 400
            if steps_in_phase <= 10:
                spike = 0.2 * math.exp(-steps_in_phase / 4.0)
                grad_target = 0.5 + spike + 0.3 * (train_loss / 0.25)
            else:
                grad_target = 0.3 + 0.3 * (train_loss / 0.25)
            grad_norm = grad_norm * 0.95 + grad_target * 0.05 + random.gauss(0, 0.015)
            head_loss = max(0.05, train_loss * 0.55 + random.gauss(0, 0.008))

        grad_norm = max(0.08, grad_norm)

        # --- Validation loss: overfit gap grows, especially after step 400 ---
        if step < 400:
            overfit_gap = 0.03 + step * 0.0002 * max(0.1, 1 - wd * 500)
        else:
            overfit_gap = 0.03 + 400 * 0.0002 * max(0.1, 1 - wd * 500)
            overfit_gap += (step - 400) * 0.0008 * max(0.1, 1 - wd * 500)
        overfit_gap = max(0.02, overfit_gap)

        val_loss = max(
            target * 1.1,
            train_loss + overfit_gap + random.gauss(0, 0.015),
        )

        # --- Accuracy from logistic mapping of loss ---
        accuracy = min(
            0.98,
            max(0.20, 1.0 / (1.0 + math.exp(2.5 * (train_loss - 0.8)))
                + random.gauss(0, 0.008)),
        )
        val_accuracy = min(
            0.96,
            max(0.18, 1.0 / (1.0 + math.exp(2.5 * (val_loss - 0.8)))
                + random.gauss(0, 0.01)),
        )

        # --- Feature drift: increases as model trains further from init ---
        drift_rate = 0.002 * min(effective_lr * 2000, 1.0)
        feature_drift = min(
            1.0,
            feature_drift + drift_rate + random.gauss(0, 0.001),
        )

        # --- Apply loss weight scaling ---
        scaled_train_loss = train_loss * loss_weight
        scaled_val_loss = val_loss * loss_weight

        # --- Build env and call kernel.apply ---
        env = {
            "step": step,
            "epoch": step // 50,
            "optimizer": opt,
            "mutable_state": mutable_state,
            "metrics": {
                "train_loss": round(scaled_train_loss, 6),
                "val_loss": round(scaled_val_loss, 6),
                "accuracy": round(accuracy, 4),
                "val_accuracy": round(val_accuracy, 4),
                "grad_norm": round(grad_norm, 4),
                "lr": lr,
                "weight_decay": wd,
                "head_loss": round(head_loss, 6),
                "feature_drift": round(feature_drift, 6),
            },
        }
        kernel.apply(env, events=["train_step_end"])

        # --- Feature capture (every 20 steps) ---
        if step % 20 == 0:
            progress = step / max_steps
            spread = max(0.25, 0.6 - progress * 0.4)
            center_shift = progress * 1.5

            activations = []
            for cls in range(5):
                cx = math.cos(cls * 2 * math.pi / 5) * (1.5 + center_shift)
                cy = math.sin(cls * 2 * math.pi / 5) * (1.5 + center_shift)
                cz = 0.5 * math.sin(cls * math.pi / 5 + step * 0.005)
                activations.append([
                    round(cx + random.gauss(0, spread), 4),
                    round(cy + random.gauss(0, spread), 4),
                    round(cz + random.gauss(0, spread * 0.5), 4),
                ])
            _write_jsonl(features_path, {
                "step": step,
                "layer": "backbone.layer4",
                "activations": activations,
            })

        if _stop_event is not None and _stop_event.is_set():
            break
        time.sleep(step_delay)

    # --- Finalize ---
    final_env = {
        "step": max_steps,
        "epoch": max_steps // 50,
        "optimizer": opt,
        "mutable_state": mutable_state,
    }
    kernel.close(final_env)


def run_finetune_demo(
    *,
    host: str = "0.0.0.0",
    port: int = 8421,
    max_steps: int = 600,
    step_delay: float = 0.12,
    run_dir: Optional[str] = None,
) -> None:
    """Launch finetune demo: recipe-driven LR control + dashboard server."""
    if run_dir is None:
        run_dir = tempfile.mkdtemp(prefix="hotcb_finetune_")

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
    w("  +================================================+\n")
    w("  |     hotcb Finetune Demo                        |\n")
    w("  |     Transfer Learning Control Plane             |\n")
    w("  +================================================+\n")
    w(f"\n  Run dir:   {run_dir}\n")
    w(f"  Dashboard: http://localhost:{port}\n")
    w(f"  Training:  {max_steps} steps @ {step_delay}s/step\n\n")
    w("  This demo simulates finetuning a pretrained model on a\n")
    w("  small 5-class dataset using HotKernel integration.\n\n")
    w("  Recipe schedule:\n")
    w("    Step 200: Reduce LR to 2e-4 (fine-tuning phase)\n")
    w("    Step 400: Reduce LR to 5e-5 (mitigate overfitting)\n\n")
    w("  Watch for:\n")
    w("    * Fast convergence in phase 1 (default LR)\n")
    w("    * Convergence rate change at LR mutations (steps 200, 400)\n")
    w("    * Growing val/train gap after step 400 (overfitting)\n")
    w("    * Interactive LR/weight_decay overrides from dashboard\n\n")
    w("  Press Ctrl+C to stop.\n\n")

    train_thread = threading.Thread(
        target=_finetune_training,
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
    run_finetune_demo()
