"""
hotcb finetune demo — Transfer learning with backbone freeze/unfreeze control.

Demonstrates:
- Pretrained backbone finetuning on a small 5-class dataset (1000 samples)
- Live backbone freeze/unfreeze toggle via recipe and interactive commands
- Overfitting dynamics on small datasets
- Feature drift monitoring from pretrained initialization

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


def _read_recipe(path: str) -> list[dict]:
    """Read all recipe entries from a JSONL file."""
    entries: list[dict] = []
    if not os.path.exists(path):
        return entries
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def _finetune_training(
    run_dir: str,
    *,
    max_steps: int = 600,
    step_delay: float = 0.12,
    _stop_event: Optional[threading.Event] = None,
) -> None:
    """Finetuning simulation: pretrained backbone on a small 5-class dataset.

    Simulates transfer learning where a pretrained ImageNet backbone is
    finetuned on 1000 samples across 5 classes. The backbone starts frozen
    (only classifier head trains), and a recipe unfreezes it at step 200.

    Training dynamics:
    - Frozen phase (steps 1-200): fast head convergence, low grad norm
    - Unfrozen phase (steps 200+): full model trains, loss spike then faster
      convergence, higher grad norm
    - Overfitting risk after step 400 on the small dataset
    """
    metrics_path = os.path.join(run_dir, "hotcb.metrics.jsonl")
    commands_path = os.path.join(run_dir, "hotcb.commands.jsonl")
    applied_path = os.path.join(run_dir, "hotcb.applied.jsonl")
    features_path = os.path.join(run_dir, "hotcb.features.jsonl")
    recipe_path = os.path.join(run_dir, "hotcb.recipe.jsonl")

    # Write the finetune recipe
    recipe_entries = [
        {
            "at": {"step": 200},
            "module": "cb",
            "op": "set_params",
            "params": {"backbone_frozen": False},
            "description": "Unfreeze backbone for full finetuning",
        },
        {
            "at": {"step": 400},
            "module": "opt",
            "op": "set_params",
            "params": {"lr": 1e-4},
            "description": "Reduce LR for fine-grained tuning, mitigate overfitting",
        },
    ]
    with open(recipe_path, "w", encoding="utf-8") as f:
        for entry in recipe_entries:
            f.write(json.dumps(entry) + "\n")

    # Hyperparameters
    lr = 5e-4
    wd = 5e-4
    backbone_frozen = True

    # Training state
    train_loss = 2.5 + random.uniform(-0.05, 0.05)
    val_loss = 2.7 + random.uniform(-0.05, 0.05)
    head_loss = 2.5 + random.uniform(-0.03, 0.03)
    grad_norm = 0.5 + random.uniform(-0.05, 0.05)
    feature_drift = 0.0  # how much backbone features have shifted
    loss_weight = 1.0  # adjustable via loss module

    # Convergence targets
    # Frozen phase can only get so far (head-only has limited capacity)
    frozen_target_loss = 0.75
    # Unfrozen phase can converge much further
    unfrozen_target_loss = 0.12

    # Track the step at which backbone was unfrozen (for transition dynamics)
    unfreeze_step: Optional[int] = None
    # Pre-unfreeze loss snapshot (for spike calculation)
    pre_unfreeze_loss: Optional[float] = None

    cmd_offset = 0
    recipe_applied: set[int] = set()

    for step in range(1, max_steps + 1):
        if _stop_event is not None and _stop_event.is_set():
            break

        # --- Process recipe entries ---
        current_recipe = _read_recipe(recipe_path)
        for idx, entry in enumerate(current_recipe):
            if idx in recipe_applied:
                continue
            at_step = entry.get("at", {}).get("step", 0)
            if step >= at_step:
                module = entry.get("module", "")
                op = entry.get("op", "")
                params = entry.get("params", {})
                if module == "cb" and op == "set_params":
                    if "backbone_frozen" in params:
                        new_frozen = bool(params["backbone_frozen"])
                        if backbone_frozen and not new_frozen:
                            # Transitioning from frozen to unfrozen
                            unfreeze_step = step
                            pre_unfreeze_loss = train_loss
                        elif not backbone_frozen and new_frozen:
                            # Re-freezing backbone
                            unfreeze_step = None
                        backbone_frozen = new_frozen
                elif module == "opt" and op == "set_params":
                    if "lr" in params:
                        lr = float(params["lr"])
                    if "weight_decay" in params:
                        wd = float(params["weight_decay"])
                elif module == "loss" and op == "set_params":
                    if "weight" in params:
                        loss_weight = float(params["weight"])
                _write_jsonl(applied_path, {
                    "step": step,
                    "module": module,
                    "op": op,
                    "params": params,
                    "decision": "applied",
                    "status": "applied",
                    "source": "recipe",
                    "description": entry.get("description", ""),
                })
                recipe_applied.add(idx)

        # --- Process interactive commands ---
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
                    "step": step,
                    "module": "opt",
                    "op": "set_params",
                    "params": {"lr": lr, "weight_decay": wd},
                    "decision": "applied",
                    "status": "applied",
                    "source": "interactive",
                })
            elif module == "cb" and op == "set_params":
                if "backbone_frozen" in params:
                    new_frozen = bool(params["backbone_frozen"])
                    if backbone_frozen and not new_frozen:
                        unfreeze_step = step
                        pre_unfreeze_loss = train_loss
                    elif not backbone_frozen and new_frozen:
                        unfreeze_step = None
                    backbone_frozen = new_frozen
                _write_jsonl(applied_path, {
                    "step": step,
                    "module": "cb",
                    "op": "set_params",
                    "params": {"backbone_frozen": backbone_frozen},
                    "decision": "applied",
                    "status": "applied",
                    "source": "interactive",
                })
            elif module == "loss" and op == "set_params":
                if "weight" in params:
                    loss_weight = float(params["weight"])
                _write_jsonl(applied_path, {
                    "step": step,
                    "module": "loss",
                    "op": "set_params",
                    "params": params,
                    "decision": "applied",
                    "status": "applied",
                    "source": "interactive",
                })

        # --- Simulate training dynamics ---

        # Effective learning rate with warmup (first 30 steps)
        warmup_factor = min(1.0, step / 30.0)
        effective_lr = lr * warmup_factor

        if backbone_frozen:
            # === FROZEN BACKBONE: only head trains ===
            # Fast initial convergence but plateaus early
            target = frozen_target_loss
            convergence_rate = 0.012 * min(effective_lr * 1000, 1.0)

            noise = random.gauss(0, 0.012 * max(train_loss, 0.05))
            train_loss = max(
                target * 0.95,
                train_loss * (1 - convergence_rate) + noise,
            )

            # Head loss tracks train loss closely when frozen
            head_loss = train_loss + random.gauss(0, 0.005)

            # Grad norm stays low (only head parameters)
            grad_target = 0.3 + 0.2 * (train_loss / 2.5)
            grad_norm = grad_norm * 0.97 + grad_target * 0.03 + random.gauss(0, 0.02)
            grad_norm = max(0.08, grad_norm)

            # No feature drift when backbone is frozen
            feature_drift = max(0.0, feature_drift * 0.999 + random.gauss(0, 0.001))

        else:
            # === UNFROZEN BACKBONE: full model trains ===
            target = unfrozen_target_loss
            steps_since_unfreeze = (
                step - unfreeze_step if unfreeze_step is not None else step
            )

            # Loss spike right after unfreezing (new gradient flow destabilizes)
            if steps_since_unfreeze <= 15 and pre_unfreeze_loss is not None:
                # Sharp spike then rapid recovery
                spike_magnitude = 0.25 * math.exp(-steps_since_unfreeze / 5.0)
                spike_noise = random.gauss(0, 0.02)
                train_loss = pre_unfreeze_loss + spike_magnitude + spike_noise
            else:
                # Faster convergence than frozen phase
                convergence_rate = 0.018 * min(effective_lr * 1000, 1.0)
                noise = random.gauss(0, 0.008 * max(train_loss, 0.03))
                train_loss = max(
                    target * 0.9,
                    train_loss * (1 - convergence_rate) + noise,
                )

            # Head loss diverges from total as backbone contributes
            head_loss = max(
                0.05,
                train_loss * 0.6 + random.gauss(0, 0.01),
            )

            # Grad norm jumps up then decays
            if steps_since_unfreeze <= 20:
                # Initial jump when backbone unfreezes
                grad_target = 3.0 + 1.5 * math.exp(-steps_since_unfreeze / 8.0)
            else:
                grad_target = 1.0 + 2.0 * (train_loss / 1.0)
            grad_norm = grad_norm * 0.92 + grad_target * 0.08 + random.gauss(0, 0.05)
            grad_norm = max(0.15, grad_norm)

            # Feature drift increases as backbone weights change
            drift_rate = 0.003 * min(effective_lr * 2000, 1.0)
            feature_drift = min(
                1.0,
                feature_drift + drift_rate + random.gauss(0, 0.001),
            )

        # --- Validation metrics ---
        # Small dataset = overfitting risk, especially after step 400
        if step < 400:
            overfit_gap = 0.03 + step * 0.0002 * (1 - wd * 500)
        else:
            # Overfitting accelerates on the small dataset
            overfit_gap = 0.03 + 400 * 0.0002 * (1 - wd * 500)
            overfit_gap += (step - 400) * 0.0008 * (1 - wd * 500)

        overfit_gap = max(0.02, overfit_gap)

        val_loss = max(
            target * 1.1 if not backbone_frozen else frozen_target_loss * 1.05,
            train_loss + overfit_gap + random.gauss(0, 0.015),
        )

        # Accuracy: derived from loss with logistic mapping
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

        # Apply loss weight scaling
        scaled_train_loss = train_loss * loss_weight
        scaled_val_loss = val_loss * loss_weight

        record = {
            "step": step,
            "epoch": step // 50,
            "wall_time": time.time(),
            "metrics": {
                "train_loss": round(scaled_train_loss, 6),
                "val_loss": round(scaled_val_loss, 6),
                "accuracy": round(accuracy, 4),
                "val_accuracy": round(val_accuracy, 4),
                "grad_norm": round(grad_norm, 4),
                "lr": lr,
                "backbone_frozen": 1 if backbone_frozen else 0,
                "head_loss": round(head_loss, 6),
                "feature_drift": round(feature_drift, 6),
            },
            "hp": {
                "lr": lr,
                "weight_decay": wd,
                "backbone_frozen": backbone_frozen,
            },
        }
        _write_jsonl(metrics_path, record)

        # --- Feature capture (every 20 steps) ---
        if step % 20 == 0:
            # Simulate PCA-reduced activations
            # Frozen: features stay clustered near pretrained position
            # Unfrozen: features spread then re-cluster in new arrangement
            if backbone_frozen:
                spread = 0.4 + 0.1 * random.random()
                center_shift = 0.0
            else:
                steps_unfrozen = (
                    step - unfreeze_step if unfreeze_step is not None else 0
                )
                # Features initially scatter then re-cluster
                spread = max(0.25, 0.8 - steps_unfrozen * 0.003)
                center_shift = min(2.0, steps_unfrozen * 0.01)

            activations = []
            for cls in range(5):
                # Each class gets a distinct cluster center
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

    # Final summary
    _write_jsonl(metrics_path, {
        "step": max_steps,
        "metrics": {
            "train_loss": round(scaled_train_loss, 6),
            "val_loss": round(scaled_val_loss, 6),
            "accuracy": round(accuracy, 4),
            "val_accuracy": round(val_accuracy, 4),
        },
    })


def run_finetune_demo(
    *,
    host: str = "0.0.0.0",
    port: int = 8421,
    max_steps: int = 600,
    step_delay: float = 0.12,
    run_dir: Optional[str] = None,
) -> None:
    """Launch finetune demo: backbone freeze/unfreeze training + dashboard server."""
    if run_dir is None:
        run_dir = tempfile.mkdtemp(prefix="hotcb_finetune_")

    os.makedirs(run_dir, exist_ok=True)
    for fname in [
        "hotcb.commands.jsonl",
        "hotcb.applied.jsonl",
        "hotcb.metrics.jsonl",
        "hotcb.recipe.jsonl",
        "hotcb.features.jsonl",
    ]:
        path = os.path.join(run_dir, fname)
        if not os.path.exists(path):
            open(path, "w").close()

    with open(os.path.join(run_dir, "hotcb.freeze.json"), "w") as f:
        json.dump({"mode": "off"}, f)

    import sys
    w = sys.stderr.write
    w("\n")
    w("  ╔══════════════════════════════════════════════╗\n")
    w("  ║     hotcb Finetune Demo                     ║\n")
    w("  ║     Transfer Learning Control Plane          ║\n")
    w("  ╚══════════════════════════════════════════════╝\n")
    w(f"\n  Run dir:   {run_dir}\n")
    w(f"  Dashboard: http://localhost:{port}\n")
    w(f"  Training:  {max_steps} steps @ {step_delay}s/step\n\n")
    w("  This demo simulates finetuning a pretrained backbone on a\n")
    w("  small 5-class dataset (1000 samples).\n\n")
    w("  Recipe schedule:\n")
    w("    Step 200: Unfreeze backbone (frozen -> full finetuning)\n")
    w("    Step 400: Reduce LR to 1e-4 (mitigate overfitting)\n\n")
    w("  Watch for:\n")
    w("    * Loss spike at backbone unfreeze (step 200)\n")
    w("    * Grad norm jump when backbone gradients start flowing\n")
    w("    * Feature drift increasing after unfreeze\n")
    w("    * Val loss diverging from train loss after step 400\n\n")
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
    run_server(run_dir=run_dir, host=host, port=port, poll_interval=0.3)


if __name__ == "__main__":
    run_finetune_demo()
