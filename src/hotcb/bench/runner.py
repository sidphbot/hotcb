"""
Benchmark runner — executes tasks under different conditions and collects results.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..kernel import HotKernel
from ..metrics.collector import MetricsCollector
from .tasks import BenchmarkTask


@dataclass
class BenchmarkResult:
    """Outcome of a single benchmark run."""

    task_name: str
    condition: str                      # "baseline", "auto_tune", "recipe_replay"
    final_metrics: dict                 # {metric_name: value}
    total_steps: int
    steps_to_target: Optional[int]      # None if target not reached
    total_time_sec: float
    intervention_count: int
    recipe_path: Optional[str] = None   # path to generated recipe


def _evaluate_accuracy(model, val_loader, device) -> tuple[float, float]:
    """Run validation and return (val_loss, val_accuracy%)."""
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += loss_fn(out, y).item() * y.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    avg_loss = total_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


class BenchmarkRunner:
    """
    Runs a :class:`BenchmarkTask` under various conditions (baseline, hotcb
    auto-tune, recipe replay) and collects :class:`BenchmarkResult` objects.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = os.path.abspath(output_dir)
        self.results: list[BenchmarkResult] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_run_dir(self, task_name: str, condition: str) -> str:
        d = os.path.join(self.output_dir, f"{task_name}_{condition}")
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def _train_loop(
        task: BenchmarkTask,
        kernel: Optional[HotKernel],
        max_steps: int,
    ) -> tuple[dict, Optional[int], int]:
        """
        PyTorch training loop with epoch tracking, LR scheduling,
        and periodic validation support.

        Returns (final_metrics, steps_to_target, intervention_count).
        """
        device = torch.device("cpu")
        model = task.create_model()
        model.to(device)
        optimizer = task.create_optimizer(model)

        # Optional LR scheduler
        scheduler = None
        if task.create_scheduler is not None:
            scheduler = task.create_scheduler(optimizer, task.epochs)

        # Optional validation loader
        val_loader = None
        if task.create_val_dataloader is not None:
            val_loader = task.create_val_dataloader()

        metrics: dict = {}
        steps_to_target: Optional[int] = None
        intervention_count = 0
        last_loss = float("inf")
        val_loss = float("inf")
        val_accuracy = 0.0

        global_step = 0
        epoch = 0

        while global_step < max_steps:
            epoch += 1
            train_loader = task.create_dataloader()

            for x, y in train_loader:
                if global_step >= max_steps:
                    break

                global_step += 1
                x, y = x.to(device), y.to(device)

                model.train()
                out = model(x)
                loss = task.loss_fn(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                last_loss = loss_val

                env = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": loss_val,
                    "train_loss": loss_val,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "optimizer": optimizer,
                    "model": model,
                    "metrics": {
                        "loss": loss_val,
                        "train_loss": loss_val,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                    },
                }

                if kernel is not None:
                    kernel.apply(env, ["train_step_end"])

                # Check target (for val_accuracy tasks, use val_accuracy)
                if steps_to_target is None and task.target_metric is not None:
                    if task.target_metric_name == "val_accuracy":
                        if val_accuracy >= task.target_metric:
                            steps_to_target = global_step
                    else:
                        if loss_val <= task.target_metric:
                            steps_to_target = global_step

            # End-of-epoch: LR scheduler step
            if scheduler is not None:
                scheduler.step()

            # End-of-epoch: validation
            if (
                val_loader is not None
                and epoch % task.val_every_n_epochs == 0
            ):
                val_loss, val_accuracy = _evaluate_accuracy(
                    model, val_loader, device,
                )

                env_val = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": last_loss,
                    "train_loss": last_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "optimizer": optimizer,
                    "model": model,
                    "metrics": {
                        "loss": last_loss,
                        "train_loss": last_loss,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                    },
                }

                if kernel is not None:
                    kernel.apply(env_val, ["val_epoch_end"])

                # Re-check target after validation
                if steps_to_target is None and task.target_metric is not None:
                    if task.target_metric_name == "val_accuracy":
                        if val_accuracy >= task.target_metric:
                            steps_to_target = global_step
                    elif task.target_metric_name == "val_loss":
                        if val_loss <= task.target_metric:
                            steps_to_target = global_step

        metrics = {
            "loss": last_loss,
            "train_loss": last_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epochs": epoch,
        }

        # Count interventions from applied ledger
        if kernel is not None:
            try:
                applied_path = kernel.applied_path
                if os.path.exists(applied_path):
                    with open(applied_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                intervention_count += 1
            except Exception:
                pass

        return metrics, steps_to_target, intervention_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_baseline(self, task: BenchmarkTask) -> BenchmarkResult:
        """Run with fixed hyperparameters, no hotcb intervention."""
        max_steps = task.max_steps
        t0 = time.monotonic()
        metrics, steps_to_target, _ = self._train_loop(task, kernel=None, max_steps=max_steps)
        elapsed = time.monotonic() - t0

        result = BenchmarkResult(
            task_name=task.name,
            condition="baseline",
            final_metrics=metrics,
            total_steps=max_steps,
            steps_to_target=steps_to_target,
            total_time_sec=elapsed,
            intervention_count=0,
        )
        self.results.append(result)
        return result

    def run_with_hotcb(self, task: BenchmarkTask, tune_mode: str = "active") -> BenchmarkResult:
        """Run with hotcb kernel (and optional tune controller) attached."""
        run_dir = self._make_run_dir(task.name, "auto_tune")
        max_steps = task.max_steps

        collector = MetricsCollector(
            path=os.path.join(run_dir, "hotcb.metrics.jsonl"),
        )
        kernel = HotKernel(
            run_dir=run_dir,
            metrics_collector=collector,
        )

        t0 = time.monotonic()
        metrics, steps_to_target, intervention_count = self._train_loop(
            task, kernel=kernel, max_steps=max_steps,
        )
        kernel.close({"step": max_steps})
        elapsed = time.monotonic() - t0

        recipe_path = os.path.join(run_dir, "hotcb.recipe.jsonl")
        if not os.path.exists(recipe_path):
            recipe_path = None

        result = BenchmarkResult(
            task_name=task.name,
            condition="auto_tune",
            final_metrics=metrics,
            total_steps=max_steps,
            steps_to_target=steps_to_target,
            total_time_sec=elapsed,
            intervention_count=intervention_count,
            recipe_path=recipe_path,
        )
        self.results.append(result)
        return result

    def run_recipe_replay(self, task: BenchmarkTask, recipe_path: str) -> BenchmarkResult:
        """Replay a recipe deterministically."""
        run_dir = self._make_run_dir(task.name, "recipe_replay")
        max_steps = task.max_steps

        collector = MetricsCollector(
            path=os.path.join(run_dir, "hotcb.metrics.jsonl"),
        )
        kernel = HotKernel(
            run_dir=run_dir,
            metrics_collector=collector,
            recipe_path=recipe_path,
        )

        t0 = time.monotonic()
        metrics, steps_to_target, intervention_count = self._train_loop(
            task, kernel=kernel, max_steps=max_steps,
        )
        kernel.close({"step": max_steps})
        elapsed = time.monotonic() - t0

        result = BenchmarkResult(
            task_name=task.name,
            condition="recipe_replay",
            final_metrics=metrics,
            total_steps=max_steps,
            steps_to_target=steps_to_target,
            total_time_sec=elapsed,
            intervention_count=intervention_count,
            recipe_path=recipe_path,
        )
        self.results.append(result)
        return result

    def run_all_conditions(self, task: BenchmarkTask) -> list[BenchmarkResult]:
        """Run baseline + auto_tune + recipe_replay for comparison."""
        results = []
        results.append(self.run_baseline(task))
        hotcb_result = self.run_with_hotcb(task)
        results.append(hotcb_result)

        if hotcb_result.recipe_path and os.path.exists(hotcb_result.recipe_path):
            results.append(self.run_recipe_replay(task, hotcb_result.recipe_path))

        return results

    def compare(self) -> dict:
        """Compare all collected results, return summary dict."""
        by_task: dict = {}
        for r in self.results:
            by_task.setdefault(r.task_name, []).append(r)

        summary: dict = {}
        for task_name, runs in by_task.items():
            conditions = {}
            for r in runs:
                conditions[r.condition] = {
                    "final_metrics": r.final_metrics,
                    "total_steps": r.total_steps,
                    "steps_to_target": r.steps_to_target,
                    "total_time_sec": round(r.total_time_sec, 4),
                    "intervention_count": r.intervention_count,
                }
            summary[task_name] = conditions

        return summary
