"""
Autopilot evaluation — replicate a published benchmark, then try to beat it.

Usage::

    from hotcb.bench.eval_autopilot import AutopilotEval

    eval = AutopilotEval(output_dir="./eval_output")

    # Phase 1: Replicate published result with standard config
    baseline = eval.run_published_baseline("cifar10_resnet20")

    # Phase 2: Run with autopilot trying to beat it
    autopilot = eval.run_autopilot_challenge("cifar10_resnet20")

    # Phase 3: Compare
    print(eval.report())
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from .runner import BenchmarkResult, BenchmarkRunner
from .tasks import BUILTIN_TASKS, BenchmarkTask
from ..server.autopilot import AutopilotEngine, AutopilotRule


# Default community guidelines for autopilot challenge
_DEFAULT_RULES = [
    AutopilotRule(
        rule_id="plateau_lr_reduce",
        condition="plateau",
        metric_name="val_loss",
        params={"window": 5, "epsilon": 0.005, "cooldown": 10},
        action={"module": "opt", "op": "set_params", "id": "main",
                "params": {"lr": "__current__ * 0.5"}},
        confidence="high",
        description="Halve LR when val_loss plateaus for 5 evaluations.",
    ),
    AutopilotRule(
        rule_id="divergence_lr_emergency",
        condition="divergence",
        metric_name="val_loss",
        params={"window": 3, "threshold": 1.0, "cooldown": 5},
        action={"module": "opt", "op": "set_params", "id": "main",
                "params": {"lr": "__current__ * 0.1"}},
        confidence="high",
        description="Emergency LR cut (10x) when val_loss spikes.",
    ),
    AutopilotRule(
        rule_id="overfitting_wd_increase",
        condition="overfitting",
        metric_name="val_loss",
        params={"ratio_threshold": 0.5, "cooldown": 20},
        action={"module": "opt", "op": "set_params", "id": "main",
                "params": {"weight_decay": "__current__ * 2.0"}},
        confidence="medium",
        description="Double weight_decay when train/val ratio indicates overfitting.",
    ),
]


class AutopilotEval:
    """Evaluate hotcb autopilot against published benchmarks."""

    def __init__(self, output_dir: str = "./eval_output") -> None:
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self._baseline_result: Optional[BenchmarkResult] = None
        self._autopilot_result: Optional[BenchmarkResult] = None
        self._autopilot_actions: list = []

    def _get_task(self, task_name: str) -> BenchmarkTask:
        factory = BUILTIN_TASKS.get(task_name)
        if factory is None:
            raise ValueError(
                f"Unknown task: {task_name!r}. "
                f"Available: {list(BUILTIN_TASKS)}"
            )
        return factory()

    # -----------------------------------------------------------------
    # Phase 1: Published baseline
    # -----------------------------------------------------------------

    def run_published_baseline(self, task_name: str) -> BenchmarkResult:
        """Run with exact published config (no hotcb intervention).

        This replicates the published benchmark for verification.
        """
        task = self._get_task(task_name)
        runner = BenchmarkRunner(
            output_dir=os.path.join(self.output_dir, "baseline"),
        )
        result = runner.run_baseline(task)
        self._baseline_result = result
        self._save_result("baseline", result)
        return result

    # -----------------------------------------------------------------
    # Phase 2: Autopilot challenge
    # -----------------------------------------------------------------

    def run_autopilot_challenge(
        self,
        task_name: str,
        guidelines_path: str | None = None,
    ) -> BenchmarkResult:
        """Run with hotcb autopilot enabled, starting from same config.

        The autopilot uses community guidelines to make live adjustments:
        - Plateau detection -> LR reduction
        - Divergence detection -> emergency LR cut
        - Overfitting detection -> WD increase

        If *guidelines_path* is provided, rules are loaded from that YAML file
        instead of the built-in defaults.
        """
        task = self._get_task(task_name)
        run_dir = os.path.join(self.output_dir, "autopilot")
        os.makedirs(run_dir, exist_ok=True)

        # Set up autopilot engine
        engine = AutopilotEngine(run_dir=run_dir, mode="auto")

        if guidelines_path is not None:
            engine.load_guidelines(guidelines_path)
        else:
            for rule in _DEFAULT_RULES:
                engine.add_rule(rule)

        # Run with hotcb kernel + autopilot evaluation at val_epoch_end
        from ..kernel import HotKernel
        from ..metrics.collector import MetricsCollector

        collector = MetricsCollector(
            path=os.path.join(run_dir, "hotcb.metrics.jsonl"),
        )
        kernel = HotKernel(
            run_dir=run_dir,
            metrics_collector=collector,
        )

        max_steps = task.max_steps
        t0 = time.monotonic()

        # Use the runner's training loop but hook autopilot into it
        # We wrap the kernel.apply to also feed metrics to the autopilot
        _original_apply = kernel.apply

        all_actions = []

        def _apply_with_autopilot(env, events):
            _original_apply(env, events)
            # Feed autopilot at val_epoch_end events
            if "val_epoch_end" in events:
                m = env.get("metrics", {})
                step = env.get("step", 0)
                actions = engine.evaluate(step, m)
                all_actions.extend(actions)

        kernel.apply = _apply_with_autopilot

        metrics, steps_to_target, intervention_count = BenchmarkRunner._train_loop(
            task, kernel=kernel, max_steps=max_steps,
        )
        kernel.close({"step": max_steps})
        elapsed = time.monotonic() - t0

        intervention_count += len(all_actions)
        self._autopilot_actions = all_actions

        result = BenchmarkResult(
            task_name=task.name,
            condition="autopilot",
            final_metrics=metrics,
            total_steps=max_steps,
            steps_to_target=steps_to_target,
            total_time_sec=elapsed,
            intervention_count=intervention_count,
        )
        self._autopilot_result = result
        self._save_result("autopilot", result)
        return result

    # -----------------------------------------------------------------
    # Phase 3: Report
    # -----------------------------------------------------------------

    def report(self) -> str:
        """Return a formatted comparison table."""
        lines = []
        lines.append("=" * 72)
        lines.append("  Autopilot Evaluation Report")
        lines.append("=" * 72)

        headers = ["Metric", "Baseline", "Autopilot", "Delta"]
        col_w = [20, 16, 16, 16]

        def _row(cells):
            return "  ".join(c.ljust(w) for c, w in zip(cells, col_w))

        lines.append("")
        lines.append(_row(headers))
        lines.append(_row(["-" * w for w in col_w]))

        b = self._baseline_result
        a = self._autopilot_result

        if b is None and a is None:
            lines.append("  (no results yet)")
            lines.append("")
            return "\n".join(lines)

        def _fmt(v, precision=4):
            if v is None:
                return "N/A"
            if isinstance(v, float):
                return f"{v:.{precision}f}"
            return str(v)

        def _delta(bv, av, higher_better=False):
            if bv is None or av is None:
                return "N/A"
            if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
                d = av - bv
                sign = "+" if d >= 0 else ""
                better = (d > 0) if higher_better else (d < 0)
                marker = " *" if better else ""
                return f"{sign}{d:.4f}{marker}"
            return "N/A"

        bm = b.final_metrics if b else {}
        am = a.final_metrics if a else {}

        metric_rows = [
            ("val_loss", False),
            ("val_accuracy", True),
            ("train_loss", False),
        ]
        for name, higher_better in metric_rows:
            bv = bm.get(name)
            av = am.get(name)
            lines.append(_row([
                name,
                _fmt(bv),
                _fmt(av),
                _delta(bv, av, higher_better),
            ]))

        lines.append(_row([
            "total_time(s)",
            _fmt(b.total_time_sec if b else None, 2),
            _fmt(a.total_time_sec if a else None, 2),
            _delta(
                b.total_time_sec if b else None,
                a.total_time_sec if a else None,
            ),
        ]))
        lines.append(_row([
            "interventions",
            _fmt(b.intervention_count if b else None),
            _fmt(a.intervention_count if a else None),
            "",
        ]))
        lines.append(_row([
            "steps_to_target",
            _fmt(b.steps_to_target if b else None),
            _fmt(a.steps_to_target if a else None),
            "",
        ]))

        lines.append("")

        if self._autopilot_actions:
            lines.append(f"  Autopilot actions fired: {len(self._autopilot_actions)}")
            for act in self._autopilot_actions[:10]:
                lines.append(
                    f"    step {act.step}: [{act.status}] {act.rule_id} "
                    f"— {act.condition_met[:60]}"
                )
            if len(self._autopilot_actions) > 10:
                lines.append(
                    f"    ... and {len(self._autopilot_actions) - 10} more"
                )
            lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines)

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def _save_result(self, phase: str, result: BenchmarkResult) -> None:
        path = os.path.join(self.output_dir, f"{phase}_result.json")
        data = {
            "task_name": result.task_name,
            "condition": result.condition,
            "final_metrics": result.final_metrics,
            "total_steps": result.total_steps,
            "steps_to_target": result.steps_to_target,
            "total_time_sec": round(result.total_time_sec, 4),
            "intervention_count": result.intervention_count,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
