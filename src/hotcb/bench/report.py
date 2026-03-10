"""
Benchmark report generation — CSV, JSON and text table output.
"""
from __future__ import annotations

import csv
import json
import io
import os
from typing import Optional

from .runner import BenchmarkResult


class BenchmarkReport:
    """Formats and exports a list of :class:`BenchmarkResult` objects."""

    def __init__(self, results: list[BenchmarkResult]) -> None:
        self.results = list(results)

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return results as a structured dict suitable for JSON serialisation."""
        rows = []
        for r in self.results:
            rows.append({
                "task_name": r.task_name,
                "condition": r.condition,
                "final_metrics": r.final_metrics,
                "total_steps": r.total_steps,
                "steps_to_target": r.steps_to_target,
                "total_time_sec": round(r.total_time_sec, 4),
                "intervention_count": r.intervention_count,
                "recipe_path": r.recipe_path,
            })
        return {"results": rows}

    def to_json(self, path: str) -> None:
        """Export results as JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    _CSV_FIELDS = [
        "task_name", "condition", "total_steps", "steps_to_target",
        "total_time_sec", "intervention_count", "final_loss", "recipe_path",
    ]

    def to_csv(self, path: str) -> None:
        """Export results as CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDS)
            writer.writeheader()
            for r in self.results:
                writer.writerow({
                    "task_name": r.task_name,
                    "condition": r.condition,
                    "total_steps": r.total_steps,
                    "steps_to_target": r.steps_to_target,
                    "total_time_sec": round(r.total_time_sec, 4),
                    "intervention_count": r.intervention_count,
                    "final_loss": r.final_metrics.get("loss", ""),
                    "recipe_path": r.recipe_path or "",
                })

    # ------------------------------------------------------------------
    # Text table
    # ------------------------------------------------------------------

    def summary_table(self) -> str:
        """Return a formatted text table for terminal display."""
        headers = ["Task", "Condition", "Steps", "ToTarget", "Time(s)", "Intervs", "Loss"]
        rows = []
        for r in self.results:
            loss_val = r.final_metrics.get("loss")
            loss_str = f"{loss_val:.6f}" if loss_val is not None else "N/A"
            rows.append([
                r.task_name,
                r.condition,
                str(r.total_steps),
                str(r.steps_to_target) if r.steps_to_target is not None else "-",
                f"{r.total_time_sec:.3f}",
                str(r.intervention_count),
                loss_str,
            ])

        # compute column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        def _fmt_row(cells: list[str]) -> str:
            parts = []
            for cell, w in zip(cells, col_widths):
                parts.append(cell.ljust(w))
            return "  ".join(parts)

        lines = [_fmt_row(headers)]
        lines.append("  ".join("-" * w for w in col_widths))
        for row in rows:
            lines.append(_fmt_row(row))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LaTeX
    # ------------------------------------------------------------------

    def to_latex(self, path: str) -> None:
        """Export results as a LaTeX table suitable for papers."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

        headers = ["Task", "Condition", "Steps", "To Target", "Time (s)",
                    "Interventions", "Loss"]
        col_spec = "l l r r r r r"

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Benchmark Results}",
            r"\label{tab:bench_results}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            " & ".join(rf"\textbf{{{h}}}" for h in headers) + r" \\",
            r"\midrule",
        ]

        for r in self.results:
            loss_val = r.final_metrics.get("loss")
            loss_str = f"{loss_val:.6f}" if loss_val is not None else "--"
            target_str = str(r.steps_to_target) if r.steps_to_target is not None else "--"
            task_esc = r.task_name.replace("_", r"\_")
            cond_esc = r.condition.replace("_", r"\_")
            cells = [
                task_esc,
                cond_esc,
                str(r.total_steps),
                target_str,
                f"{r.total_time_sec:.2f}",
                str(r.intervention_count),
                loss_str,
            ]
            lines.append(" & ".join(cells) + r" \\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # Matplotlib figures
    # ------------------------------------------------------------------

    def save_figures(self, output_dir: str) -> list[str]:
        """Generate matplotlib bar charts comparing conditions. Returns paths."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return []

        os.makedirs(output_dir, exist_ok=True)
        paths: list[str] = []

        # Group by task
        by_task: dict[str, list[BenchmarkResult]] = {}
        for r in self.results:
            by_task.setdefault(r.task_name, []).append(r)

        for task_name, runs in by_task.items():
            conditions = [r.condition for r in runs]
            losses = [r.final_metrics.get("loss", 0) for r in runs]
            times = [r.total_time_sec for r in runs]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle(task_name.replace("_", " ").title())

            ax1.bar(conditions, losses, color=["#3b82f6", "#10b981", "#f59e0b"][:len(runs)])
            ax1.set_ylabel("Final Loss")
            ax1.set_title("Loss by Condition")

            ax2.bar(conditions, times, color=["#3b82f6", "#10b981", "#f59e0b"][:len(runs)])
            ax2.set_ylabel("Time (s)")
            ax2.set_title("Training Time")

            plt.tight_layout()
            path = os.path.join(output_dir, f"{task_name}_comparison.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)

        return paths
