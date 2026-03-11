"""Tests for hotcb.bench — Benchmarking module."""
from __future__ import annotations

import csv
import json
import os
import tempfile

import pytest
import torch

from hotcb.bench.tasks import BenchmarkTask, BUILTIN_TASKS, _make_quadratic_data, _make_classification_data
from hotcb.bench.runner import BenchmarkRunner, BenchmarkResult
from hotcb.bench.report import BenchmarkReport


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestBenchmarkTasks:
    def test_builtin_tasks_exist(self):
        assert "synthetic_quadratic" in BUILTIN_TASKS
        assert "synthetic_classification" in BUILTIN_TASKS

    def test_quadratic_task_creates_valid_components(self):
        task = BUILTIN_TASKS["synthetic_quadratic"]()
        model = task.create_model()
        assert isinstance(model, torch.nn.Module)
        optimizer = task.create_optimizer(model)
        assert isinstance(optimizer, torch.optim.Optimizer)
        loader = task.create_dataloader()
        batch = next(iter(loader))
        x, y = batch
        assert x.shape[1] == 1
        assert y.shape[1] == 1

    def test_classification_task_creates_valid_components(self):
        task = BUILTIN_TASKS["synthetic_classification"]()
        model = task.create_model()
        optimizer = task.create_optimizer(model)
        loader = task.create_dataloader()
        x, y = next(iter(loader))
        assert x.shape[1] == 20
        assert y.dtype == torch.long

    def test_quadratic_data_generation(self):
        batches = _make_quadratic_data(n_samples=64, batch_size=16)
        assert len(batches) == 4
        assert batches[0][0].shape == (16, 1)

    def test_classification_data_generation(self):
        batches = _make_classification_data(n_samples=64, n_features=10, n_classes=2, batch_size=16)
        assert len(batches) == 4

    def test_quadratic_forward_pass(self):
        task = BUILTIN_TASKS["synthetic_quadratic"](max_steps=5)
        model = task.create_model()
        x, y = next(iter(task.create_dataloader()))
        out = model(x)
        loss = task.loss_fn(out, y)
        assert loss.item() > 0


class TestBenchmarkRunner:
    def test_run_baseline(self, tmp_dir):
        task = BUILTIN_TASKS["synthetic_quadratic"](max_steps=20)
        runner = BenchmarkRunner(output_dir=tmp_dir)
        result = runner.run_baseline(task)
        assert isinstance(result, BenchmarkResult)
        assert result.task_name == "synthetic_quadratic"
        assert result.condition == "baseline"
        assert result.total_steps == 20
        assert result.total_time_sec > 0
        assert result.intervention_count == 0
        assert "loss" in result.final_metrics

    def test_run_with_hotcb(self, tmp_dir):
        task = BUILTIN_TASKS["synthetic_quadratic"](max_steps=20)
        runner = BenchmarkRunner(output_dir=tmp_dir)
        result = runner.run_with_hotcb(task)
        assert result.condition == "auto_tune"
        assert result.total_steps == 20
        assert "loss" in result.final_metrics
        # Metrics JSONL should exist
        metrics_path = os.path.join(
            tmp_dir, "synthetic_quadratic_auto_tune", "hotcb.metrics.jsonl"
        )
        assert os.path.exists(metrics_path)

    def test_run_baseline_classification(self, tmp_dir):
        task = BUILTIN_TASKS["synthetic_classification"](max_steps=15)
        runner = BenchmarkRunner(output_dir=tmp_dir)
        result = runner.run_baseline(task)
        assert result.total_steps == 15

    def test_results_accumulate(self, tmp_dir):
        task = BUILTIN_TASKS["synthetic_quadratic"](max_steps=10)
        runner = BenchmarkRunner(output_dir=tmp_dir)
        runner.run_baseline(task)
        runner.run_with_hotcb(task)
        assert len(runner.results) == 2

    def test_compare(self, tmp_dir):
        task = BUILTIN_TASKS["synthetic_quadratic"](max_steps=10)
        runner = BenchmarkRunner(output_dir=tmp_dir)
        runner.run_baseline(task)
        runner.run_with_hotcb(task)
        summary = runner.compare()
        assert "synthetic_quadratic" in summary
        assert "baseline" in summary["synthetic_quadratic"]
        assert "auto_tune" in summary["synthetic_quadratic"]

    def test_steps_to_target(self, tmp_dir):
        # With very high target, should not be reached in 10 steps
        task = BUILTIN_TASKS["synthetic_quadratic"](max_steps=10)
        task.target_metric = 0.0001  # very ambitious
        runner = BenchmarkRunner(output_dir=tmp_dir)
        result = runner.run_baseline(task)
        # steps_to_target may or may not be None depending on convergence
        assert isinstance(result.steps_to_target, (int, type(None)))


class TestBenchmarkReport:
    def _make_results(self) -> list[BenchmarkResult]:
        return [
            BenchmarkResult(
                task_name="task_a", condition="baseline",
                final_metrics={"loss": 0.5}, total_steps=100,
                steps_to_target=50, total_time_sec=1.234,
                intervention_count=0,
            ),
            BenchmarkResult(
                task_name="task_a", condition="auto_tune",
                final_metrics={"loss": 0.3}, total_steps=100,
                steps_to_target=30, total_time_sec=2.345,
                intervention_count=5, recipe_path="/tmp/recipe.jsonl",
            ),
        ]

    def test_to_dict(self):
        report = BenchmarkReport(self._make_results())
        d = report.to_dict()
        assert "results" in d
        assert len(d["results"]) == 2
        assert d["results"][0]["task_name"] == "task_a"

    def test_to_json(self, tmp_dir):
        report = BenchmarkReport(self._make_results())
        path = os.path.join(tmp_dir, "report.json")
        report.to_json(path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data["results"]) == 2

    def test_to_csv(self, tmp_dir):
        report = BenchmarkReport(self._make_results())
        path = os.path.join(tmp_dir, "report.csv")
        report.to_csv(path)
        assert os.path.exists(path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["task_name"] == "task_a"
        assert rows[0]["condition"] == "baseline"

    def test_summary_table(self):
        report = BenchmarkReport(self._make_results())
        table = report.summary_table()
        assert isinstance(table, str)
        assert "task_a" in table
        assert "baseline" in table
        assert "auto_tune" in table
        # Should have header + separator + 2 data rows
        lines = table.strip().split("\n")
        assert len(lines) == 4

    def test_empty_report(self):
        report = BenchmarkReport([])
        assert report.to_dict() == {"results": []}
        table = report.summary_table()
        # Just header + separator
        lines = table.strip().split("\n")
        assert len(lines) == 2

    def test_to_latex(self, tmp_dir):
        report = BenchmarkReport(self._make_results())
        path = os.path.join(tmp_dir, "report.tex")
        report.to_latex(path)
        assert os.path.exists(path)
        with open(path) as f:
            tex = f.read()
        assert r"\begin{table}" in tex
        assert r"\toprule" in tex
        assert r"\bottomrule" in tex
        assert "task\_a" in tex  # underscores escaped
        assert "baseline" in tex
        assert "auto\_tune" in tex
        assert "0.500000" in tex  # loss value

    def test_to_latex_empty(self, tmp_dir):
        report = BenchmarkReport([])
        path = os.path.join(tmp_dir, "empty.tex")
        report.to_latex(path)
        with open(path) as f:
            tex = f.read()
        assert r"\begin{table}" in tex
        assert r"\end{table}" in tex

    def test_save_figures(self, tmp_dir):
        matplotlib = pytest.importorskip("matplotlib")
        report = BenchmarkReport(self._make_results())
        fig_dir = os.path.join(tmp_dir, "figs")
        paths = report.save_figures(fig_dir)
        assert len(paths) == 1
        assert os.path.exists(paths[0])
        assert paths[0].endswith(".png")

    def test_save_figures_no_matplotlib(self, tmp_dir, monkeypatch):
        """save_figures returns empty list when matplotlib missing."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        report = BenchmarkReport(self._make_results())
        paths = report.save_figures(os.path.join(tmp_dir, "figs2"))
        assert paths == []
