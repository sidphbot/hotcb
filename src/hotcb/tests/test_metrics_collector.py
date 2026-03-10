"""Tests for hotcb.metrics.collector — MetricsCollector."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from hotcb.metrics.collector import MetricsCollector, MetricSnapshot


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def metrics_path(tmp_dir):
    return os.path.join(tmp_dir, "hotcb.metrics.jsonl")


class TestMetricsCollectorBasic:
    """Core collect/persist/ring-buffer functionality."""

    def test_collect_from_env_dict(self, metrics_path):
        mc = MetricsCollector(metrics_path)
        env = {"step": 1, "epoch": 0, "loss": 0.5, "val_loss": 0.3}
        snap = mc.collect(env)
        assert snap is not None
        assert snap.step == 1
        assert snap.epoch == 0
        assert snap.metrics["loss"] == 0.5
        assert snap.metrics["val_loss"] == 0.3

    def test_collect_from_metrics_subdict(self, metrics_path):
        mc = MetricsCollector(metrics_path)
        env = {"step": 1, "metrics": {"train_loss": 0.4, "lr": 1e-3}}
        snap = mc.collect(env)
        assert snap is not None
        assert snap.metrics["train_loss"] == 0.4
        assert snap.metrics["lr"] == pytest.approx(1e-3)

    def test_collect_from_metric_callable(self, metrics_path):
        mc = MetricsCollector(metrics_path, extra_metric_names=["custom_metric"])
        values = {"loss": 0.5, "custom_metric": 0.9}
        env = {"step": 1, "metric": lambda name, default=None: values.get(name, default)}
        snap = mc.collect(env)
        assert snap is not None
        assert snap.metrics["loss"] == 0.5
        assert snap.metrics["custom_metric"] == 0.9

    def test_persist_to_jsonl(self, metrics_path):
        mc = MetricsCollector(metrics_path)
        mc.collect({"step": 1, "loss": 0.5})
        mc.collect({"step": 2, "loss": 0.4})

        assert os.path.exists(metrics_path)
        with open(metrics_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 2
        assert lines[0]["step"] == 1
        assert lines[0]["metrics"]["loss"] == 0.5
        assert lines[1]["step"] == 2

    def test_ring_buffer(self, metrics_path):
        mc = MetricsCollector(metrics_path, ring_size=3)
        for i in range(5):
            mc.collect({"step": i, "loss": float(i)})
        assert len(mc.recent) == 3
        assert mc.recent[0].step == 2
        assert mc.recent[-1].step == 4

    def test_tail(self, metrics_path):
        mc = MetricsCollector(metrics_path)
        for i in range(10):
            mc.collect({"step": i, "loss": float(i)})
        tail = mc.tail(last_n=3)
        assert len(tail) == 3
        assert tail[0].step == 7

    def test_discovered_names(self, metrics_path):
        mc = MetricsCollector(metrics_path)
        mc.collect({"step": 1, "loss": 0.5})
        mc.collect({"step": 2, "metrics": {"val_loss": 0.3}})
        assert "loss" in mc.discovered_names
        assert "val_loss" in mc.discovered_names


class TestMetricsCollectorFiltering:
    """Whitelist / blacklist / decimation."""

    def test_whitelist(self, metrics_path):
        mc = MetricsCollector(metrics_path, whitelist={"loss"})
        env = {"step": 1, "loss": 0.5, "val_loss": 0.3}
        snap = mc.collect(env)
        assert "loss" in snap.metrics
        assert "val_loss" not in snap.metrics

    def test_blacklist(self, metrics_path):
        mc = MetricsCollector(metrics_path, blacklist={"val_loss"})
        env = {"step": 1, "loss": 0.5, "val_loss": 0.3}
        snap = mc.collect(env)
        assert "loss" in snap.metrics
        assert "val_loss" not in snap.metrics

    def test_every_n_steps(self, metrics_path):
        mc = MetricsCollector(metrics_path, every_n_steps=3)
        results = []
        for i in range(9):
            r = mc.collect({"step": i, "loss": float(i)})
            results.append(r)
        # Steps collected: step_counter 3, 6, 9 → idx 2, 5, 8
        collected = [r for r in results if r is not None]
        assert len(collected) == 3

    def test_empty_env_no_snapshot(self, metrics_path):
        mc = MetricsCollector(metrics_path)
        snap = mc.collect({"step": 1})
        assert snap is None  # no metrics found


class TestMetricsCollectorEdgeCases:
    """Edge cases and robustness."""

    def test_tensor_like_values(self, metrics_path):
        """Simulate torch.Tensor with .item() method."""
        class FakeTensor:
            def __init__(self, val):
                self._val = val
            def item(self):
                return self._val

        mc = MetricsCollector(metrics_path)
        env = {"step": 1, "loss": FakeTensor(0.42)}
        snap = mc.collect(env)
        assert snap is not None
        assert snap.metrics["loss"] == pytest.approx(0.42)

    def test_non_numeric_values_skipped(self, metrics_path):
        mc = MetricsCollector(metrics_path)
        env = {"step": 1, "metrics": {"name": "experiment1", "loss": 0.5}}
        snap = mc.collect(env)
        assert "name" not in snap.metrics
        assert snap.metrics["loss"] == 0.5

    def test_missing_path_directory_created(self, tmp_dir):
        deep_path = os.path.join(tmp_dir, "sub", "deep", "metrics.jsonl")
        mc = MetricsCollector(deep_path)
        mc.collect({"step": 1, "loss": 0.5})
        assert os.path.exists(deep_path)

    def test_snapshot_frozen(self, metrics_path):
        mc = MetricsCollector(metrics_path)
        snap = mc.collect({"step": 1, "loss": 0.5})
        with pytest.raises(AttributeError):
            snap.step = 99  # frozen dataclass


class TestMetricsCollectorKernelIntegration:
    """Test MetricsCollector integration with HotKernel."""

    def test_kernel_with_collector(self, tmp_dir):
        from hotcb.kernel import HotKernel

        metrics_path = os.path.join(tmp_dir, "hotcb.metrics.jsonl")
        mc = MetricsCollector(metrics_path)
        kernel = HotKernel(run_dir=tmp_dir, metrics_collector=mc)

        # Simulate training steps
        for i in range(5):
            env = {"step": i, "epoch": 0, "loss": 1.0 - i * 0.1}
            kernel.apply(env, events=["train_batch_end"])

        # Verify metrics were collected
        assert len(mc.recent) > 0
        assert os.path.exists(metrics_path)

    def test_kernel_without_collector(self, tmp_dir):
        """Kernel works fine without a collector (zero overhead path)."""
        from hotcb.kernel import HotKernel

        kernel = HotKernel(run_dir=tmp_dir)
        env = {"step": 1, "epoch": 0, "loss": 0.5}
        kernel.apply(env, events=["train_batch_end"])
        # No crash, no metrics file
        assert not os.path.exists(os.path.join(tmp_dir, "hotcb.metrics.jsonl"))

    def test_collector_failure_does_not_crash_kernel(self, tmp_dir):
        """If collector raises, training continues."""
        from hotcb.kernel import HotKernel

        class BrokenCollector:
            def collect(self, env):
                raise RuntimeError("boom")

        kernel = HotKernel(run_dir=tmp_dir, metrics_collector=BrokenCollector())
        env = {"step": 1, "epoch": 0, "loss": 0.5}
        # Should not raise
        kernel.apply(env, events=["train_batch_end"])
