"""Tests for hotcb.server.manifolds and hotcb.metrics.features."""
from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pytest

from hotcb.server.manifolds import ManifoldEngine, ManifoldResult, TrajectoryStats, available_methods


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _make_metric_records(n: int = 50) -> list[dict]:
    """Synthetic multi-metric records with smooth trajectories."""
    records = []
    for i in range(n):
        records.append({
            "step": i,
            "epoch": 0,
            "wall_time": 1000.0 + i,
            "metrics": {
                "loss": 1.0 * math.exp(-0.03 * i) + 0.01 * (i % 3),
                "val_loss": 1.1 * math.exp(-0.025 * i) + 0.02 * (i % 5),
                "accuracy": 0.5 + 0.4 * (1 - math.exp(-0.05 * i)),
            },
        })
    return records


class TestManifoldEngineBasic:
    def test_empty_engine_returns_empty_result(self):
        engine = ManifoldEngine()
        result = engine.compute_metric_manifold()
        assert result.points == []
        assert result.steps == []
        assert result.metric_names == []

    def test_update_metrics(self):
        engine = ManifoldEngine()
        records = _make_metric_records(10)
        engine.update_metrics(records)
        assert len(engine._metric_history) == 10

    def test_update_interventions(self):
        engine = ManifoldEngine()
        engine.update_interventions([{"step": 5}, {"step": 10}])
        assert 5 in engine._intervention_steps
        assert 10 in engine._intervention_steps

    def test_ring_buffer_trimming(self):
        engine = ManifoldEngine(max_points=20)
        engine.update_metrics(_make_metric_records(50))
        assert len(engine._metric_history) == 20

    def test_skips_records_without_metrics(self):
        engine = ManifoldEngine()
        engine.update_metrics([{"step": 1}, {"step": 2, "metrics": {}}])
        assert len(engine._metric_history) == 0


class TestManifoldPCA:
    def test_pca_produces_valid_result(self):
        engine = ManifoldEngine()
        engine.update_metrics(_make_metric_records(30))
        result = engine.compute_metric_manifold(method="pca", n_components=3)
        assert isinstance(result, ManifoldResult)
        assert len(result.points) == 30
        assert len(result.steps) == 30
        assert len(result.points[0]) == 3
        assert result.method == "pca"
        assert "loss" in result.metric_names

    def test_pca_2d(self):
        engine = ManifoldEngine()
        engine.update_metrics(_make_metric_records(20))
        result = engine.compute_metric_manifold(method="pca", n_components=2)
        assert len(result.points[0]) == 2

    def test_explained_variance_returned(self):
        engine = ManifoldEngine()
        engine.update_metrics(_make_metric_records(30))
        result = engine.compute_metric_manifold(method="pca", n_components=3)
        assert result.explained_variance is not None
        assert len(result.explained_variance) == 3
        assert all(0 <= v <= 1 for v in result.explained_variance)

    def test_intervention_marking(self):
        engine = ManifoldEngine()
        engine.update_metrics(_make_metric_records(20))
        engine.update_interventions([{"step": 5}, {"step": 15}])
        result = engine.compute_metric_manifold()
        assert result.is_intervention[5] is True
        assert result.is_intervention[15] is True
        assert result.is_intervention[0] is False

    def test_single_metric_works(self):
        engine = ManifoldEngine()
        records = [{"step": i, "metrics": {"loss": float(i)}} for i in range(10)]
        engine.update_metrics(records)
        result = engine.compute_metric_manifold(n_components=1)
        assert len(result.points) == 10

    def test_single_point_returns_without_crash(self):
        engine = ManifoldEngine()
        engine.update_metrics([{"step": 0, "metrics": {"loss": 1.0, "val_loss": 0.9}}])
        result = engine.compute_metric_manifold()
        assert len(result.points) == 1

    def test_forward_fill_missing_metrics(self):
        engine = ManifoldEngine()
        records = [
            {"step": 0, "metrics": {"loss": 1.0, "val_loss": 0.9}},
            {"step": 1, "metrics": {"loss": 0.8}},  # val_loss missing
            {"step": 2, "metrics": {"loss": 0.6, "val_loss": 0.7}},
        ]
        engine.update_metrics(records)
        result = engine.compute_metric_manifold()
        assert len(result.points) == 3  # no rows dropped

    def test_unknown_method_falls_back_to_pca(self):
        engine = ManifoldEngine()
        engine.update_metrics(_make_metric_records(20))
        result = engine.compute_metric_manifold(method="nonexistent")
        assert result.method == "pca"


class TestTrajectoryStats:
    def test_empty_returns_zeros(self):
        engine = ManifoldEngine()
        stats = engine.compute_trajectory_stats()
        assert stats.total_distance == 0.0
        assert stats.velocities == []

    def test_single_point_returns_zeros(self):
        engine = ManifoldEngine()
        engine.update_metrics([{"step": 0, "metrics": {"loss": 1.0}}])
        stats = engine.compute_trajectory_stats()
        assert stats.total_distance == 0.0

    def test_trajectory_stats_computed(self):
        engine = ManifoldEngine()
        engine.update_metrics(_make_metric_records(20))
        stats = engine.compute_trajectory_stats()
        assert isinstance(stats, TrajectoryStats)
        assert stats.total_distance > 0
        assert len(stats.velocities) == 19  # N-1
        assert stats.mean_velocity > 0

    def test_intervention_impacts(self):
        engine = ManifoldEngine()
        engine.update_metrics(_make_metric_records(20))
        engine.update_interventions([{"step": 10}])
        stats = engine.compute_trajectory_stats()
        assert len(stats.intervention_impacts) >= 1
        impact = stats.intervention_impacts[0]
        assert impact["step"] == 10
        assert "velocity_before" in impact
        assert "velocity_after" in impact


class TestAvailableMethods:
    def test_pca_always_available(self):
        methods = available_methods()
        assert "pca" in methods


class TestFeatureCapture:
    def test_register_and_capture(self):
        import torch
        import torch.nn as nn
        from hotcb.metrics.features import FeatureCapture

        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )
        # Name: "0" for first layer
        capture = FeatureCapture(every_n_steps=1, max_samples=16, pre_reduce_dim=8)
        capture.register(model, layer_names=["0"])
        assert len(capture._hooks) == 1

        # Run a forward pass
        x = torch.randn(16, 10)
        model(x)

        # Step to capture
        capture.step(1)
        snap = capture.latest_snapshot()
        assert snap is not None
        assert snap.step == 1
        assert snap.layer_name == "0"
        assert len(snap.activations) <= 16

    def test_every_n_steps_decimation(self):
        import torch
        import torch.nn as nn
        from hotcb.metrics.features import FeatureCapture

        model = nn.Linear(10, 20)
        capture = FeatureCapture(every_n_steps=3, max_samples=8)
        capture.register(model, layer_names=[""])  # empty won't resolve

        # The root module for nn.Linear is itself
        capture._hooks.clear()
        capture._layer_names.clear()

        # Manually register on the model itself
        def hook(mod, inp, out):
            capture._latest_activations["linear"] = out
        handle = model.register_forward_hook(hook)
        capture._hooks.append(handle)
        capture._layer_names.append("linear")

        for step in range(1, 10):
            model(torch.randn(8, 10))
            capture.step(step)

        # Only steps 3, 6, 9 should have captures
        assert capture.snapshot_count == 3

    def test_unregister(self):
        import torch.nn as nn
        from hotcb.metrics.features import FeatureCapture

        model = nn.Linear(5, 5)
        capture = FeatureCapture()
        capture.register(model, layer_names=[])
        capture.unregister()
        assert len(capture._hooks) == 0

    def test_persistence(self, tmp_dir):
        import torch
        import torch.nn as nn
        from hotcb.metrics.features import FeatureCapture

        path = os.path.join(tmp_dir, "features.jsonl")
        model = nn.Linear(10, 20)
        capture = FeatureCapture(every_n_steps=1, output_path=path, max_samples=4)

        def hook(mod, inp, out):
            capture._latest_activations["layer"] = out
        handle = model.register_forward_hook(hook)
        capture._hooks.append(handle)
        capture._layer_names.append("layer")

        model(torch.randn(4, 10))
        capture.step(1)

        assert os.path.exists(path)
        with open(path) as f:
            rec = json.loads(f.readline())
        assert rec["step"] == 1
        assert rec["layer_name"] == "layer"

    def test_missing_layer_skipped(self):
        import torch.nn as nn
        from hotcb.metrics.features import FeatureCapture

        model = nn.Linear(5, 5)
        capture = FeatureCapture()
        capture.register(model, layer_names=["nonexistent.layer"])
        assert len(capture._hooks) == 0


class TestManifoldRESTEndpoints:
    @pytest.fixture
    def client(self, tmp_dir):
        from hotcb.server.app import create_app
        from starlette.testclient import TestClient
        from hotcb.util import append_jsonl

        # Write some metrics
        metrics_path = os.path.join(tmp_dir, "hotcb.metrics.jsonl")
        for rec in _make_metric_records(30):
            append_jsonl(metrics_path, rec)
        open(os.path.join(tmp_dir, "hotcb.commands.jsonl"), "w").close()

        app = create_app(tmp_dir, poll_interval=60)
        # Manually seed the manifold engine
        app.state.manifold_engine.update_metrics(_make_metric_records(30))
        return TestClient(app)

    def test_metric_manifold_endpoint(self, client):
        r = client.get("/api/manifolds/metric?method=pca&n_components=2")
        assert r.status_code == 200
        data = r.json()
        assert len(data["points"]) == 30
        assert data["method"] == "pca"

    def test_trajectory_endpoint(self, client):
        r = client.get("/api/manifolds/trajectory")
        assert r.status_code == 200
        data = r.json()
        assert data["total_distance"] > 0
        assert len(data["velocities"]) == 29

    def test_available_methods_endpoint(self, client):
        r = client.get("/api/manifolds/available-methods")
        assert r.status_code == 200
        assert "pca" in r.json()["methods"]
