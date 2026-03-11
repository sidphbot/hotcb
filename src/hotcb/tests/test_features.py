"""Tests for hotcb.metrics.features — FeatureCapture, FeatureSnapshot, helpers."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from hotcb.metrics.features import (
    FeatureCapture,
    FeatureSnapshot,
    _quick_pca,
    _resolve_module,
    _to_numpy,
)


# ---------- helpers ----------

def _simple_model():
    """A small sequential model with named sub-modules."""
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )


def _nested_model():
    """Model with dot-path addressable sub-modules."""
    model = nn.Module()
    model.encoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
    model.decoder = nn.Linear(32, 4)
    return model


def _run_forward(model, batch_size=4, in_features=16):
    """Run a forward pass to trigger hooks."""
    x = torch.randn(batch_size, in_features)
    with torch.no_grad():
        model(x)


# ---------- FeatureSnapshot dataclass ----------

def test_snapshot_is_frozen():
    snap = FeatureSnapshot(step=10, layer_name="fc", activations=[[1.0, 2.0]])
    assert snap.step == 10
    assert snap.layer_name == "fc"
    assert snap.activations == [[1.0, 2.0]]
    with pytest.raises(AttributeError):
        snap.step = 20


def test_snapshot_equality():
    a = FeatureSnapshot(step=1, layer_name="x", activations=[[0.0]])
    b = FeatureSnapshot(step=1, layer_name="x", activations=[[0.0]])
    assert a == b


# ---------- FeatureCapture init ----------

def test_default_init():
    fc = FeatureCapture()
    assert fc.every_n_steps == 50
    assert fc.max_samples == 256
    assert fc.pre_reduce_dim == 64
    assert fc.ring_size == 100
    assert fc.output_path is None
    assert fc.snapshot_count == 0


def test_custom_init():
    fc = FeatureCapture(every_n_steps=10, max_samples=32, pre_reduce_dim=8,
                        ring_size=5, output_path="/tmp/test.jsonl")
    assert fc.every_n_steps == 10
    assert fc.max_samples == 32
    assert fc.pre_reduce_dim == 8
    assert fc.ring_size == 5
    assert fc.output_path == "/tmp/test.jsonl"


def test_every_n_steps_clamped_to_one():
    fc = FeatureCapture(every_n_steps=0)
    assert fc.every_n_steps == 1
    fc2 = FeatureCapture(every_n_steps=-5)
    assert fc2.every_n_steps == 1


# ---------- register / unregister ----------

def test_register_attaches_hooks():
    model = _simple_model()
    fc = FeatureCapture()
    fc.register(model, layer_names=["0", "2"])
    assert len(fc._hooks) == 2
    assert fc._layer_names == ["0", "2"]


def test_register_skips_missing_layer():
    model = _simple_model()
    fc = FeatureCapture()
    fc.register(model, layer_names=["nonexistent", "0"])
    assert len(fc._hooks) == 1
    assert fc._layer_names == ["0"]


def test_register_replaces_previous_hooks():
    model = _simple_model()
    fc = FeatureCapture()
    fc.register(model, layer_names=["0"])
    assert len(fc._hooks) == 1
    fc.register(model, layer_names=["0", "2"])
    assert len(fc._hooks) == 2


def test_unregister_removes_hooks():
    model = _simple_model()
    fc = FeatureCapture()
    fc.register(model, layer_names=["0", "2"])
    assert len(fc._hooks) == 2
    fc.unregister()
    assert len(fc._hooks) == 0
    assert len(fc._layer_names) == 0
    assert len(fc._latest_activations) == 0


def test_hooks_capture_activations_on_forward():
    model = _simple_model()
    fc = FeatureCapture()
    fc.register(model, layer_names=["0"])
    _run_forward(model)
    assert "0" in fc._latest_activations
    act = fc._latest_activations["0"]
    assert act.shape == (4, 32)


# ---------- step — capture and skip ----------

def test_step_captures_at_correct_interval():
    model = _simple_model()
    fc = FeatureCapture(every_n_steps=10, pre_reduce_dim=8)
    fc.register(model, layer_names=["0"])

    _run_forward(model)
    fc.step(10)
    assert fc.snapshot_count == 1

    _run_forward(model)
    fc.step(20)
    assert fc.snapshot_count == 2


def test_step_skips_non_capture_steps():
    model = _simple_model()
    fc = FeatureCapture(every_n_steps=10, pre_reduce_dim=8)
    fc.register(model, layer_names=["0"])

    _run_forward(model)
    for s in [1, 2, 3, 5, 7, 9, 11]:
        fc.step(s)
    assert fc.snapshot_count == 0


def test_step_zero_is_capture_step():
    model = _simple_model()
    fc = FeatureCapture(every_n_steps=5, pre_reduce_dim=8)
    fc.register(model, layer_names=["0"])
    _run_forward(model)
    fc.step(0)
    assert fc.snapshot_count == 1


# ---------- PCA reduction ----------

def test_pca_reduction_when_features_exceed_pre_reduce_dim():
    model = _simple_model()
    # layer "0" outputs 32 features, pre_reduce_dim=4 should trigger PCA
    fc = FeatureCapture(every_n_steps=1, pre_reduce_dim=4, max_samples=256)
    fc.register(model, layer_names=["0"])

    _run_forward(model, batch_size=8)
    fc.step(1)

    snap = fc.latest_snapshot()
    assert snap is not None
    # activations should be (8, 4) after PCA
    assert len(snap.activations) == 8
    assert len(snap.activations[0]) == 4


def test_no_pca_when_features_below_pre_reduce_dim():
    model = _simple_model()
    # layer "2" outputs 8 features, pre_reduce_dim=64 -> no PCA
    fc = FeatureCapture(every_n_steps=1, pre_reduce_dim=64, max_samples=256)
    fc.register(model, layer_names=["2"])

    _run_forward(model, batch_size=4)
    fc.step(1)

    snap = fc.latest_snapshot()
    assert snap is not None
    assert len(snap.activations) == 4
    assert len(snap.activations[0]) == 8


# ---------- max_samples truncation ----------

def test_max_samples_truncation():
    model = _simple_model()
    fc = FeatureCapture(every_n_steps=1, max_samples=3, pre_reduce_dim=64)
    fc.register(model, layer_names=["2"])

    _run_forward(model, batch_size=10)
    fc.step(1)

    snap = fc.latest_snapshot()
    assert snap is not None
    assert len(snap.activations) == 3


# ---------- ring buffer ----------

def test_ring_buffer_overflow():
    model = _simple_model()
    fc = FeatureCapture(every_n_steps=1, ring_size=3, pre_reduce_dim=64)
    fc.register(model, layer_names=["2"])

    for s in range(1, 6):
        _run_forward(model, batch_size=4)
        fc.step(s)

    assert fc.snapshot_count == 3
    snaps = fc.all_snapshots()
    assert [s.step for s in snaps] == [3, 4, 5]


# ---------- latest_snapshot / all_snapshots ----------

def test_latest_snapshot_empty():
    fc = FeatureCapture()
    assert fc.latest_snapshot() is None


def test_latest_snapshot_returns_most_recent():
    model = _simple_model()
    fc = FeatureCapture(every_n_steps=1, pre_reduce_dim=64)
    fc.register(model, layer_names=["2"])

    _run_forward(model)
    fc.step(1)
    _run_forward(model)
    fc.step(2)

    snap = fc.latest_snapshot()
    assert snap is not None
    assert snap.step == 2


def test_all_snapshots_returns_list():
    model = _simple_model()
    fc = FeatureCapture(every_n_steps=1, pre_reduce_dim=64)
    fc.register(model, layer_names=["2"])

    for s in range(1, 4):
        _run_forward(model)
        fc.step(s)

    snaps = fc.all_snapshots()
    assert isinstance(snaps, list)
    assert len(snaps) == 3
    assert [s.step for s in snaps] == [1, 2, 3]


# ---------- JSONL persistence ----------

def test_jsonl_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "features.jsonl")
        model = _simple_model()
        fc = FeatureCapture(every_n_steps=1, pre_reduce_dim=64, output_path=path)
        fc.register(model, layer_names=["2"])

        for s in range(1, 4):
            _run_forward(model)
            fc.step(s)

        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3

        for i, line in enumerate(lines, start=1):
            record = json.loads(line)
            assert record["step"] == i
            assert record["layer_name"] == "2"
            assert isinstance(record["activations"], list)


def test_jsonl_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sub", "dir", "features.jsonl")
        model = _simple_model()
        fc = FeatureCapture(every_n_steps=1, pre_reduce_dim=64, output_path=path)
        fc.register(model, layer_names=["2"])

        _run_forward(model)
        fc.step(1)
        assert os.path.exists(path)


# ---------- _resolve_module ----------

def test_resolve_module_simple():
    model = _simple_model()
    mod = _resolve_module(model, "0")
    assert mod is model[0]


def test_resolve_module_nested():
    model = _nested_model()
    mod = _resolve_module(model, "encoder.0")
    assert isinstance(mod, nn.Linear)


def test_resolve_module_returns_none_for_missing():
    model = _simple_model()
    assert _resolve_module(model, "does.not.exist") is None


def test_resolve_module_partial_path_missing():
    model = _nested_model()
    assert _resolve_module(model, "encoder.99") is None


# ---------- _to_numpy ----------

def test_to_numpy_from_tensor():
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    arr = _to_numpy(t)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64
    np.testing.assert_array_almost_equal(arr, [[1.0, 2.0], [3.0, 4.0]])


def test_to_numpy_from_ndarray():
    a = np.array([[5.0, 6.0]], dtype=np.float32)
    arr = _to_numpy(a)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64
    np.testing.assert_array_almost_equal(arr, [[5.0, 6.0]])


def test_to_numpy_returns_none_for_non_convertible():
    assert _to_numpy("not a tensor") is None
    assert _to_numpy(42) is None


def test_to_numpy_from_grad_tensor():
    t = torch.tensor([1.0, 2.0], requires_grad=True)
    arr = _to_numpy(t)
    assert arr is not None
    assert arr.dtype == np.float64


# ---------- _quick_pca ----------

def test_quick_pca_reduces_dimensions():
    rng = np.random.RandomState(42)
    X = rng.randn(20, 10)
    reduced = _quick_pca(X, 3)
    assert reduced.shape == (20, 3)


def test_quick_pca_preserves_sample_count():
    rng = np.random.RandomState(0)
    X = rng.randn(5, 50)
    reduced = _quick_pca(X, 2)
    assert reduced.shape == (5, 2)


def test_quick_pca_zero_mean():
    """After PCA, the projected data should be centered (zero mean)."""
    rng = np.random.RandomState(7)
    X = rng.randn(30, 10) + 5.0  # offset
    reduced = _quick_pca(X, 4)
    col_means = reduced.mean(axis=0)
    np.testing.assert_array_almost_equal(col_means, np.zeros(4), decimal=10)


def test_quick_pca_variance_ordering():
    """First PCA component should capture >= variance of second component."""
    rng = np.random.RandomState(123)
    X = rng.randn(50, 8)
    reduced = _quick_pca(X, 4)
    variances = np.var(reduced, axis=0)
    # Variance should be non-increasing
    for i in range(len(variances) - 1):
        assert variances[i] >= variances[i + 1] - 1e-10


def test_quick_pca_single_feature_dim():
    """Edge case: cov is scalar, reshaped to (1,1)."""
    X = np.array([[1.0], [2.0], [3.0]])
    reduced = _quick_pca(X, 1)
    assert reduced.shape == (3, 1)


# ---------- multi-layer capture ----------

def test_multi_layer_capture():
    model = _simple_model()
    fc = FeatureCapture(every_n_steps=1, pre_reduce_dim=64)
    fc.register(model, layer_names=["0", "2"])

    _run_forward(model)
    fc.step(1)

    # Should capture one snapshot per layer
    snaps = fc.all_snapshots()
    assert len(snaps) == 2
    layer_names = {s.layer_name for s in snaps}
    assert layer_names == {"0", "2"}


# ---------- step with no activations ----------

def test_step_no_forward_pass_no_crash():
    """step() without a prior forward pass should not crash, just skip."""
    model = _simple_model()
    fc = FeatureCapture(every_n_steps=1, pre_reduce_dim=64)
    fc.register(model, layer_names=["0"])
    # No forward pass
    fc.step(1)
    assert fc.snapshot_count == 0
