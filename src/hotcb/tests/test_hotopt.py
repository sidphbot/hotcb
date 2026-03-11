"""Tests for HotOptController — spec section 19.6."""

from __future__ import annotations

import pytest

from hotcb.modules.opt import HotOptController, OptHandle
from hotcb.ops import HotOp


class MockOptimizer:
    def __init__(self, param_groups):
        self.param_groups = param_groups


def _make_param_groups(n=2, lr=1e-3, weight_decay=0.01):
    return [{"lr": lr, "weight_decay": weight_decay, "params": []} for _ in range(n)]


def _op(params=None, op="set_params", id="main"):
    return HotOp(module="opt", op=op, id=id, params=params)


# ------------------------------------------------------------------ #
# 1. Global lr update
# ------------------------------------------------------------------ #
class TestGlobalLrUpdate:
    def test_both_groups_updated(self):
        ctrl = HotOptController(auto_disable_on_error=True)
        groups = _make_param_groups(2)
        env = {"optimizer": MockOptimizer(groups)}

        result = ctrl.apply_op(_op(params={"lr": 1e-4}), env)

        assert result.decision == "applied"
        assert groups[0]["lr"] == pytest.approx(1e-4)
        assert groups[1]["lr"] == pytest.approx(1e-4)


# ------------------------------------------------------------------ #
# 2. Group-specific lr
# ------------------------------------------------------------------ #
class TestGroupSpecificLr:
    def test_only_target_group_changed(self):
        ctrl = HotOptController(auto_disable_on_error=True)
        groups = _make_param_groups(2, lr=1e-3)
        env = {"optimizer": MockOptimizer(groups)}

        result = ctrl.apply_op(_op(params={"group": 1, "lr": 5e-5}), env)

        assert result.decision == "applied"
        assert groups[0]["lr"] == pytest.approx(1e-3)  # unchanged
        assert groups[1]["lr"] == pytest.approx(5e-5)


# ------------------------------------------------------------------ #
# 3. Weight decay update
# ------------------------------------------------------------------ #
class TestWeightDecayUpdate:
    def test_all_groups_updated(self):
        ctrl = HotOptController(auto_disable_on_error=True)
        groups = _make_param_groups(2, weight_decay=0.01)
        env = {"optimizer": MockOptimizer(groups)}

        result = ctrl.apply_op(_op(params={"weight_decay": 0.05}), env)

        assert result.decision == "applied"
        assert groups[0]["weight_decay"] == pytest.approx(0.05)
        assert groups[1]["weight_decay"] == pytest.approx(0.05)


# ------------------------------------------------------------------ #
# 4. scheduler_scale
# ------------------------------------------------------------------ #
class TestSchedulerScale:
    def test_lr_halved(self):
        ctrl = HotOptController(auto_disable_on_error=True)
        groups = _make_param_groups(2, lr=1e-3)
        env = {"optimizer": MockOptimizer(groups)}

        result = ctrl.apply_op(_op(params={"scheduler_scale": 0.5}), env)

        assert result.decision == "applied"
        assert groups[0]["lr"] == pytest.approx(1e-3 * 0.5)
        assert groups[1]["lr"] == pytest.approx(1e-3 * 0.5)


# ------------------------------------------------------------------ #
# 5. scheduler_drop
# ------------------------------------------------------------------ #
class TestSchedulerDrop:
    def test_lr_multiplied(self):
        ctrl = HotOptController(auto_disable_on_error=True)
        groups = _make_param_groups(2, lr=1e-3)
        env = {"optimizer": MockOptimizer(groups)}

        result = ctrl.apply_op(_op(params={"scheduler_drop": 0.1}), env)

        assert result.decision == "applied"
        assert groups[0]["lr"] == pytest.approx(1e-3 * 0.1)
        assert groups[1]["lr"] == pytest.approx(1e-3 * 0.1)


# ------------------------------------------------------------------ #
# 6. clip_norm
# ------------------------------------------------------------------ #
class TestClipNorm:
    def test_stored_in_group(self):
        ctrl = HotOptController(auto_disable_on_error=True)
        groups = _make_param_groups(2)
        env = {"optimizer": MockOptimizer(groups)}

        result = ctrl.apply_op(_op(params={"clip_norm": 2.0}), env)

        assert result.decision == "applied"
        assert groups[0]["hotcb_clip_norm"] == pytest.approx(2.0)
        assert groups[1]["hotcb_clip_norm"] == pytest.approx(2.0)


# ------------------------------------------------------------------ #
# 7. Per-group mapping
# ------------------------------------------------------------------ #
class TestPerGroupMapping:
    def test_each_group_gets_specific_lr(self):
        ctrl = HotOptController(auto_disable_on_error=True)
        groups = _make_param_groups(2, lr=1e-3)
        env = {"optimizer": MockOptimizer(groups)}

        result = ctrl.apply_op(
            _op(params={"groups": {"0": {"lr": 1e-3}, "1": {"lr": 2e-3}}}), env
        )

        assert result.decision == "applied"
        assert groups[0]["lr"] == pytest.approx(1e-3)
        assert groups[1]["lr"] == pytest.approx(2e-3)


# ------------------------------------------------------------------ #
# 8. Missing optimizer
# ------------------------------------------------------------------ #
class TestMissingOptimizer:
    def test_failed_with_error(self):
        ctrl = HotOptController(auto_disable_on_error=True)
        env = {}  # no optimizer, no resolve_optimizer

        result = ctrl.apply_op(_op(params={"lr": 1e-4}), env)

        assert result.decision == "failed"
        assert "missing_optimizer" in result.error


# ------------------------------------------------------------------ #
# 9. resolve_optimizer callable
# ------------------------------------------------------------------ #
class TestResolveOptimizer:
    def test_resolver_returns_optimizer(self):
        ctrl = HotOptController(auto_disable_on_error=True)
        groups = _make_param_groups(2, lr=1e-3)
        opt = MockOptimizer(groups)
        env = {"resolve_optimizer": lambda: opt}

        result = ctrl.apply_op(_op(params={"lr": 1e-4}), env)

        assert result.decision == "applied"
        assert groups[0]["lr"] == pytest.approx(1e-4)
        assert groups[1]["lr"] == pytest.approx(1e-4)


# ------------------------------------------------------------------ #
# 10. Enable / disable
# ------------------------------------------------------------------ #
class TestEnableDisable:
    def test_disabled_handle_skips(self):
        ctrl = HotOptController(auto_disable_on_error=True)
        groups = _make_param_groups(2, lr=1e-3)
        env = {"optimizer": MockOptimizer(groups)}

        # Disable the handle
        ctrl.apply_op(_op(op="disable", id="main"), env)

        # set_params should be skipped
        result = ctrl.apply_op(_op(params={"lr": 1e-4}), env)
        assert result.decision == "skipped_noop"
        assert result.notes == "handle_disabled"
        assert groups[0]["lr"] == pytest.approx(1e-3)  # unchanged

        # Re-enable
        ctrl.apply_op(_op(op="enable", id="main"), env)

        # Now it should apply
        result = ctrl.apply_op(_op(params={"lr": 1e-4}), env)
        assert result.decision == "applied"
        assert groups[0]["lr"] == pytest.approx(1e-4)


# ------------------------------------------------------------------ #
# 11. Auto-disable on error
# ------------------------------------------------------------------ #
class TestAutoDisableOnError:
    def test_handle_disabled_after_error(self):
        ctrl = HotOptController(auto_disable_on_error=True)

        class BrokenOptimizer:
            @property
            def param_groups(self):
                raise RuntimeError("gpu exploded")

        env = {"optimizer": BrokenOptimizer()}

        result = ctrl.apply_op(_op(params={"lr": 1e-4}), env)

        assert result.decision == "failed"
        handle = ctrl.handles["main"]
        assert handle.enabled is False
        assert handle.last_error == "gpu exploded"


# ------------------------------------------------------------------ #
# 12. Status
# ------------------------------------------------------------------ #
class TestStatus:
    def test_status_structure(self):
        ctrl = HotOptController(auto_disable_on_error=True)
        groups = _make_param_groups(2)
        env = {"optimizer": MockOptimizer(groups)}

        ctrl.apply_op(_op(params={"lr": 1e-4}), env)

        status = ctrl.status()

        assert "main" in status
        entry = status["main"]
        assert "enabled" in entry
        assert "last_params" in entry
        assert "last_error" in entry
        assert entry["enabled"] is True
        assert entry["last_params"]["lr"] == pytest.approx(1e-4)
        assert entry["last_error"] is None
