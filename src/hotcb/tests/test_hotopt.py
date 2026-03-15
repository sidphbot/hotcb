"""Tests for optimizer param mutations via kernel default stream + MutableState."""

from __future__ import annotations

import os

import pytest

from hotcb.actuators import optimizer_actuators, mutable_state, ApplyResult
from hotcb.actuators.actuator import ActuatorState, ActuatorType, HotcbActuator
from hotcb.actuators.state import MutableState
from hotcb.kernel import HotKernel
from hotcb.ops import HotOp


class MockOptimizer:
    def __init__(self, param_groups):
        self.param_groups = param_groups


def _make_param_groups(n=2, lr=1e-3, weight_decay=0.01):
    return [{"lr": lr, "weight_decay": weight_decay, "params": []} for _ in range(n)]


def _kernel_with_opt(run_dir, opt, extra_actuators=None):
    """Build a kernel with MutableState containing optimizer actuators."""
    acts = optimizer_actuators(opt)
    if extra_actuators:
        acts += extra_actuators
    ms = mutable_state(acts)
    return HotKernel(run_dir=run_dir, debounce_steps=1, mutable_state=ms)


def _op(params=None, op="set_params", id="main"):
    return HotOp(module="opt", op=op, id=id, params=params)


# ------------------------------------------------------------------ #
# 1. Global lr update
# ------------------------------------------------------------------ #
class TestGlobalLrUpdate:
    def test_both_groups_updated(self, run_dir, make_env):
        groups = _make_param_groups(2)
        opt = MockOptimizer(groups)
        kernel = _kernel_with_opt(run_dir, opt)
        env = make_env(step=1, optimizer=opt)

        kernel._apply_single(_op(params={"lr": 1e-4}), env, "train_step", 1)

        assert groups[0]["lr"] == pytest.approx(1e-4)
        assert groups[1]["lr"] == pytest.approx(1e-4)


# ------------------------------------------------------------------ #
# 2. Weight decay update
# ------------------------------------------------------------------ #
class TestWeightDecayUpdate:
    def test_all_groups_updated(self, run_dir, make_env):
        groups = _make_param_groups(2, weight_decay=0.01)
        opt = MockOptimizer(groups)
        kernel = _kernel_with_opt(run_dir, opt)
        env = make_env(step=1, optimizer=opt)

        kernel._apply_single(_op(params={"weight_decay": 0.05}), env, "train_step", 1)

        assert groups[0]["weight_decay"] == pytest.approx(0.05)
        assert groups[1]["weight_decay"] == pytest.approx(0.05)


# ------------------------------------------------------------------ #
# 3. Scheduler coordination
# ------------------------------------------------------------------ #
class TestSchedulerCoordination:
    def test_scheduler_base_lrs_updated(self, run_dir, make_env):
        groups = _make_param_groups(1, lr=1e-3)
        opt = MockOptimizer(groups)

        class FakeScheduler:
            def __init__(self):
                self.base_lrs = [1e-3]

        sched = FakeScheduler()
        kernel = _kernel_with_opt(run_dir, opt)
        env = make_env(step=1, optimizer=opt, scheduler=sched)

        kernel._apply_single(_op(params={"lr": 5e-4}), env, "train_step", 1)

        assert groups[0]["lr"] == pytest.approx(5e-4)
        assert sched.base_lrs[0] == pytest.approx(5e-4)


# ------------------------------------------------------------------ #
# 4. Missing optimizer (no MutableState)
# ------------------------------------------------------------------ #
class TestMissingMutableState:
    def test_failed_with_no_mutable_state(self, run_dir, make_env, read_ledger):
        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)  # no mutable_state
        env = make_env(step=1)

        kernel._apply_single(_op(params={"lr": 1e-4}), env, "train_step", 1)

        ledger = read_ledger()
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "failed"
        assert "no_mutable_state" in ledger[0]["error"]


# ------------------------------------------------------------------ #
# 5. Unknown param key
# ------------------------------------------------------------------ #
class TestUnknownParamKey:
    def test_unknown_param_fails(self, run_dir, make_env, read_ledger):
        groups = _make_param_groups(1, lr=1e-3)
        opt = MockOptimizer(groups)
        kernel = _kernel_with_opt(run_dir, opt)
        env = make_env(step=1, optimizer=opt)

        kernel._apply_single(
            _op(params={"key": "nonexistent", "value": 1.0}),
            env, "train_step", 1,
        )

        ledger = read_ledger()
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "failed"
        assert "unknown_param" in ledger[0]["error"]


# ------------------------------------------------------------------ #
# 6. Enable / disable via default stream
# ------------------------------------------------------------------ #
class TestEnableDisable:
    def test_disabled_actuator_rejects_set_params(self, run_dir, make_env, read_ledger):
        groups = _make_param_groups(1, lr=1e-3)
        opt = MockOptimizer(groups)
        kernel = _kernel_with_opt(run_dir, opt)
        env = make_env(step=1, optimizer=opt)

        # Disable the lr actuator
        kernel._apply_single(
            HotOp(module="opt", op="disable", params={"key": "lr"}),
            env, "train_step", 1,
        )

        # set_params should fail
        kernel._apply_single(_op(params={"lr": 1e-4}), env, "train_step", 2)
        assert groups[0]["lr"] == pytest.approx(1e-3)  # unchanged

        ledger = read_ledger()
        set_entries = [e for e in ledger if e["op"] == "set_params"]
        assert len(set_entries) == 1
        assert set_entries[0]["decision"] == "failed"
        assert "disabled" in set_entries[0]["error"]

    def test_re_enable_then_apply(self, run_dir, make_env, read_ledger):
        groups = _make_param_groups(1, lr=1e-3)
        opt = MockOptimizer(groups)
        kernel = _kernel_with_opt(run_dir, opt)
        env = make_env(step=1, optimizer=opt)

        # Disable then re-enable
        kernel._apply_single(
            HotOp(module="opt", op="disable", params={"key": "lr"}),
            env, "train_step", 1,
        )
        kernel._apply_single(
            HotOp(module="opt", op="enable", params={"key": "lr"}),
            env, "train_step", 2,
        )

        # Now it should apply
        kernel._apply_single(_op(params={"lr": 1e-4}), env, "train_step", 3)
        assert groups[0]["lr"] == pytest.approx(1e-4)


# ------------------------------------------------------------------ #
# 7. Ledger records preserve module field
# ------------------------------------------------------------------ #
class TestLedgerFormat:
    def test_ledger_preserves_module_field(self, run_dir, make_env, read_ledger):
        groups = _make_param_groups(1, lr=1e-3)
        opt = MockOptimizer(groups)
        kernel = _kernel_with_opt(run_dir, opt)
        env = make_env(step=1, optimizer=opt)

        kernel._apply_single(_op(params={"lr": 5e-4}), env, "train_step", 1)

        ledger = read_ledger()
        assert len(ledger) == 1
        assert ledger[0]["module"] == "opt"
        assert ledger[0]["decision"] == "applied"


# ------------------------------------------------------------------ #
# 8. New format: explicit key+value
# ------------------------------------------------------------------ #
class TestNewKeyValueFormat:
    def test_explicit_key_value(self, run_dir, make_env, read_ledger):
        groups = _make_param_groups(1, lr=1e-3)
        opt = MockOptimizer(groups)
        kernel = _kernel_with_opt(run_dir, opt)
        env = make_env(step=1, optimizer=opt)

        kernel._apply_single(
            HotOp(module="opt", op="set_params", params={"key": "lr", "value": 5e-4}),
            env, "train_step", 1,
        )

        assert groups[0]["lr"] == pytest.approx(5e-4)
        ledger = read_ledger()
        assert ledger[0]["decision"] == "applied"


# ------------------------------------------------------------------ #
# 9. Validation error
# ------------------------------------------------------------------ #
class TestValidationError:
    def test_out_of_bounds_rejected(self, run_dir, make_env, read_ledger):
        groups = _make_param_groups(1, lr=1e-3)
        opt = MockOptimizer(groups)
        kernel = _kernel_with_opt(run_dir, opt)
        env = make_env(step=1, optimizer=opt)

        # lr actuator has max_value=1.0 by default
        kernel._apply_single(
            _op(params={"key": "lr", "value": 5.0}),
            env, "train_step", 1,
        )

        assert groups[0]["lr"] == pytest.approx(1e-3)  # unchanged
        ledger = read_ledger()
        assert ledger[0]["decision"] == "failed"
        assert "above max" in ledger[0]["error"]


# ------------------------------------------------------------------ #
# 10. Betas set
# ------------------------------------------------------------------ #
class TestBetasSet:
    def test_betas_via_mutable_state(self, run_dir, make_env, read_ledger):
        groups = [{"lr": 1e-3, "weight_decay": 0.01, "betas": (0.9, 0.999)}]
        opt = MockOptimizer(groups)
        kernel = _kernel_with_opt(run_dir, opt)
        env = make_env(step=1, optimizer=opt)

        kernel._apply_single(
            _op(params={"key": "betas", "value": (0.8, 0.99)}),
            env, "train_step", 1,
        )

        assert groups[0]["betas"] == pytest.approx((0.8, 0.99))
        ledger = read_ledger()
        assert ledger[0]["decision"] == "applied"


# ------------------------------------------------------------------ #
# 11. Multiple params in single op
# ------------------------------------------------------------------ #
class TestMultipleParams:
    def test_lr_and_wd_set_together(self, run_dir, make_env, read_ledger):
        groups = _make_param_groups(1, lr=1e-3, weight_decay=0.01)
        opt = MockOptimizer(groups)
        kernel = _kernel_with_opt(run_dir, opt)
        env = make_env(step=1, optimizer=opt)

        kernel._apply_single(
            _op(params={"lr": 5e-4, "weight_decay": 0.05}),
            env, "train_step", 1,
        )

        assert groups[0]["lr"] == pytest.approx(5e-4)
        assert groups[0]["weight_decay"] == pytest.approx(0.05)


# ------------------------------------------------------------------ #
# 12. Apply error produces error in ledger
# ------------------------------------------------------------------ #
class TestApplyError:
    def test_apply_fn_failure_recorded(self, run_dir, make_env, read_ledger):
        def _bad_apply(value, env):
            raise RuntimeError("gpu exploded")

        bad_act = HotcbActuator(
            param_key="bad_param",
            type=ActuatorType.FLOAT,
            apply_fn=_bad_apply,
            min_value=0.0,
            max_value=10.0,
            current_value=1.0,
        )
        ms = mutable_state([bad_act])
        kernel = HotKernel(run_dir=run_dir, debounce_steps=1, mutable_state=ms)
        env = make_env(step=1)

        kernel._apply_single(
            HotOp(module="opt", op="set_params", params={"key": "bad_param", "value": 2.0}),
            env, "train_step", 1,
        )

        ledger = read_ledger()
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "failed"
        assert "gpu exploded" in ledger[0]["error"]
