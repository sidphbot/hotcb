"""Tests for loss weight mutations via kernel default stream + MutableState."""
from __future__ import annotations

import os

import pytest

from hotcb.actuators import loss_actuators, mutable_state, ApplyResult
from hotcb.actuators.actuator import ActuatorState, ActuatorType, HotcbActuator
from hotcb.actuators.state import MutableState
from hotcb.kernel import HotKernel
from hotcb.ops import HotOp


def _kernel_with_loss(run_dir, loss_weights, extra_actuators=None, **bounds):
    """Build a kernel with MutableState containing loss actuators."""
    acts = loss_actuators(loss_weights, **bounds)
    if extra_actuators:
        acts += extra_actuators
    ms = mutable_state(acts)
    return HotKernel(run_dir=run_dir, debounce_steps=1, mutable_state=ms), loss_weights


def _op(op="set_params", params=None, id="main"):
    return HotOp(module="loss", op=op, id=id, params=params)


# ------------------------------------------------------------------ #
# 1. Weight mutation via direct key
# ------------------------------------------------------------------ #
class TestWeightsMutation:
    def test_set_weight_by_key(self, run_dir, make_env, read_ledger):
        weights = {"distill": 1.0, "depth": 1.0}
        kernel, weights = _kernel_with_loss(run_dir, weights)
        env = make_env(step=1)

        kernel._apply_single(
            _op(params={"key": "distill", "value": 0.2}), env, "train_step", 1,
        )

        assert weights["distill"] == pytest.approx(0.2)
        ledger = read_ledger()
        assert ledger[0]["decision"] == "applied"

    def test_set_weight_direct_format(self, run_dir, make_env, read_ledger):
        weights = {"distill": 1.0, "depth": 1.0}
        kernel, weights = _kernel_with_loss(run_dir, weights)
        env = make_env(step=1)

        kernel._apply_single(
            _op(params={"distill": 0.2, "depth": 1.5}), env, "train_step", 1,
        )

        assert weights["distill"] == pytest.approx(0.2)
        assert weights["depth"] == pytest.approx(1.5)
        ledger = read_ledger()
        assert ledger[0]["decision"] == "applied"


# ------------------------------------------------------------------ #
# 2. Bounds enforcement
# ------------------------------------------------------------------ #
class TestBounds:
    def test_out_of_bounds_rejected(self, run_dir, make_env, read_ledger):
        weights = {"distill": 1.0}
        kernel, weights = _kernel_with_loss(run_dir, weights, global_bounds=(0.0, 10.0))
        env = make_env(step=1)

        kernel._apply_single(
            _op(params={"key": "distill", "value": 50.0}), env, "train_step", 1,
        )

        assert weights["distill"] == pytest.approx(1.0)  # unchanged
        ledger = read_ledger()
        assert ledger[0]["decision"] == "failed"
        assert "above max" in ledger[0]["error"]


# ------------------------------------------------------------------ #
# 3. Missing MutableState
# ------------------------------------------------------------------ #
class TestMissingMutableState:
    def test_no_mutable_state_fails(self, run_dir, make_env, read_ledger):
        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)  # no mutable_state
        env = make_env(step=1)

        kernel._apply_single(
            _op(params={"key": "distill", "value": 0.2}), env, "train_step", 1,
        )

        ledger = read_ledger()
        assert ledger[0]["decision"] == "failed"
        assert "no_mutable_state" in ledger[0]["error"]


# ------------------------------------------------------------------ #
# 4. Unknown param key
# ------------------------------------------------------------------ #
class TestUnknownKey:
    def test_unknown_key_fails(self, run_dir, make_env, read_ledger):
        weights = {"distill": 1.0}
        kernel, _ = _kernel_with_loss(run_dir, weights)
        env = make_env(step=1)

        kernel._apply_single(
            _op(params={"key": "nonexistent", "value": 0.5}), env, "train_step", 1,
        )

        ledger = read_ledger()
        assert ledger[0]["decision"] == "failed"
        assert "unknown_param" in ledger[0]["error"]


# ------------------------------------------------------------------ #
# 5. Enable / disable
# ------------------------------------------------------------------ #
class TestEnableDisable:
    def test_disabled_rejects_set(self, run_dir, make_env, read_ledger):
        weights = {"distill": 1.0}
        kernel, weights = _kernel_with_loss(run_dir, weights)
        env = make_env(step=1)

        kernel._apply_single(
            HotOp(module="loss", op="disable", params={"key": "distill"}),
            env, "train_step", 1,
        )
        kernel._apply_single(
            _op(params={"key": "distill", "value": 0.2}), env, "train_step", 2,
        )

        assert weights["distill"] == pytest.approx(1.0)  # unchanged
        ledger = read_ledger()
        set_entries = [e for e in ledger if e["op"] == "set_params"]
        assert set_entries[0]["decision"] == "failed"
        assert "disabled" in set_entries[0]["error"]

    def test_re_enable_then_apply(self, run_dir, make_env, read_ledger):
        weights = {"distill": 1.0}
        kernel, weights = _kernel_with_loss(run_dir, weights)
        env = make_env(step=1)

        kernel._apply_single(
            HotOp(module="loss", op="disable", params={"key": "distill"}),
            env, "train_step", 1,
        )
        kernel._apply_single(
            HotOp(module="loss", op="enable", params={"key": "distill"}),
            env, "train_step", 2,
        )
        kernel._apply_single(
            _op(params={"key": "distill", "value": 0.3}), env, "train_step", 3,
        )

        assert weights["distill"] == pytest.approx(0.3)


# ------------------------------------------------------------------ #
# 6. Ledger format
# ------------------------------------------------------------------ #
class TestLedgerFormat:
    def test_module_preserved_in_ledger(self, run_dir, make_env, read_ledger):
        weights = {"distill": 1.0}
        kernel, _ = _kernel_with_loss(run_dir, weights)
        env = make_env(step=1)

        kernel._apply_single(
            _op(params={"key": "distill", "value": 0.5}), env, "train_step", 1,
        )

        ledger = read_ledger()
        assert ledger[0]["module"] == "loss"
        assert ledger[0]["decision"] == "applied"


# ------------------------------------------------------------------ #
# 7. Multiple weights in single op
# ------------------------------------------------------------------ #
class TestMultipleWeights:
    def test_set_multiple_weights(self, run_dir, make_env, read_ledger):
        weights = {"distill": 1.0, "depth": 1.0, "kl": 0.5}
        kernel, weights = _kernel_with_loss(run_dir, weights)
        env = make_env(step=1)

        kernel._apply_single(
            _op(params={"distill": 0.2, "depth": 1.5, "kl": 0.8}),
            env, "train_step", 1,
        )

        assert weights["distill"] == pytest.approx(0.2)
        assert weights["depth"] == pytest.approx(1.5)
        assert weights["kl"] == pytest.approx(0.8)
        ledger = read_ledger()
        assert ledger[0]["decision"] == "applied"


# ------------------------------------------------------------------ #
# 8. Apply error
# ------------------------------------------------------------------ #
class TestApplyError:
    def test_apply_fn_failure_recorded(self, run_dir, make_env, read_ledger):
        def _bad_apply(value, env):
            raise RuntimeError("broken")

        bad_act = HotcbActuator(
            param_key="bad_weight",
            type=ActuatorType.FLOAT,
            apply_fn=_bad_apply,
            group="loss",
            min_value=0.0,
            max_value=100.0,
            current_value=1.0,
        )
        ms = mutable_state([bad_act])
        kernel = HotKernel(run_dir=run_dir, debounce_steps=1, mutable_state=ms)
        env = make_env(step=1)

        kernel._apply_single(
            HotOp(module="loss", op="set_params", params={"key": "bad_weight", "value": 2.0}),
            env, "train_step", 1,
        )

        ledger = read_ledger()
        assert ledger[0]["decision"] == "failed"
        assert "broken" in ledger[0]["error"]


# ------------------------------------------------------------------ #
# 9. Unknown op
# ------------------------------------------------------------------ #
class TestUnknownOp:
    def test_unknown_op_ignored(self, run_dir, make_env, read_ledger):
        weights = {"distill": 1.0}
        kernel, _ = _kernel_with_loss(run_dir, weights)
        env = make_env(step=1)

        kernel._apply_single(
            HotOp(module="loss", op="reset", id="main"), env, "train_step", 1,
        )

        ledger = read_ledger()
        assert ledger[0]["decision"] == "ignored"
        assert "unknown_op" in (ledger[0].get("notes") or "")


# ------------------------------------------------------------------ #
# 10. Custom module routes through default stream
# ------------------------------------------------------------------ #
class TestCustomModule:
    def test_custom_module_routes_to_mutable_state(self, run_dir, make_env, read_ledger):
        """Any unknown module name routes through MutableState if it has the key."""
        custom_act = HotcbActuator(
            param_key="dropout",
            type=ActuatorType.FLOAT,
            apply_fn=lambda v, e: ApplyResult(success=True),
            group="custom",
            min_value=0.0,
            max_value=1.0,
            current_value=0.5,
        )
        ms = mutable_state([custom_act])
        kernel = HotKernel(run_dir=run_dir, debounce_steps=1, mutable_state=ms)
        env = make_env(step=1)

        kernel._apply_single(
            HotOp(module="custom", op="set_params", params={"key": "dropout", "value": 0.3}),
            env, "train_step", 1,
        )

        ledger = read_ledger()
        assert ledger[0]["decision"] == "applied"
        assert ledger[0]["module"] == "custom"


# ------------------------------------------------------------------ #
# 11. No params produces error
# ------------------------------------------------------------------ #
class TestNoParams:
    def test_empty_params_fails(self, run_dir, make_env, read_ledger):
        weights = {"distill": 1.0}
        kernel, _ = _kernel_with_loss(run_dir, weights)
        env = make_env(step=1)

        kernel._apply_single(
            _op(params={}), env, "train_step", 1,
        )

        ledger = read_ledger()
        assert ledger[0]["decision"] == "failed"
        assert "no_params" in ledger[0]["error"]


# ------------------------------------------------------------------ #
# 12. Partial success with errors
# ------------------------------------------------------------------ #
class TestPartialSuccess:
    def test_mixed_known_and_unknown_keys(self, run_dir, make_env, read_ledger):
        weights = {"distill": 1.0}
        kernel, weights = _kernel_with_loss(run_dir, weights)
        env = make_env(step=1)

        kernel._apply_single(
            _op(params={"distill": 0.5, "nonexistent": 2.0}),
            env, "train_step", 1,
        )

        # Should succeed for known key, note error for unknown
        assert weights["distill"] == pytest.approx(0.5)
        ledger = read_ledger()
        assert ledger[0]["decision"] == "applied"
        assert "unknown_param" in (ledger[0].get("notes") or "")


# ------------------------------------------------------------------ #
# 13. Freeze enforcement for default stream
# ------------------------------------------------------------------ #
class TestFreezeEnforcement:
    def test_freeze_blocks_default_stream(self, run_dir, make_env, write_freeze, read_ledger):
        weights = {"distill": 1.0}
        write_freeze(mode="prod")
        kernel, weights = _kernel_with_loss(run_dir, weights)
        env = make_env(step=1)

        kernel._apply_single(
            _op(params={"key": "distill", "value": 0.2}),
            env, "train_step", 1,
        )

        assert weights["distill"] == pytest.approx(1.0)  # unchanged
        ledger = read_ledger()
        assert ledger[0]["decision"] == "ignored_freeze"


# ------------------------------------------------------------------ #
# 14. Mutation tracking
# ------------------------------------------------------------------ #
class TestMutationTracking:
    def test_mutation_recorded_in_actuator(self, run_dir, make_env):
        weights = {"distill": 1.0}
        kernel, weights = _kernel_with_loss(run_dir, weights)
        env = make_env(step=1)

        kernel._apply_single(
            _op(params={"key": "distill", "value": 0.5}), env, "train_step", 10,
        )

        act = kernel._mutable_state.get("distill")
        assert act is not None
        assert len(act.mutations) == 1
        assert act.mutations[0].step == 10
        assert act.mutations[0].new_value == pytest.approx(0.5)
        assert act.state == ActuatorState.UNVERIFIED
