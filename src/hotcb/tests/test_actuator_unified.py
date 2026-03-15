"""Comprehensive tests for the unified actuator types (Phase 2).

Tests cover: ActuatorType validation, state machine transitions, mutation
tracking, apply_fn behaviour, snapshot/restore, describe_space, and the
convenience constructors (optimizer_actuators, loss_actuators, mutable_state).
"""

from __future__ import annotations

import pytest

from hotcb.actuators import (
    ApplyResult,
    ValidationResult,
    mutable_state,
    optimizer_actuators,
    loss_actuators,
)
from hotcb.actuators.actuator import (
    ActuatorState,
    ActuatorType,
    HotcbActuator,
    Mutation,
    _INIT_SENTINEL,
)
from hotcb.actuators.state import MutableState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockOptimizer:
    def __init__(self, lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999)):
        self.param_groups = [{"lr": lr, "weight_decay": weight_decay, "betas": betas}]


class MockScheduler:
    def __init__(self, base_lrs):
        self.base_lrs = list(base_lrs)


def _noop_apply(value, env):
    return ApplyResult(success=True, detail={"applied": value})


def _failing_apply(value, env):
    return ApplyResult(success=False, error="intentional_failure")


def _raising_apply(value, env):
    raise RuntimeError("boom")


def _make_float_actuator(key="x", min_v=0.0, max_v=1.0, current=0.5, apply_fn=None):
    return HotcbActuator(
        param_key=key,
        type=ActuatorType.FLOAT,
        apply_fn=apply_fn or _noop_apply,
        min_value=min_v,
        max_value=max_v,
        current_value=current,
    )


# ===================================================================
# ActuatorType & validation
# ===================================================================

class TestActuatorValidation:

    def test_float_actuator_validate_in_bounds(self):
        act = HotcbActuator(
            param_key="x", type=ActuatorType.FLOAT, apply_fn=_noop_apply,
            min_value=0.0, max_value=1.0,
        )
        vr = act.validate(0.5)
        assert vr.valid
        assert vr.errors == []

    def test_float_actuator_validate_out_of_bounds(self):
        act = HotcbActuator(
            param_key="x", type=ActuatorType.FLOAT, apply_fn=_noop_apply,
            min_value=0.0, max_value=1.0,
        )
        vr = act.validate(1.5)
        assert not vr.valid
        assert any("above max" in e for e in vr.errors)

        vr_low = act.validate(-0.1)
        assert not vr_low.valid
        assert any("below min" in e for e in vr_low.errors)

    def test_log_float_actuator_validate(self):
        act = HotcbActuator(
            param_key="lr", type=ActuatorType.LOG_FLOAT, apply_fn=_noop_apply,
            min_value=1e-7, max_value=1.0,
        )
        vr = act.validate(1e-4)
        assert vr.valid

        vr_neg = act.validate(-1.0)
        assert not vr_neg.valid
        assert any("log_float must be positive" in e for e in vr_neg.errors)

    def test_bool_actuator_validate(self):
        act = HotcbActuator(
            param_key="flag", type=ActuatorType.BOOL, apply_fn=_noop_apply,
        )
        assert act.validate(True).valid
        assert act.validate(False).valid

        vr = act.validate("yes")
        assert not vr.valid
        assert any("expected bool" in e for e in vr.errors)

    def test_int_actuator_validate(self):
        act = HotcbActuator(
            param_key="n", type=ActuatorType.INT, apply_fn=_noop_apply,
            min_value=0, max_value=100,
        )
        assert act.validate(50).valid

        vr_float = act.validate(50.5)
        assert not vr_float.valid
        assert any("expected int" in e for e in vr_float.errors)

        vr_bool = act.validate(True)
        assert not vr_bool.valid
        assert any("expected int" in e for e in vr_bool.errors)

        vr_oob = act.validate(101)
        assert not vr_oob.valid
        assert any("above max" in e for e in vr_oob.errors)

    def test_choice_actuator_validate(self):
        act = HotcbActuator(
            param_key="opt", type=ActuatorType.CHOICE, apply_fn=_noop_apply,
            choices=["adam", "sgd", "adamw"],
        )
        assert act.validate("adam").valid

        vr = act.validate("rmsprop")
        assert not vr.valid
        assert any("not in choices" in e for e in vr.errors)

    def test_tuple_actuator_validate(self):
        act = HotcbActuator(
            param_key="betas", type=ActuatorType.TUPLE, apply_fn=_noop_apply,
        )
        assert act.validate((0.9, 0.999)).valid
        assert act.validate([0.9, 0.999]).valid

        vr = act.validate("not a tuple")
        assert not vr.valid
        assert any("expected tuple/list" in e for e in vr.errors)


# ===================================================================
# State machine
# ===================================================================

class TestStateMachine:

    def test_initial_state_is_init(self):
        act = HotcbActuator(
            param_key="x", type=ActuatorType.FLOAT, apply_fn=_noop_apply,
        )
        assert act.state == ActuatorState.INIT
        assert act.current_value is _INIT_SENTINEL

    def test_initialize_transitions_to_untouched(self):
        opt = MockOptimizer(lr=1e-3)
        acts = optimizer_actuators(opt)
        ms = MutableState(acts)

        lr_act = ms.get("lr")
        assert lr_act is not None
        assert lr_act.state == ActuatorState.INIT  # set by constructor, but current_value populated

        ms.initialize(env={})
        assert lr_act.state == ActuatorState.UNTOUCHED

    def test_apply_transitions_to_unverified(self):
        act = _make_float_actuator(current=0.5)
        ms = MutableState([act])
        ms.initialize(env={})
        assert act.state == ActuatorState.UNTOUCHED

        result = ms.apply("x", 0.7, {}, step=10)
        assert result.success
        assert act.state == ActuatorState.UNVERIFIED

    def test_verify_transitions_to_verified(self):
        act = _make_float_actuator(key="x", current=0.5)
        act.metrics_dict_name = "x"
        ms = MutableState([act])
        ms.initialize(env={})
        ms.apply("x", 0.7, {}, step=10)
        assert act.state == ActuatorState.UNVERIFIED

        verified = ms.verify("x", {"x": 0.7})
        assert verified
        assert act.state == ActuatorState.VERIFIED

    def test_apply_after_verified_goes_back_to_unverified(self):
        act = _make_float_actuator(key="x", current=0.5)
        act.metrics_dict_name = "x"
        ms = MutableState([act])
        ms.initialize(env={})
        ms.apply("x", 0.7, {}, step=10)
        ms.verify("x", {"x": 0.7})
        assert act.state == ActuatorState.VERIFIED

        result = ms.apply("x", 0.3, {}, step=20)
        assert result.success
        assert act.state == ActuatorState.UNVERIFIED

    def test_disabled_actuator_rejects_apply(self):
        act = _make_float_actuator(current=0.5)
        ms = MutableState([act])
        ms.initialize(env={})
        ms.disable("x")
        assert act.state == ActuatorState.DISABLED

        result = ms.apply("x", 0.7, {}, step=10)
        assert not result.success
        assert "actuator_disabled" in result.error

    def test_disable_actuator(self):
        act = _make_float_actuator(current=0.5)
        ms = MutableState([act])
        ms.initialize(env={})
        assert act.state == ActuatorState.UNTOUCHED

        ms.disable("x")
        assert act.state == ActuatorState.DISABLED

    def test_enable_after_disable(self):
        act = _make_float_actuator(current=0.5)
        ms = MutableState([act])
        ms.initialize(env={})
        ms.disable("x")
        assert act.state == ActuatorState.DISABLED

        ms.enable("x")
        assert act.state == ActuatorState.UNTOUCHED


# ===================================================================
# Mutation tracking
# ===================================================================

class TestMutationTracking:

    def test_mutation_recorded_on_apply(self):
        act = _make_float_actuator(current=0.5)
        ms = MutableState([act])
        ms.initialize(env={})

        result = ms.apply("x", 0.7, {}, step=10)
        assert result.success
        assert len(act.mutations) == 1
        m = act.mutations[0]
        assert m.step == 10
        assert m.old_value == 0.5
        assert m.new_value == 0.7
        assert m.verified is False

    def test_multiple_mutations_accumulated(self):
        act = _make_float_actuator(current=0.5)
        ms = MutableState([act])
        ms.initialize(env={})

        ms.apply("x", 0.6, {}, step=10)
        ms.apply("x", 0.7, {}, step=20)
        ms.apply("x", 0.8, {}, step=30)

        assert len(act.mutations) == 3
        assert [m.new_value for m in act.mutations] == [0.6, 0.7, 0.8]

    def test_last_changed_step_updated(self):
        act = _make_float_actuator(current=0.5)
        ms = MutableState([act])
        ms.initialize(env={})

        ms.apply("x", 0.7, {}, step=50)
        assert act.last_changed_step == 50


# ===================================================================
# apply_fn behaviour
# ===================================================================

class TestApplyFn:

    def test_apply_fn_receives_value_and_env(self):
        received = {}

        def capture_apply(value, env):
            received["value"] = value
            received["env"] = env
            return ApplyResult(success=True)

        act = HotcbActuator(
            param_key="x", type=ActuatorType.FLOAT, apply_fn=capture_apply,
            min_value=0.0, max_value=1.0, current_value=0.5,
        )
        ms = MutableState([act])
        ms.initialize(env={})

        test_env = {"key": "val"}
        ms.apply("x", 0.8, test_env, step=1)

        assert received["value"] == 0.8
        assert received["env"] is test_env

    def test_apply_fn_failure_does_not_mutate_state(self):
        act = HotcbActuator(
            param_key="x", type=ActuatorType.FLOAT, apply_fn=_failing_apply,
            min_value=0.0, max_value=1.0, current_value=0.5,
        )
        ms = MutableState([act])
        ms.initialize(env={})

        result = ms.apply("x", 0.7, {}, step=10)
        assert not result.success
        # State should not change
        assert act.current_value == 0.5
        assert len(act.mutations) == 0
        assert act.state == ActuatorState.UNTOUCHED

    def test_apply_fn_exception_caught(self):
        act = HotcbActuator(
            param_key="x", type=ActuatorType.FLOAT, apply_fn=_raising_apply,
            min_value=0.0, max_value=1.0, current_value=0.5,
        )
        ms = MutableState([act])
        ms.initialize(env={})

        result = ms.apply("x", 0.7, {}, step=10)
        assert not result.success
        assert "apply_fn_exception" in result.error
        assert "boom" in result.error
        # State not corrupted
        assert act.current_value == 0.5
        assert len(act.mutations) == 0
        assert act.state == ActuatorState.UNTOUCHED


# ===================================================================
# Snapshot / restore
# ===================================================================

class TestSnapshotRestore:

    def test_snapshot_all(self):
        a1 = _make_float_actuator(key="lr", current=1e-3)
        a2 = _make_float_actuator(key="wd", current=1e-4)
        ms = MutableState([a1, a2])
        ms.initialize(env={})

        snap = ms.snapshot_all()
        assert "lr" in snap
        assert "wd" in snap
        assert snap["lr"]["value"] == 1e-3
        assert snap["lr"]["state"] == "untouched"
        assert snap["wd"]["value"] == 1e-4

    def test_restore_from_snapshot(self):
        # Track the "live" value so we can verify restore actually calls apply_fn
        live = {"lr": 1e-3, "wd": 1e-4}

        def make_apply(key):
            def _apply(value, env):
                live[key] = value
                return ApplyResult(success=True)
            return _apply

        a1 = HotcbActuator(
            param_key="lr", type=ActuatorType.FLOAT, apply_fn=make_apply("lr"),
            min_value=0.0, max_value=1.0, current_value=1e-3,
        )
        a2 = HotcbActuator(
            param_key="wd", type=ActuatorType.FLOAT, apply_fn=make_apply("wd"),
            min_value=0.0, max_value=1.0, current_value=1e-4,
        )
        ms = MutableState([a1, a2])
        ms.initialize(env={})

        # Snapshot
        snap = ms.snapshot_all()

        # Apply mutations
        ms.apply("lr", 5e-4, {}, step=10)
        ms.apply("wd", 5e-5, {}, step=10)
        assert live["lr"] == 5e-4
        assert live["wd"] == 5e-5

        # Restore
        results = ms.restore_all(snap, {})
        assert results["lr"].success
        assert results["wd"].success
        assert live["lr"] == 1e-3
        assert live["wd"] == 1e-4
        assert a1.current_value == 1e-3
        assert a2.current_value == 1e-4


# ===================================================================
# describe_space
# ===================================================================

class TestDescribeSpace:

    def test_describe_space_includes_all_fields(self):
        act = HotcbActuator(
            param_key="lr",
            type=ActuatorType.LOG_FLOAT,
            apply_fn=_noop_apply,
            label="Learning Rate",
            group="optimizer",
            min_value=1e-7,
            max_value=1.0,
            step_size=0.01,
            log_base=10.0,
            current_value=1e-3,
        )
        d = act.describe_space()

        assert d["param_key"] == "lr"
        assert d["type"] == "log_float"
        assert d["label"] == "Learning Rate"
        assert d["group"] == "optimizer"
        assert d["min"] == 1e-7
        assert d["max"] == 1.0
        assert d["step"] == 0.01
        assert d["log_base"] == 10.0
        assert d["choices"] is None
        assert d["current"] == 1e-3
        assert d["state"] == "init"

    def test_describe_space_current_none_for_init_sentinel(self):
        act = HotcbActuator(
            param_key="x", type=ActuatorType.FLOAT, apply_fn=_noop_apply,
        )
        d = act.describe_space()
        assert d["current"] is None

    def test_describe_space_log_base_only_for_log_float(self):
        act_float = HotcbActuator(
            param_key="x", type=ActuatorType.FLOAT, apply_fn=_noop_apply,
        )
        assert act_float.describe_space()["log_base"] is None

        act_log = HotcbActuator(
            param_key="y", type=ActuatorType.LOG_FLOAT, apply_fn=_noop_apply,
            log_base=2.0,
        )
        assert act_log.describe_space()["log_base"] == 2.0

    def test_describe_all(self):
        a1 = _make_float_actuator(key="a")
        a2 = _make_float_actuator(key="b")
        a3 = _make_float_actuator(key="c")
        ms = MutableState([a1, a2, a3])

        descs = ms.describe_all()
        assert len(descs) == 3
        keys = [d["param_key"] for d in descs]
        assert keys == ["a", "b", "c"]

    def test_describe_all_excludes_disabled(self):
        a1 = _make_float_actuator(key="a")
        a2 = _make_float_actuator(key="b")
        ms = MutableState([a1, a2])
        ms.disable("b")

        descs = ms.describe_all()
        assert len(descs) == 1
        assert descs[0]["param_key"] == "a"


# ===================================================================
# Convenience constructors — optimizer_actuators
# ===================================================================

class TestOptimizerActuators:

    def test_optimizer_actuators_from_torch_optimizer(self):
        opt = MockOptimizer(lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
        acts = optimizer_actuators(opt)

        assert len(acts) == 3
        keys = {a.param_key for a in acts}
        assert keys == {"lr", "weight_decay", "betas"}

        lr_act = next(a for a in acts if a.param_key == "lr")
        assert lr_act.type == ActuatorType.LOG_FLOAT
        assert lr_act.current_value == 1e-3
        assert lr_act.group == "optimizer"

        wd_act = next(a for a in acts if a.param_key == "weight_decay")
        assert wd_act.type == ActuatorType.LOG_FLOAT
        assert wd_act.current_value == 1e-4

        betas_act = next(a for a in acts if a.param_key == "betas")
        assert betas_act.type == ActuatorType.TUPLE
        assert betas_act.current_value == (0.9, 0.999)

    def test_optimizer_actuators_bounds(self):
        opt = MockOptimizer()
        acts = optimizer_actuators(opt, lr_bounds=(1e-6, 0.1), wd_bounds=(0.0, 0.5))

        lr_act = next(a for a in acts if a.param_key == "lr")
        assert lr_act.min_value == 1e-6
        assert lr_act.max_value == 0.1

        wd_act = next(a for a in acts if a.param_key == "weight_decay")
        assert wd_act.min_value == 0.0
        assert wd_act.max_value == 0.5

    def test_optimizer_actuators_apply_fn_sets_param_groups(self):
        opt = MockOptimizer(lr=1e-3)
        # Add a second param group
        opt.param_groups.append({"lr": 1e-3, "weight_decay": 1e-4, "betas": (0.9, 0.999)})
        acts = optimizer_actuators(opt)

        lr_act = next(a for a in acts if a.param_key == "lr")
        result = lr_act.apply_fn(5e-4, {"optimizer": opt})
        assert result.success

        # All param groups updated
        for g in opt.param_groups:
            assert g["lr"] == 5e-4

    def test_optimizer_actuators_apply_fn_coordinates_scheduler(self):
        opt = MockOptimizer(lr=1e-3)
        sched = MockScheduler(base_lrs=[1e-3])
        acts = optimizer_actuators(opt)

        lr_act = next(a for a in acts if a.param_key == "lr")
        result = lr_act.apply_fn(5e-4, {"optimizer": opt, "scheduler": sched})
        assert result.success

        assert opt.param_groups[0]["lr"] == 5e-4
        assert sched.base_lrs == [5e-4]

    def test_optimizer_actuators_wd_apply_fn(self):
        opt = MockOptimizer(weight_decay=1e-4)
        opt.param_groups.append({"lr": 1e-3, "weight_decay": 1e-4, "betas": (0.9, 0.999)})
        acts = optimizer_actuators(opt)

        wd_act = next(a for a in acts if a.param_key == "weight_decay")
        result = wd_act.apply_fn(5e-5, {})
        assert result.success
        for g in opt.param_groups:
            assert g["weight_decay"] == 5e-5

    def test_optimizer_actuators_betas_apply_fn(self):
        opt = MockOptimizer(betas=(0.9, 0.999))
        acts = optimizer_actuators(opt)

        betas_act = next(a for a in acts if a.param_key == "betas")
        result = betas_act.apply_fn([0.85, 0.99], {})
        assert result.success
        assert opt.param_groups[0]["betas"] == (0.85, 0.99)

    def test_optimizer_without_betas(self):
        """Optimizer without betas (e.g. SGD) produces only lr + wd actuators."""
        opt = MockOptimizer()
        del opt.param_groups[0]["betas"]
        acts = optimizer_actuators(opt)
        keys = {a.param_key for a in acts}
        assert "betas" not in keys
        assert "lr" in keys
        assert "weight_decay" in keys


# ===================================================================
# Convenience constructors — loss_actuators
# ===================================================================

class TestLossActuators:

    def test_loss_actuators_from_dict(self):
        weights = {"recon": 1.0, "kl": 0.5, "perceptual": 0.3}
        acts = loss_actuators(weights)

        assert len(acts) == 3
        keys = {a.param_key for a in acts}
        assert keys == {"recon", "kl", "perceptual"}

        for a in acts:
            assert a.type == ActuatorType.FLOAT
            assert a.group == "loss"

        recon_act = next(a for a in acts if a.param_key == "recon")
        assert recon_act.current_value == 1.0

    def test_loss_actuators_apply_fn_mutates_dict(self):
        weights = {"recon": 1.0, "kl": 0.5}
        acts = loss_actuators(weights)

        recon_act = next(a for a in acts if a.param_key == "recon")
        result = recon_act.apply_fn(2.0, {})
        assert result.success
        assert weights["recon"] == 2.0  # original dict mutated

    def test_loss_actuators_bounds(self):
        weights = {"recon": 1.0, "kl": 0.5}
        acts = loss_actuators(weights, global_bounds=(0.0, 10.0))

        for a in acts:
            assert a.min_value == 0.0
            assert a.max_value == 10.0

    def test_loss_actuators_key_bounds(self):
        weights = {"recon": 1.0, "kl": 0.5}
        acts = loss_actuators(weights, key_bounds={"kl": (0.0, 2.0)})

        kl_act = next(a for a in acts if a.param_key == "kl")
        recon_act = next(a for a in acts if a.param_key == "recon")

        assert kl_act.min_value == 0.0
        assert kl_act.max_value == 2.0
        # recon uses global_bounds default
        assert recon_act.min_value == 0.0
        assert recon_act.max_value == 100.0


# ===================================================================
# Convenience constructors — mutable_state
# ===================================================================

class TestMutableStateConstructor:

    def test_mutable_state_constructor(self):
        a1 = _make_float_actuator(key="lr")
        a2 = _make_float_actuator(key="wd")
        a3 = _make_float_actuator(key="recon_w")

        ms = mutable_state([a1, a2, a3])
        assert isinstance(ms, MutableState)
        assert ms.keys() == ["lr", "wd", "recon_w"]
        assert len(ms) == 3
        assert "lr" in ms
        assert "missing" not in ms


# ===================================================================
# MutableState container basics
# ===================================================================

class TestMutableStateContainer:

    def test_get_returns_actuator(self):
        act = _make_float_actuator(key="x")
        ms = MutableState([act])
        assert ms.get("x") is act

    def test_get_returns_none_for_missing(self):
        ms = MutableState([])
        assert ms.get("missing") is None

    def test_apply_unknown_key(self):
        ms = MutableState([])
        result = ms.apply("missing", 1.0, {}, step=0)
        assert not result.success
        assert "unknown_param" in result.error

    def test_verify_nonexistent_key(self):
        ms = MutableState([])
        assert not ms.verify("missing", {})

    def test_verify_no_metrics_dict_name(self):
        act = _make_float_actuator(key="x", current=0.5)
        ms = MutableState([act])
        ms.initialize(env={})
        ms.apply("x", 0.7, {}, step=10)
        # No metrics_dict_name set
        assert not ms.verify("x", {"x": 0.7})

    def test_verify_wrong_state(self):
        act = _make_float_actuator(key="x", current=0.5)
        act.metrics_dict_name = "x"
        ms = MutableState([act])
        ms.initialize(env={})
        # In UNTOUCHED, not UNVERIFIED
        assert not ms.verify("x", {"x": 0.5})

    def test_disable_nonexistent_key(self):
        ms = MutableState([])
        # Should not raise
        ms.disable("missing")

    def test_enable_nonexistent_key(self):
        ms = MutableState([])
        # Should not raise
        ms.enable("missing")

    def test_enable_non_disabled_is_noop(self):
        act = _make_float_actuator(key="x")
        ms = MutableState([act])
        ms.initialize(env={})
        assert act.state == ActuatorState.UNTOUCHED
        ms.enable("x")  # Should be noop since not disabled
        assert act.state == ActuatorState.UNTOUCHED


# ===================================================================
# Integration: full end-to-end flow
# ===================================================================

class TestIntegration:

    def test_full_optimizer_flow(self):
        """Full lifecycle: create, initialize, apply, verify, snapshot, restore."""
        opt = MockOptimizer(lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
        sched = MockScheduler(base_lrs=[1e-3])
        acts = optimizer_actuators(opt)
        ms = MutableState(acts)

        # Initialize
        ms.initialize(env={})
        for a in acts:
            assert a.state == ActuatorState.UNTOUCHED

        # Snapshot
        snap = ms.snapshot_all()

        # Apply lr change
        env = {"optimizer": opt, "scheduler": sched}
        result = ms.apply("lr", 5e-4, env, step=10)
        assert result.success
        assert opt.param_groups[0]["lr"] == 5e-4
        assert sched.base_lrs == [5e-4]

        lr_act = ms.get("lr")
        assert lr_act.state == ActuatorState.UNVERIFIED
        assert lr_act.current_value == 5e-4

        # Verify
        ms.verify("lr", {"lr": 5e-4})
        assert lr_act.state == ActuatorState.VERIFIED

        # Restore snapshot
        results = ms.restore_all(snap, env)
        assert all(r.success for r in results.values())
        assert opt.param_groups[0]["lr"] == 1e-3

    def test_full_loss_flow(self):
        """Full lifecycle for loss actuators."""
        weights = {"recon": 1.0, "kl": 0.5}
        acts = loss_actuators(weights)
        ms = MutableState(acts)

        ms.initialize(env={})

        # Apply
        result = ms.apply("recon", 2.0, {}, step=5)
        assert result.success
        assert weights["recon"] == 2.0

        # Describe
        descs = ms.describe_all()
        assert len(descs) == 2
        recon_desc = next(d for d in descs if d["param_key"] == "recon")
        assert recon_desc["current"] == 2.0
        assert recon_desc["state"] == "unverified"

    def test_mixed_actuators(self):
        """Optimizer + loss actuators in one MutableState."""
        opt = MockOptimizer(lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
        weights = {"recon": 1.0, "kl": 0.5}

        all_acts = optimizer_actuators(opt) + loss_actuators(weights)
        ms = MutableState(all_acts)
        ms.initialize(env={})

        assert len(ms) == 5  # lr, wd, betas, recon, kl
        assert ms.keys() == ["lr", "weight_decay", "betas", "recon", "kl"]

        descs = ms.describe_all()
        groups = {d["group"] for d in descs}
        assert groups == {"optimizer", "loss"}
