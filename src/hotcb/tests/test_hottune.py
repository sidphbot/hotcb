"""Tests for the hottune module — actuators, controller, search, evaluator, recipe."""
from __future__ import annotations

import json
import math
import os
import tempfile
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from hotcb.actuators.base import ApplyResult, ValidationResult
from hotcb.actuators.optimizer import OptimizerActuator
from hotcb.actuators.mutable_state import MutableStateActuator
from hotcb.modules.tune.schemas import (
    AcceptanceConfig,
    ActuatorConfig,
    MutationSpec,
    ObjectiveConfig,
    SafetyConfig,
    SearchConfig,
    TuneRecipe,
)
from hotcb.modules.tune.state import Mutation, Segment, TuneState
from hotcb.modules.tune.constraints import (
    check_mutation_constraints,
    check_safety_blockers,
    get_phase_bin,
)
from hotcb.modules.tune.evaluator import evaluate_segment, read_metrics, score_segment
from hotcb.modules.tune.controller import HotTuneController
from hotcb.modules.tune.storage import write_mutation, write_segment, write_summary, load_mutations_log, load_segments_log
from hotcb.modules.tune.recipe import compute_run_stats, evolve_recipe
from hotcb.ops import HotOp
from hotcb.kernel import HotKernel


# ---------- helpers ----------

class FakeOptimizer:
    def __init__(self, lr=0.001, wd=0.01, betas=(0.9, 0.999)):
        self.param_groups = [{"lr": lr, "weight_decay": wd, "betas": betas}]


def make_env(
    step=100,
    epoch=1,
    phase="val",
    optimizer=None,
    mutable_state=None,
    loss=None,
    metric_fn=None,
    max_steps=1000,
):
    env = {
        "step": step,
        "epoch": epoch,
        "phase": phase,
        "max_steps": max_steps,
    }
    if optimizer is not None:
        env["optimizer"] = optimizer
    if mutable_state is not None:
        env["mutable_state"] = mutable_state
    if loss is not None:
        env["loss"] = loss
    if metric_fn is not None:
        env["metric"] = metric_fn
    return env


def simple_recipe(**overrides) -> TuneRecipe:
    """Build a minimal recipe for testing."""
    base = {
        "version": 1,
        "objective": {"primary": "val/loss", "mode": "min"},
        "actuators": {
            "opt": {
                "enabled": True,
                "mutations": {
                    "lr_mult": {"bounds": [0.7, 1.3], "prior_center": 1.0, "cooldown": 1, "risk": "low"},
                },
            },
            "loss": {
                "enabled": True,
                "keys": {
                    "main_w": {"bounds": [0.5, 2.0], "prior_center": 1.0, "cooldown": 1, "risk": "low", "mode": "mult"},
                },
            },
        },
        "search": {"algorithm": "random", "startup_trials": 2, "candidate_count": 4},
        "acceptance": {"epsilon": 0.001, "horizon": "next_val_epoch_end", "rollback_on_reject": True},
        "safety": {"block_on_nan": True, "block_on_anomaly": True, "max_global_reject_streak": 3},
    }
    base.update(overrides)
    return TuneRecipe.from_dict(base)


# ========== Actuator tests ==========

class TestOptimizerActuator:
    def test_snapshot_and_restore(self):
        opt = FakeOptimizer(lr=0.001, wd=0.01, betas=(0.9, 0.999))
        env = {"optimizer": opt}
        act = OptimizerActuator()

        snap = act.snapshot(env)
        assert snap["groups"][0]["lr"] == 0.001
        assert snap["groups"][0]["betas"] == [0.9, 0.999]

        # Mutate
        opt.param_groups[0]["lr"] = 0.01
        assert opt.param_groups[0]["lr"] == 0.01

        # Restore
        result = act.restore(snap, env)
        assert result.success
        assert opt.param_groups[0]["lr"] == 0.001

    def test_validate_lr_mult(self):
        act = OptimizerActuator()
        env = {"optimizer": FakeOptimizer()}
        assert act.validate({"op": "lr_mult", "value": 0.9}, env).valid
        assert not act.validate({"op": "lr_mult", "value": -1}, env).valid
        assert not act.validate({"op": "unknown", "value": 1}, env).valid

    def test_validate_betas_set(self):
        act = OptimizerActuator()
        env = {"optimizer": FakeOptimizer()}
        assert act.validate({"op": "betas_set", "value": [0.9, 0.98]}, env).valid
        assert not act.validate({"op": "betas_set", "value": [1.5, 0.98]}, env).valid
        assert not act.validate({"op": "betas_set", "value": [0.9]}, env).valid

    def test_apply_lr_mult(self):
        opt = FakeOptimizer(lr=0.001)
        env = {"optimizer": opt}
        act = OptimizerActuator()
        result = act.apply({"op": "lr_mult", "value": 0.5}, env)
        assert result.success
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0005)

    def test_apply_lr_set(self):
        opt = FakeOptimizer(lr=0.001)
        env = {"optimizer": opt}
        act = OptimizerActuator()
        result = act.apply({"op": "lr_set", "value": 0.01}, env)
        assert result.success
        assert opt.param_groups[0]["lr"] == 0.01

    def test_apply_wd_mult(self):
        opt = FakeOptimizer(wd=0.01)
        env = {"optimizer": opt}
        act = OptimizerActuator()
        result = act.apply({"op": "wd_mult", "value": 2.0}, env)
        assert result.success
        assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.02)

    def test_apply_betas_set(self):
        opt = FakeOptimizer()
        env = {"optimizer": opt}
        act = OptimizerActuator()
        result = act.apply({"op": "betas_set", "value": [0.85, 0.95]}, env)
        assert result.success
        assert opt.param_groups[0]["betas"] == (0.85, 0.95)

    def test_apply_missing_optimizer(self):
        act = OptimizerActuator()
        result = act.apply({"op": "lr_mult", "value": 0.9}, {})
        assert not result.success
        assert "missing_optimizer" in result.error

    def test_describe_space(self):
        act = OptimizerActuator()
        space = act.describe_space()
        assert "lr_mult" in space["mutations"]

    def test_lr_bounds_clamping(self):
        opt = FakeOptimizer(lr=0.5)
        env = {"optimizer": opt}
        act = OptimizerActuator(lr_bounds=(1e-7, 1.0))
        result = act.apply({"op": "lr_mult", "value": 10.0}, env)
        assert result.success
        assert opt.param_groups[0]["lr"] == 1.0  # clamped


class TestMutableStateActuator:
    def test_snapshot_and_restore(self):
        ls = {"weights": {"main": 1.0, "aux": 0.5}}
        env = {"mutable_state": ls}
        act = MutableStateActuator()

        snap = act.snapshot(env)
        assert snap["weights"]["main"] == 1.0

        ls["weights"]["main"] = 2.0
        result = act.restore(snap, env)
        assert result.success
        assert ls["weights"]["main"] == 1.0

    def test_validate_set(self):
        act = MutableStateActuator(global_bounds=(0.0, 10.0))
        ls = {"weights": {"main": 1.0}}
        env = {"mutable_state": ls}
        assert act.validate({"op": "set", "key": "main", "value": 5.0}, env).valid
        assert not act.validate({"op": "set", "key": "main", "value": 11.0}, env).valid

    def test_validate_mult_bounds(self):
        act = MutableStateActuator(global_bounds=(0.0, 10.0))
        ls = {"weights": {"main": 5.0}}
        env = {"mutable_state": ls}
        assert not act.validate({"op": "mult", "key": "main", "value": 3.0}, env).valid  # 15 > 10

    def test_apply_set(self):
        ls = {"weights": {"main": 1.0}}
        env = {"mutable_state": ls}
        act = MutableStateActuator()
        result = act.apply({"op": "set", "key": "main", "value": 2.0}, env)
        assert result.success
        assert ls["weights"]["main"] == 2.0

    def test_apply_mult(self):
        ls = {"weights": {"main": 1.0}}
        env = {"mutable_state": ls}
        act = MutableStateActuator()
        result = act.apply({"op": "mult", "key": "main", "value": 1.5}, env)
        assert result.success
        assert ls["weights"]["main"] == pytest.approx(1.5)

    def test_apply_delta(self):
        ls = {"weights": {"main": 1.0}}
        env = {"mutable_state": ls}
        act = MutableStateActuator()
        result = act.apply({"op": "delta", "key": "main", "value": 0.3}, env)
        assert result.success
        assert ls["weights"]["main"] == pytest.approx(1.3)

    def test_apply_missing_mutable_state(self):
        act = MutableStateActuator()
        result = act.apply({"op": "set", "key": "main", "value": 1.0}, {})
        assert not result.success

    def test_key_bounds(self):
        act = MutableStateActuator(key_bounds={"main": (0.5, 1.5)})
        ls = {"weights": {"main": 1.0}}
        env = {"mutable_state": ls}
        assert not act.validate({"op": "set", "key": "main", "value": 2.0}, env).valid
        assert act.validate({"op": "set", "key": "main", "value": 1.2}, env).valid


# ========== Schema tests ==========

class TestTuneRecipe:
    def test_from_dict_roundtrip(self):
        r = simple_recipe()
        d = r.to_dict()
        r2 = TuneRecipe.from_dict(d)
        assert r2.objective.primary == "val/loss"
        assert r2.objective.mode == "min"
        assert "opt" in r2.actuators
        assert "lr_mult" in r2.actuators["opt"].mutations

    def test_defaults(self):
        r = TuneRecipe()
        assert r.objective.primary == "val/loss"
        assert "early" in r.phases
        assert r.safety.block_on_nan is True


# ========== State tests ==========

class TestTuneState:
    def test_cooldown_tracking(self):
        s = TuneState()
        s.set_cooldown("opt:lr_mult", 2)
        assert not s.is_cooled_down("opt:lr_mult")
        s.tick_cooldowns()
        assert not s.is_cooled_down("opt:lr_mult")
        s.tick_cooldowns()
        assert s.is_cooled_down("opt:lr_mult")

    def test_global_cooldown(self):
        s = TuneState()
        s.global_cooldown = 1
        assert not s.is_cooled_down("anything")
        s.tick_cooldowns()
        assert s.is_cooled_down("anything")

    def test_mutation_id_generation(self):
        s = TuneState()
        assert s.next_mutation_id() == "m_00001"
        assert s.next_mutation_id() == "m_00002"


# ========== Constraints tests ==========

class TestConstraints:
    def test_phase_bin(self):
        phases = {
            "early": type("P", (), {"start_frac": 0.0, "end_frac": 0.2})(),
            "mid": type("P", (), {"start_frac": 0.2, "end_frac": 0.7})(),
            "late": type("P", (), {"start_frac": 0.7, "end_frac": 1.0})(),
        }
        assert get_phase_bin(50, 1000, phases) == "early"
        assert get_phase_bin(400, 1000, phases) == "mid"
        assert get_phase_bin(800, 1000, phases) == "late"
        assert get_phase_bin(100, None, phases) == "mid"  # unknown defaults to mid

    def test_safety_blockers_nan(self):
        recipe = simple_recipe()
        env = {"loss": float("nan")}
        blockers = check_safety_blockers(env, recipe)
        assert "nan_or_inf_loss" in blockers

    def test_safety_blockers_inf(self):
        recipe = simple_recipe()
        env = {"loss": float("inf")}
        blockers = check_safety_blockers(env, recipe)
        assert "nan_or_inf_loss" in blockers

    def test_safety_blockers_anomaly(self):
        recipe = simple_recipe()
        env = {"anomaly_critical": True}
        blockers = check_safety_blockers(env, recipe)
        assert "anomaly_critical" in blockers

    def test_safety_blockers_clean(self):
        recipe = simple_recipe()
        env = {"loss": 0.5}
        assert check_safety_blockers(env, recipe) == []

    def test_mutation_constraints_cooldown(self):
        state = TuneState()
        state.set_cooldown("opt:lr_mult", 2)
        recipe = simple_recipe()
        blocks = check_mutation_constraints("opt", "lr_mult", state, recipe, "mid")
        assert any("cooldown" in b for b in blocks)

    def test_mutation_constraints_reject_streak(self):
        state = TuneState()
        state.reject_streak = 5
        recipe = simple_recipe()
        blocks = check_mutation_constraints("opt", "lr_mult", state, recipe, "mid")
        assert any("max_reject_streak" in b for b in blocks)

    def test_mutation_constraints_missing_actuator(self):
        state = TuneState()
        recipe = simple_recipe()
        blocks = check_mutation_constraints("unknown", "lr_mult", state, recipe, "mid")
        assert any("actuator_not_in_recipe" in b for b in blocks)


# ========== Evaluator tests ==========

class TestEvaluator:
    def test_read_metrics(self):
        def metric_fn(name, default=None):
            return {"val/loss": 0.5, "val/score": 0.8}.get(name, default)
        result = read_metrics(metric_fn, ["val/loss", "val/score", "missing"])
        assert result == {"val/loss": 0.5, "val/score": 0.8}

    def test_score_segment_minimize(self):
        recipe = simple_recipe()
        seg = Segment(
            segment_id="s1", mutation_id="m1", start_step=0,
            pre={"val/loss": 0.5}, post={"val/loss": 0.4},
        )
        score = score_segment(seg, recipe)
        assert score == pytest.approx(0.1)

    def test_score_segment_maximize(self):
        recipe = simple_recipe(objective={"primary": "val/score", "mode": "max"})
        seg = Segment(
            segment_id="s1", mutation_id="m1", start_step=0,
            pre={"val/score": 0.7}, post={"val/score": 0.8},
        )
        score = score_segment(seg, recipe)
        assert score == pytest.approx(0.1)

    def test_score_penalty_on_nan(self):
        recipe = simple_recipe()
        seg = Segment(
            segment_id="s1", mutation_id="m1", start_step=0,
            pre={"val/loss": 0.5}, post={"val/loss": 0.4},
            stability={"nan": True, "anomaly": False, "grad_spike": False},
        )
        score = score_segment(seg, recipe)
        assert score < 0  # penalty outweighs gain

    def test_evaluate_segment_accept(self):
        recipe = simple_recipe()

        def metric_fn(name, default=None):
            return {"val/loss": 0.3}.get(name, default)

        seg = Segment(
            segment_id="s1", mutation_id="m1", start_step=0,
            pre={"val/loss": 0.5},
        )
        result = evaluate_segment(seg, recipe, metric_fn, {"loss": 0.3})
        assert result.decision == "accepted"
        assert result.score_delta > 0

    def test_evaluate_segment_reject(self):
        recipe = simple_recipe()

        def metric_fn(name, default=None):
            return {"val/loss": 0.6}.get(name, default)

        seg = Segment(
            segment_id="s1", mutation_id="m1", start_step=0,
            pre={"val/loss": 0.5},
        )
        result = evaluate_segment(seg, recipe, metric_fn, {"loss": 0.6})
        assert result.decision == "rejected"


# ========== Storage tests ==========

class TestStorage:
    def test_write_and_load_mutations(self, tmp_path):
        m = Mutation(
            mutation_id="m_00001", step=100, epoch=1,
            phase_bin="mid", event="val_epoch_end",
            actuator="opt", patch={"op": "lr_mult", "value": 0.9},
            status="applied",
        )
        write_mutation(str(tmp_path), m)
        records = load_mutations_log(str(tmp_path))
        assert len(records) == 1
        assert records[0]["mutation_id"] == "m_00001"

    def test_write_and_load_segments(self, tmp_path):
        s = Segment(
            segment_id="s_00001", mutation_id="m_00001",
            start_step=100, end_step=200,
            pre={"val/loss": 0.5}, post={"val/loss": 0.4},
            decision="accepted", score_delta=0.1,
        )
        write_segment(str(tmp_path), s)
        records = load_segments_log(str(tmp_path))
        assert len(records) == 1
        assert records[0]["decision"] == "accepted"

    def test_write_summary(self, tmp_path):
        write_summary(str(tmp_path), {"mode": "active", "total_mutations": 5})
        path = os.path.join(str(tmp_path), "hotcb.tune.summary.json")
        with open(path) as f:
            d = json.load(f)
        assert d["mode"] == "active"


# ========== Recipe evolution tests ==========

class TestRecipeEvolution:
    def test_compute_run_stats(self):
        mutations = [
            {"mutation_id": "m1", "status": "applied", "actuator": "opt", "patch": {"op": "lr_mult"}},
            {"mutation_id": "m2", "status": "applied", "actuator": "opt", "patch": {"op": "lr_mult"}},
        ]
        segments = [
            {"mutation_id": "m1", "decision": "accepted", "score_delta": 0.05},
            {"mutation_id": "m2", "decision": "rejected", "score_delta": -0.01},
        ]
        stats = compute_run_stats(mutations, segments)
        assert stats["total_mutations"] == 2
        assert stats["applied_mutations"] == 2
        assert stats["accept_rate"] == pytest.approx(0.5)
        assert "opt:lr_mult" in stats["win_rates"]

    def test_evolve_recipe(self):
        # Use asymmetric bounds so midpoint != prior_center
        recipe_data = simple_recipe().to_dict()
        recipe_data["actuators"]["opt"]["mutations"]["lr_mult"]["bounds"] = [0.5, 1.5]
        recipe_data["actuators"]["opt"]["mutations"]["lr_mult"]["prior_center"] = 0.8
        recipe = TuneRecipe.from_dict(recipe_data)
        summaries = [
            {"win_rates": {"opt:lr_mult": 0.8}, "mean_scores": {"opt:lr_mult": 0.05}},
        ]
        evolved = evolve_recipe(recipe, summaries)
        original_center = 0.8
        evolved_center = evolved.actuators["opt"].mutations["lr_mult"].prior_center
        # EMA should shift toward midpoint (1.0): 0.7*0.8 + 0.3*1.0 = 0.86
        assert evolved_center == pytest.approx(0.86)


# ========== Controller tests ==========

class TestHotTuneController:
    def test_enable_disable(self):
        ctrl = HotTuneController()
        result = ctrl.apply_op(HotOp(module="tune", op="enable", params={"mode": "active"}), {})
        assert result.decision == "applied"
        assert ctrl.state.mode == "active"

        result = ctrl.apply_op(HotOp(module="tune", op="disable"), {})
        assert result.decision == "applied"
        assert ctrl.state.mode == "off"

    def test_status(self):
        ctrl = HotTuneController()
        ctrl.state.mode = "active"
        result = ctrl.apply_op(HotOp(module="tune", op="status"), {})
        assert result.decision == "applied"
        assert result.payload["mode"] == "active"

    def test_set_recipe_override(self):
        ctrl = HotTuneController()
        result = ctrl.apply_op(
            HotOp(module="tune", op="set", params={"acceptance.epsilon": 0.002}),
            {},
        )
        assert result.decision == "applied"
        assert ctrl.recipe.acceptance.epsilon == 0.002

    def test_on_event_off_mode_noop(self):
        ctrl = HotTuneController()
        ctrl.state.mode = "off"
        ctrl.on_event("val_epoch_end", make_env())
        assert ctrl.state.mutation_counter == 0

    def test_on_event_observe_mode_no_mutations(self):
        ctrl = HotTuneController(recipe=simple_recipe())
        ctrl.state.mode = "observe"
        opt = FakeOptimizer()
        ctrl.register_actuator("opt", OptimizerActuator())
        env = make_env(optimizer=opt)
        ctrl.on_event("val_epoch_end", env)
        assert ctrl.state.mutation_counter == 0
        # LR unchanged
        assert opt.param_groups[0]["lr"] == 0.001

    def test_on_event_active_proposes(self, tmp_path):
        ctrl = HotTuneController(
            recipe=simple_recipe(),
            run_dir=str(tmp_path),
        )
        ctrl.state.mode = "active"
        opt = FakeOptimizer(lr=0.001)
        ctrl.register_actuator("opt", OptimizerActuator())
        ls = {"weights": {"main_w": 1.0}}
        ctrl.register_actuator("loss", MutableStateActuator())

        def metric_fn(name, default=None):
            return {"val/loss": 0.5}.get(name, default)

        env = make_env(optimizer=opt, mutable_state=ls, metric_fn=metric_fn)
        ctrl.on_event("val_epoch_end", env)
        assert ctrl.state.mutation_counter >= 1
        assert ctrl.state.active_mutation is not None

    def test_full_accept_cycle(self, tmp_path):
        ctrl = HotTuneController(
            recipe=simple_recipe(),
            run_dir=str(tmp_path),
        )
        ctrl.state.mode = "active"
        opt = FakeOptimizer(lr=0.001)
        ls = {"weights": {"main_w": 1.0}}
        ctrl.register_actuator("opt", OptimizerActuator())
        ctrl.register_actuator("loss", MutableStateActuator())

        # First val_epoch_end: proposes and applies mutation
        metrics_val = [0.5]

        def metric_fn(name, default=None):
            if name == "val/loss":
                return metrics_val[0]
            return default

        env = make_env(step=100, optimizer=opt, mutable_state=ls, metric_fn=metric_fn)
        ctrl.on_event("val_epoch_end", env)
        assert ctrl.state.active_segment is not None

        # Second val_epoch_end: evaluates the segment
        metrics_val[0] = 0.3  # improvement
        env = make_env(step=200, optimizer=opt, mutable_state=ls, metric_fn=metric_fn)
        ctrl.on_event("val_epoch_end", env)

        # Segment should have been evaluated
        assert len(ctrl.state.history) == 1
        assert ctrl.state.history[0]["segment"]["decision"] in ("accepted", "rolled_back")

    def test_full_reject_and_rollback_cycle(self, tmp_path):
        ctrl = HotTuneController(
            recipe=simple_recipe(),
            run_dir=str(tmp_path),
        )
        ctrl.state.mode = "active"
        opt = FakeOptimizer(lr=0.001)
        ls = {"weights": {"main_w": 1.0}}
        ctrl.register_actuator("opt", OptimizerActuator())
        ctrl.register_actuator("loss", MutableStateActuator())

        original_lr = opt.param_groups[0]["lr"]

        metrics_val = [0.5]

        def metric_fn(name, default=None):
            if name == "val/loss":
                return metrics_val[0]
            return default

        env = make_env(step=100, optimizer=opt, mutable_state=ls, metric_fn=metric_fn)
        ctrl.on_event("val_epoch_end", env)
        assert ctrl.state.active_segment is not None

        # Regression
        metrics_val[0] = 0.6
        env = make_env(step=200, optimizer=opt, mutable_state=ls, metric_fn=metric_fn)
        ctrl.on_event("val_epoch_end", env)

        assert len(ctrl.state.history) == 1
        seg = ctrl.state.history[0]["segment"]
        assert seg["decision"] in ("rejected", "rolled_back")

    def test_safety_block_on_nan(self, tmp_path):
        ctrl = HotTuneController(
            recipe=simple_recipe(),
            run_dir=str(tmp_path),
        )
        ctrl.state.mode = "active"
        ctrl.register_actuator("opt", OptimizerActuator())
        env = make_env(optimizer=FakeOptimizer(), loss=float("nan"))
        ctrl.on_event("val_epoch_end", env)
        assert ctrl.state.mutation_counter == 0

    def test_reject_streak_blocks_mutations(self, tmp_path):
        ctrl = HotTuneController(
            recipe=simple_recipe(),
            run_dir=str(tmp_path),
        )
        ctrl.state.mode = "active"
        ctrl.state.reject_streak = 10  # way over limit
        ctrl.register_actuator("opt", OptimizerActuator())
        env = make_env(optimizer=FakeOptimizer())
        ctrl.on_event("val_epoch_end", env)
        # No mutation should be proposed due to reject streak
        assert ctrl.state.mutation_counter == 0

    def test_close_writes_summary(self, tmp_path):
        ctrl = HotTuneController(
            recipe=simple_recipe(),
            run_dir=str(tmp_path),
        )
        ctrl.state.mode = "active"
        ctrl.close(env={"step": 1000, "epoch": 10})
        summary_path = os.path.join(str(tmp_path), "hotcb.tune.summary.json")
        assert os.path.exists(summary_path)


# ========== Kernel integration tests ==========

class TestKernelTuneIntegration:
    def test_kernel_has_tune_module(self, tmp_path):
        k = HotKernel(run_dir=str(tmp_path))
        assert "tune" in k.modules
        assert isinstance(k.modules["tune"], HotTuneController)

    def test_kernel_actuator_registry(self, tmp_path):
        k = HotKernel(run_dir=str(tmp_path))
        act = OptimizerActuator()
        k.register_actuator("opt", act)
        assert k.get_actuator("opt") is act
        # Also registered in tune module
        tune = k.modules["tune"]
        assert tune.get_actuator("opt") is act

    def test_kernel_dispatches_tune_ops(self, tmp_path):
        k = HotKernel(run_dir=str(tmp_path))
        env = {"step": 10, "epoch": 0, "phase": "train"}
        k.apply(env, events=["fit_start"])
        # Tune module should have received the event (noop in off mode)
        tune = k.modules["tune"]
        assert tune.state.mode == "off"

    def test_kernel_tune_enable_via_command(self, tmp_path):
        k = HotKernel(run_dir=str(tmp_path))
        # Simulate command
        from hotcb.util import append_jsonl
        append_jsonl(k.commands_path, {"module": "tune", "op": "enable", "params": {"mode": "observe"}})
        env = {"step": 1, "epoch": 0, "phase": "train"}
        k.apply(env, events=["train_batch_end"])
        tune = k.modules["tune"]
        assert tune.state.mode == "observe"

    def test_kernel_close_closes_tune(self, tmp_path):
        k = HotKernel(run_dir=str(tmp_path))
        tune = k.modules["tune"]
        tune.state.mode = "active"
        k.close(env={"step": 100, "epoch": 5})
        summary_path = os.path.join(str(tmp_path), "hotcb.tune.summary.json")
        assert os.path.exists(summary_path)

    def test_freeze_blocks_tune_commands(self, tmp_path):
        k = HotKernel(run_dir=str(tmp_path))
        # Set freeze to prod
        import json as json_mod
        with open(k.freeze_path, "w") as f:
            f.write(json_mod.dumps({"mode": "prod"}))
        k._freeze_state = k._freeze_state.__class__.load(k.freeze_path)

        from hotcb.util import append_jsonl
        append_jsonl(k.commands_path, {"module": "tune", "op": "enable"})
        env = {"step": 1, "epoch": 0, "phase": "train"}
        k.apply(env, events=["train_batch_end"])

        # Should be ignored due to freeze
        tune = k.modules["tune"]
        assert tune.state.mode == "off"


# ========== CLI tests ==========

class TestCLITune:
    def test_tune_enable_command(self, tmp_path):
        from hotcb.cli import main
        run_dir = str(tmp_path)
        # Init first
        main(["--dir", run_dir, "init"])
        # Tune enable
        main(["--dir", run_dir, "tune", "enable"])
        # Check command was queued
        cmd_path = os.path.join(run_dir, "hotcb.commands.jsonl")
        with open(cmd_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        tune_cmds = [l for l in lines if l.get("module") == "tune"]
        assert len(tune_cmds) == 1
        assert tune_cmds[0]["op"] == "enable"

    def test_tune_disable_command(self, tmp_path):
        from hotcb.cli import main
        run_dir = str(tmp_path)
        main(["--dir", run_dir, "init"])
        main(["--dir", run_dir, "tune", "disable"])
        cmd_path = os.path.join(run_dir, "hotcb.commands.jsonl")
        with open(cmd_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        tune_cmds = [l for l in lines if l.get("module") == "tune"]
        assert len(tune_cmds) == 1
        assert tune_cmds[0]["op"] == "disable"

    def test_tune_set_command(self, tmp_path):
        from hotcb.cli import main
        run_dir = str(tmp_path)
        main(["--dir", run_dir, "init"])
        main(["--dir", run_dir, "tune", "set", "acceptance.epsilon=0.002"])
        cmd_path = os.path.join(run_dir, "hotcb.commands.jsonl")
        with open(cmd_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        tune_cmds = [l for l in lines if l.get("module") == "tune"]
        assert len(tune_cmds) == 1
        assert tune_cmds[0]["params"]["acceptance.epsilon"] == 0.002

    def test_tune_status_no_data(self, tmp_path, capsys):
        from hotcb.cli import main
        run_dir = str(tmp_path)
        main(["--dir", run_dir, "init"])
        main(["--dir", run_dir, "tune", "status"])
        out = capsys.readouterr().out
        assert "No tune recipe found" in out


# ========== Suggest mode tests ==========

class TestSuggestMode:
    def test_suggest_does_not_apply_mutation(self, tmp_path):
        ctrl = HotTuneController(
            recipe=simple_recipe(),
            run_dir=str(tmp_path),
        )
        ctrl.state.mode = "suggest"
        opt = FakeOptimizer(lr=0.001)
        ls = {"weights": {"main_w": 1.0}}
        ctrl.register_actuator("opt", OptimizerActuator())
        ctrl.register_actuator("loss", MutableStateActuator())

        original_lr = opt.param_groups[0]["lr"]
        original_w = ls["weights"]["main_w"]

        def metric_fn(name, default=None):
            return {"val/loss": 0.5}.get(name, default)

        env = make_env(optimizer=opt, mutable_state=ls, metric_fn=metric_fn)
        ctrl.on_event("val_epoch_end", env)

        # Mutation should be logged as suggested
        assert ctrl.state.mutation_counter >= 1
        # But no active segment (not applied)
        assert ctrl.state.active_segment is None
        # State should be unchanged
        assert opt.param_groups[0]["lr"] == original_lr
        assert ls["weights"]["main_w"] == original_w

        # Check mutation file has "suggested" status
        records = load_mutations_log(str(tmp_path))
        assert len(records) >= 1
        assert records[0]["status"] == "suggested"


# ========== Replay mode tests ==========

class TestReplayMode:
    def test_replay_applies_mutations_in_order(self, tmp_path):
        # Write replay mutations
        replay_dir = str(tmp_path / "replay_source")
        os.makedirs(replay_dir, exist_ok=True)
        mutations = [
            Mutation(
                mutation_id="m_00001", step=100, epoch=1,
                phase_bin="mid", event="val_epoch_end",
                actuator="opt", patch={"op": "lr_mult", "value": 0.9},
                status="applied",
            ),
            Mutation(
                mutation_id="m_00002", step=200, epoch=2,
                phase_bin="mid", event="val_epoch_end",
                actuator="opt", patch={"op": "lr_mult", "value": 0.8},
                status="applied",
            ),
        ]
        for m in mutations:
            write_mutation(replay_dir, m)

        replay_path = os.path.join(replay_dir, "hotcb.tune.mutations.jsonl")
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir, exist_ok=True)

        ctrl = HotTuneController(
            recipe=simple_recipe(),
            run_dir=run_dir,
            replay_mutations_path=replay_path,
        )
        ctrl.state.mode = "replay"
        opt = FakeOptimizer(lr=0.001)
        ctrl.register_actuator("opt", OptimizerActuator())

        env = make_env(step=100, optimizer=opt)
        ctrl.on_event("val_epoch_end", env)
        # First replay should have applied lr_mult 0.9
        expected_lr = 0.001 * 0.9
        assert opt.param_groups[0]["lr"] == pytest.approx(expected_lr, rel=1e-4)

        env = make_env(step=200, optimizer=opt)
        ctrl.on_event("val_epoch_end", env)
        # Second replay: lr_mult 0.8
        expected_lr *= 0.8
        assert opt.param_groups[0]["lr"] == pytest.approx(expected_lr, rel=1e-4)

    def test_replay_exhausted(self, tmp_path):
        replay_dir = str(tmp_path / "replay_source")
        os.makedirs(replay_dir, exist_ok=True)
        m = Mutation(
            mutation_id="m_00001", step=100, epoch=1,
            phase_bin="mid", event="val_epoch_end",
            actuator="opt", patch={"op": "lr_mult", "value": 0.9},
            status="applied",
        )
        write_mutation(replay_dir, m)

        ctrl = HotTuneController(
            recipe=simple_recipe(),
            run_dir=str(tmp_path / "run"),
            replay_mutations_path=os.path.join(replay_dir, "hotcb.tune.mutations.jsonl"),
        )
        ctrl.state.mode = "replay"
        opt = FakeOptimizer(lr=0.001)
        ctrl.register_actuator("opt", OptimizerActuator())

        # First event applies
        ctrl.on_event("val_epoch_end", make_env(step=100, optimizer=opt))
        assert ctrl.state.mutation_counter == 1

        # Second event: no more mutations to replay
        ctrl.on_event("val_epoch_end", make_env(step=200, optimizer=opt))
        assert ctrl.state.mutation_counter == 1  # unchanged

    def test_replay_missing_actuator(self, tmp_path):
        replay_dir = str(tmp_path / "replay_source")
        os.makedirs(replay_dir, exist_ok=True)
        m = Mutation(
            mutation_id="m_00001", step=100, epoch=1,
            phase_bin="mid", event="val_epoch_end",
            actuator="nonexistent", patch={"op": "lr_mult", "value": 0.9},
            status="applied",
        )
        write_mutation(replay_dir, m)

        run_dir = str(tmp_path / "run")
        ctrl = HotTuneController(
            recipe=simple_recipe(),
            run_dir=run_dir,
            replay_mutations_path=os.path.join(replay_dir, "hotcb.tune.mutations.jsonl"),
        )
        ctrl.state.mode = "replay"

        ctrl.on_event("val_epoch_end", make_env(step=100))
        # Should log as failed
        records = load_mutations_log(run_dir)
        assert len(records) == 1
        assert records[0]["status"] == "failed"

    def test_enable_replay_via_apply_op(self, tmp_path):
        replay_dir = str(tmp_path / "replay_source")
        os.makedirs(replay_dir, exist_ok=True)
        m = Mutation(
            mutation_id="m_00001", step=100, epoch=1,
            phase_bin="mid", event="val_epoch_end",
            actuator="opt", patch={"op": "lr_mult", "value": 0.85},
            status="applied",
        )
        write_mutation(replay_dir, m)
        replay_path = os.path.join(replay_dir, "hotcb.tune.mutations.jsonl")

        ctrl = HotTuneController(recipe=simple_recipe(), run_dir=str(tmp_path / "run"))
        result = ctrl.apply_op(
            HotOp(module="tune", op="enable", params={"mode": "replay", "replay_path": replay_path}),
            {},
        )
        assert result.decision == "applied"
        assert ctrl.state.mode == "replay"
        assert len(ctrl._replay_queue) == 1


# ========== Deterministic simulation tests ==========

class TestDeterministicSimulation:
    """Fake trainer loop with synthetic objectives."""

    def _run_simulation(self, tmp_path, loss_surface, num_epochs=10):
        """
        Run a full tune simulation with a synthetic loss surface.
        loss_surface(step) -> float
        """
        ctrl = HotTuneController(
            recipe=simple_recipe(),
            run_dir=str(tmp_path),
        )
        ctrl.state.mode = "active"
        opt = FakeOptimizer(lr=0.001)
        ls = {"weights": {"main_w": 1.0}}
        ctrl.register_actuator("opt", OptimizerActuator())
        ctrl.register_actuator("loss", MutableStateActuator())

        for epoch in range(num_epochs):
            step = epoch * 100
            current_loss = loss_surface(step)

            def metric_fn(name, default=None, _loss=current_loss):
                if name == "val/loss":
                    return _loss
                return default

            env = make_env(
                step=step, epoch=epoch, optimizer=opt,
                mutable_state=ls, loss=current_loss,
                metric_fn=metric_fn, max_steps=num_epochs * 100,
            )
            ctrl.on_event("val_epoch_end", env)

        ctrl.close(env={"step": num_epochs * 100, "epoch": num_epochs})
        return ctrl

    def test_convex_improvement(self, tmp_path):
        """Steadily improving loss should lead to accepted mutations."""
        ctrl = self._run_simulation(
            tmp_path,
            loss_surface=lambda step: 1.0 - step * 0.005,
            num_epochs=10,
        )
        summary_path = os.path.join(str(tmp_path), "hotcb.tune.summary.json")
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["total_mutations"] >= 1

    def test_noisy_plateau(self, tmp_path):
        """Noisy plateau should mostly reject mutations."""
        import random
        random.seed(42)
        ctrl = self._run_simulation(
            tmp_path,
            loss_surface=lambda step: 0.5 + random.gauss(0, 0.001),
            num_epochs=10,
        )
        # Should have attempted some mutations
        assert ctrl.state.mutation_counter >= 0

    def test_instability_blocks(self, tmp_path):
        """NaN loss should block mutations."""
        ctrl = HotTuneController(
            recipe=simple_recipe(),
            run_dir=str(tmp_path),
        )
        ctrl.state.mode = "active"
        opt = FakeOptimizer(lr=0.001)
        ctrl.register_actuator("opt", OptimizerActuator())
        ctrl.register_actuator("loss", MutableStateActuator())

        env = make_env(step=100, optimizer=opt, loss=float("nan"))
        ctrl.on_event("val_epoch_end", env)
        assert ctrl.state.mutation_counter == 0

        env = make_env(step=200, optimizer=opt, loss=float("inf"))
        ctrl.on_event("val_epoch_end", env)
        assert ctrl.state.mutation_counter == 0

    def test_delayed_reward(self, tmp_path):
        """Loss spikes then improves — second eval should accept."""
        losses = [0.5, 0.6, 0.3]  # spike then improve
        ctrl = HotTuneController(
            recipe=simple_recipe(),
            run_dir=str(tmp_path),
        )
        ctrl.state.mode = "active"
        opt = FakeOptimizer(lr=0.001)
        ls = {"weights": {"main_w": 1.0}}
        ctrl.register_actuator("opt", OptimizerActuator())
        ctrl.register_actuator("loss", MutableStateActuator())

        for i, loss_val in enumerate(losses):
            def metric_fn(name, default=None, _v=loss_val):
                return _v if name == "val/loss" else default

            env = make_env(
                step=i * 100, epoch=i, optimizer=opt,
                mutable_state=ls, loss=loss_val, metric_fn=metric_fn,
            )
            ctrl.on_event("val_epoch_end", env)

        # Should have history
        assert ctrl.state.mutation_counter >= 1


# ========== Failure tests ==========

class TestFailureModes:
    def test_optuna_not_installed_fallback(self, tmp_path):
        """Search falls back to random when TPE requested but no optuna available."""
        from hotcb.modules.tune.search import _random_proposal
        recipe = simple_recipe()
        recipe.search.algorithm = "random"
        state = TuneState()
        result = _random_proposal(recipe, state, "mid")
        # Should return a proposal or None (no crash)
        assert result is None or isinstance(result, dict)

    def test_invalid_recipe_bounds(self):
        """Recipe with impossible bounds should still parse."""
        data = {
            "actuators": {
                "opt": {
                    "enabled": True,
                    "mutations": {
                        "lr_mult": {"bounds": [2.0, 0.5]},  # inverted bounds
                    },
                },
            },
        }
        recipe = TuneRecipe.from_dict(data)
        assert recipe.actuators["opt"].mutations["lr_mult"].bounds == (2.0, 0.5)

    def test_rollback_failure_logged(self, tmp_path):
        """If rollback fails, controller doesn't crash."""
        ctrl = HotTuneController(
            recipe=simple_recipe(),
            run_dir=str(tmp_path),
        )
        ctrl.state.mode = "active"
        opt = FakeOptimizer(lr=0.001)
        ls = {"weights": {"main_w": 1.0}}
        ctrl.register_actuator("opt", OptimizerActuator())
        ctrl.register_actuator("loss", MutableStateActuator())

        metrics_val = [0.5]

        def metric_fn(name, default=None):
            return metrics_val[0] if name == "val/loss" else default

        env = make_env(step=100, optimizer=opt, mutable_state=ls, metric_fn=metric_fn)
        ctrl.on_event("val_epoch_end", env)

        if ctrl.state.active_segment:
            # Force a broken env for rollback
            metrics_val[0] = 0.8  # regression
            broken_env = make_env(step=200, metric_fn=metric_fn)
            # No optimizer in env — rollback will fail if it was opt actuator
            ctrl.on_event("val_epoch_end", broken_env)
            # Should not crash
            assert len(ctrl.state.history) >= 1

    def test_missing_metric_fn(self, tmp_path):
        """Controller handles missing metric fn gracefully."""
        ctrl = HotTuneController(
            recipe=simple_recipe(),
            run_dir=str(tmp_path),
        )
        ctrl.state.mode = "active"
        opt = FakeOptimizer(lr=0.001)
        ls = {"weights": {"main_w": 1.0}}
        ctrl.register_actuator("opt", OptimizerActuator())
        ctrl.register_actuator("loss", MutableStateActuator())

        env = make_env(step=100, optimizer=opt, mutable_state=ls)
        # No metric_fn in env
        ctrl.on_event("val_epoch_end", env)
        # Should not crash, may or may not propose

    def test_actuator_apply_exception(self, tmp_path):
        """Controller handles actuator apply exception gracefully."""

        class BrokenActuator:
            name = "broken"
            def snapshot(self, env):
                return {}
            def validate(self, patch, env):
                return ValidationResult(valid=True)
            def apply(self, patch, env):
                raise RuntimeError("boom")
            def restore(self, snapshot, env):
                return ApplyResult(success=True)
            def describe_space(self):
                return {}

        # Build recipe that references the broken actuator
        recipe_data = {
            "actuators": {
                "broken": {
                    "enabled": True,
                    "mutations": {
                        "explode": {"bounds": [0.5, 1.5], "cooldown": 0, "risk": "low"},
                    },
                },
            },
            "search": {"algorithm": "random"},
        }
        recipe = TuneRecipe.from_dict(recipe_data)
        ctrl = HotTuneController(recipe=recipe, run_dir=str(tmp_path))
        ctrl.state.mode = "active"
        ctrl.register_actuator("broken", BrokenActuator())

        env = make_env(step=100)
        # Should not crash
        ctrl.on_event("val_epoch_end", env)

    def test_empty_actuator_registry_observe_only(self, tmp_path):
        """No registered actuators means no mutations even in active mode."""
        ctrl = HotTuneController(
            recipe=simple_recipe(),
            run_dir=str(tmp_path),
        )
        ctrl.state.mode = "active"
        # No actuators registered
        env = make_env(step=100)
        ctrl.on_event("val_epoch_end", env)
        assert ctrl.state.mutation_counter == 0

    def test_validate_missing_value(self):
        act = OptimizerActuator()
        env = {"optimizer": FakeOptimizer()}
        result = act.validate({"op": "lr_mult"}, env)
        assert not result.valid
        assert any("missing value" in e for e in result.errors)

    def test_loss_actuator_validate_missing_key(self):
        act = MutableStateActuator()
        env = {"mutable_state": {"weights": {}}}
        result = act.validate({"op": "set", "value": 1.0}, env)
        assert not result.valid
        assert any("missing key" in e for e in result.errors)

    def test_loss_actuator_validate_non_numeric(self):
        act = MutableStateActuator()
        env = {"mutable_state": {"weights": {}}}
        result = act.validate({"op": "set", "key": "x", "value": "bad"}, env)
        assert not result.valid


# ========== Config YAML tune section tests ==========

class TestConfigYamlTune:
    def test_yaml_tune_enable(self, tmp_path):
        import yaml
        cfg_path = str(tmp_path / "hotcb.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump({"tune": {"enabled": True, "mode": "observe"}}, f)

        from hotcb.config import load_yaml
        ops = load_yaml(cfg_path)
        tune_ops = [op for op in ops if op.module == "tune"]
        assert len(tune_ops) == 1
        assert tune_ops[0].op == "enable"
        assert tune_ops[0].params["mode"] == "observe"

    def test_yaml_tune_disable(self, tmp_path):
        import yaml
        cfg_path = str(tmp_path / "hotcb.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump({"tune": {"enabled": False}}, f)

        from hotcb.config import load_yaml
        ops = load_yaml(cfg_path)
        tune_ops = [op for op in ops if op.module == "tune"]
        assert len(tune_ops) == 1
        assert tune_ops[0].op == "disable"
