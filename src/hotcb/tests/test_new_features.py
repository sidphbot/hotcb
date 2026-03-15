"""Tests for recently added hotcb features:

1. Replay source capture fallback (cb.py)
2. Traceback in ledger entries (opt, loss, kernel)
3. Strict replay policy enforcement (kernel.close())
4. New CLI commands (status, sugar enable/disable/set, recipe validate)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import textwrap
from types import SimpleNamespace

import pytest

from hotcb.kernel import HotKernel
from hotcb.modules.cb import CallbackModule, _capture_source
from hotcb.actuators import optimizer_actuators, loss_actuators, mutable_state, ApplyResult
from hotcb.actuators.actuator import ActuatorType, HotcbActuator
from hotcb.ops import HotOp
from hotcb.cli import (
    cmd_status,
    cmd_sugar_enable,
    cmd_sugar_disable,
    cmd_sugar_set,
    cmd_recipe_validate,
    _infer_module,
)
from hotcb.ops import CallbackTarget


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

MINIMAL_CB_SRC = textwrap.dedent("""\
    class TestCb:
        def __init__(self, **kwargs):
            pass
        def handle(self, event, env):
            pass
        def set_params(self, **kwargs):
            pass
""")


def _write_cb_file(directory: str, source: str, name: str = "cb.py") -> str:
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(source)
    return path


def _sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mock_optimizer(lr: float = 0.01, weight_decay: float = 0.0):
    return SimpleNamespace(param_groups=[{"lr": lr, "weight_decay": weight_decay}])


# ===========================================================================
# 1. Replay source capture fallback
# ===========================================================================

class TestReplaySourceCaptureFallback:
    """When source=replay, cb module should use captured_path from raw record."""

    def test_replay_overrides_target_path_with_captured(
        self, run_dir, make_env, write_recipe, write_freeze, read_ledger, tmp_path,
    ):
        """If op.source=replay and captured_path exists, cb_op.target.path is overridden."""
        # Write a callback source file and capture it
        cb_path = _write_cb_file(str(tmp_path), MINIMAL_CB_SRC)
        sources_dir = os.path.join(run_dir, "hotcb.sources")
        capture = _capture_source(cb_path, sources_dir)
        assert capture is not None
        captured_path = capture["captured_path"]

        # Create a recipe entry that references the captured source
        recipe_path = write_recipe({
            "at": {"step": 1, "event": "train_step_end"},
            "module": "cb",
            "op": "load",
            "id": "replayed_cb",
            "target": {"kind": "python_file", "path": cb_path, "symbol": "TestCb"},
            "source_capture": {
                "sha256": capture["sha256"],
                "captured_path": captured_path,
            },
        })
        write_freeze(mode="replay", recipe_path=recipe_path)

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        env = make_env(step=1)
        kernel.apply(env, events=["train_step_end"])

        ledger = read_ledger()
        load_entries = [e for e in ledger if e.get("op") == "load"]
        assert len(load_entries) == 1
        assert load_entries[0]["decision"] == "applied"
        assert load_entries[0]["source"] == "replay"

    def test_replay_capture_missing_fallback(
        self, run_dir, make_env, write_recipe, write_freeze, read_ledger, tmp_path,
    ):
        """If captured_path doesn't exist, payload should contain capture_missing_fallback."""
        cb_path = _write_cb_file(str(tmp_path), MINIMAL_CB_SRC)
        nonexistent_captured = os.path.join(str(tmp_path), "no_such_file.py")

        recipe_path = write_recipe({
            "at": {"step": 1, "event": "train_step_end"},
            "module": "cb",
            "op": "load",
            "id": "replayed_cb",
            "target": {"kind": "python_file", "path": cb_path, "symbol": "TestCb"},
            "source_capture": {
                "sha256": "deadbeef",
                "captured_path": nonexistent_captured,
            },
        })
        write_freeze(mode="replay", recipe_path=recipe_path)

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        env = make_env(step=1)
        kernel.apply(env, events=["train_step_end"])

        ledger = read_ledger()
        load_entries = [e for e in ledger if e.get("op") == "load"]
        assert len(load_entries) == 1
        payload = load_entries[0].get("payload", {})
        assert payload.get("capture_missing_fallback") is True

    def test_external_source_capture_still_works(
        self, run_dir, make_env, write_commands, read_ledger, tmp_path,
    ):
        """When source != replay (normal external load), source capture behavior is unchanged."""
        cb_path = _write_cb_file(str(tmp_path), MINIMAL_CB_SRC)
        write_commands({
            "module": "cb",
            "op": "load",
            "id": "ext_cb",
            "target": {"kind": "python_file", "path": cb_path, "symbol": "TestCb"},
        })

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        env = make_env(step=1)
        kernel.apply(env, events=["train_step_end"])

        ledger = read_ledger()
        load_entries = [e for e in ledger if e.get("op") == "load"]
        assert len(load_entries) == 1
        assert load_entries[0]["decision"] == "applied"
        assert load_entries[0]["source"] == "external"

        payload = load_entries[0].get("payload", {})
        assert "source_capture" in payload
        sc = payload["source_capture"]
        expected_sha = _sha256_of(MINIMAL_CB_SRC.encode("utf-8"))
        assert sc["sha256"] == expected_sha
        assert os.path.isfile(sc["captured_path"])


# ===========================================================================
# 2. Traceback in ledger entries
# ===========================================================================

class TestTracebackInModuleResult:
    """When set_params fails with an exception, error info is recorded."""

    def test_opt_apply_fn_failure_has_error(self):
        """Optimizer actuator apply_fn failure produces error in ApplyResult."""
        def _bad_apply(value, env):
            raise RuntimeError("gpu exploded")

        bad_act = HotcbActuator(
            param_key="bad_lr",
            type=ActuatorType.FLOAT,
            apply_fn=_bad_apply,
            min_value=0.0,
            max_value=1.0,
            current_value=0.01,
        )
        from hotcb.actuators.state import MutableState
        ms = MutableState([bad_act])
        result = ms.apply("bad_lr", 0.001, {}, step=1)
        assert not result.success
        assert result.error is not None
        assert "gpu exploded" in result.error

    def test_loss_apply_fn_failure_has_error(self):
        """Loss actuator apply_fn failure produces error in ApplyResult."""
        def _bad_apply(value, env):
            raise TypeError("intentional error for test")

        bad_act = HotcbActuator(
            param_key="bad_weight",
            type=ActuatorType.FLOAT,
            apply_fn=_bad_apply,
            min_value=0.0,
            max_value=100.0,
            current_value=1.0,
        )
        from hotcb.actuators.state import MutableState
        ms = MutableState([bad_act])
        result = ms.apply("bad_weight", 0.5, {}, step=1)
        assert not result.success
        assert "intentional error" in result.error

    def test_kernel_ledger_has_error_on_opt_failure(
        self, run_dir, make_env, write_commands, read_ledger,
    ):
        """When opt set_params fails through the kernel, ledger entry has error."""
        def _bad_apply(value, env):
            raise RuntimeError("gpu exploded")

        bad_act = HotcbActuator(
            param_key="lr",
            type=ActuatorType.FLOAT,
            apply_fn=_bad_apply,
            min_value=0.0,
            max_value=1.0,
            current_value=0.01,
        )
        ms = mutable_state([bad_act])

        write_commands({
            "module": "opt",
            "op": "set_params",
            "params": {"key": "lr", "value": 0.001},
        })
        kernel = HotKernel(run_dir=run_dir, debounce_steps=1, mutable_state=ms)
        env = make_env(step=1)
        kernel.apply(env, ["train_step_end"])

        ledger = read_ledger()
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "failed"
        assert ledger[0]["error"] is not None
        assert "gpu exploded" in ledger[0]["error"]

    def test_kernel_ledger_has_error_on_loss_failure(
        self, run_dir, make_env, write_commands, read_ledger,
    ):
        """When loss set_params fails through the kernel, ledger entry has error."""
        def _bad_apply(value, env):
            raise TypeError("intentional error for test")

        bad_act = HotcbActuator(
            param_key="kl",
            type=ActuatorType.FLOAT,
            apply_fn=_bad_apply,
            min_value=0.0,
            max_value=100.0,
            current_value=1.0,
        )
        ms = mutable_state([bad_act])

        write_commands({
            "module": "loss",
            "op": "set_params",
            "params": {"key": "kl", "value": 0.5},
        })
        kernel = HotKernel(run_dir=run_dir, debounce_steps=1, mutable_state=ms)
        env = make_env(step=1)
        kernel.apply(env, ["train_step_end"])

        ledger = read_ledger()
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "failed"
        assert ledger[0]["error"] is not None
        assert "intentional error" in ledger[0]["error"]

    def test_successful_op_has_no_traceback(
        self, run_dir, make_env, write_commands, read_ledger,
    ):
        """Successful ops should have traceback=None in the ledger."""
        optimizer = _mock_optimizer(lr=0.01)
        ms = mutable_state(optimizer_actuators(optimizer))

        write_commands({
            "module": "opt",
            "op": "set_params",
            "params": {"lr": 0.001},
        })
        kernel = HotKernel(run_dir=run_dir, debounce_steps=1, mutable_state=ms)
        env = make_env(step=1, optimizer=optimizer)
        kernel.apply(env, ["train_step_end"])

        ledger = read_ledger()
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "applied"
        assert ledger[0]["traceback"] is None


# ===========================================================================
# 3. Strict replay policy enforcement (kernel.close())
# ===========================================================================

class TestStrictReplayPolicyClose:
    """kernel.close() enforces replay policy on remaining recipe entries."""

    def test_strict_policy_raises_on_remaining_entries(
        self, run_dir, make_env, write_recipe, write_freeze, read_ledger,
    ):
        """With policy=strict and unconsumed entries, close() raises RuntimeError."""
        recipe_path = write_recipe(
            {
                "at": {"step": 100, "event": "train_step_end"},
                "module": "opt",
                "op": "set_params",
                "id": "main",
                "params": {"lr": 1e-5},
            },
            {
                "at": {"step": 200, "event": "train_step_end"},
                "module": "opt",
                "op": "set_params",
                "id": "main",
                "params": {"lr": 2e-5},
            },
        )
        write_freeze(mode="replay", recipe_path=recipe_path, policy="strict")

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        optimizer = _mock_optimizer(lr=0.01)

        # Only run up to step 5 -- recipe entries at 100 and 200 are not consumed
        for step in range(1, 6):
            env = make_env(step=step, optimizer=optimizer)
            kernel.apply(env, ["train_step_end"])

        with pytest.raises(RuntimeError, match="Strict replay policy"):
            kernel.close(env=make_env(step=5))

        ledger = read_ledger()
        missed = [e for e in ledger if e.get("notes") == "strict_policy_missed"]
        assert len(missed) == 2
        for entry in missed:
            assert entry["decision"] == "failed"
            assert "missed_step" in entry["error"]

    def test_best_effort_does_not_raise_on_remaining(
        self, run_dir, make_env, write_recipe, write_freeze, read_ledger,
    ):
        """With policy=best_effort and unconsumed entries, close() logs but does not raise."""
        recipe_path = write_recipe(
            {
                "at": {"step": 100, "event": "train_step_end"},
                "module": "opt",
                "op": "set_params",
                "id": "main",
                "params": {"lr": 1e-5},
            },
        )
        write_freeze(mode="replay", recipe_path=recipe_path, policy="best_effort")

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        optimizer = _mock_optimizer(lr=0.01)

        for step in range(1, 6):
            env = make_env(step=step, optimizer=optimizer)
            kernel.apply(env, ["train_step_end"])

        # Should not raise
        kernel.close(env=make_env(step=5))

        ledger = read_ledger()
        missed = [e for e in ledger if e.get("notes") == "best_effort_missed"]
        assert len(missed) == 1
        assert missed[0]["decision"] == "failed"
        assert "missed_step" in missed[0]["error"]

    def test_all_entries_consumed_close_is_noop(
        self, run_dir, make_env, write_recipe, write_freeze, read_ledger,
    ):
        """When all recipe entries are consumed, close() does nothing."""
        recipe_path = write_recipe(
            {
                "at": {"step": 2, "event": "train_step_end"},
                "module": "opt",
                "op": "set_params",
                "id": "main",
                "params": {"lr": 5e-5},
            },
        )
        write_freeze(mode="replay", recipe_path=recipe_path, policy="strict")

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        optimizer = _mock_optimizer(lr=0.01)

        # Run past the recipe entry at step 2
        for step in range(1, 5):
            env = make_env(step=step, optimizer=optimizer)
            kernel.apply(env, ["train_step_end"])

        # All entries consumed; close should not raise
        ledger_before = read_ledger()
        kernel.close(env=make_env(step=4))
        ledger_after = read_ledger()

        # No new ledger entries from close()
        assert len(ledger_after) == len(ledger_before)

    def test_non_replay_mode_close_is_noop(
        self, run_dir, make_env, read_ledger,
    ):
        """When mode is not replay, close() does nothing."""
        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)

        # Mode is 'off' by default
        kernel.close(env=make_env(step=10))

        ledger = read_ledger()
        assert len(ledger) == 0

    def test_strict_close_ledger_entries_have_correct_source(
        self, run_dir, make_env, write_recipe, write_freeze, read_ledger,
    ):
        """Missed entries written by close() have source=replay."""
        recipe_path = write_recipe(
            {
                "at": {"step": 50, "event": "train_step_end"},
                "module": "loss",
                "op": "set_params",
                "id": "main",
                "params": {"kl_w": 0.1},
            },
        )
        write_freeze(mode="replay", recipe_path=recipe_path, policy="strict")

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        env = make_env(step=1)
        kernel.apply(env, ["train_step_end"])

        with pytest.raises(RuntimeError):
            kernel.close(env=make_env(step=1))

        ledger = read_ledger()
        missed = [e for e in ledger if e.get("notes") == "strict_policy_missed"]
        assert len(missed) == 1
        assert missed[0]["source"] == "replay"
        assert missed[0]["module"] == "loss"


# ===========================================================================
# 4. New CLI commands
# ===========================================================================

class TestCmdStatus:
    """Tests for the status CLI command."""

    def test_status_prints_freeze_mode(self, tmp_path, capsys):
        """cmd_status reads freeze state and prints mode."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir, exist_ok=True)

        freeze_path = os.path.join(run_dir, "hotcb.freeze.json")
        with open(freeze_path, "w", encoding="utf-8") as f:
            json.dump({"mode": "prod", "policy": "strict"}, f)

        args = argparse.Namespace(dir=run_dir)
        cmd_status(args)

        captured = capsys.readouterr()
        assert "Freeze mode: prod" in captured.out
        assert "policy: strict" in captured.out

    def test_status_prints_applied_ledger_summary(self, tmp_path, capsys):
        """cmd_status reads applied ledger and prints per-handle summary."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir, exist_ok=True)

        # Create empty freeze file
        freeze_path = os.path.join(run_dir, "hotcb.freeze.json")
        with open(freeze_path, "w", encoding="utf-8") as f:
            f.write("{}")

        applied_path = os.path.join(run_dir, "hotcb.applied.jsonl")
        with open(applied_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "seq": 1, "step": 10, "event": "train_step_end",
                "module": "opt", "op": "set_params", "id": "main",
                "decision": "applied",
            }) + "\n")
            f.write(json.dumps({
                "seq": 2, "step": 20, "event": "train_step_end",
                "module": "cb", "op": "enable", "id": "my_cb",
                "decision": "applied",
            }) + "\n")

        args = argparse.Namespace(dir=run_dir)
        cmd_status(args)

        captured = capsys.readouterr()
        assert "Freeze mode: off" in captured.out
        assert "2 handles" in captured.out
        assert "opt:main" in captured.out
        assert "cb:my_cb" in captured.out

    def test_status_no_applied_ledger(self, tmp_path, capsys):
        """cmd_status without applied ledger prints appropriate message."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir, exist_ok=True)

        # Empty freeze file
        freeze_path = os.path.join(run_dir, "hotcb.freeze.json")
        with open(freeze_path, "w", encoding="utf-8") as f:
            f.write("{}")

        args = argparse.Namespace(dir=run_dir)
        cmd_status(args)

        captured = capsys.readouterr()
        assert "No applied ledger" in captured.out


class TestCmdSugarEnable:
    """Tests for the sugar enable CLI command."""

    def test_writes_cb_enable_command(self, tmp_path):
        """cmd_sugar_enable writes module=cb, op=enable to commands file."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir, exist_ok=True)

        args = argparse.Namespace(dir=run_dir, id="my_callback")
        cmd_sugar_enable(args)

        cmd_path = os.path.join(run_dir, "hotcb.commands.jsonl")
        with open(cmd_path, "r") as f:
            rec = json.loads(f.readline())

        assert rec["module"] == "cb"
        assert rec["op"] == "enable"
        assert rec["id"] == "my_callback"


class TestCmdSugarDisable:
    """Tests for the sugar disable CLI command."""

    def test_writes_cb_disable_command(self, tmp_path):
        """cmd_sugar_disable writes module=cb, op=disable to commands file."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir, exist_ok=True)

        args = argparse.Namespace(dir=run_dir, id="my_callback")
        cmd_sugar_disable(args)

        cmd_path = os.path.join(run_dir, "hotcb.commands.jsonl")
        with open(cmd_path, "r") as f:
            rec = json.loads(f.readline())

        assert rec["module"] == "cb"
        assert rec["op"] == "disable"
        assert rec["id"] == "my_callback"


class TestCmdSugarSet:
    """Tests for the sugar set CLI command (auto-routing)."""

    def test_routes_lr_to_opt(self, tmp_path):
        """Setting lr=1e-4 auto-routes to opt module."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir, exist_ok=True)

        args = argparse.Namespace(dir=run_dir, id="main", kv=["lr=0.0001"])
        cmd_sugar_set(args)

        cmd_path = os.path.join(run_dir, "hotcb.commands.jsonl")
        with open(cmd_path, "r") as f:
            rec = json.loads(f.readline())

        assert rec["module"] == "opt"
        assert rec["op"] == "set_params"
        assert rec["params"]["lr"] == pytest.approx(1e-4)

    def test_routes_distill_w_to_loss(self, tmp_path):
        """Setting distill_w=0.2 auto-routes to loss module (ends with _w)."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir, exist_ok=True)

        args = argparse.Namespace(dir=run_dir, id="main", kv=["distill_w=0.2"])
        cmd_sugar_set(args)

        cmd_path = os.path.join(run_dir, "hotcb.commands.jsonl")
        with open(cmd_path, "r") as f:
            rec = json.loads(f.readline())

        assert rec["module"] == "loss"
        assert rec["op"] == "set_params"
        assert rec["params"]["distill_w"] == pytest.approx(0.2)

    def test_ambiguous_keys_raises(self, tmp_path):
        """Keys that match neither opt nor loss should raise SystemExit."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir, exist_ok=True)

        args = argparse.Namespace(dir=run_dir, id="main", kv=["unknown_param=42"])
        with pytest.raises(SystemExit, match="Cannot auto-route"):
            cmd_sugar_set(args)

    def test_infer_module_opt_keys(self):
        """_infer_module correctly identifies opt keys."""
        assert _infer_module({"lr"}) == "opt"
        assert _infer_module({"weight_decay"}) == "opt"
        assert _infer_module({"clip_norm"}) == "opt"
        assert _infer_module({"scheduler_scale"}) == "opt"

    def test_infer_module_loss_keys(self):
        """_infer_module correctly identifies loss keys."""
        assert _infer_module({"kl_w"}) == "loss"
        assert _infer_module({"terms.main"}) == "loss"
        assert _infer_module({"ramps.warmup"}) == "loss"


class TestCmdRecipeValidate:
    """Tests for the recipe validate CLI command."""

    def test_valid_recipe_passes(self, tmp_path, capsys):
        """A well-formed recipe should pass validation."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir, exist_ok=True)

        recipe_path = os.path.join(run_dir, "recipe.jsonl")
        with open(recipe_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "at": {"step": 10, "event": "train_step_end"},
                "module": "opt",
                "op": "set_params",
                "params": {"lr": 0.001},
            }) + "\n")
            f.write(json.dumps({
                "at": {"step": 20, "event": "train_step_end"},
                "module": "loss",
                "op": "set_params",
                "params": {"kl_w": 0.5},
            }) + "\n")

        args = argparse.Namespace(dir=run_dir, recipe=recipe_path)
        cmd_recipe_validate(args)

        captured = capsys.readouterr()
        assert "2 entries, valid" in captured.out

    def test_invalid_recipe_raises(self, tmp_path):
        """A recipe with missing required fields should raise SystemExit."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir, exist_ok=True)

        recipe_path = os.path.join(run_dir, "recipe.jsonl")
        with open(recipe_path, "w", encoding="utf-8") as f:
            # Missing 'at' and 'op' fields
            f.write(json.dumps({"module": "opt"}) + "\n")

        args = argparse.Namespace(dir=run_dir, recipe=recipe_path)
        with pytest.raises(SystemExit):
            cmd_recipe_validate(args)

    def test_missing_recipe_file_raises(self, tmp_path):
        """Validating a non-existent recipe should raise SystemExit."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir, exist_ok=True)

        args = argparse.Namespace(dir=run_dir, recipe=os.path.join(run_dir, "no_such.jsonl"))
        with pytest.raises(SystemExit):
            cmd_recipe_validate(args)

    def test_invalid_json_in_recipe_raises(self, tmp_path):
        """A recipe with malformed JSON should raise SystemExit."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir, exist_ok=True)

        recipe_path = os.path.join(run_dir, "recipe.jsonl")
        with open(recipe_path, "w", encoding="utf-8") as f:
            f.write("not valid json\n")

        args = argparse.Namespace(dir=run_dir, recipe=recipe_path)
        with pytest.raises(SystemExit):
            cmd_recipe_validate(args)

    def test_invalid_module_in_recipe_raises(self, tmp_path):
        """A recipe with an invalid module name should raise SystemExit."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir, exist_ok=True)

        recipe_path = os.path.join(run_dir, "recipe.jsonl")
        with open(recipe_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "at": {"step": 1, "event": "train_step_end"},
                "module": "core",  # core is not allowed in recipe
                "op": "freeze",
            }) + "\n")

        args = argparse.Namespace(dir=run_dir, recipe=recipe_path)
        with pytest.raises(SystemExit):
            cmd_recipe_validate(args)

    def test_missing_step_in_at_raises(self, tmp_path):
        """A recipe entry where 'at' lacks 'step' should raise SystemExit."""
        run_dir = str(tmp_path / "run")
        os.makedirs(run_dir, exist_ok=True)

        recipe_path = os.path.join(run_dir, "recipe.jsonl")
        with open(recipe_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "at": {"event": "train_step_end"},  # missing 'step'
                "module": "opt",
                "op": "set_params",
            }) + "\n")

        args = argparse.Namespace(dir=run_dir, recipe=recipe_path)
        with pytest.raises(SystemExit):
            cmd_recipe_validate(args)
