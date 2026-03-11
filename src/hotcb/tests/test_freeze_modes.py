"""Unit tests for hotcb freeze modes (spec 19.3)."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from hotcb.kernel import HotKernel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_optimizer(lr: float = 1e-4, weight_decay: float = 0.0):
    """Return a minimal mock optimizer with param_groups."""
    return SimpleNamespace(param_groups=[{"lr": lr, "weight_decay": weight_decay}])


def _write_adjust(run_dir: str, patches: list) -> str:
    """Write an adjustment overlay JSON file and return its path."""
    path = os.path.join(run_dir, "hotcb.adjust.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"patches": patches}, f)
    return path


# ---------------------------------------------------------------------------
# 1. Freeze prod -- external ops are ignored
# ---------------------------------------------------------------------------

class TestFreezeProd:
    def test_external_opt_ignored(self, run_dir, make_env, write_commands, write_freeze, read_ledger):
        """In prod mode, an external opt set_params command is ignored."""
        write_freeze(mode="prod")

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)

        opt = _make_optimizer(lr=1e-4)
        env = make_env(step=1, optimizer=opt)

        # Append an external opt command after kernel init so cursor sees it
        write_commands({"module": "opt", "op": "set_params", "id": "main", "params": {"lr": 5e-4}})

        kernel.apply(env, ["train_step_end"])

        # Optimizer state must be unchanged
        assert opt.param_groups[0]["lr"] == 1e-4

        # Ledger must record decision="ignored_freeze"
        ledger = read_ledger()
        opt_entries = [e for e in ledger if e["module"] == "opt"]
        assert len(opt_entries) == 1
        assert opt_entries[0]["decision"] == "ignored_freeze"
        assert opt_entries[0]["source"] == "external"


# ---------------------------------------------------------------------------
# 2. Freeze prod allows core ops
# ---------------------------------------------------------------------------

class TestFreezeProdAllowsCore:
    def test_core_freeze_unfreeze_in_prod(self, run_dir, make_env, write_commands, write_freeze, read_ledger):
        """Core ops (freeze/unfreeze) are still applied even in prod mode."""
        write_freeze(mode="prod")

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        env = make_env(step=1)

        # Send a core unfreeze op
        write_commands({"module": "core", "op": "unfreeze"})
        kernel.apply(env, ["train_step_end"])

        ledger = read_ledger()
        core_entries = [e for e in ledger if e["module"] == "core"]
        assert len(core_entries) == 1
        assert core_entries[0]["decision"] == "applied"
        assert core_entries[0]["op"] == "unfreeze"

    def test_core_freeze_op_in_prod(self, run_dir, make_env, write_commands, write_freeze, read_ledger):
        """A core freeze op (changing mode) works even under existing prod freeze."""
        write_freeze(mode="prod")

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        env = make_env(step=1)

        # Send a core freeze op to switch to replay mode
        write_commands({"module": "core", "op": "freeze", "mode": "replay"})
        kernel.apply(env, ["train_step_end"])

        ledger = read_ledger()
        core_entries = [e for e in ledger if e["module"] == "core"]
        assert len(core_entries) == 1
        assert core_entries[0]["decision"] == "applied"


# ---------------------------------------------------------------------------
# 3. Freeze replay -- recipe ops applied, external ops ignored
# ---------------------------------------------------------------------------

class TestFreezeReplay:
    def test_replay_applies_recipe_ignores_external(
        self, run_dir, make_env, write_commands, write_recipe, write_freeze, read_ledger
    ):
        """
        In replay mode:
        - recipe ops at matching step are applied (source=replay, decision=applied)
        - external ops are ignored (decision=ignored_replay)
        """
        recipe_path = write_recipe(
            {
                "at": {"step": 3, "event": "train_step_end"},
                "module": "opt",
                "op": "set_params",
                "id": "main",
                "params": {"lr": 3e-5},
            }
        )
        write_freeze(mode="replay", recipe_path=recipe_path)

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)

        opt = _make_optimizer(lr=1e-4)

        # Append an external conflicting opt command
        write_commands({"module": "opt", "op": "set_params", "id": "main", "params": {"lr": 9e-4}})

        # Run steps 1..5
        for step in range(1, 6):
            env = make_env(step=step, optimizer=opt)
            kernel.apply(env, ["train_step_end"])

        ledger = read_ledger()

        # The external command is read at step 1 and must be ignored_replay
        ext_entries = [e for e in ledger if e["source"] == "external" and e["module"] == "opt"]
        assert len(ext_entries) == 1
        assert ext_entries[0]["decision"] == "ignored_replay"

        # The recipe entry at step 3 must be applied
        replay_entries = [e for e in ledger if e["source"] == "replay" and e["module"] == "opt"]
        assert len(replay_entries) == 1
        assert replay_entries[0]["decision"] == "applied"
        assert replay_entries[0]["step"] == 3

        # Optimizer lr should have been updated by the recipe
        assert opt.param_groups[0]["lr"] == pytest.approx(3e-5)


# ---------------------------------------------------------------------------
# 4. Freeze replay_adjusted -- overlay patches modify recipe params
# ---------------------------------------------------------------------------

class TestFreezeReplayAdjusted:
    def test_adjusted_lr(self, run_dir, make_env, write_recipe, write_freeze, read_ledger):
        """
        replay_adjusted applies the adjustment overlay so lr=3e-5 becomes lr=2e-5.
        """
        recipe_path = write_recipe(
            {
                "at": {"step": 3, "event": "train_step_end"},
                "module": "opt",
                "op": "set_params",
                "id": "main",
                "params": {"lr": 3e-5},
            }
        )

        adjust_path = _write_adjust(
            run_dir,
            patches=[
                {
                    "match": {"module": "opt", "at_step": 3},
                    "replace_params": {"lr": 2e-5},
                }
            ],
        )

        write_freeze(mode="replay_adjusted", recipe_path=recipe_path, adjust_path=adjust_path)

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        opt = _make_optimizer(lr=1e-4)

        for step in range(1, 6):
            env = make_env(step=step, optimizer=opt)
            kernel.apply(env, ["train_step_end"])

        # lr must be the adjusted value, not the original recipe value
        assert opt.param_groups[0]["lr"] == pytest.approx(2e-5)

        ledger = read_ledger()
        replay_entries = [e for e in ledger if e["source"] == "replay" and e["module"] == "opt"]
        assert len(replay_entries) == 1
        assert replay_entries[0]["decision"] == "applied"
        assert replay_entries[0]["step"] == 3


# ---------------------------------------------------------------------------
# 5. Unfreeze -- ops ignored while frozen, applied after unfreeze
# ---------------------------------------------------------------------------

class TestUnfreeze:
    def test_unfreeze_restores_normal_operation(
        self, run_dir, make_env, write_commands, write_freeze, read_ledger
    ):
        """
        Start in prod mode: external ops are ignored.
        Send a core unfreeze op: subsequent external ops should be applied.
        """
        write_freeze(mode="prod")

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        opt = _make_optimizer(lr=1e-4)

        # Step 1: external opt command -- should be ignored
        write_commands({"module": "opt", "op": "set_params", "id": "main", "params": {"lr": 5e-4}})
        env1 = make_env(step=1, optimizer=opt)
        kernel.apply(env1, ["train_step_end"])

        assert opt.param_groups[0]["lr"] == 1e-4  # unchanged

        # Step 2: core unfreeze
        write_commands({"module": "core", "op": "unfreeze"})
        env2 = make_env(step=2, optimizer=opt)
        kernel.apply(env2, ["train_step_end"])

        # Step 3: external opt command -- should now be applied
        write_commands({"module": "opt", "op": "set_params", "id": "main", "params": {"lr": 7e-4}})
        env3 = make_env(step=3, optimizer=opt)
        kernel.apply(env3, ["train_step_end"])

        assert opt.param_groups[0]["lr"] == pytest.approx(7e-4)

        ledger = read_ledger()

        # Verify the first opt op was ignored
        opt_entries = [e for e in ledger if e["module"] == "opt"]
        assert opt_entries[0]["decision"] == "ignored_freeze"

        # Verify unfreeze was applied
        core_entries = [e for e in ledger if e["module"] == "core" and e["op"] == "unfreeze"]
        assert len(core_entries) == 1
        assert core_entries[0]["decision"] == "applied"

        # Verify the second opt op was applied
        assert opt_entries[1]["decision"] == "applied"
