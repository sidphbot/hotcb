"""Tests for hotcb recipe export + replay system (spec 19.4 + overlay 11)."""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import pytest

from hotcb.recipe import RecipePlayer, RecipeEntry, apply_overlay, write_effective_recipe
from hotcb.cli import cmd_recipe_export
from hotcb.kernel import HotKernel


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _write_applied_ledger(run_dir: str, entries: List[dict]) -> str:
    """Write synthetic applied ledger entries to the standard path."""
    path = os.path.join(run_dir, "hotcb.applied.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return path


def _read_recipe(path: str) -> List[dict]:
    """Read recipe JSONL and return list of dicts."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _make_recipe_entry(step: int, event: str = "train_step_end",
                       module: str = "opt", op: str = "set_params",
                       id_: str = "main", params: dict | None = None) -> dict:
    """Build a single recipe JSONL record."""
    rec = {
        "at": {"step": step, "event": event},
        "module": module,
        "op": op,
        "id": id_,
    }
    if params is not None:
        rec["params"] = params
    return rec


def _make_ledger_entry(seq: int, step: int, event: str, module: str,
                       op: str, id_: str, decision: str,
                       payload: dict | None = None) -> dict:
    """Build a single applied-ledger record."""
    return {
        "seq": seq,
        "step": step,
        "event": event,
        "module": module,
        "op": op,
        "id": id_,
        "source": "external",
        "decision": decision,
        "payload": payload or {},
    }


# ===================================================================
# 1. Recipe Export -- only 'applied' entries, ordered by seq
# ===================================================================

class TestRecipeExport:

    def test_export_filters_applied_only(self, run_dir):
        """Only decision=='applied' entries from cb/opt/loss modules are exported."""
        ledger = [
            _make_ledger_entry(1, 10, "train_step_end", "opt", "set_params", "main",
                               "applied", {"params": {"lr": 0.01}}),
            _make_ledger_entry(2, 11, "train_step_end", "opt", "set_params", "main",
                               "ignored_freeze", {"params": {"lr": 0.02}}),
            _make_ledger_entry(3, 12, "train_step_end", "loss", "set_params", "main",
                               "failed", {"params": {"weight": 1.0}}),
            _make_ledger_entry(4, 13, "train_step_end", "cb", "enable", "ckpt",
                               "applied", {}),
            # core module entries should be excluded even when applied
            _make_ledger_entry(5, 14, "train_step_end", "core", "freeze", "x",
                               "applied", {"mode": "prod"}),
        ]
        _write_applied_ledger(run_dir, ledger)

        out_path = os.path.join(run_dir, "recipe_out.jsonl")
        args = argparse.Namespace(dir=run_dir, out=out_path)
        cmd_recipe_export(args)

        recipes = _read_recipe(out_path)
        assert len(recipes) == 2
        assert recipes[0]["module"] == "opt"
        assert recipes[1]["module"] == "cb"

    def test_export_ordering_by_seq(self, run_dir):
        """Exported entries preserve insertion (seq) order from the ledger."""
        ledger = [
            _make_ledger_entry(3, 30, "train_step_end", "opt", "set_params", "main",
                               "applied", {"params": {"lr": 0.03}}),
            _make_ledger_entry(1, 10, "train_step_end", "opt", "set_params", "main",
                               "applied", {"params": {"lr": 0.01}}),
            _make_ledger_entry(2, 20, "train_step_end", "opt", "set_params", "main",
                               "applied", {"params": {"lr": 0.02}}),
        ]
        _write_applied_ledger(run_dir, ledger)

        out_path = os.path.join(run_dir, "recipe_out.jsonl")
        args = argparse.Namespace(dir=run_dir, out=out_path)
        cmd_recipe_export(args)

        recipes = _read_recipe(out_path)
        # Order matches ledger file order (seq 3, 1, 2 as written)
        steps = [r["at"]["step"] for r in recipes]
        assert steps == [30, 10, 20]


# ===================================================================
# 2. at.step and at.event mapping
# ===================================================================

class TestExportMapping:

    def test_step_and_event_mapped_correctly(self, run_dir):
        """Exported recipe maps step and event from the ledger entry."""
        ledger = [
            _make_ledger_entry(1, 42, "epoch_end", "opt", "set_params", "main",
                               "applied", {"params": {"lr": 0.1}}),
            _make_ledger_entry(2, 99, "train_step_end", "cb", "enable", "ckpt",
                               "applied", {}),
        ]
        _write_applied_ledger(run_dir, ledger)

        out_path = os.path.join(run_dir, "recipe_out.jsonl")
        args = argparse.Namespace(dir=run_dir, out=out_path)
        cmd_recipe_export(args)

        recipes = _read_recipe(out_path)
        assert recipes[0]["at"]["step"] == 42
        assert recipes[0]["at"]["event"] == "epoch_end"
        assert recipes[1]["at"]["step"] == 99
        assert recipes[1]["at"]["event"] == "train_step_end"


# ===================================================================
# 3. Replay -- two ops at same (step, event)
# ===================================================================

class TestReplayMatching:

    def test_two_ops_same_step_event(self, run_dir, write_recipe, write_freeze, make_env, read_ledger):
        """Two recipe entries at the same (step, event) both fire in recipe order."""
        recipe_path = write_recipe(
            _make_recipe_entry(5, "train_step_end", "opt", "set_params", "a",
                               params={"lr": 0.01}),
            _make_recipe_entry(5, "train_step_end", "opt", "set_params", "b",
                               params={"lr": 0.02}),
        )
        write_freeze(mode="replay", recipe_path=recipe_path)

        kernel = HotKernel(run_dir)
        for s in range(1, 7):
            env = make_env(step=s)
            kernel.apply(env, ["train_step_end"])

        ledger = read_ledger()
        replay_applied = [e for e in ledger if e.get("source") == "replay"]
        assert len(replay_applied) == 2
        assert replay_applied[0]["id"] == "a"
        assert replay_applied[1]["id"] == "b"
        assert replay_applied[0]["step"] == 5
        assert replay_applied[1]["step"] == 5

    # ===============================================================
    # 4. Ops at specific steps only
    # ===============================================================

    def test_ops_fire_at_correct_steps(self, run_dir, write_recipe, write_freeze, make_env, read_ledger):
        """Recipe ops at steps 2, 5, 8 fire exactly at those steps."""
        recipe_path = write_recipe(
            _make_recipe_entry(2, "train_step_end", "opt", "set_params", "s2",
                               params={"lr": 0.1}),
            _make_recipe_entry(5, "train_step_end", "opt", "set_params", "s5",
                               params={"lr": 0.2}),
            _make_recipe_entry(8, "train_step_end", "opt", "set_params", "s8",
                               params={"lr": 0.3}),
        )
        write_freeze(mode="replay", recipe_path=recipe_path)

        kernel = HotKernel(run_dir)
        for s in range(1, 11):
            env = make_env(step=s)
            kernel.apply(env, ["train_step_end"])

        ledger = read_ledger()
        replay_entries = [e for e in ledger if e.get("source") == "replay"]
        fired_steps = [e["step"] for e in replay_entries]
        assert fired_steps == [2, 5, 8]
        assert [e["id"] for e in replay_entries] == ["s2", "s5", "s8"]


# ===================================================================
# 5. Step offset
# ===================================================================

class TestStepOffset:

    def test_step_offset_shifts_firing(self, run_dir, write_recipe, write_freeze, make_env, read_ledger):
        """Recipe entry at step=3 with step_offset=2 fires at step=5."""
        recipe_path = write_recipe(
            _make_recipe_entry(3, "train_step_end", "opt", "set_params", "off",
                               params={"lr": 0.05}),
        )
        write_freeze(mode="replay", recipe_path=recipe_path, step_offset=2)

        kernel = HotKernel(run_dir)
        for s in range(1, 8):
            env = make_env(step=s)
            kernel.apply(env, ["train_step_end"])

        ledger = read_ledger()
        replay_entries = [e for e in ledger if e.get("source") == "replay"]
        assert len(replay_entries) == 1
        assert replay_entries[0]["step"] == 5


# ===================================================================
# 6. Overlay: replace_params
# ===================================================================

class TestOverlayReplaceParams:

    def test_replace_lr(self):
        """replace_params replaces lr value for matching entry."""
        entries = [
            RecipeEntry(
                at_step=5, at_event="train_step_end",
                record=_make_recipe_entry(5, "train_step_end", "opt", "set_params", "main",
                                          params={"lr": 0.01}),
            ),
        ]
        adjust = {
            "patches": [
                {
                    "match": {"module": "opt", "at_step": 5},
                    "replace_params": {"lr": 0.99},
                }
            ]
        }
        result = apply_overlay(entries, adjust)
        assert len(result) == 1
        assert result[0].record["params"]["lr"] == 0.99

    def test_replace_params_via_player(self, tmp_path):
        """RecipePlayer with adjust file returns modified op."""
        recipe_path = str(tmp_path / "recipe.jsonl")
        adjust_path = str(tmp_path / "adjust.json")

        with open(recipe_path, "w") as f:
            f.write(json.dumps(_make_recipe_entry(
                5, "train_step_end", "opt", "set_params", "main",
                params={"lr": 0.01})) + "\n")

        with open(adjust_path, "w") as f:
            json.dump({"patches": [
                {"match": {"module": "opt", "at_step": 5},
                 "replace_params": {"lr": 0.99}}
            ]}, f)

        player = RecipePlayer(recipe_path, adjust_path=adjust_path)
        ops = player.ops_for(5, "train_step_end")
        assert len(ops) == 1
        assert ops[0].params["lr"] == 0.99


# ===================================================================
# 7. Overlay: shift_step
# ===================================================================

class TestOverlayShiftStep:

    def test_shift_step_plus_3(self):
        """shift_step=+3 moves entry from step 5 to step 8."""
        entries = [
            RecipeEntry(
                at_step=5, at_event="train_step_end",
                record=_make_recipe_entry(5, "train_step_end", "opt", "set_params", "main",
                                          params={"lr": 0.01}),
            ),
        ]
        adjust = {
            "patches": [
                {"match": {"module": "opt"}, "shift_step": 3}
            ]
        }
        result = apply_overlay(entries, adjust)
        assert len(result) == 1
        assert result[0].at_step == 8
        assert result[0].record["at"]["step"] == 8

    def test_shifted_entry_fires_at_new_step(self, tmp_path):
        """After shift_step=+3, entry originally at step 5 fires at step 8."""
        recipe_path = str(tmp_path / "recipe.jsonl")
        adjust_path = str(tmp_path / "adjust.json")

        with open(recipe_path, "w") as f:
            f.write(json.dumps(_make_recipe_entry(
                5, "train_step_end", "opt", "set_params", "main",
                params={"lr": 0.01})) + "\n")

        with open(adjust_path, "w") as f:
            json.dump({"patches": [
                {"match": {"module": "opt"}, "shift_step": 3}
            ]}, f)

        player = RecipePlayer(recipe_path, adjust_path=adjust_path)

        # Should NOT fire at step 5
        assert player.ops_for(5, "train_step_end") == []
        # Should fire at step 8
        ops = player.ops_for(8, "train_step_end")
        assert len(ops) == 1
        assert ops[0].params["lr"] == 0.01


# ===================================================================
# 8. Overlay: drop
# ===================================================================

class TestOverlayDrop:

    def test_drop_by_module(self):
        """drop=true removes matching entries."""
        entries = [
            RecipeEntry(at_step=1, at_event="train_step_end",
                        record=_make_recipe_entry(1, "train_step_end", "cb", "enable", "ckpt")),
            RecipeEntry(at_step=2, at_event="train_step_end",
                        record=_make_recipe_entry(2, "train_step_end", "opt", "set_params", "main",
                                                  params={"lr": 0.01})),
        ]
        adjust = {
            "patches": [
                {"match": {"module": "cb"}, "drop": True}
            ]
        }
        result = apply_overlay(entries, adjust)
        assert len(result) == 1
        assert result[0].record["module"] == "opt"

    def test_dropped_entry_absent_in_replay(self, tmp_path):
        """Dropped entry does not appear in RecipePlayer."""
        recipe_path = str(tmp_path / "recipe.jsonl")
        adjust_path = str(tmp_path / "adjust.json")

        with open(recipe_path, "w") as f:
            f.write(json.dumps(_make_recipe_entry(1, "train_step_end", "cb", "enable", "ckpt")) + "\n")
            f.write(json.dumps(_make_recipe_entry(2, "train_step_end", "opt", "set_params", "main",
                                                  params={"lr": 0.01})) + "\n")

        with open(adjust_path, "w") as f:
            json.dump({"patches": [
                {"match": {"module": "cb"}, "drop": True}
            ]}, f)

        player = RecipePlayer(recipe_path, adjust_path=adjust_path)
        assert player.ops_for(1, "train_step_end") == []
        ops = player.ops_for(2, "train_step_end")
        assert len(ops) == 1
        assert ops[0].module == "opt"


# ===================================================================
# 9. Overlay: insert
# ===================================================================

class TestOverlayInsert:

    def test_insert_new_entry(self):
        """insert adds a new entry to the recipe."""
        entries = [
            RecipeEntry(at_step=5, at_event="train_step_end",
                        record=_make_recipe_entry(5, "train_step_end", "opt", "set_params", "main",
                                                  params={"lr": 0.01})),
        ]
        adjust = {
            "patches": [
                {
                    "insert": {
                        "at": {"step": 1, "event": "train_step_end"},
                        "module": "cb",
                        "op": "enable",
                        "id": "warmup",
                    }
                }
            ]
        }
        result = apply_overlay(entries, adjust)
        assert len(result) == 2
        # After sort, step=1 comes first
        assert result[0].at_step == 1
        assert result[0].record["module"] == "cb"
        assert result[0].record["id"] == "warmup"
        assert result[1].at_step == 5

    def test_inserted_entry_appears_in_replay(self, tmp_path):
        """Inserted entry is available via RecipePlayer."""
        recipe_path = str(tmp_path / "recipe.jsonl")
        adjust_path = str(tmp_path / "adjust.json")

        with open(recipe_path, "w") as f:
            f.write(json.dumps(_make_recipe_entry(
                5, "train_step_end", "opt", "set_params", "main",
                params={"lr": 0.01})) + "\n")

        with open(adjust_path, "w") as f:
            json.dump({"patches": [
                {"insert": {
                    "at": {"step": 1, "event": "train_step_end"},
                    "module": "cb", "op": "enable", "id": "warmup",
                }}
            ]}, f)

        player = RecipePlayer(recipe_path, adjust_path=adjust_path)
        ops = player.ops_for(1, "train_step_end")
        assert len(ops) == 1
        assert ops[0].module == "cb"
        assert ops[0].id == "warmup"


# ===================================================================
# 10. Overlay: transform_params (scale)
# ===================================================================

class TestOverlayTransformParams:

    def test_scale_lr_by_half(self):
        """transform_params with scale halves lr."""
        entries = [
            RecipeEntry(at_step=3, at_event="train_step_end",
                        record=_make_recipe_entry(3, "train_step_end", "opt", "set_params", "main",
                                                  params={"lr": 0.1})),
        ]
        adjust = {
            "patches": [
                {
                    "match": {"module": "opt"},
                    "transform_params": {"scale": {"lr": 0.5}},
                }
            ]
        }
        result = apply_overlay(entries, adjust)
        assert len(result) == 1
        assert result[0].record["params"]["lr"] == pytest.approx(0.05)

    def test_transform_via_player(self, tmp_path):
        """RecipePlayer returns ops with scaled params."""
        recipe_path = str(tmp_path / "recipe.jsonl")
        adjust_path = str(tmp_path / "adjust.json")

        with open(recipe_path, "w") as f:
            f.write(json.dumps(_make_recipe_entry(
                3, "train_step_end", "opt", "set_params", "main",
                params={"lr": 0.1})) + "\n")

        with open(adjust_path, "w") as f:
            json.dump({"patches": [
                {"match": {"module": "opt"},
                 "transform_params": {"scale": {"lr": 0.5}}}
            ]}, f)

        player = RecipePlayer(recipe_path, adjust_path=adjust_path)
        ops = player.ops_for(3, "train_step_end")
        assert len(ops) == 1
        assert ops[0].params["lr"] == pytest.approx(0.05)


# ===================================================================
# 11. Effective recipe snapshot
# ===================================================================

class TestEffectiveRecipe:

    def test_write_effective_recipe_after_overlay(self, tmp_path):
        """write_effective_recipe writes correct JSONL after overlay transformations."""
        entries = [
            RecipeEntry(at_step=2, at_event="train_step_end",
                        record=_make_recipe_entry(2, "train_step_end", "opt", "set_params", "main",
                                                  params={"lr": 0.1})),
            RecipeEntry(at_step=5, at_event="train_step_end",
                        record=_make_recipe_entry(5, "train_step_end", "cb", "enable", "ckpt")),
        ]
        adjust = {
            "patches": [
                {"match": {"module": "opt"}, "replace_params": {"lr": 0.42}},
                {"match": {"module": "cb"}, "drop": True},
                {"insert": {
                    "at": {"step": 1, "event": "train_step_end"},
                    "module": "loss", "op": "set_params", "id": "main",
                    "params": {"weight": 2.0},
                }},
            ]
        }

        result = apply_overlay(entries, adjust)
        out_path = str(tmp_path / "effective.jsonl")
        write_effective_recipe(result, out_path)

        written = _read_recipe(out_path)
        assert len(written) == 2
        # Sorted: step 1 (inserted loss), step 2 (modified opt)
        assert written[0]["module"] == "loss"
        assert written[0]["at"]["step"] == 1
        assert written[0]["params"]["weight"] == 2.0
        assert written[1]["module"] == "opt"
        assert written[1]["params"]["lr"] == 0.42

    def test_effective_recipe_via_player(self, tmp_path):
        """RecipePlayer writes effective recipe snapshot when path provided."""
        recipe_path = str(tmp_path / "recipe.jsonl")
        adjust_path = str(tmp_path / "adjust.json")
        effective_path = str(tmp_path / "effective.jsonl")

        with open(recipe_path, "w") as f:
            f.write(json.dumps(_make_recipe_entry(
                3, "train_step_end", "opt", "set_params", "main",
                params={"lr": 0.1})) + "\n")

        with open(adjust_path, "w") as f:
            json.dump({"patches": [
                {"match": {"module": "opt"}, "replace_params": {"lr": 0.77}}
            ]}, f)

        player = RecipePlayer(recipe_path, adjust_path=adjust_path,
                              effective_recipe_path=effective_path)

        assert os.path.exists(effective_path)
        written = _read_recipe(effective_path)
        assert len(written) == 1
        assert written[0]["params"]["lr"] == 0.77
