"""Tests for hotcb.server.recipe_editor — Recipe CRUD and replay preview."""
from __future__ import annotations

import json
import os
import pytest
from pathlib import Path

from hotcb.server.recipe_editor import RecipeEditor, RecipeEntry, _entry_to_dict, router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ENTRIES = [
    {"at": {"step": 100, "event": "train_step_end"}, "module": "opt", "op": "set_params", "id": "main", "params": {"lr": 0.001}},
    {"at": {"step": 200, "event": "train_step_end"}, "module": "loss", "op": "set_params", "id": "main", "params": {"recon_w": 0.5}},
    {"at": {"step": 300, "event": "train_step_end"}, "module": "cb", "op": "enable", "id": "grad_clip"},
    {"at": {"step": 500, "event": "val_epoch_end"}, "module": "opt", "op": "set_params", "id": "main", "params": {"lr": 0.0001}},
]


@pytest.fixture
def recipe_path(tmp_path: Path) -> str:
    path = str(tmp_path / "test_recipe.jsonl")
    with open(path, "w") as f:
        for entry in SAMPLE_ENTRIES:
            f.write(json.dumps(entry) + "\n")
    return path


@pytest.fixture
def editor(recipe_path: str) -> RecipeEditor:
    return RecipeEditor(recipe_path)


@pytest.fixture
def empty_recipe_path(tmp_path: Path) -> str:
    path = str(tmp_path / "empty_recipe.jsonl")
    with open(path, "w") as f:
        pass  # empty file
    return path


# ---------------------------------------------------------------------------
# RecipeEditor — load / save
# ---------------------------------------------------------------------------

class TestLoadSave:
    def test_load_parses_entries(self, editor: RecipeEditor):
        assert len(editor.entries) == 4
        assert editor.entries[0].at_step == 100
        assert editor.entries[0].module == "opt"
        assert editor.entries[0].op == "set_params"
        assert editor.entries[0].entry_id == "main"
        assert editor.entries[0].params == {"lr": 0.001}

    def test_load_sets_indices(self, editor: RecipeEditor):
        for i, entry in enumerate(editor.entries):
            assert entry.index == i

    def test_load_preserves_raw(self, editor: RecipeEditor):
        assert editor.entries[0].raw == SAMPLE_ENTRIES[0]

    def test_save_roundtrip(self, editor: RecipeEditor, tmp_path: Path):
        out_path = str(tmp_path / "saved.jsonl")
        editor.path = out_path
        editor.save()
        reloaded = RecipeEditor(out_path)
        assert len(reloaded.entries) == len(editor.entries)
        for orig, loaded in zip(editor.entries, reloaded.entries):
            assert orig.raw == loaded.raw

    def test_load_empty_file(self, empty_recipe_path: str):
        editor = RecipeEditor(empty_recipe_path)
        assert len(editor.entries) == 0

    def test_load_nonexistent_file(self, tmp_path: Path):
        editor = RecipeEditor(str(tmp_path / "nonexistent.jsonl"))
        assert len(editor.entries) == 0

    def test_load_skips_bad_lines(self, tmp_path: Path):
        path = str(tmp_path / "bad.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps(SAMPLE_ENTRIES[0]) + "\n")
            f.write("not valid json\n")
            f.write(json.dumps(SAMPLE_ENTRIES[1]) + "\n")
        editor = RecipeEditor(path)
        assert len(editor.entries) == 2


# ---------------------------------------------------------------------------
# Add / Remove / Update / Move
# ---------------------------------------------------------------------------

class TestCRUD:
    def test_add_appends(self, editor: RecipeEditor):
        new = {"at": {"step": 600, "event": "train_step_end"}, "module": "opt", "op": "disable", "id": "main"}
        entry = editor.add(new)
        assert len(editor.entries) == 5
        assert entry.index == 4
        assert entry.at_step == 600

    def test_add_at_position(self, editor: RecipeEditor):
        new = {"at": {"step": 150, "event": "train_step_end"}, "module": "opt", "op": "disable"}
        entry = editor.add(new, position=1)
        assert len(editor.entries) == 5
        assert editor.entries[1].at_step == 150
        # indices should be reindexed
        for i, e in enumerate(editor.entries):
            assert e.index == i

    def test_add_at_position_zero(self, editor: RecipeEditor):
        new = {"at": {"step": 50, "event": "train_step_end"}, "module": "opt", "op": "enable"}
        editor.add(new, position=0)
        assert editor.entries[0].at_step == 50

    def test_remove(self, editor: RecipeEditor):
        removed = editor.remove(1)
        assert removed.at_step == 200
        assert len(editor.entries) == 3
        for i, e in enumerate(editor.entries):
            assert e.index == i

    def test_remove_first(self, editor: RecipeEditor):
        editor.remove(0)
        assert len(editor.entries) == 3
        assert editor.entries[0].at_step == 200

    def test_remove_last(self, editor: RecipeEditor):
        editor.remove(3)
        assert len(editor.entries) == 3

    def test_remove_invalid_index(self, editor: RecipeEditor):
        with pytest.raises(IndexError):
            editor.remove(10)
        with pytest.raises(IndexError):
            editor.remove(-1)

    def test_update_params(self, editor: RecipeEditor):
        updated = editor.update(0, {"params": {"lr": 0.01}})
        assert updated.params == {"lr": 0.01}
        assert updated.raw["params"] == {"lr": 0.01}

    def test_update_step(self, editor: RecipeEditor):
        updated = editor.update(0, {"at_step": 150})
        assert updated.at_step == 150
        assert updated.raw["at"]["step"] == 150

    def test_update_event(self, editor: RecipeEditor):
        updated = editor.update(0, {"at_event": "val_epoch_end"})
        assert updated.at_event == "val_epoch_end"
        assert updated.raw["at"]["event"] == "val_epoch_end"

    def test_update_module_and_op(self, editor: RecipeEditor):
        updated = editor.update(0, {"module": "loss", "op": "disable"})
        assert updated.module == "loss"
        assert updated.op == "disable"

    def test_update_id(self, editor: RecipeEditor):
        updated = editor.update(0, {"id": "aux"})
        assert updated.entry_id == "aux"

    def test_update_invalid_index(self, editor: RecipeEditor):
        with pytest.raises(IndexError):
            editor.update(10, {"params": {}})

    def test_move_forward(self, editor: RecipeEditor):
        # Move entry at index 0 to index 2
        editor.move(0, 2)
        assert editor.entries[2].at_step == 100  # the originally-first entry
        assert len(editor.entries) == 4
        for i, e in enumerate(editor.entries):
            assert e.index == i

    def test_move_backward(self, editor: RecipeEditor):
        editor.move(3, 0)
        assert editor.entries[0].at_step == 500
        assert len(editor.entries) == 4

    def test_move_same_position(self, editor: RecipeEditor):
        editor.move(1, 1)
        assert editor.entries[1].at_step == 200

    def test_move_invalid_from(self, editor: RecipeEditor):
        with pytest.raises(IndexError):
            editor.move(10, 0)

    def test_move_invalid_to(self, editor: RecipeEditor):
        with pytest.raises(IndexError):
            editor.move(0, 10)


# ---------------------------------------------------------------------------
# shift_steps
# ---------------------------------------------------------------------------

class TestShiftSteps:
    def test_shift_all(self, editor: RecipeEditor):
        count = editor.shift_steps(50)
        assert count == 4
        assert editor.entries[0].at_step == 150
        assert editor.entries[1].at_step == 250
        assert editor.entries[2].at_step == 350
        assert editor.entries[3].at_step == 550

    def test_shift_range(self, editor: RecipeEditor):
        count = editor.shift_steps(100, from_index=1, to_index=3)
        assert count == 2
        assert editor.entries[0].at_step == 100  # unchanged
        assert editor.entries[1].at_step == 300  # shifted
        assert editor.entries[2].at_step == 400  # shifted
        assert editor.entries[3].at_step == 500  # unchanged

    def test_shift_negative(self, editor: RecipeEditor):
        editor.shift_steps(-50, from_index=0, to_index=2)
        assert editor.entries[0].at_step == 50
        assert editor.entries[1].at_step == 150

    def test_shift_from_index_only(self, editor: RecipeEditor):
        count = editor.shift_steps(10, from_index=2)
        assert count == 2
        assert editor.entries[2].at_step == 310
        assert editor.entries[3].at_step == 510

    def test_shift_updates_raw(self, editor: RecipeEditor):
        editor.shift_steps(25)
        assert editor.entries[0].raw["at"]["step"] == 125


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

class TestValidate:
    def test_valid_recipe(self, editor: RecipeEditor):
        errors = editor.validate()
        assert errors == []

    def test_missing_module(self, editor: RecipeEditor):
        editor.add({"at": {"step": 600}, "op": "enable"})
        errors = editor.validate()
        assert any("missing 'module'" in e for e in errors)

    def test_unknown_module(self, editor: RecipeEditor):
        editor.add({"at": {"step": 600}, "module": "bogus", "op": "enable"})
        errors = editor.validate()
        assert any("unknown module" in e for e in errors)

    def test_missing_op(self, editor: RecipeEditor):
        editor.add({"at": {"step": 600}, "module": "opt"})
        errors = editor.validate()
        assert any("missing 'op'" in e for e in errors)

    def test_unknown_op(self, editor: RecipeEditor):
        editor.add({"at": {"step": 600}, "module": "opt", "op": "explode"})
        errors = editor.validate()
        assert any("unknown op" in e for e in errors)

    def test_out_of_order_steps(self, editor: RecipeEditor):
        # Insert an entry with step 50 at the end
        editor.add({"at": {"step": 50, "event": "train_step_end"}, "module": "opt", "op": "enable"})
        errors = editor.validate()
        assert any("out of order" in e for e in errors)

    def test_negative_step(self, editor: RecipeEditor):
        editor.add({"at": {"step": -10, "event": "train_step_end"}, "module": "opt", "op": "enable"}, position=0)
        errors = editor.validate()
        assert any("negative step" in e for e in errors)


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

class TestDiff:
    def test_identical_recipes(self, editor: RecipeEditor, recipe_path: str):
        diffs = editor.diff(recipe_path)
        assert diffs == []

    def test_diff_detects_changes(self, editor: RecipeEditor, tmp_path: Path):
        other_path = str(tmp_path / "other.jsonl")
        modified = [dict(e) for e in SAMPLE_ENTRIES]
        modified[0] = {**modified[0], "params": {"lr": 0.999}}
        with open(other_path, "w") as f:
            for entry in modified:
                f.write(json.dumps(entry) + "\n")
        diffs = editor.diff(other_path)
        assert len(diffs) == 1
        assert diffs[0]["type"] == "changed"
        assert diffs[0]["index"] == 0

    def test_diff_detects_added(self, editor: RecipeEditor, tmp_path: Path):
        other_path = str(tmp_path / "other.jsonl")
        extra = SAMPLE_ENTRIES + [{"at": {"step": 700}, "module": "opt", "op": "disable"}]
        with open(other_path, "w") as f:
            for entry in extra:
                f.write(json.dumps(entry) + "\n")
        diffs = editor.diff(other_path)
        assert len(diffs) == 1
        assert diffs[0]["type"] == "added"

    def test_diff_detects_removed(self, editor: RecipeEditor, tmp_path: Path):
        other_path = str(tmp_path / "other.jsonl")
        shorter = SAMPLE_ENTRIES[:2]
        with open(other_path, "w") as f:
            for entry in shorter:
                f.write(json.dumps(entry) + "\n")
        diffs = editor.diff(other_path)
        assert len(diffs) == 2
        assert all(d["type"] == "removed" for d in diffs)


# ---------------------------------------------------------------------------
# timeline
# ---------------------------------------------------------------------------

class TestTimeline:
    def test_timeline_length(self, editor: RecipeEditor):
        tl = editor.timeline()
        assert len(tl) == 4

    def test_timeline_fields(self, editor: RecipeEditor):
        tl = editor.timeline()
        first = tl[0]
        assert first["step"] == 100
        assert first["event"] == "train_step_end"
        assert first["module"] == "opt"
        assert first["op"] == "set_params"
        assert first["id"] == "main"
        assert "summary" in first

    def test_timeline_summary_content(self, editor: RecipeEditor):
        tl = editor.timeline()
        assert "opt.set_params" in tl[0]["summary"]
        assert "lr=0.001" in tl[0]["summary"]

    def test_timeline_no_id(self, editor: RecipeEditor):
        editor.add({"at": {"step": 700}, "module": "opt", "op": "enable"})
        tl = editor.timeline()
        last = tl[-1]
        assert last["id"] is None


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_creates_file(self, editor: RecipeEditor, tmp_path: Path):
        out = str(tmp_path / "exported.jsonl")
        editor.export(out)
        assert os.path.exists(out)

    def test_export_roundtrip(self, editor: RecipeEditor, tmp_path: Path):
        out = str(tmp_path / "exported.jsonl")
        editor.export(out)
        reloaded = RecipeEditor(out)
        assert len(reloaded.entries) == len(editor.entries)
        for orig, loaded in zip(editor.entries, reloaded.entries):
            assert orig.raw == loaded.raw

    def test_export_nested_dir(self, editor: RecipeEditor, tmp_path: Path):
        out = str(tmp_path / "sub" / "dir" / "exported.jsonl")
        editor.export(out)
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# REST endpoints via TestClient
# ---------------------------------------------------------------------------

class TestRESTEndpoints:
    @pytest.fixture
    def client(self, recipe_path: str):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.state.recipe_editor = RecipeEditor(recipe_path)
        app.include_router(router)
        return TestClient(app)

    @pytest.fixture
    def empty_client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.state.recipe_editor = None
        app.include_router(router)
        return TestClient(app)

    def test_list_entries(self, client):
        resp = client.get("/api/recipe/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["entries"]) == 4

    def test_list_no_editor(self, empty_client):
        resp = empty_client.get("/api/recipe/")
        assert resp.status_code == 400

    def test_load_recipe(self, client, recipe_path: str):
        resp = client.post("/api/recipe/load", json={"path": recipe_path})
        assert resp.status_code == 200
        assert resp.json()["count"] == 4

    def test_add_entry(self, client):
        entry = {"at": {"step": 600, "event": "train_step_end"}, "module": "opt", "op": "disable"}
        resp = client.post("/api/recipe/entry", json={"entry": entry})
        assert resp.status_code == 200
        assert resp.json()["entry"]["at_step"] == 600
        # Check it's actually in the list
        resp2 = client.get("/api/recipe/")
        assert len(resp2.json()["entries"]) == 5

    def test_add_entry_at_position(self, client):
        entry = {"at": {"step": 150}, "module": "opt", "op": "enable"}
        resp = client.post("/api/recipe/entry", json={"entry": entry, "position": 1})
        assert resp.status_code == 200
        resp2 = client.get("/api/recipe/")
        assert resp2.json()["entries"][1]["at_step"] == 150

    def test_remove_entry(self, client):
        resp = client.delete("/api/recipe/entry/0")
        assert resp.status_code == 200
        assert resp.json()["entry"]["at_step"] == 100
        resp2 = client.get("/api/recipe/")
        assert len(resp2.json()["entries"]) == 3

    def test_remove_invalid_index(self, client):
        resp = client.delete("/api/recipe/entry/99")
        assert resp.status_code == 404

    def test_update_entry(self, client):
        resp = client.put("/api/recipe/entry/0", json={"changes": {"params": {"lr": 0.05}}})
        assert resp.status_code == 200
        assert resp.json()["entry"]["params"] == {"lr": 0.05}

    def test_update_invalid_index(self, client):
        resp = client.put("/api/recipe/entry/99", json={"changes": {"params": {}}})
        assert resp.status_code == 404

    def test_move_entry(self, client):
        resp = client.post("/api/recipe/move", json={"from": 0, "to": 2})
        assert resp.status_code == 200
        resp2 = client.get("/api/recipe/")
        assert resp2.json()["entries"][2]["at_step"] == 100

    def test_move_invalid(self, client):
        resp = client.post("/api/recipe/move", json={"from": 0, "to": 99})
        assert resp.status_code == 400

    def test_shift_steps(self, client):
        resp = client.post("/api/recipe/shift", json={"offset": 50, "from_index": 0})
        assert resp.status_code == 200
        assert resp.json()["count"] == 4
        resp2 = client.get("/api/recipe/")
        assert resp2.json()["entries"][0]["at_step"] == 150

    def test_validate(self, client):
        resp = client.get("/api/recipe/validate")
        assert resp.status_code == 200
        assert resp.json()["valid"] is True

    def test_diff(self, client, recipe_path: str):
        resp = client.post("/api/recipe/diff", json={"other_path": recipe_path})
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_timeline(self, client):
        resp = client.get("/api/recipe/timeline")
        assert resp.status_code == 200
        tl = resp.json()["timeline"]
        assert len(tl) == 4
        assert tl[0]["step"] == 100

    def test_export(self, client, tmp_path: Path):
        out = str(tmp_path / "api_export.jsonl")
        resp = client.post("/api/recipe/export", json={"path": out})
        assert resp.status_code == 200
        assert os.path.exists(out)

    def test_save(self, client):
        # Modify then save
        client.put("/api/recipe/entry/0", json={"changes": {"params": {"lr": 0.05}}})
        resp = client.post("/api/recipe/save")
        assert resp.status_code == 200
        assert resp.json()["status"] == "saved"
