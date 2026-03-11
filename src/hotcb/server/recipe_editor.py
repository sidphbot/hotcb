"""
hotcb.server.recipe_editor — Recipe CRUD and replay preview.

Provides a RecipeEditor class for loading, editing, and saving JSONL recipe
files, plus a FastAPI router exposing these operations as REST endpoints.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

VALID_MODULES = {"cb", "opt", "loss", "tune"}
VALID_OPS = {
    "cb": {"enable", "disable", "load", "unload", "set_params"},
    "opt": {"set_params", "enable", "disable"},
    "loss": {"set_params", "enable", "disable"},
    "tune": {"enable", "disable", "set"},
}


@dataclass
class RecipeEntry:
    index: int
    at_step: int
    at_event: str
    module: str
    op: str
    entry_id: Optional[str]
    params: Optional[dict]
    raw: dict


def _parse_entry(raw: dict, index: int) -> RecipeEntry:
    at = raw.get("at", {})
    return RecipeEntry(
        index=index,
        at_step=int(at.get("step", 0)),
        at_event=str(at.get("event", "train_step_end")),
        module=str(raw.get("module", "")),
        op=str(raw.get("op", "")),
        entry_id=raw.get("id"),
        params=raw.get("params"),
        raw=raw,
    )


def _entry_to_dict(entry: RecipeEntry) -> dict:
    return {
        "index": entry.index,
        "at_step": entry.at_step,
        "at_event": entry.at_event,
        "module": entry.module,
        "op": entry.op,
        "entry_id": entry.entry_id,
        "params": entry.params,
        "raw": entry.raw,
    }


# ---------------------------------------------------------------------------
# RecipeEditor
# ---------------------------------------------------------------------------

class RecipeEditor:
    """CRUD editor for hotcb JSONL recipe files."""

    def __init__(self, path: str):
        self.path = path
        self.entries: list[RecipeEntry] = []
        self.load()

    def load(self) -> None:
        """Parse JSONL file into RecipeEntry list."""
        self.entries = []
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                        self.entries.append(_parse_entry(raw, len(self.entries)))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            self.entries = []

    def save(self) -> None:
        """Write entries back to JSONL, re-indexing."""
        self._reindex()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(json.dumps(entry.raw, ensure_ascii=False) + "\n")

    def add(self, entry: dict, position: Optional[int] = None) -> RecipeEntry:
        """Insert a new entry. Append if position is None."""
        idx = position if position is not None else len(self.entries)
        idx = max(0, min(idx, len(self.entries)))
        rec = _parse_entry(entry, idx)
        self.entries.insert(idx, rec)
        self._reindex()
        return self.entries[idx]

    def remove(self, index: int) -> RecipeEntry:
        """Delete entry by index."""
        if index < 0 or index >= len(self.entries):
            raise IndexError(f"Index {index} out of range (0..{len(self.entries) - 1})")
        removed = self.entries.pop(index)
        self._reindex()
        return removed

    def update(self, index: int, changes: dict) -> RecipeEntry:
        """Modify entry fields (params, step, etc.)."""
        if index < 0 or index >= len(self.entries):
            raise IndexError(f"Index {index} out of range (0..{len(self.entries) - 1})")
        entry = self.entries[index]
        raw = copy.deepcopy(entry.raw)

        if "at_step" in changes:
            raw.setdefault("at", {})["step"] = int(changes["at_step"])
        if "at_event" in changes:
            raw.setdefault("at", {})["event"] = str(changes["at_event"])
        if "module" in changes:
            raw["module"] = str(changes["module"])
        if "op" in changes:
            raw["op"] = str(changes["op"])
        if "id" in changes:
            raw["id"] = changes["id"]
        if "params" in changes:
            raw["params"] = changes["params"]

        updated = _parse_entry(raw, index)
        self.entries[index] = updated
        return updated

    def move(self, from_index: int, to_index: int) -> None:
        """Reorder an entry from one position to another."""
        if from_index < 0 or from_index >= len(self.entries):
            raise IndexError(f"from_index {from_index} out of range")
        if to_index < 0 or to_index >= len(self.entries):
            raise IndexError(f"to_index {to_index} out of range")
        entry = self.entries.pop(from_index)
        self.entries.insert(to_index, entry)
        self._reindex()

    def shift_steps(self, offset: int, from_index: int = 0, to_index: Optional[int] = None) -> int:
        """Offset step numbers for a range of entries. Returns count modified."""
        end = to_index if to_index is not None else len(self.entries)
        end = min(end, len(self.entries))
        start = max(from_index, 0)
        count = 0
        for i in range(start, end):
            entry = self.entries[i]
            new_step = entry.at_step + offset
            raw = copy.deepcopy(entry.raw)
            raw.setdefault("at", {})["step"] = new_step
            self.entries[i] = _parse_entry(raw, i)
            count += 1
        return count

    def validate(self) -> list[str]:
        """Check entries for required fields, valid modules, step ordering."""
        errors: list[str] = []
        prev_step = -1
        for entry in self.entries:
            if not entry.module:
                errors.append(f"Entry {entry.index}: missing 'module'")
            elif entry.module not in VALID_MODULES:
                errors.append(
                    f"Entry {entry.index}: unknown module '{entry.module}', "
                    f"expected one of {sorted(VALID_MODULES)}"
                )
            if not entry.op:
                errors.append(f"Entry {entry.index}: missing 'op'")
            elif entry.module in VALID_OPS:
                allowed = VALID_OPS[entry.module]
                if entry.op not in allowed:
                    errors.append(
                        f"Entry {entry.index}: unknown op '{entry.op}' for "
                        f"module '{entry.module}', expected one of {sorted(allowed)}"
                    )
            if entry.at_step < 0:
                errors.append(f"Entry {entry.index}: negative step {entry.at_step}")
            if entry.at_step < prev_step:
                errors.append(
                    f"Entry {entry.index}: step {entry.at_step} is out of order "
                    f"(previous was {prev_step})"
                )
            prev_step = entry.at_step
        return errors

    def diff(self, other_path: str) -> list[dict]:
        """Compare with another recipe file, return list of differences."""
        other = RecipeEditor(other_path)
        diffs: list[dict] = []

        max_len = max(len(self.entries), len(other.entries))
        for i in range(max_len):
            if i >= len(self.entries):
                diffs.append({
                    "index": i,
                    "type": "added",
                    "entry": _entry_to_dict(other.entries[i]),
                })
            elif i >= len(other.entries):
                diffs.append({
                    "index": i,
                    "type": "removed",
                    "entry": _entry_to_dict(self.entries[i]),
                })
            elif self.entries[i].raw != other.entries[i].raw:
                diffs.append({
                    "index": i,
                    "type": "changed",
                    "before": _entry_to_dict(self.entries[i]),
                    "after": _entry_to_dict(other.entries[i]),
                })
        return diffs

    def timeline(self) -> list[dict]:
        """Return simplified timeline view for UI."""
        timeline: list[dict] = []
        for entry in self.entries:
            timeline.append({
                "index": entry.index,
                "step": entry.at_step,
                "event": entry.at_event,
                "module": entry.module,
                "op": entry.op,
                "id": entry.entry_id,
                "summary": _summarize(entry),
            })
        return timeline

    def export(self, output_path: str) -> None:
        """Export current entries to a new JSONL file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(json.dumps(entry.raw, ensure_ascii=False) + "\n")

    def _reindex(self) -> None:
        for i, entry in enumerate(self.entries):
            entry.index = i


def _summarize(entry: RecipeEntry) -> str:
    parts = [f"{entry.module}.{entry.op}"]
    if entry.entry_id:
        parts.append(f"id={entry.entry_id}")
    if entry.params:
        param_strs = [f"{k}={v}" for k, v in entry.params.items()]
        parts.append(", ".join(param_strs))
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class LoadRequest(BaseModel):
    path: str


class AddEntryRequest(BaseModel):
    entry: Dict[str, Any]
    position: Optional[int] = None


class UpdateEntryRequest(BaseModel):
    changes: Dict[str, Any]


class MoveRequest(BaseModel):
    from_index: int = Field(..., alias="from")
    to_index: int = Field(..., alias="to")

    model_config = {"populate_by_name": True}


class ShiftRequest(BaseModel):
    offset: int
    from_index: int = 0
    to_index: Optional[int] = None


class DiffRequest(BaseModel):
    other_path: str


class ImportRequest(BaseModel):
    path: str


class ExportRequest(BaseModel):
    path: str


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/recipe", tags=["recipe"])


def _get_editor(request: Request) -> RecipeEditor:
    editor = getattr(request.app.state, "recipe_editor", None)
    run_dir = getattr(request.app.state, "run_dir", None)
    recipe_path = os.path.join(run_dir, "hotcb.recipe.jsonl") if run_dir else None

    if editor is None and recipe_path:
        # Auto-create editor if recipe file exists (or create empty one)
        if not os.path.exists(recipe_path):
            open(recipe_path, "w").close()
        editor = RecipeEditor(recipe_path)
        request.app.state.recipe_editor = editor
    elif editor is not None:
        # Reload from disk so we pick up changes from schedule/save-as-recipe
        editor.load()

    if editor is None:
        raise HTTPException(status_code=400, detail="No recipe loaded")
    return editor


@router.get("/")
async def list_entries(request: Request):
    editor = _get_editor(request)
    return {"entries": [_entry_to_dict(e) for e in editor.entries]}


@router.post("/load")
async def load_recipe(body: LoadRequest, request: Request):
    editor = RecipeEditor(body.path)
    request.app.state.recipe_editor = editor
    return {"status": "loaded", "path": body.path, "count": len(editor.entries)}


@router.post("/entry")
async def add_entry(body: AddEntryRequest, request: Request):
    editor = _get_editor(request)
    entry = editor.add(body.entry, body.position)
    editor.save()
    return {"status": "added", "entry": _entry_to_dict(entry)}


@router.delete("/entry/{index}")
async def remove_entry(index: int, request: Request):
    editor = _get_editor(request)
    try:
        removed = editor.remove(index)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))
    editor.save()
    return {"status": "removed", "entry": _entry_to_dict(removed)}


@router.put("/entry/{index}")
async def update_entry(index: int, body: UpdateEntryRequest, request: Request):
    editor = _get_editor(request)
    try:
        updated = editor.update(index, body.changes)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))
    editor.save()
    return {"status": "updated", "entry": _entry_to_dict(updated)}


@router.post("/move")
async def move_entry(body: MoveRequest, request: Request):
    editor = _get_editor(request)
    try:
        editor.move(body.from_index, body.to_index)
    except IndexError as e:
        raise HTTPException(status_code=400, detail=str(e))
    editor.save()
    return {"status": "moved", "from": body.from_index, "to": body.to_index}


@router.post("/shift")
async def shift_steps(body: ShiftRequest, request: Request):
    editor = _get_editor(request)
    count = editor.shift_steps(body.offset, body.from_index, body.to_index)
    editor.save()
    return {"status": "shifted", "count": count, "offset": body.offset}


@router.get("/validate")
async def validate_recipe(request: Request):
    editor = _get_editor(request)
    errors = editor.validate()
    return {"valid": len(errors) == 0, "errors": errors}


@router.post("/diff")
async def diff_recipe(body: DiffRequest, request: Request):
    editor = _get_editor(request)
    diffs = editor.diff(body.other_path)
    return {"diffs": diffs, "count": len(diffs)}


@router.get("/timeline")
async def get_timeline(request: Request):
    editor = _get_editor(request)
    return {"timeline": editor.timeline()}


@router.post("/import")
async def import_recipe(body: ImportRequest, request: Request):
    """Import entries from an external recipe file."""
    from fastapi.responses import JSONResponse
    if not os.path.exists(body.path):
        return JSONResponse(status_code=404, content={"error": f"File not found: {body.path}"})
    editor = _get_editor(request)
    imported = 0
    with open(body.path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                editor.add(entry)
                imported += 1
            except Exception:
                pass
    if imported > 0:
        editor.save()
    return {"status": "imported", "count": imported}


@router.post("/export")
async def export_recipe(body: ExportRequest, request: Request):
    editor = _get_editor(request)
    editor.export(body.path)
    return {"status": "exported", "path": body.path, "count": len(editor.entries)}


@router.post("/save")
async def save_recipe(request: Request):
    editor = _get_editor(request)
    editor.save()
    return {"status": "saved", "path": editor.path, "count": len(editor.entries)}
