from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ops import CallbackTarget, HotOp
from .util import safe_mtime


@dataclass
class RecipeEntry:
    at_step: int
    at_event: str
    record: dict


def _parse_recipe_line(d: dict) -> RecipeEntry:
    at = d.get("at", {})
    step = int(at.get("step", 0))
    event = str(at.get("event", "train_step_end"))
    return RecipeEntry(at_step=step, at_event=event, record=d)


# ---------------------------------------------------------------------------
# Overlay / adjustment patch system (spec §11)
# ---------------------------------------------------------------------------

def _load_adjust_file(path: str) -> dict:
    """Load a YAML or JSON adjustment overlay file."""
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            return {}
    # default: JSON
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _patch_matches(match: dict, entry: RecipeEntry) -> bool:
    """Check if a patch rule's match criteria apply to a recipe entry."""
    rec = entry.record
    if "module" in match and rec.get("module") != match["module"]:
        return False
    if "op" in match and rec.get("op") != match["op"]:
        return False
    if "id" in match and rec.get("id") != match["id"]:
        return False
    if "at_event" in match and entry.at_event != match["at_event"]:
        return False
    # exact step match
    if "at_step" in match and entry.at_step != int(match["at_step"]):
        return False
    # step range match
    if "step_min" in match and entry.at_step < int(match["step_min"]):
        return False
    if "step_max" in match and entry.at_step > int(match["step_max"]):
        return False
    # payload key existence
    if "has_param" in match:
        params = rec.get("params") or {}
        if match["has_param"] not in params:
            return False
    return True


def _apply_replace_params(entry: RecipeEntry, replace: dict) -> RecipeEntry:
    """Replace specific params in a recipe entry."""
    rec = copy.deepcopy(entry.record)
    params = rec.setdefault("params", {})
    params.update(replace)
    at = rec.get("at", {})
    return RecipeEntry(at_step=int(at.get("step", entry.at_step)),
                       at_event=str(at.get("event", entry.at_event)),
                       record=rec)


def _apply_transform_params(entry: RecipeEntry, transform: dict) -> RecipeEntry:
    """Apply bulk transforms (scale factors) to params."""
    rec = copy.deepcopy(entry.record)
    params = rec.get("params") or {}
    scale = transform.get("scale", {})
    for key, factor in scale.items():
        if key in params:
            params[key] = float(params[key]) * float(factor)
    add = transform.get("add", {})
    for key, delta in add.items():
        if key in params:
            params[key] = float(params[key]) + float(delta)
    rec["params"] = params
    at = rec.get("at", {})
    return RecipeEntry(at_step=int(at.get("step", entry.at_step)),
                       at_event=str(at.get("event", entry.at_event)),
                       record=rec)


def _apply_shift_step(entry: RecipeEntry, shift: int) -> RecipeEntry:
    """Shift a recipe entry's step by a delta."""
    rec = copy.deepcopy(entry.record)
    new_step = entry.at_step + int(shift)
    rec.setdefault("at", {})["step"] = new_step
    return RecipeEntry(at_step=new_step, at_event=entry.at_event, record=rec)


def _make_insert_entry(insert_spec: dict) -> RecipeEntry:
    """Create a new recipe entry from an insert spec."""
    rec = copy.deepcopy(insert_spec)
    at = rec.get("at", {})
    step = int(at.get("step", 0))
    event = str(at.get("event", "train_step_end"))
    return RecipeEntry(at_step=step, at_event=event, record=rec)


def apply_overlay(entries: List[RecipeEntry], adjust_data: dict) -> List[RecipeEntry]:
    """
    Apply adjustment overlay patches to recipe entries.

    Supports 5 patch rule types (spec §11.1):
    1. replace_params  - surgical param replacement
    2. shift_step      - delay/advance an action
    3. drop            - remove matching entries
    4. insert          - add new entries
    5. transform_params - bulk transforms (scale/add)

    Returns a new list of entries (does not mutate originals).
    """
    patches = adjust_data.get("patches", [])
    if not patches:
        return list(entries)

    result = list(entries)

    # Track nth-occurrence counters per patch
    for patch in patches:
        # Handle inserts first (they don't match existing entries)
        if "insert" in patch:
            result.append(_make_insert_entry(patch["insert"]))
            continue

        match = patch.get("match", {})
        nth = match.get("nth")  # optional: only apply to Nth match (0-indexed)
        match_count = 0

        new_result: List[RecipeEntry] = []
        for entry in result:
            if not _patch_matches(match, entry):
                new_result.append(entry)
                continue

            # Check nth occurrence filter
            if nth is not None and match_count != int(nth):
                match_count += 1
                new_result.append(entry)
                continue
            match_count += 1

            # Drop
            if patch.get("drop", False):
                continue  # skip this entry

            modified = entry

            # Replace params
            if "replace_params" in patch:
                modified = _apply_replace_params(modified, patch["replace_params"])

            # Transform params (scale/add)
            if "transform_params" in patch:
                modified = _apply_transform_params(modified, patch["transform_params"])

            # Shift step
            if "shift_step" in patch:
                modified = _apply_shift_step(modified, int(patch["shift_step"]))

            new_result.append(modified)

        result = new_result

    # Re-sort after modifications
    result.sort(key=lambda x: (x.at_step, x.at_event))
    return result


def write_effective_recipe(entries: List[RecipeEntry], path: str) -> None:
    """Write the effective (post-overlay) recipe to a JSONL file."""
    from .util import ensure_dir
    import os
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry.record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# RecipePlayer
# ---------------------------------------------------------------------------

class RecipePlayer:
    """
    Recipe replayer. Injects hotcb ops at matching (step,event).
    Supports adjustment overlays for replay-adjusted mode.
    """

    def __init__(self, recipe_path: Optional[str], adjust_path: Optional[str] = None,
                 step_offset: int = 0, effective_recipe_path: Optional[str] = None) -> None:
        self.recipe_path = recipe_path
        self.adjust_path = adjust_path
        self.step_offset = int(step_offset or 0)
        self.effective_recipe_path = effective_recipe_path
        self._entries: List[RecipeEntry] = []
        self._idx = 0
        self._recipe_mtime = 0.0
        self._adjust_mtime = 0.0
        if recipe_path:
            self._load(recipe_path, adjust_path)

    def _load(self, recipe_path: str, adjust_path: Optional[str]) -> None:
        try:
            with open(recipe_path, "r", encoding="utf-8") as f:
                self._entries = [_parse_recipe_line(json.loads(line)) for line in f if line.strip()]
            self._entries.sort(key=lambda x: (x.at_step, x.at_event))
            self._recipe_mtime = safe_mtime(recipe_path)
        except FileNotFoundError:
            self._entries = []
        except Exception as exc:
            import logging
            logging.getLogger("hotcb.recipe").warning("Failed to load recipe %s: %s", recipe_path, exc)
            self._entries = []

        # Apply overlay if adjust file exists
        if adjust_path:
            self._adjust_mtime = safe_mtime(adjust_path)
            try:
                adjust_data = _load_adjust_file(adjust_path)
                self._entries = apply_overlay(self._entries, adjust_data)
                # Optionally write effective recipe snapshot
                if self.effective_recipe_path:
                    try:
                        write_effective_recipe(self._entries, self.effective_recipe_path)
                    except Exception:
                        pass  # best-effort
            except FileNotFoundError:
                pass
            except Exception:
                pass  # overlay errors don't block replay

    def reload_if_needed(self, recipe_path: Optional[str], adjust_path: Optional[str], step_offset: int) -> None:
        if recipe_path != self.recipe_path:
            self.recipe_path = recipe_path
            self.adjust_path = adjust_path
            self.step_offset = int(step_offset or 0)
            self._idx = 0
            if recipe_path:
                self._load(recipe_path, adjust_path)
            return

        step_offset = int(step_offset or 0)
        if step_offset != self.step_offset:
            self.step_offset = step_offset

        if adjust_path != self.adjust_path:
            self.adjust_path = adjust_path

        reload = False
        if recipe_path and safe_mtime(recipe_path) != self._recipe_mtime:
            reload = True
        if adjust_path and safe_mtime(adjust_path) != self._adjust_mtime:
            reload = True
        if reload and recipe_path:
            self._idx = 0
            self._load(recipe_path, adjust_path)

    def ops_for(self, step: int, event: str) -> List[HotOp]:
        """
        Return replay ops scheduled for (step,event). Advances internal cursor.
        """
        target_step = step - self.step_offset
        out: List[HotOp] = []
        while self._idx < len(self._entries):
            entry = self._entries[self._idx]
            if entry.at_step > target_step:
                break
            if entry.at_step == target_step and entry.at_event == event:
                rec = entry.record
                target = None
                if rec.get("target"):
                    t = rec["target"]
                    target = CallbackTarget(kind=str(t["kind"]), path=str(t["path"]), symbol=str(t["symbol"]))
                op = HotOp(
                    module=str(rec.get("module")),
                    op=str(rec.get("op")),
                    id=str(rec.get("id")) if rec.get("id") is not None else None,
                    params=rec.get("params"),
                    init=rec.get("init"),
                    enabled=rec.get("enabled"),
                    target=target,
                    source="replay",
                    raw=rec,
                )
                out.append(op)
                self._idx += 1
                continue

            # if step has passed, advance pointer to avoid stalling
            if entry.at_step < target_step:
                self._idx += 1
                continue
            break
        return out

    @property
    def remaining_entries(self) -> List[RecipeEntry]:
        """Return entries that haven't been consumed yet (for strict policy checks)."""
        return self._entries[self._idx:]

    @property
    def all_entries(self) -> List[RecipeEntry]:
        """Return all loaded entries."""
        return list(self._entries)
