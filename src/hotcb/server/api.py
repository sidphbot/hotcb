"""
hotcb.server.api — REST API router for interactive training controls.

All command endpoints communicate with the training process by appending
JSON objects to ``hotcb.commands.jsonl`` in the run directory.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ..util import append_jsonl

router = APIRouter(prefix="/api", tags=["commands"])


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class OptSetRequest(BaseModel):
    id: str = "main"
    params: Dict[str, Any] = Field(..., min_length=1)


class LossSetRequest(BaseModel):
    id: str = "main"
    params: Dict[str, Any] = Field(..., min_length=1)


class TuneModeRequest(BaseModel):
    mode: str = Field(..., pattern=r"^(active|observe|suggest|off)$")


class FreezeRequest(BaseModel):
    mode: str = Field(..., pattern=r"^(off|prod|replay|replay_adjusted)$")
    recipe_path: Optional[str] = None
    adjust_path: Optional[str] = None
    policy: str = "best_effort"
    step_offset: int = 0


class ScheduleRequest(BaseModel):
    at_step: int = Field(..., gt=0)
    module: str = Field(..., pattern=r"^(opt|loss|cb|tune)$")
    op: str
    id: str = "main"
    params: Dict[str, Any] = Field(default_factory=dict)


class ValidateRequest(BaseModel):
    module: str = Field(..., pattern=r"^(opt|loss|cb|tune)$")
    op: str
    id: str = "main"
    params: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cmd_path(request: Request) -> str:
    run_dir: str = request.app.state.run_dir
    return os.path.join(run_dir, "hotcb.commands.jsonl")


def _append(request: Request, obj: Dict[str, Any]) -> None:
    append_jsonl(_cmd_path(request), obj)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/opt/set")
async def opt_set(body: OptSetRequest, request: Request):
    cmd = {"module": "opt", "op": "set_params", "id": body.id, "params": body.params}
    _append(request, cmd)
    return {"status": "queued", "command": cmd}


@router.post("/loss/set")
async def loss_set(body: LossSetRequest, request: Request):
    cmd = {"module": "loss", "op": "set_params", "id": body.id, "params": body.params}
    _append(request, cmd)
    return {"status": "queued", "command": cmd}


@router.get("/loss/params")
async def loss_params(request: Request):
    """Return current loss weight values from the latest metrics or applied ledger."""
    run_dir: str = request.app.state.run_dir

    # Strategy 1: read from latest metrics (most up-to-date)
    metrics_path = os.path.join(run_dir, "hotcb.metrics.jsonl")
    latest_metrics: Dict[str, Any] = {}
    if os.path.exists(metrics_path):
        last_line = ""
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    last_line = stripped
        if last_line:
            try:
                rec = json.loads(last_line)
                latest_metrics = rec.get("metrics", {})
            except json.JSONDecodeError:
                pass

    # Strategy 2: read last applied loss params
    applied_path = os.path.join(run_dir, "hotcb.applied.jsonl")
    last_applied: Dict[str, Any] = {}
    if os.path.exists(applied_path):
        with open(applied_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if rec.get("module") == "loss" and rec.get("decision") == "applied":
                    p = rec.get("params") or rec.get("payload")
                    if p:
                        last_applied.update(p)

    # Strategy 3: read from capabilities file for key names
    from ..capabilities import TrainingCapabilities
    caps = TrainingCapabilities.load(run_dir)
    mutable_keys = list(caps.mutable_state_keys) if caps and caps.mutable_state_keys else []

    # Extract weight-related metrics (w/* prefix from training loop logging)
    current_weights: Dict[str, float] = {}
    for k, v in latest_metrics.items():
        if k.startswith("train_w/"):
            key = k.replace("train_w/", "")
            current_weights[key] = v

    return {
        "mutable_state_keys": mutable_keys,
        "current_weights": current_weights,
        "last_applied": last_applied,
    }


@router.post("/tune/mode")
async def tune_mode(body: TuneModeRequest, request: Request):
    op = "enable" if body.mode != "off" else "disable"
    cmd: Dict[str, Any] = {"module": "tune", "op": op}
    if op == "enable":
        cmd["params"] = {"mode": body.mode}
    _append(request, cmd)
    return {"status": "queued", "command": cmd}


@router.get("/cb/list")
async def cb_list(request: Request):
    """Return list of registered callbacks."""
    run_dir: str = request.app.state.run_dir
    applied_path = os.path.join(run_dir, "hotcb.applied.jsonl")

    # Start with server-side registry
    cb_registry = getattr(request.app.state, 'cb_registry', {})
    cbs: Dict[str, Dict[str, Any]] = dict(cb_registry)  # copy

    # Merge from applied ledger
    if os.path.exists(applied_path):
        with open(applied_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("module") == "cb":
                    cb_id = rec.get("id", "unknown")
                    if cb_id == "unknown":
                        continue  # skip entries with no proper id
                    op = rec.get("op", "")
                    cbs[cb_id] = {
                        "id": cb_id,
                        "enabled": op not in ("disable", "unload"),
                        "last_op": op,
                        "last_step": rec.get("step", 0),
                        "params": rec.get("params", {}),
                    }
    return {"callbacks": list(cbs.values())}


class CbLoadRequest(BaseModel):
    id: str = Field(..., min_length=1)
    path: str = Field(..., min_length=1)


@router.post("/cb/load")
async def cb_load(body: CbLoadRequest, request: Request):
    """Load a new callback from a host file path."""
    cmd = {"module": "cb", "op": "load", "id": body.id, "params": {"path": body.path}}
    _append(request, cmd)
    # Also track in server-side registry
    cb_registry = getattr(request.app.state, 'cb_registry', {})
    cb_registry[body.id] = {
        "id": body.id,
        "enabled": True,
        "last_op": "load",
        "last_step": 0,
        "params": {"path": body.path},
    }
    request.app.state.cb_registry = cb_registry
    return {"status": "queued", "command": cmd}


class CbSetParamsRequest(BaseModel):
    id: str = "main"
    params: Dict[str, Any] = Field(..., min_length=1)


@router.post("/cb/set_params")
async def cb_set_params(body: CbSetParamsRequest, request: Request):
    """Set parameters on a callback."""
    cmd = {"module": "cb", "op": "set_params", "id": body.id, "params": body.params}
    _append(request, cmd)
    return {"status": "queued", "command": cmd}


@router.post("/cb/{cb_id}/enable")
async def cb_enable(cb_id: str, request: Request):
    cmd = {"module": "cb", "op": "enable", "id": cb_id}
    _append(request, cmd)
    cb_registry = getattr(request.app.state, 'cb_registry', {})
    if cb_id in cb_registry:
        cb_registry[cb_id]["enabled"] = True
        cb_registry[cb_id]["last_op"] = "enable"
    return {"status": "queued", "command": cmd}


@router.post("/cb/{cb_id}/disable")
async def cb_disable(cb_id: str, request: Request):
    cmd = {"module": "cb", "op": "disable", "id": cb_id}
    _append(request, cmd)
    cb_registry = getattr(request.app.state, 'cb_registry', {})
    if cb_id in cb_registry:
        cb_registry[cb_id]["enabled"] = False
        cb_registry[cb_id]["last_op"] = "disable"
    return {"status": "queued", "command": cmd}


@router.post("/cb/{cb_id}/unload")
async def cb_unload(cb_id: str, request: Request):
    """Unload a callback, removing it from the active set."""
    cmd = {"module": "cb", "op": "unload", "id": cb_id}
    _append(request, cmd)
    cb_registry = getattr(request.app.state, 'cb_registry', {})
    cb_registry.pop(cb_id, None)
    request.app.state.cb_registry = cb_registry
    return {"status": "queued", "command": cmd}


@router.post("/freeze")
async def freeze(body: FreezeRequest, request: Request):
    run_dir: str = request.app.state.run_dir
    freeze_path = os.path.join(run_dir, "hotcb.freeze.json")
    cfg = {
        "mode": body.mode,
        "recipe_path": body.recipe_path,
        "adjust_path": body.adjust_path,
        "policy": body.policy,
        "step_offset": body.step_offset,
    }
    os.makedirs(os.path.dirname(freeze_path) or ".", exist_ok=True)
    with open(freeze_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(cfg))
    return {"status": "written", "path": freeze_path, "config": cfg}


@router.post("/schedule")
async def schedule(body: ScheduleRequest, request: Request):
    cmd: Dict[str, Any] = {
        "at": {"step": body.at_step},
        "module": body.module,
        "op": body.op,
        "id": body.id,
    }
    if body.params:
        cmd["params"] = body.params
    run_dir: str = request.app.state.run_dir
    recipe_path = os.path.join(run_dir, "hotcb.recipe.jsonl")
    append_jsonl(recipe_path, cmd)
    # Ensure recipe editor picks up the new entry on next fetch
    editor = getattr(request.app.state, "recipe_editor", None)
    if editor is not None:
        editor.load()
    return {"status": "scheduled", "command": cmd}


@router.post("/validate")
async def validate(body: ValidateRequest, request: Request):
    """Dry-run validation of a mutation against known bounds."""
    errors: List[str] = []

    # Basic structural validation
    valid_ops = {
        "opt": {"set_params", "enable", "disable"},
        "loss": {"set_params", "enable", "disable"},
        "cb": {"enable", "disable", "load", "unload", "set_params"},
        "tune": {"enable", "disable", "set"},
    }
    allowed = valid_ops.get(body.module, set())
    if body.op not in allowed:
        errors.append(f"Unknown op '{body.op}' for module '{body.module}'. Allowed: {sorted(allowed)}")

    if body.op == "set_params" and not body.params:
        errors.append("set_params requires non-empty params")

    # Check for known param bounds (actuator-style validation)
    if body.module == "opt" and body.op == "set_params":
        lr = body.params.get("lr")
        if lr is not None:
            if not isinstance(lr, (int, float)) or lr <= 0:
                errors.append(f"lr must be a positive number, got {lr!r}")
        wd = body.params.get("weight_decay")
        if wd is not None:
            if not isinstance(wd, (int, float)) or wd < 0:
                errors.append(f"weight_decay must be non-negative, got {wd!r}")

    if errors:
        return {"valid": False, "errors": errors}
    return {"valid": True, "errors": []}


# ---------------------------------------------------------------------------
# Natural-language chat endpoint (vibe-coder mode)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)


# Rule-based NL → command translation
_NL_PATTERNS: List[Dict[str, Any]] = [
    {"keywords": ["learn faster", "speed up", "faster learning"],
     "cmd": {"module": "opt", "op": "set_params", "id": "main",
             "params": {"lr": "__current__ * 2.0"}},
     "reply": "Doubling the learning rate to speed up learning."},
    {"keywords": ["learn slower", "slow down", "slower learning", "more careful"],
     "cmd": {"module": "opt", "op": "set_params", "id": "main",
             "params": {"lr": "__current__ * 0.5"}},
     "reply": "Halving the learning rate for more careful learning."},
    {"keywords": ["reduce overfitting", "less overfitting", "regularize", "regularise"],
     "cmd": {"module": "opt", "op": "set_params", "id": "main",
             "params": {"weight_decay": "__current__ * 2.0"}},
     "reply": "Doubling weight decay to reduce overfitting."},
    {"keywords": ["stop tuning", "disable tuning", "turn off tune", "stop tune"],
     "cmd": {"module": "tune", "op": "disable"},
     "reply": "Disabling auto-tune."},
    {"keywords": ["start tuning", "enable tuning", "turn on tune", "start tune"],
     "cmd": {"module": "tune", "op": "enable", "params": {"mode": "active"}},
     "reply": "Enabling auto-tune in active mode."},
    {"keywords": ["freeze", "lock", "production mode"],
     "cmd": None,
     "reply": "To freeze training, use the freeze toggle in the control bar."},
    {"keywords": ["pause", "stop training"],
     "cmd": None,
     "reply": "Training pause is not directly supported. Use freeze mode to lock parameters."},
]


def _match_nl(message: str) -> Optional[Dict[str, Any]]:
    """Match a natural-language message to a known command pattern."""
    msg_lower = message.lower().strip()
    for pattern in _NL_PATTERNS:
        for kw in pattern["keywords"]:
            if kw in msg_lower:
                return pattern
    return None


@router.post("/chat")
async def chat(body: ChatRequest, request: Request):
    """Translate a natural-language message into a hotcb command."""
    match = _match_nl(body.message)
    if match is None:
        return {
            "reply": (
                f"I don't understand \"{body.message}\". "
                "Try things like: \"learn faster\", \"reduce overfitting\", "
                "\"start tuning\", or \"slow down\"."
            ),
            "command": None,
        }

    cmd = match.get("cmd")
    if cmd is not None:
        _append(request, cmd)
        return {"reply": match["reply"], "command": cmd, "status": "queued"}
    return {"reply": match["reply"], "command": None}


# ---------------------------------------------------------------------------
# Save applied mutations as recipe
# ---------------------------------------------------------------------------

@router.post("/applied/save-as-recipe")
async def save_applied_as_recipe(request: Request):
    """Convert applied mutation history into a recipe file."""
    run_dir: str = request.app.state.run_dir
    applied_path = os.path.join(run_dir, "hotcb.applied.jsonl")
    recipe_path = os.path.join(run_dir, "hotcb.recipe.jsonl")

    if not os.path.exists(applied_path):
        return {"status": "error", "detail": "No applied mutations found"}

    entries: List[Dict[str, Any]] = []
    with open(applied_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Only convert successfully applied mutations
            if rec.get("status") != "applied" and rec.get("decision") != "applied":
                continue
            entry: Dict[str, Any] = {
                "at": {"step": rec.get("step", 0)},
                "module": rec.get("module", "opt"),
                "op": rec.get("op", "set_params"),
            }
            if rec.get("params"):
                entry["params"] = rec["params"]
            if rec.get("description"):
                entry["description"] = rec["description"]
            entries.append(entry)

    with open(recipe_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Reload the recipe editor so it picks up the new file
    from .recipe_editor import RecipeEditor
    request.app.state.recipe_editor = RecipeEditor(recipe_path)

    return {
        "status": "saved",
        "path": recipe_path,
        "count": len(entries),
    }
