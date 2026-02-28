from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

import pytest


@pytest.fixture
def run_dir(tmp_path):
    """Create a temporary run directory with standard hotcb layout."""
    d = tmp_path / "run"
    d.mkdir()
    return str(d)


@pytest.fixture
def make_env():
    """Factory for creating env dicts used by kernel.apply()."""
    def _make(step: int = 0, logs: List[str] | None = None, **extra: Any) -> Dict[str, Any]:
        logs = logs if logs is not None else []

        def _log(s: str) -> None:
            logs.append(s)

        env: Dict[str, Any] = {"step": step, "log": _log}
        env.update(extra)
        return env
    return _make


@pytest.fixture
def write_commands(run_dir):
    """Helper to append command dicts to hotcb.commands.jsonl."""
    cmd_path = os.path.join(run_dir, "hotcb.commands.jsonl")

    def _write(*cmds: dict) -> str:
        with open(cmd_path, "a", encoding="utf-8") as f:
            for c in cmds:
                f.write(json.dumps(c) + "\n")
        return cmd_path

    return _write


@pytest.fixture
def write_freeze(run_dir):
    """Helper to write hotcb.freeze.json."""
    freeze_path = os.path.join(run_dir, "hotcb.freeze.json")

    def _write(**kwargs) -> str:
        with open(freeze_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(kwargs))
        return freeze_path

    return _write


@pytest.fixture
def write_recipe(run_dir):
    """Helper to write recipe entries to hotcb.recipe.jsonl."""
    recipe_path = os.path.join(run_dir, "hotcb.recipe.jsonl")

    def _write(*entries: dict) -> str:
        with open(recipe_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        return recipe_path

    return _write


@pytest.fixture
def read_ledger(run_dir):
    """Helper to read all entries from hotcb.applied.jsonl."""
    ledger_path = os.path.join(run_dir, "hotcb.applied.jsonl")

    def _read() -> List[dict]:
        if not os.path.exists(ledger_path):
            return []
        entries = []
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    return _read
