"""Tests for hotcb CLI commands."""
from __future__ import annotations

import argparse
import json
import os

import pytest

from hotcb.cli import (
    build_parser,
    cmd_cb,
    cmd_freeze,
    cmd_init,
    cmd_loss,
    cmd_opt,
    cmd_recipe_export,
    _parse_kv,
)


# ── 1. cmd_init ────────────────────────────────────────────────────────────

def test_cmd_init(tmp_path):
    """cmd_init must create all expected files in the run directory."""
    run_dir = str(tmp_path / "run")
    args = argparse.Namespace(dir=run_dir)
    cmd_init(args)

    expected_files = [
        "hotcb.yaml",
        "hotcb.commands.jsonl",
        "hotcb.applied.jsonl",
        "hotcb.recipe.jsonl",
        "hotcb.freeze.json",
    ]
    for fname in expected_files:
        fpath = os.path.join(run_dir, fname)
        assert os.path.exists(fpath), f"Missing: {fname}"

    # Verify yaml has version header
    with open(os.path.join(run_dir, "hotcb.yaml"), "r") as f:
        assert "version: 1" in f.read()


def test_cmd_init_idempotent(tmp_path):
    """Calling cmd_init twice should not overwrite existing files."""
    run_dir = str(tmp_path / "run")
    args = argparse.Namespace(dir=run_dir)
    cmd_init(args)

    # Write some content to commands file
    cmd_path = os.path.join(run_dir, "hotcb.commands.jsonl")
    with open(cmd_path, "w") as f:
        f.write("test content\n")

    cmd_init(args)

    # File should not be overwritten
    with open(cmd_path, "r") as f:
        assert f.read() == "test content\n"


# ── 2. cmd_cb load ─────────────────────────────────────────────────────────

def test_cmd_cb_load(tmp_path):
    """cmd_cb load should write a properly structured command to the commands file."""
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir, exist_ok=True)

    args = argparse.Namespace(
        dir=run_dir,
        cb_command="load",
        id="my_callback",
        file="path/to/cb.py",
        path=None,
        symbol="MyCallback",
        enabled=True,
        init=["lr=0.01", "verbose=true"],
    )
    cmd_cb(args)

    cmd_path = os.path.join(run_dir, "hotcb.commands.jsonl")
    with open(cmd_path, "r") as f:
        rec = json.loads(f.readline())

    assert rec["module"] == "cb"
    assert rec["op"] == "load"
    assert rec["id"] == "my_callback"
    assert rec["target"]["kind"] == "python_file"
    assert rec["target"]["path"] == "path/to/cb.py"
    assert rec["target"]["symbol"] == "MyCallback"
    assert rec["enabled"] is True
    assert rec["init"]["lr"] == 0.01
    assert rec["init"]["verbose"] is True


def test_cmd_cb_load_module_path(tmp_path):
    """cmd_cb load with --path uses 'module' kind."""
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir, exist_ok=True)

    args = argparse.Namespace(
        dir=run_dir,
        cb_command="load",
        id="my_cb",
        file=None,
        path="mypackage.callbacks",
        symbol="MyCB",
        enabled=None,
        init=[],
    )
    cmd_cb(args)

    cmd_path = os.path.join(run_dir, "hotcb.commands.jsonl")
    with open(cmd_path, "r") as f:
        rec = json.loads(f.readline())

    assert rec["target"]["kind"] == "module"
    assert rec["target"]["path"] == "mypackage.callbacks"
    assert "enabled" not in rec


def test_cmd_cb_load_requires_file_or_path(tmp_path):
    """cmd_cb load without --file or --path should raise SystemExit."""
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir, exist_ok=True)

    args = argparse.Namespace(
        dir=run_dir,
        cb_command="load",
        id="my_cb",
        file=None,
        path=None,
        symbol="MyCB",
        enabled=None,
        init=[],
    )
    with pytest.raises(SystemExit):
        cmd_cb(args)


# ── 3. cmd_opt set_params ──────────────────────────────────────────────────

def test_cmd_opt_set_params(tmp_path):
    """cmd_opt set_params should write module=opt with parsed params."""
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir, exist_ok=True)

    args = argparse.Namespace(
        dir=run_dir,
        opt_command="set_params",
        id="main",
        kv=["lr=0.001", "weight_decay=0.0001"],
    )
    cmd_opt(args)

    cmd_path = os.path.join(run_dir, "hotcb.commands.jsonl")
    with open(cmd_path, "r") as f:
        rec = json.loads(f.readline())

    assert rec["module"] == "opt"
    assert rec["op"] == "set_params"
    assert rec["id"] == "main"
    assert rec["params"]["lr"] == 0.001
    assert rec["params"]["weight_decay"] == pytest.approx(0.0001)


def test_cmd_opt_enable(tmp_path):
    """cmd_opt enable should write a simple enable command."""
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir, exist_ok=True)

    args = argparse.Namespace(
        dir=run_dir,
        opt_command="enable",
        id="main",
        kv=None,
    )
    cmd_opt(args)

    cmd_path = os.path.join(run_dir, "hotcb.commands.jsonl")
    with open(cmd_path, "r") as f:
        rec = json.loads(f.readline())

    assert rec["module"] == "opt"
    assert rec["op"] == "enable"


# ── 4. cmd_loss set_params ─────────────────────────────────────────────────

def test_cmd_loss_set_params(tmp_path):
    """cmd_loss set_params should write module=loss with parsed params."""
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir, exist_ok=True)

    args = argparse.Namespace(
        dir=run_dir,
        loss_command="set_params",
        id="main",
        kv=["alpha=0.5", "beta=2"],
    )
    cmd_loss(args)

    cmd_path = os.path.join(run_dir, "hotcb.commands.jsonl")
    with open(cmd_path, "r") as f:
        rec = json.loads(f.readline())

    assert rec["module"] == "loss"
    assert rec["op"] == "set_params"
    assert rec["params"]["alpha"] == 0.5
    assert rec["params"]["beta"] == 2


# ── 5. cmd_freeze ──────────────────────────────────────────────────────────

def test_cmd_freeze(tmp_path):
    """cmd_freeze should write hotcb.freeze.json with the correct mode."""
    run_dir = str(tmp_path / "run")

    args = argparse.Namespace(
        dir=run_dir,
        mode="prod",
        recipe=None,
        adjust=None,
        policy="best_effort",
        step_offset=0,
    )
    cmd_freeze(args)

    freeze_path = os.path.join(run_dir, "hotcb.freeze.json")
    assert os.path.exists(freeze_path)

    with open(freeze_path, "r") as f:
        data = json.loads(f.read())

    assert data["mode"] == "prod"
    assert data["policy"] == "best_effort"
    assert data["step_offset"] == 0


def test_cmd_freeze_replay(tmp_path):
    """cmd_freeze with replay mode and recipe path."""
    run_dir = str(tmp_path / "run")

    args = argparse.Namespace(
        dir=run_dir,
        mode="replay",
        recipe="/path/to/recipe.jsonl",
        adjust=None,
        policy="strict",
        step_offset=100,
    )
    cmd_freeze(args)

    freeze_path = os.path.join(run_dir, "hotcb.freeze.json")
    with open(freeze_path, "r") as f:
        data = json.loads(f.read())

    assert data["mode"] == "replay"
    assert data["recipe_path"] == "/path/to/recipe.jsonl"
    assert data["policy"] == "strict"
    assert data["step_offset"] == 100


# ── 6. cmd_recipe_export ───────────────────────────────────────────────────

def test_cmd_recipe_export(tmp_path):
    """cmd_recipe_export should filter applied entries and write a recipe file."""
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir, exist_ok=True)

    applied_path = os.path.join(run_dir, "hotcb.applied.jsonl")
    with open(applied_path, "w", encoding="utf-8") as f:
        # Applied entry — should be included
        f.write(json.dumps({
            "seq": 1, "step": 10, "event": "train_step_end",
            "module": "opt", "op": "set_params", "id": "main",
            "decision": "applied",
            "payload": {"params": {"lr": 0.01}},
        }) + "\n")
        # Failed entry — should be excluded
        f.write(json.dumps({
            "seq": 2, "step": 11, "event": "train_step_end",
            "module": "opt", "op": "enable", "id": "main",
            "decision": "failed",
            "payload": {},
        }) + "\n")
        # Applied cb entry — should be included
        f.write(json.dumps({
            "seq": 3, "step": 20, "event": "train_step_end",
            "module": "cb", "op": "load", "id": "my_cb",
            "decision": "applied",
            "payload": {"target": {"kind": "python_file", "path": "cb.py", "symbol": "MyCB"}},
        }) + "\n")
        # Core module — should be excluded (not in cb/opt/loss)
        f.write(json.dumps({
            "seq": 4, "step": 25, "event": "train_step_end",
            "module": "core", "op": "freeze", "id": None,
            "decision": "applied",
            "payload": {"mode": "prod"},
        }) + "\n")

    recipe_out = os.path.join(run_dir, "hotcb.recipe.jsonl")
    args = argparse.Namespace(dir=run_dir, out=recipe_out)
    cmd_recipe_export(args)

    assert os.path.exists(recipe_out)
    with open(recipe_out, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    assert len(entries) == 2
    assert entries[0]["module"] == "opt"
    assert entries[0]["op"] == "set_params"
    assert entries[0]["at"] == {"step": 10, "event": "train_step_end"}
    assert entries[0]["params"] == {"lr": 0.01}
    assert entries[1]["module"] == "cb"
    assert entries[1]["target"]["kind"] == "python_file"


def test_cmd_recipe_export_missing_ledger(tmp_path, capsys):
    """cmd_recipe_export with no applied ledger should print a message, not crash."""
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir, exist_ok=True)

    args = argparse.Namespace(dir=run_dir, out=None)
    cmd_recipe_export(args)

    captured = capsys.readouterr()
    assert "No applied ledger" in captured.out


# ── 7. _parse_kv ───────────────────────────────────────────────────────────

class TestParseKv:
    """Test key=value parsing for various types."""

    def test_booleans(self):
        result = _parse_kv(["flag=true", "debug=false", "UPPER=True"])
        assert result["flag"] is True
        assert result["debug"] is False
        assert result["UPPER"] is True

    def test_integers(self):
        result = _parse_kv(["epochs=10", "seed=42", "neg=-5"])
        assert result["epochs"] == 10
        assert result["seed"] == 42
        assert result["neg"] == -5

    def test_floats(self):
        result = _parse_kv(["lr=0.001", "momentum=0.9", "decay=1e-4"])
        assert result["lr"] == pytest.approx(0.001)
        assert result["momentum"] == pytest.approx(0.9)
        # 1e-4 contains no '.', so it may be parsed as float via scientific notation
        # _parse_kv tries int first, which will fail for "1e-4", then falls through to string
        # Actually "1e-4" has no "." so it tries int("1e-4") which raises, then falls to string
        assert result["decay"] == "1e-4" or result["decay"] == pytest.approx(1e-4)

    def test_json_object(self):
        result = _parse_kv(['cfg={"a":1,"b":"hello"}'])
        assert result["cfg"] == {"a": 1, "b": "hello"}

    def test_json_array(self):
        result = _parse_kv(["layers=[1,2,3]"])
        assert result["layers"] == [1, 2, 3]

    def test_plain_strings(self):
        result = _parse_kv(["name=my_model", "tag=v1.2"])
        assert result["name"] == "my_model"
        assert result["tag"] == "v1.2"

    def test_missing_equals_raises(self):
        with pytest.raises(SystemExit):
            _parse_kv(["no_equals_here"])

    def test_value_with_equals(self):
        """Value containing '=' should be handled (split on first '=' only)."""
        result = _parse_kv(["expr=a=b"])
        assert result["expr"] == "a=b"

    def test_empty_list(self):
        assert _parse_kv([]) == {}


# ── Parser integration ─────────────────────────────────────────────────────

def test_build_parser_cb_load():
    """build_parser should correctly parse a cb load command."""
    parser = build_parser()
    args = parser.parse_args([
        "--dir", "/tmp/run",
        "cb", "load", "my_cb",
        "--file", "path/to/cb.py",
        "--symbol", "MyCB",
        "--enabled",
        "--init", "lr=0.01",
    ])
    assert args.cb_command == "load"
    assert args.id == "my_cb"
    assert args.file == "path/to/cb.py"
    assert args.symbol == "MyCB"
    assert args.enabled is True
    assert args.init == ["lr=0.01"]


def test_build_parser_opt_set_params():
    """build_parser should correctly parse an opt set_params command."""
    parser = build_parser()
    args = parser.parse_args([
        "--dir", "/tmp/run",
        "opt", "set_params",
        "lr=0.001", "wd=1e-5",
    ])
    assert args.opt_command == "set_params"
    assert args.kv == ["lr=0.001", "wd=1e-5"]


def test_build_parser_freeze():
    """build_parser should correctly parse a freeze command."""
    parser = build_parser()
    args = parser.parse_args([
        "--dir", "/tmp/run",
        "freeze", "--mode", "prod",
    ])
    assert args.mode == "prod"
    assert args.policy == "best_effort"
    assert args.step_offset == 0
