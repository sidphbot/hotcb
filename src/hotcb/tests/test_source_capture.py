"""Tests for hotcb source capture / versioning system (spec §19.5)."""
from __future__ import annotations

import hashlib
import json
import os
import textwrap

import pytest

from hotcb.kernel import HotKernel
from hotcb.modules.cb import _capture_source

# ---------------------------------------------------------------------------
# Helpers
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
    """Write a callback source file and return its absolute path."""
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(source)
    return path


def _sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# 1. Capture on load
# ---------------------------------------------------------------------------

class TestCaptureOnLoad:
    """Loading a python_file callback through the kernel must capture its source."""

    def test_captured_file_exists(self, run_dir, make_env, write_commands, read_ledger, tmp_path):
        cb_path = _write_cb_file(str(tmp_path), MINIMAL_CB_SRC)
        write_commands({
            "module": "cb",
            "op": "load",
            "id": "test_cb",
            "target": {"kind": "python_file", "path": cb_path, "symbol": "TestCb"},
        })

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        env = make_env(step=1)
        kernel.apply(env, events=["train_step_end"])

        sources_dir = os.path.join(run_dir, "hotcb.sources")
        expected_sha = _sha256_of(MINIMAL_CB_SRC.encode("utf-8"))
        captured = os.path.join(sources_dir, f"{expected_sha}.py")

        assert os.path.isfile(captured), f"Captured file not found at {captured}"

    def test_ledger_contains_source_capture(self, run_dir, make_env, write_commands, read_ledger, tmp_path):
        cb_path = _write_cb_file(str(tmp_path), MINIMAL_CB_SRC)
        write_commands({
            "module": "cb",
            "op": "load",
            "id": "test_cb",
            "target": {"kind": "python_file", "path": cb_path, "symbol": "TestCb"},
        })

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        env = make_env(step=1)
        kernel.apply(env, events=["train_step_end"])

        entries = read_ledger()
        load_entries = [e for e in entries if e.get("op") == "load"]
        assert len(load_entries) == 1

        payload = load_entries[0].get("payload", {})
        assert "source_capture" in payload, f"payload missing source_capture: {payload}"

        sc = payload["source_capture"]
        expected_sha = _sha256_of(MINIMAL_CB_SRC.encode("utf-8"))
        assert sc["sha256"] == expected_sha
        assert sc["captured_path"].endswith(f"{expected_sha}.py")


# ---------------------------------------------------------------------------
# 2. Captured file content matches original
# ---------------------------------------------------------------------------

class TestCapturedContentMatches:
    """The captured file must be byte-identical to the original source."""

    def test_byte_identical(self, run_dir, make_env, write_commands, read_ledger, tmp_path):
        cb_path = _write_cb_file(str(tmp_path), MINIMAL_CB_SRC)
        write_commands({
            "module": "cb",
            "op": "load",
            "id": "test_cb",
            "target": {"kind": "python_file", "path": cb_path, "symbol": "TestCb"},
        })

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        kernel.apply(make_env(step=1), events=["train_step_end"])

        sources_dir = os.path.join(run_dir, "hotcb.sources")
        sha = _sha256_of(MINIMAL_CB_SRC.encode("utf-8"))
        captured = os.path.join(sources_dir, f"{sha}.py")

        with open(cb_path, "rb") as f:
            original_bytes = f.read()
        with open(captured, "rb") as f:
            captured_bytes = f.read()

        assert captured_bytes == original_bytes


# ---------------------------------------------------------------------------
# 3. Replay uses captured version
# ---------------------------------------------------------------------------

class TestReplayUsesCaptured:
    """After capture, modifying the original file must not affect the captured copy."""

    def test_captured_differs_from_modified_original(self, run_dir, make_env, write_commands, read_ledger, tmp_path):
        cb_path = _write_cb_file(str(tmp_path), MINIMAL_CB_SRC)
        write_commands({
            "module": "cb",
            "op": "load",
            "id": "test_cb",
            "target": {"kind": "python_file", "path": cb_path, "symbol": "TestCb"},
        })

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        kernel.apply(make_env(step=1), events=["train_step_end"])

        # Read back captured path from ledger
        entries = read_ledger()
        load_entry = [e for e in entries if e.get("op") == "load"][0]
        captured_path = load_entry["payload"]["source_capture"]["captured_path"]

        # Now modify the original file
        modified_src = MINIMAL_CB_SRC.replace("TestCb", "ModifiedCb")
        with open(cb_path, "w", encoding="utf-8") as f:
            f.write(modified_src)

        # The captured file must still contain the original source
        with open(captured_path, "rb") as f:
            captured_bytes = f.read()

        assert captured_bytes == MINIMAL_CB_SRC.encode("utf-8")
        assert b"ModifiedCb" not in captured_bytes

    def test_capture_source_returns_captured_path(self, tmp_path):
        """_capture_source returns captured_path that points to an independent copy."""
        out_dir = str(tmp_path / "sources")
        cb_path = _write_cb_file(str(tmp_path), MINIMAL_CB_SRC)

        result = _capture_source(cb_path, out_dir)
        assert result is not None
        captured_path = result["captured_path"]

        # Modify original
        with open(cb_path, "w", encoding="utf-8") as f:
            f.write("# completely replaced\n")

        # Captured copy unchanged
        with open(captured_path, "rb") as f:
            assert f.read() == MINIMAL_CB_SRC.encode("utf-8")


# ---------------------------------------------------------------------------
# 4. Fallback when captured file missing
# ---------------------------------------------------------------------------

class TestFallbackCapturedMissing:
    """_capture_source returns None when the source file does not exist."""

    def test_returns_none_for_missing_file(self, tmp_path):
        out_dir = str(tmp_path / "sources")
        missing_path = str(tmp_path / "nonexistent.py")

        result = _capture_source(missing_path, out_dir)
        assert result is None

    def test_returns_none_does_not_create_dir(self, tmp_path):
        out_dir = str(tmp_path / "sources")
        missing_path = str(tmp_path / "nonexistent.py")

        _capture_source(missing_path, out_dir)
        # out_dir should not have been created since there was nothing to write
        assert not os.path.exists(out_dir)


# ---------------------------------------------------------------------------
# 5. Idempotent capture
# ---------------------------------------------------------------------------

class TestIdempotentCapture:
    """Loading the same file twice must not produce duplicate captured files."""

    def test_same_sha_single_file(self, tmp_path):
        """Two captures of the same content produce the same path, file written once."""
        out_dir = str(tmp_path / "sources")
        cb_path = _write_cb_file(str(tmp_path), MINIMAL_CB_SRC)

        r1 = _capture_source(cb_path, out_dir)
        assert r1 is not None
        mtime1 = os.path.getmtime(r1["captured_path"])

        r2 = _capture_source(cb_path, out_dir)
        assert r2 is not None

        # Same sha and path
        assert r1["sha256"] == r2["sha256"]
        assert r1["captured_path"] == r2["captured_path"]

        # File was not rewritten (mtime unchanged)
        mtime2 = os.path.getmtime(r2["captured_path"])
        assert mtime1 == mtime2

    def test_idempotent_via_kernel(self, run_dir, make_env, write_commands, read_ledger, tmp_path):
        """Loading the same callback twice through the kernel writes the captured file once."""
        cb_path = _write_cb_file(str(tmp_path), MINIMAL_CB_SRC)
        cmd = {
            "module": "cb",
            "op": "load",
            "id": "test_cb",
            "target": {"kind": "python_file", "path": cb_path, "symbol": "TestCb"},
        }

        write_commands(cmd)
        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        kernel.apply(make_env(step=1), events=["train_step_end"])

        sources_dir = os.path.join(run_dir, "hotcb.sources")
        sha = _sha256_of(MINIMAL_CB_SRC.encode("utf-8"))
        captured = os.path.join(sources_dir, f"{sha}.py")
        mtime_after_first = os.path.getmtime(captured)

        # Second load of the same file
        write_commands(cmd)
        kernel.apply(make_env(step=2), events=["train_step_end"])

        mtime_after_second = os.path.getmtime(captured)
        assert mtime_after_first == mtime_after_second

        # Only one file in sources dir
        files = os.listdir(sources_dir)
        assert len(files) == 1
