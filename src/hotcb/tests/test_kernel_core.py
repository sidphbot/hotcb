"""
Unit tests for hotcb core ingestion, routing, and ledger system (spec 19.2).
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from hotcb.kernel import HotKernel


# ---------------------------------------------------------------------------
# 1. JSONL tailing
# ---------------------------------------------------------------------------

class TestJsonlTailing:
    def test_read_initial_batch(self, run_dir, make_env, write_commands, read_ledger):
        """Write 3 commands, apply once, verify exactly 3 are read."""
        write_commands(
            {"module": "opt", "op": "enable", "id": "a"},
            {"module": "opt", "op": "enable", "id": "b"},
            {"module": "opt", "op": "enable", "id": "c"},
        )
        kernel = HotKernel(run_dir, debounce_steps=1)
        env = make_env(step=1, optimizer=_mock_optimizer())
        kernel.apply(env, ["on_train_begin"])

        ledger = read_ledger()
        assert len(ledger) == 3

    def test_incremental_read(self, run_dir, make_env, write_commands, read_ledger):
        """Write 3, apply, then write 2 more, apply again -- only 2 new."""
        write_commands(
            {"module": "opt", "op": "enable", "id": "a"},
            {"module": "opt", "op": "enable", "id": "b"},
            {"module": "opt", "op": "enable", "id": "c"},
        )
        kernel = HotKernel(run_dir, debounce_steps=1)
        env = make_env(step=1, optimizer=_mock_optimizer())
        kernel.apply(env, ["on_train_begin"])

        assert len(read_ledger()) == 3

        write_commands(
            {"module": "opt", "op": "enable", "id": "d"},
            {"module": "opt", "op": "enable", "id": "e"},
        )
        kernel.apply(env, ["on_train_begin"])

        ledger = read_ledger()
        assert len(ledger) == 5

    def test_cursor_increments(self, run_dir, make_env, write_commands):
        """Cursor offset grows after each apply."""
        write_commands({"module": "opt", "op": "enable", "id": "a"})
        kernel = HotKernel(run_dir, debounce_steps=1)
        env = make_env(step=1, optimizer=_mock_optimizer())

        offset_before = kernel._cmd_cursor.offset
        kernel.apply(env, ["step"])
        offset_after = kernel._cmd_cursor.offset
        assert offset_after > offset_before

        write_commands({"module": "opt", "op": "enable", "id": "b"})
        kernel.apply(env, ["step"])
        offset_final = kernel._cmd_cursor.offset
        assert offset_final > offset_after


# ---------------------------------------------------------------------------
# 2. Debounce
# ---------------------------------------------------------------------------

class TestDebounce:
    def test_debounce_skips_until_threshold(self, run_dir, make_env, write_commands, read_ledger):
        """With debounce_steps=5, first 4 calls produce nothing; 5th processes ops."""
        write_commands({"module": "opt", "op": "enable", "id": "x"})
        kernel = HotKernel(run_dir, debounce_steps=5)
        env = make_env(step=1, optimizer=_mock_optimizer())

        for _ in range(4):
            kernel.apply(env, ["step"])
        assert len(read_ledger()) == 0

        # 5th call triggers poll
        kernel.apply(env, ["step"])
        assert len(read_ledger()) == 1

    def test_debounce_cycles(self, run_dir, make_env, write_commands, read_ledger):
        """After the first trigger at step 5, next trigger is at step 10."""
        write_commands({"module": "opt", "op": "enable", "id": "x"})
        kernel = HotKernel(run_dir, debounce_steps=5)
        env = make_env(step=1, optimizer=_mock_optimizer())

        # Steps 1-5: trigger at 5
        for _ in range(5):
            kernel.apply(env, ["step"])
        assert len(read_ledger()) == 1

        # Write a new command for the next cycle
        write_commands({"module": "opt", "op": "enable", "id": "y"})

        # Steps 6-9: no trigger
        for _ in range(4):
            kernel.apply(env, ["step"])
        assert len(read_ledger()) == 1

        # Step 10: trigger again
        kernel.apply(env, ["step"])
        assert len(read_ledger()) == 2


# ---------------------------------------------------------------------------
# 3. Poll interval (time gate)
# ---------------------------------------------------------------------------

class TestPollInterval:
    def test_time_gate_blocks_frequent_polls(self, run_dir, make_env, write_commands, read_ledger):
        """With poll_interval_sec > 0, rapid apply calls are gated."""
        write_commands({"module": "opt", "op": "enable", "id": "a"})
        kernel = HotKernel(run_dir, debounce_steps=1, poll_interval_sec=10.0)
        env = make_env(step=1, optimizer=_mock_optimizer())

        # First call passes the time gate (last_poll_t=0)
        kernel.apply(env, ["step"])
        assert len(read_ledger()) == 1

        # Write more, but time gate blocks
        write_commands({"module": "opt", "op": "enable", "id": "b"})
        kernel.apply(env, ["step"])
        assert len(read_ledger()) == 1  # still 1

    def test_time_gate_allows_after_interval(self, run_dir, make_env, write_commands, read_ledger):
        """After enough time elapses, poll succeeds."""
        write_commands({"module": "opt", "op": "enable", "id": "a"})
        kernel = HotKernel(run_dir, debounce_steps=1, poll_interval_sec=0.5)
        env = make_env(step=1, optimizer=_mock_optimizer())

        kernel.apply(env, ["step"])
        assert len(read_ledger()) == 1

        write_commands({"module": "opt", "op": "enable", "id": "b"})

        # Immediately: blocked
        kernel.apply(env, ["step"])
        assert len(read_ledger()) == 1

        # Advance time past the interval by patching
        kernel._last_poll_t -= 1.0
        kernel.apply(env, ["step"])
        assert len(read_ledger()) == 2


# ---------------------------------------------------------------------------
# 4. Routing
# ---------------------------------------------------------------------------

class TestRouting:
    def test_route_to_opt(self, run_dir, make_env, write_commands, read_ledger):
        """Ops with module='opt' reach the opt controller."""
        optimizer = _mock_optimizer(lr=0.01)
        write_commands({"module": "opt", "op": "set_params", "params": {"lr": 0.002}})
        kernel = HotKernel(run_dir, debounce_steps=1)
        env = make_env(step=1, optimizer=optimizer)
        kernel.apply(env, ["step"])

        ledger = read_ledger()
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "applied"
        assert ledger[0]["module"] == "opt"
        # Verify the optimizer was actually mutated
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.002)

    def test_route_to_loss(self, run_dir, make_env, write_commands, read_ledger):
        """Ops with module='loss' reach the loss controller."""
        mutable_state = {"weights": {}, "terms": {}, "ramps": {}}
        write_commands({"module": "loss", "op": "set_params", "params": {"kl_w": 0.5}})
        kernel = HotKernel(run_dir, debounce_steps=1)
        env = make_env(step=1, mutable_state=mutable_state)
        kernel.apply(env, ["step"])

        ledger = read_ledger()
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "applied"
        assert ledger[0]["module"] == "loss"
        assert mutable_state["weights"]["kl"] == 0.5

    def test_route_to_core(self, run_dir, make_env, write_commands, read_ledger):
        """Ops with module='core' are handled by the kernel itself."""
        write_commands({"module": "core", "op": "unfreeze"})
        kernel = HotKernel(run_dir, debounce_steps=1)
        env = make_env(step=1)
        kernel.apply(env, ["step"])

        ledger = read_ledger()
        assert len(ledger) == 1
        assert ledger[0]["module"] == "core"
        assert ledger[0]["decision"] == "applied"

    def test_route_to_cb(self, run_dir, make_env, write_commands, read_ledger):
        """Ops with module='cb' reach the callback module."""
        # Just sending an enable op to cb; we don't need a real callback loaded.
        write_commands({"module": "cb", "op": "enable", "id": "my_cb"})
        kernel = HotKernel(run_dir, debounce_steps=1)
        env = make_env(step=1)
        kernel.apply(env, ["step"])

        ledger = read_ledger()
        assert len(ledger) == 1
        assert ledger[0]["module"] == "cb"

    def test_opt_missing_optimizer_fails(self, run_dir, make_env, write_commands, read_ledger):
        """set_params on opt without an optimizer produces failed."""
        write_commands({"module": "opt", "op": "set_params", "params": {"lr": 0.1}})
        kernel = HotKernel(run_dir, debounce_steps=1)
        env = make_env(step=1)  # no optimizer
        kernel.apply(env, ["step"])

        ledger = read_ledger()
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "failed"
        assert "missing_optimizer" in (ledger[0].get("error") or "")


# ---------------------------------------------------------------------------
# 5. Ledger correctness
# ---------------------------------------------------------------------------

class TestLedgerCorrectness:
    def test_one_record_per_op(self, run_dir, make_env, write_commands, read_ledger):
        """Every processed op produces exactly one ledger record."""
        write_commands(
            {"module": "opt", "op": "enable", "id": "a"},
            {"module": "opt", "op": "enable", "id": "b"},
            {"module": "opt", "op": "enable", "id": "c"},
        )
        kernel = HotKernel(run_dir, debounce_steps=1)
        env = make_env(step=1, optimizer=_mock_optimizer())
        kernel.apply(env, ["step"])

        assert len(read_ledger()) == 3

    def test_seq_monotonically_increasing(self, run_dir, make_env, write_commands, read_ledger):
        """Seq values are monotonically increasing."""
        write_commands(
            {"module": "opt", "op": "enable", "id": "a"},
            {"module": "opt", "op": "enable", "id": "b"},
        )
        kernel = HotKernel(run_dir, debounce_steps=1)
        env = make_env(step=1, optimizer=_mock_optimizer())
        kernel.apply(env, ["step"])

        write_commands({"module": "opt", "op": "enable", "id": "c"})
        kernel.apply(env, ["step"])

        ledger = read_ledger()
        seqs = [e["seq"] for e in ledger]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)  # all unique
        # Strictly increasing
        for i in range(1, len(seqs)):
            assert seqs[i] > seqs[i - 1]

    def test_required_fields_populated(self, run_dir, make_env, write_commands, read_ledger):
        """step, event, source, decision fields are present in every ledger record."""
        write_commands({"module": "opt", "op": "enable", "id": "a"})
        kernel = HotKernel(run_dir, debounce_steps=1)
        env = make_env(step=42, optimizer=_mock_optimizer())
        kernel.apply(env, ["on_batch_end"])

        ledger = read_ledger()
        assert len(ledger) == 1
        entry = ledger[0]
        assert entry["step"] == 42
        assert entry["event"] == "on_batch_end"
        assert entry["source"] == "external"
        assert entry["decision"] == "applied"
        assert "seq" in entry

    def test_failure_includes_error_text(self, run_dir, make_env, write_commands, read_ledger):
        """Failed ops have error text in the ledger."""
        write_commands({"module": "opt", "op": "set_params", "params": {"lr": 0.1}})
        kernel = HotKernel(run_dir, debounce_steps=1)
        env = make_env(step=1)  # no optimizer -> failure
        kernel.apply(env, ["step"])

        ledger = read_ledger()
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "failed"
        assert ledger[0]["error"] is not None
        assert len(ledger[0]["error"]) > 0


# ---------------------------------------------------------------------------
# 6. Unknown module
# ---------------------------------------------------------------------------

class TestUnknownModule:
    def test_unknown_module_fails(self, run_dir, make_env, write_commands, read_ledger):
        """Op with an unknown module is recorded as failed with error text."""
        write_commands({"module": "xyz", "op": "enable", "id": "thing"})
        kernel = HotKernel(run_dir, debounce_steps=1)
        env = make_env(step=1)
        kernel.apply(env, ["step"])

        ledger = read_ledger()
        assert len(ledger) == 1
        assert ledger[0]["decision"] == "failed"
        assert "unknown_module" in ledger[0]["error"]
        assert "xyz" in ledger[0]["error"]

    def test_unknown_module_still_produces_ledger_record(self, run_dir, make_env, write_commands, read_ledger):
        """Unknown modules still get a ledger entry with correct fields."""
        write_commands({"module": "xyz", "op": "do_thing"})
        kernel = HotKernel(run_dir, debounce_steps=1)
        env = make_env(step=5)
        kernel.apply(env, ["on_epoch_end"])

        ledger = read_ledger()
        assert len(ledger) == 1
        entry = ledger[0]
        assert entry["module"] == "xyz"
        assert entry["step"] == 5
        assert entry["event"] == "on_epoch_end"
        assert entry["source"] == "external"
        assert entry["seq"] >= 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockOptimizer:
    """Minimal mock optimizer with param_groups."""

    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0):
        self.param_groups = [{"lr": lr, "weight_decay": weight_decay}]


def _mock_optimizer(lr: float = 0.01, weight_decay: float = 0.0) -> _MockOptimizer:
    return _MockOptimizer(lr=lr, weight_decay=weight_decay)
