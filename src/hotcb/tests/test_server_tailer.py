"""Tests for hotcb.server.tailer — JsonlTailer."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile

import pytest

from hotcb.server.tailer import JsonlTailer
from hotcb.util import append_jsonl


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestJsonlTailerBasic:
    """Core tailing functionality."""

    def test_watch_registers_target(self, tmp_dir):
        tailer = JsonlTailer()
        path = os.path.join(tmp_dir, "test.jsonl")
        tailer.watch("test", path)
        assert "test" in tailer._targets
        assert tailer._targets["test"].cursor.offset == 0

    def test_watch_from_end(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.jsonl")
        # Write some data first
        append_jsonl(path, {"a": 1})
        append_jsonl(path, {"a": 2})
        size = os.path.getsize(path)

        tailer = JsonlTailer()
        tailer.watch_from_end("test", path)
        assert tailer._targets["test"].cursor.offset == size

    @pytest.mark.asyncio
    async def test_poll_once_reads_new_records(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.jsonl")
        tailer = JsonlTailer()
        tailer.watch("test", path)

        # No file yet
        result = await tailer.poll_once()
        assert result == {}

        # Write records
        append_jsonl(path, {"step": 1})
        append_jsonl(path, {"step": 2})

        result = await tailer.poll_once()
        assert "test" in result
        assert len(result["test"]) == 2
        assert result["test"][0]["step"] == 1

    @pytest.mark.asyncio
    async def test_incremental_reads(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.jsonl")
        tailer = JsonlTailer()
        tailer.watch("test", path)

        append_jsonl(path, {"step": 1})
        r1 = await tailer.poll_once()
        assert len(r1["test"]) == 1

        append_jsonl(path, {"step": 2})
        append_jsonl(path, {"step": 3})
        r2 = await tailer.poll_once()
        assert len(r2["test"]) == 2
        assert r2["test"][0]["step"] == 2

    @pytest.mark.asyncio
    async def test_subscriber_receives_records(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.jsonl")
        tailer = JsonlTailer()
        tailer.watch("test", path)

        received = []

        async def on_records(name, records):
            received.extend(records)

        tailer.subscribe("test", on_records)

        append_jsonl(path, {"step": 1})
        append_jsonl(path, {"step": 2})
        await tailer.poll_once()

        assert len(received) == 2
        assert received[0]["step"] == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.jsonl")
        tailer = JsonlTailer()
        tailer.watch("test", path)

        received = []

        async def on_records(name, records):
            received.extend(records)

        tailer.subscribe("test", on_records)
        tailer.unsubscribe("test", on_records)

        append_jsonl(path, {"step": 1})
        await tailer.poll_once()
        assert len(received) == 0

    def test_subscribe_unknown_raises(self, tmp_dir):
        tailer = JsonlTailer()
        with pytest.raises(ValueError, match="No watched file"):
            tailer.subscribe("nonexistent", lambda n, r: None)

    @pytest.mark.asyncio
    async def test_subscriber_error_does_not_break_tailer(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.jsonl")
        tailer = JsonlTailer()
        tailer.watch("test", path)

        good_received = []

        async def bad_sub(name, records):
            raise RuntimeError("boom")

        async def good_sub(name, records):
            good_received.extend(records)

        tailer.subscribe("test", bad_sub)
        tailer.subscribe("test", good_sub)

        append_jsonl(path, {"step": 1})
        await tailer.poll_once()
        # Good subscriber still got the records
        assert len(good_received) == 1

    @pytest.mark.asyncio
    async def test_multiple_targets(self, tmp_dir):
        path_a = os.path.join(tmp_dir, "a.jsonl")
        path_b = os.path.join(tmp_dir, "b.jsonl")
        tailer = JsonlTailer()
        tailer.watch("a", path_a)
        tailer.watch("b", path_b)

        received_a, received_b = [], []

        async def on_a(name, records):
            received_a.extend(records)

        async def on_b(name, records):
            received_b.extend(records)

        tailer.subscribe("a", on_a)
        tailer.subscribe("b", on_b)

        append_jsonl(path_a, {"source": "a"})
        append_jsonl(path_b, {"source": "b1"})
        append_jsonl(path_b, {"source": "b2"})

        await tailer.poll_once()
        assert len(received_a) == 1
        assert len(received_b) == 2

    def test_get_cursor_offsets(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.jsonl")
        tailer = JsonlTailer()
        tailer.watch("test", path)
        offsets = tailer.get_cursor_offsets()
        assert offsets["test"] == 0

    @pytest.mark.asyncio
    async def test_run_and_stop(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.jsonl")
        tailer = JsonlTailer(poll_interval=0.05)
        tailer.watch("test", path)

        received = []

        async def on_records(name, records):
            received.extend(records)

        tailer.subscribe("test", on_records)

        task = asyncio.create_task(tailer.run())
        await asyncio.sleep(0.05)  # let the task start
        assert tailer.is_running

        # Write while running
        append_jsonl(path, {"step": 1})
        await asyncio.sleep(0.15)
        assert len(received) >= 1

        tailer.stop()
        await asyncio.sleep(0.1)
        assert not tailer.is_running
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
