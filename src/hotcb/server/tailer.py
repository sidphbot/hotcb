"""
File tailer — incrementally reads JSONL files and pushes new records
to registered async callbacks (WebSocket broadcast, projection engine, etc.).

Uses polling (no watchdog dependency) with configurable interval.
Reuses the existing ``FileCursor`` / ``read_new_jsonl`` infrastructure.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

from ..util import FileCursor, read_new_jsonl

log = logging.getLogger("hotcb.server.tailer")

# Type alias for async subscriber callbacks
Subscriber = Callable[[str, List[dict]], Coroutine[Any, Any, None]]


@dataclass
class TailTarget:
    """A single JSONL file being tailed."""
    name: str
    cursor: FileCursor
    subscribers: List[Subscriber] = field(default_factory=list)


class JsonlTailer:
    """
    Background async task that polls one or more JSONL files and
    pushes new records to subscribers.

    Usage::

        tailer = JsonlTailer(poll_interval=0.5)
        tailer.watch("metrics", "/run/hotcb.metrics.jsonl")
        tailer.watch("applied", "/run/hotcb.applied.jsonl")
        tailer.subscribe("metrics", my_ws_broadcast)

        # In FastAPI lifespan:
        task = asyncio.create_task(tailer.run())
        ...
        tailer.stop()
        await task
    """

    def __init__(self, poll_interval: float = 0.5) -> None:
        self._poll_interval = poll_interval
        self._targets: Dict[str, TailTarget] = {}
        self._running = False

    def watch(self, name: str, path: str) -> None:
        """Register a JSONL file to tail."""
        self._targets[name] = TailTarget(
            name=name,
            cursor=FileCursor(path=path, offset=0),
        )

    def watch_from_end(self, name: str, path: str) -> None:
        """Register a JSONL file to tail, starting from the current end."""
        import os
        offset = 0
        try:
            offset = os.path.getsize(path)
        except OSError:
            pass
        self._targets[name] = TailTarget(
            name=name,
            cursor=FileCursor(path=path, offset=offset),
        )

    def subscribe(self, name: str, callback: Subscriber) -> None:
        """Add a subscriber for a watched file."""
        target = self._targets.get(name)
        if target is None:
            raise ValueError(f"No watched file named {name!r}")
        target.subscribers.append(callback)

    def unsubscribe(self, name: str, callback: Subscriber) -> None:
        """Remove a subscriber."""
        target = self._targets.get(name)
        if target is not None:
            try:
                target.subscribers.remove(callback)
            except ValueError:
                pass

    async def run(self) -> None:
        """Poll loop — call as an asyncio task."""
        self._running = True
        while self._running:
            for target in self._targets.values():
                await self._poll_target(target)
            await asyncio.sleep(self._poll_interval)

    def stop(self) -> None:
        """Signal the poll loop to exit."""
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def poll_once(self) -> Dict[str, List[dict]]:
        """
        Single poll pass — useful for testing.
        Returns dict of {name: [new_records]}.
        """
        result: Dict[str, List[dict]] = {}
        for target in self._targets.values():
            records = await self._poll_target(target)
            if records:
                result[target.name] = records
        return result

    async def _poll_target(self, target: TailTarget) -> List[dict]:
        """Read new records from one target and dispatch to subscribers."""
        try:
            # Run blocking file I/O in a thread to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            records, new_cursor = await loop.run_in_executor(
                None, read_new_jsonl, target.cursor
            )
            target.cursor = new_cursor
        except Exception as e:
            log.warning("Tailer error on %s: %s", target.name, e)
            return []

        if not records:
            return []

        for sub in target.subscribers:
            try:
                await sub(target.name, records)
            except Exception as e:
                log.warning("Subscriber error on %s: %s", target.name, e)

        return records

    def get_cursor_offsets(self) -> Dict[str, int]:
        """Return current byte offsets for all targets (diagnostic)."""
        return {name: t.cursor.offset for name, t in self._targets.items()}
