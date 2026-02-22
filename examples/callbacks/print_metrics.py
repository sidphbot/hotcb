from __future__ import annotations
from typing import Any, Dict


class PrintMetricsCallback:
    def __init__(self, id: str, every: int = 50, prefix: str = "[metrics]") -> None:
        self.id = id
        self.every = int(every)
        self.prefix = str(prefix)

    def set_params(self, **kwargs: Any) -> None:
        if "every" in kwargs:
            self.every = int(kwargs["every"])
        if "prefix" in kwargs:
            self.prefix = str(kwargs["prefix"])

    def handle(self, event: str, env: Dict[str, Any]) -> None:
        step = int(env.get("step", 0))
        if self.every <= 0 or (step % self.every) != 0:
            return
        log = env.get("log", print)
        phase = env.get("phase")
        log(f"{self.prefix} id={self.id} event={event} step={step} phase={phase}")