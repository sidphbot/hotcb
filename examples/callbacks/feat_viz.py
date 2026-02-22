from __future__ import annotations
from typing import Any, Dict
import os
import json


class FeatureVizCallback:
    def __init__(self, id: str, every: int = 200, out_dir: str = "debug/features") -> None:
        self.id = id
        self.every = int(every)
        self.out_dir = out_dir

    def set_params(self, **kwargs: Any) -> None:
        if "every" in kwargs:
            self.every = int(kwargs["every"])
        if "out_dir" in kwargs:
            self.out_dir = str(kwargs["out_dir"])

    def handle(self, event: str, env: Dict[str, Any]) -> None:
        step = int(env.get("step", 0))
        if self.every <= 0 or (step % self.every) != 0:
            return

        # Example artifact: dump lightweight metadata
        os.makedirs(self.out_dir, exist_ok=True)
        payload = {
            "id": self.id,
            "event": event,
            "step": step,
            "phase": env.get("phase"),
            "framework": env.get("framework"),
        }
        p = os.path.join(self.out_dir, f"feat_viz_step_{step:07d}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        log = env.get("log", print)
        log(f"[feat_viz] wrote {p}")