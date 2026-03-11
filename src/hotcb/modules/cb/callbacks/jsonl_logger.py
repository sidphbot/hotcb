# src/hotcb/callbacks/jsonl_logger.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import os

from .utils import get_in, get_log, safe_int, tensor_basic_stats, to_float


class JSONLLoggerCallback:
    """
    Append compact training/eval records to a JSONL file for later analysis.

    This is a "poor man's telemetry": you can plot it after training with pandas,
    send it to a dashboard, or use it for regression comparisons.

    Parameters
    ----------
    id:
        Callback identifier.

    every:
        Write every N steps.

    out_path:
        Path to JSONL output file. Directories are created automatically.

    scalars:
        List of env paths that should be recorded as float scalars.
        Each path is resolved via get_in(env, path). If convertible to float,
        it is recorded under the same key name.

        Typical entries:
          - "loss"
          - "metrics.eval_loss"

    tensor_paths:
        List of env paths for which to also record tensor_basic_stats.
        The stats dict is written under "<path>.stats" key.

    Output record format
    --------------------
    Each JSON line includes:
      - id, event, step, epoch, phase, framework
      - scalar values (best-effort)
      - tensor stats entries (best-effort)

    Example (YAML)
    -------------
    callbacks:
      jsonl:
        enabled: true
        target: { kind: module, path: hotcb.callbacks.jsonl_logger, symbol: JSONLLoggerCallback }
        init:
          every: 20
          out_path: runs/exp1/metrics.jsonl
          scalars: ["loss"]
          tensor_paths: ["outputs.logits"]

    Example (CLI)
    -------------
    $ hotcb --dir runs/exp1 enable jsonl
    $ hotcb --dir runs/exp1 set jsonl out_path=runs/exp1/metrics.jsonl every=10
    """

    def __init__(
        self,
        id: str,
        every: int = 10,
        out_path: str = "hotcb_metrics.jsonl",
        scalars: Optional[List[str]] = None,
        tensor_paths: Optional[List[str]] = None,
    ) -> None:
        self.id = id
        self.every = int(every)
        self.out_path = str(out_path)
        self.scalars = scalars or ["loss"]
        self.tensor_paths = tensor_paths or []

    def set_params(self, **kwargs: Any) -> None:
        """
        Supported hot params
        --------------------
        every: int
        out_path: str
        scalars: list[str] or comma-separated string
        tensor_paths: list[str] or comma-separated string
        """
        if "every" in kwargs:
            self.every = int(kwargs["every"])
        if "out_path" in kwargs:
            self.out_path = str(kwargs["out_path"])
        if "scalars" in kwargs:
            v = kwargs["scalars"]
            self.scalars = [s.strip() for s in v.split(",")] if isinstance(v, str) else list(v)
        if "tensor_paths" in kwargs:
            v = kwargs["tensor_paths"]
            self.tensor_paths = [s.strip() for s in v.split(",")] if isinstance(v, str) else list(v)

    def handle(self, event: str, env: Dict[str, Any]) -> None:
        """
        Write one JSONL record at configured frequency.

        Parameters
        ----------
        event:
            Event name string.

        env:
            Environment dict; recorded fields are best-effort.
        """
        step = safe_int(env.get("step", 0))
        if self.every <= 0 or step % self.every != 0:
            return

        rec: Dict[str, Any] = {
            "id": self.id,
            "event": event,
            "step": step,
            "epoch": env.get("epoch"),
            "phase": env.get("phase"),
            "framework": env.get("framework"),
        }

        for p in self.scalars:
            v = get_in(env, p)
            fv = to_float(v)
            if fv is not None:
                rec[p] = fv

        for p in self.tensor_paths:
            st = tensor_basic_stats(get_in(env, p))
            if st is not None:
                rec[f"{p}.stats"] = st

        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        log = get_log(env)
        log(f"[hotcb:{self.id}] wrote jsonl step={step} -> {self.out_path}")