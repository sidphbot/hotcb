from __future__ import annotations
from typing import Any, Dict, Optional, List

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from hotcb import HotController


class HotHFCallback(TrainerCallback):
    """
    HF Trainer adapter.

    We'll poll/apply at step end and eval end by default.
    """
    def __init__(
        self,
        controller: HotController,
        train_events: Optional[List[str]] = None,
        eval_events: Optional[List[str]] = None,
    ) -> None:
        self.controller = controller
        self.train_events = train_events or ["train_step_end"]
        self.eval_events = eval_events or ["eval_end"]

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any):
        env = self._env(args, state, control, phase="train_begin", **kwargs)
        self.controller.apply(env, events=["train_begin"])
        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any):
        env = self._env(args, state, control, phase="train", **kwargs)
        self.controller.apply(env, events=self.train_events)
        return control

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs: Any):
        env = self._env(args, state, control, phase="eval", metrics=metrics, **kwargs)
        self.controller.apply(env, events=self.eval_events)
        return control

    def _env(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, phase: str, **extra: Any) -> Dict[str, Any]:
        def _log(s: str) -> None:
            print(s)

        env: Dict[str, Any] = {
            "framework": "hf",
            "phase": phase,
            "step": int(getattr(state, "global_step", 0)),
            "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
            "args": args,
            "state": state,
            "control": control,
            "log": _log,
        }
        env.update(extra)
        return env