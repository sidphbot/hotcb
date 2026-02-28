from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from hotcb.kernel import HotKernel


class HotCBHFCallback(TrainerCallback):
    """
    HuggingFace Trainer adapter for hotcb.

    Connects HF Trainer hooks to HotKernel, exposing optimizer and
    loss_state in the env dict for hotopt/hotloss modules.
    """

    def __init__(
        self,
        kernel: HotKernel,
        train_events: Optional[List[str]] = None,
        eval_events: Optional[List[str]] = None,
        resolve_optimizer: Optional[Callable] = None,
        loss_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.kernel = kernel
        self.train_events = train_events or ["train_step_end"]
        self.eval_events = eval_events or ["eval_end"]
        self._resolve_optimizer = resolve_optimizer
        self._loss_state = loss_state

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any):
        env = self._env(args, state, control, phase="train_begin", **kwargs)
        self.kernel.apply(env, events=["train_begin"])
        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any):
        env = self._env(args, state, control, phase="train", **kwargs)
        self.kernel.apply(env, events=self.train_events)
        return control

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs: Any):
        env = self._env(args, state, control, phase="eval", metrics=metrics, **kwargs)
        self.kernel.apply(env, events=self.eval_events)
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

        # Expose optimizer for hotopt
        if self._resolve_optimizer is not None:
            try:
                opt = self._resolve_optimizer()
                if opt is not None:
                    env["optimizer"] = opt
            except Exception:
                pass

        # Expose loss_state for hotloss
        if self._loss_state is not None:
            env["loss_state"] = self._loss_state

        env.update(extra)
        return env
