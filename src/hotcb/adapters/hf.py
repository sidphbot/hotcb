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
        self.kernel.apply(env, events=self.eval_events + ["val_epoch_end"])
        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any):
        env = self._env(args, state, control, phase="train_end", **kwargs)
        self.kernel.apply(env, events=["run_end"])
        self.kernel.close(env)
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

        env["kernel"] = self.kernel

        # Metric accessor for hottune
        eval_metrics = extra.get("metrics") or {}

        def _metric(name: str, default: Any = None) -> Any:
            # Check eval metrics passed to on_evaluate
            if name in eval_metrics:
                return eval_metrics[name]
            # Check state log history
            if state.log_history:
                for entry in reversed(state.log_history):
                    if name in entry:
                        return entry[name]
            # Normalized env fields
            if name == "lr":
                opt = env.get("optimizer")
                if opt is not None:
                    try:
                        return opt.param_groups[0]["lr"]
                    except Exception:
                        pass
            if name in ("train/loss", "loss"):
                if state.log_history:
                    for entry in reversed(state.log_history):
                        if "loss" in entry:
                            return entry["loss"]
            return default

        env["metric"] = _metric

        # Expose max_steps for phase binning
        try:
            if hasattr(args, "max_steps") and args.max_steps and args.max_steps > 0:
                env["max_steps"] = args.max_steps
        except Exception:
            pass

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
