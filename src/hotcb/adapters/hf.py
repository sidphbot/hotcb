from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from hotcb.kernel import HotKernel
from hotcb.capabilities import TrainingCapabilities, validate_mutable_state
from hotcb.actuators import optimizer_actuators, mutable_state as make_mutable_state


class HotCBHFCallback(TrainerCallback):
    """
    HuggingFace Trainer adapter for hotcb.

    Connects HF Trainer hooks to HotKernel, exposing optimizer and
    mutable_state in the env dict for hotopt/hotloss modules.

    Multi-optimizer support
    -----------------------
    If ``resolve_optimizers`` is provided (returns a list), all are
    exposed via ``env["optimizers"]``.  Falls back to ``resolve_optimizer``
    (single) for backward compat.

    Mutable state auto-detection
    ----------------------------
    If ``mutable_state`` is not explicitly provided, the adapter looks for
    a ``mutable_state`` attribute on the model passed to ``on_step_end``.
    """

    def __init__(
        self,
        kernel: HotKernel,
        train_events: Optional[List[str]] = None,
        eval_events: Optional[List[str]] = None,
        resolve_optimizer: Optional[Callable] = None,
        resolve_optimizers: Optional[Callable] = None,
        mutable_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.kernel = kernel
        self.train_events = train_events or ["train_step_end"]
        self.eval_events = eval_events or ["eval_end"]
        self._resolve_optimizer = resolve_optimizer
        self._resolve_optimizers = resolve_optimizers
        self._mutable_state = mutable_state
        self._capabilities: Optional[TrainingCapabilities] = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any):
        self._capabilities = self._detect_capabilities(args, kwargs)
        if self.kernel.run_dir:
            self._capabilities.save(self.kernel.run_dir)
        # Auto-create mutable_state from resolved optimizer if kernel doesn't have one
        if self.kernel._mutable_state is None:
            optimizers = self._resolve_all_optimizers()
            if optimizers:
                try:
                    acts = optimizer_actuators(optimizers[0])
                    self.kernel._mutable_state = make_mutable_state(acts)
                except Exception:
                    pass  # defensive — don't block training
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

    # -- capabilities detection --------------------------------------------

    def _detect_capabilities(
        self,
        args: TrainingArguments,
        kwargs: Dict[str, Any],
    ) -> TrainingCapabilities:
        opt_names: list[str] = []
        group_counts: list[int] = []
        optimizers = self._resolve_all_optimizers()
        for opt in optimizers:
            opt_names.append(type(opt).__name__)
            group_counts.append(len(opt.param_groups))

        accum = getattr(args, "gradient_accumulation_steps", 1) or 1
        clip_val = getattr(args, "max_grad_norm", None)
        clip_wired = clip_val is not None and clip_val > 0

        # Mutable state
        ls_detected = False
        ls_keys: list[str] = []
        ls = self._resolve_mutable_state_from_model(kwargs.get("model"))
        if ls is not None:
            ls_detected = True
            ls_keys = list(ls.get("weights", {}).keys())

        return TrainingCapabilities(
            framework="hf",
            num_optimizers=max(len(opt_names), 1),
            optimizer_names=tuple(opt_names),
            num_param_groups=tuple(group_counts),
            grad_accumulation_steps=accum,
            mutable_state_detected=ls_detected,
            mutable_state_keys=tuple(ls_keys),
            grad_clip_value=float(clip_val) if clip_val else None,
            grad_clip_wired=clip_wired,
        )

    # -- env builder -------------------------------------------------------

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

        if self._capabilities is not None:
            env["capabilities"] = self._capabilities

        # -- metric accessor --
        eval_metrics = extra.get("metrics") or {}

        def _metric(name: str, default: Any = None) -> Any:
            if name in eval_metrics:
                return eval_metrics[name]
            if state.log_history:
                for entry in reversed(state.log_history):
                    if name in entry:
                        return entry[name]
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

        # -- max_steps --
        try:
            if hasattr(args, "max_steps") and args.max_steps and args.max_steps > 0:
                env["max_steps"] = args.max_steps
        except Exception:
            pass

        # -- all optimizers (multi-optimizer support) --
        optimizers = self._resolve_all_optimizers()
        if optimizers:
            env["optimizer"] = optimizers[0]
            env["optimizers"] = optimizers
        elif self._resolve_optimizer is not None:
            try:
                opt = self._resolve_optimizer()
                if opt is not None:
                    env["optimizer"] = opt
                    env["optimizers"] = [opt]
            except Exception:
                pass

        # -- mutable_state (explicit > auto-detected from model) --
        if self._mutable_state is not None:
            valid, _ = validate_mutable_state(self._mutable_state)
            if valid:
                env["mutable_state"] = self._mutable_state
        else:
            model = extra.get("model")
            ls = self._resolve_mutable_state_from_model(model)
            if ls is not None:
                env["mutable_state"] = ls

        # -- metrics dict for discovery --
        if eval_metrics:
            all_metrics: Dict[str, float] = {}
            for k, v in eval_metrics.items():
                try:
                    all_metrics[k] = float(v)
                except (TypeError, ValueError):
                    pass
            if state.log_history:
                latest = state.log_history[-1]
                for k, v in latest.items():
                    if k not in all_metrics:
                        try:
                            all_metrics[k] = float(v)
                        except (TypeError, ValueError):
                            pass
            if all_metrics:
                env["metrics"] = all_metrics

        env.update(extra)
        return env

    # -- helpers -----------------------------------------------------------

    def _resolve_all_optimizers(self) -> list:
        """Return list of all optimizers."""
        if self._resolve_optimizers is not None:
            try:
                result = self._resolve_optimizers()
                if isinstance(result, (list, tuple)):
                    return list(result)
            except Exception:
                pass
        if self._resolve_optimizer is not None:
            try:
                opt = self._resolve_optimizer()
                if opt is not None:
                    return [opt]
            except Exception:
                pass
        return []

    def _resolve_mutable_state_from_model(self, model: Any) -> Optional[Dict[str, Any]]:
        """Try to find mutable_state on the model."""
        if model is None:
            return None
        if hasattr(model, "mutable_state"):
            valid, _ = validate_mutable_state(model.mutable_state)
            if valid:
                return model.mutable_state
        return None
