from __future__ import annotations

from typing import Any, Dict, Optional, List

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .. import HotController


class HotHFCallback(TrainerCallback):
    """
    HuggingFace `transformers.Trainer` adapter for `hotcb`.

    This adapter bridges the HuggingFace Trainer callback system to the
    framework-agnostic `HotController` by:

      1) Defining "safe points" in the HF training lifecycle where it is safe
         to poll and apply hot updates (enable/disable callbacks, update params,
         load new callbacks), and

      2) Constructing a stable, framework-agnostic `env` dictionary to pass to
         `HotController.apply(env, events=[...])`.

    The adapter is intentionally *minimal*:
      - It does not alter the training loop.
      - It does not assume distributed training (no DDP synchronization).
      - It does not attempt to infer or mutate optimizer / scheduler state.

    Safe points / hooks used
    ------------------------
    - `on_train_begin`: emits a one-time "train_begin" event.
    - `on_step_end`: emits events configured by `train_events` (default:
      ["train_step_end"]).
    - `on_evaluate`: emits events configured by `eval_events` (default:
      ["eval_end"]).

    These hooks are chosen because they are typically stable, called frequently
    (step end), and do not occur mid-backward.

    Parameters
    ----------
    controller:
        The `HotController` instance that owns callback registry, control-plane
        polling (YAML/JSONL), and dispatch.

        Your HF training script should create it with consistent paths, e.g.:

        >>> controller = HotController(
        ...     config_path="runs/exp1/hotcb.yaml",
        ...     commands_path="runs/exp1/hotcb.commands.jsonl",
        ...     debounce_steps=10,
        ...     log_path="runs/exp1/hotcb.log",
        ... )

    train_events:
        List of event names to dispatch at `on_step_end`.

        Values:
          - Any non-empty list of strings is accepted.
          - These strings are purely *conventional*; hotcb does not enforce a
            fixed schema.

        Recommended defaults:
          - ["train_step_end"] (default)

        Example:
          - ["train_step_end", "timing_tick"] if you want multiple callbacks to
            respond to different "channels".

    eval_events:
        List of event names to dispatch at `on_evaluate`.

        Recommended defaults:
          - ["eval_end"] (default)

        Example:
          - ["eval_end", "metrics_ready"]

    Environment (`env`) contract
    ----------------------------
    The adapter builds an `env: Dict[str, Any]` containing:

      - "framework": str
            Always "hf" for this adapter.

      - "phase": str
            One of:
              - "train_begin"
              - "train"
              - "eval"
            (You can add additional phases by extending hooks.)

      - "step": int
            Derived from `state.global_step` (falls back to 0).

      - "epoch": float
            Derived from `state.epoch` (falls back to 0.0). HF uses float epochs.

      - "args": TrainingArguments
            HF Trainer arguments.

      - "state": TrainerState
            HF trainer state.

      - "control": TrainerControl
            HF trainer control object.

      - "log": Callable[[str], None]
            A basic logging function. Defaults to `print`.

    Additionally, the adapter merges any extra keyword arguments provided by
    HuggingFace callbacks into env (e.g., `metrics` for evaluation).

    Important notes
    ---------------
    - This adapter is designed to be safe and low-coupling. It does not assume
      specific keys in `kwargs`, except those it explicitly sets/receives.
    - If you want to expose `loss` for diagnostics callbacks, the best hook is
      `on_log` or by injecting custom values via a custom Trainer. This baseline
      does not guarantee `loss` is present in env.

    Example usage
    -------------
    >>> from transformers import Trainer
    >>> from .. import HotController
    >>> from hotcb.adapters.hf import HotHFCallback
    >>>
    >>> controller = HotController(
    ...     config_path="runs/exp1/hotcb.yaml",
    ...     commands_path="runs/exp1/hotcb.commands.jsonl",
    ...     debounce_steps=10,
    ... )
    >>> trainer = Trainer(..., callbacks=[HotHFCallback(controller)])
    >>> trainer.train()

    Live control from another terminal:
    >>> hotcb --dir runs/exp1 enable timing
    >>> hotcb --dir runs/exp1 set timing every=10 window=200
    >>> hotcb --dir runs/exp1 load my_diag --file /tmp/my_diag.py --symbol MyDiag --enabled --init msg="hi"
    """

    def __init__(
        self,
        controller: HotController,
        train_events: Optional[List[str]] = None,
        eval_events: Optional[List[str]] = None,
    ) -> None:
        """
        Create a HuggingFace Trainer adapter for hotcb.

        Parameters
        ----------
        controller:
            HotController instance to poll/apply updates and dispatch events.

        train_events:
            Event names to dispatch on each training step end. If None,
            defaults to ["train_step_end"].

        eval_events:
            Event names to dispatch when evaluation completes. If None,
            defaults to ["eval_end"].

        Notes
        -----
        - `train_events` and `eval_events` should be treated as stable public
          "event channels" for your callback ecosystem. Keep them consistent
          across runs if you rely on config-driven callbacks.
        """
        self.controller = controller
        self.train_events = train_events or ["train_step_end"]
        self.eval_events = eval_events or ["eval_end"]

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ):
        """
        HF hook: called when training begins.

        This is a safe point to:
          - apply initial hotcb config/commands (load callbacks, set params),
          - dispatch a one-time "train_begin" event.

        Parameters
        ----------
        args:
            TrainingArguments for the Trainer.

        state:
            TrainerState, including `global_step`, `epoch`, etc.

        control:
            TrainerControl used by HF to influence flow (pause/stop/etc).

        **kwargs:
            HF may pass additional data depending on version and call site.
            We attach all of it into env so your callbacks can access it.

        Returns
        -------
        TrainerControl
            Must return control per HF callback contract.

        Example
        -------
        A callback might use this to create output directories:

        >>> class MyCB:
        ...   def handle(self, event, env):
        ...     if event == "train_begin":
        ...       os.makedirs("runs/exp1/artifacts", exist_ok=True)
        """
        env = self._env(args, state, control, phase="train_begin", **kwargs)
        self.controller.apply(env, events=["train_begin"])
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ):
        """
        HF hook: called at the end of a training step.

        This is the primary hotcb "tick" for HF training: it is frequent and
        occurs at a safe boundary (not mid-backward).

        Behavior
        --------
        - Build `env` with current step/epoch and HF objects.
        - Call `controller.apply(env, events=self.train_events)`.

        Parameters
        ----------
        args, state, control:
            As provided by HF.

        **kwargs:
            Extra context from HF.

        Returns
        -------
        TrainerControl
            Returned unchanged.

        Common usage patterns
        ---------------------
        - Enable/disable callbacks live:
            $ hotcb --dir runs/exp1 enable sys
            $ hotcb --dir runs/exp1 disable sys

        - Adjust frequency live:
            $ hotcb --dir runs/exp1 set timing every=10
        """
        env = self._env(args, state, control, phase="train", **kwargs)
        self.controller.apply(env, events=self.train_events)
        return control

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics=None,
        **kwargs: Any,
    ):
        """
        HF hook: called after evaluation.

        This is a good place to:
          - write evaluation artifacts,
          - log metric summaries,
          - run diagnostic callbacks that depend on evaluation metrics.

        Behavior
        --------
        - Builds env with `phase="eval"` and includes `metrics` in env.
        - Dispatches `self.eval_events` (default: ["eval_end"]).

        Parameters
        ----------
        args, state, control:
            As provided by HF.

        metrics:
            Evaluation metrics dict from HF Trainer (may be None depending on
            HF version and evaluation flow). When present, it usually contains
            keys like:
              - "eval_loss"
              - "eval_runtime"
              - "eval_samples_per_second"
              - "eval_steps_per_second"
              - plus user-defined metrics from compute_metrics()

        **kwargs:
            Extra HF context (best-effort included in env).

        Returns
        -------
        TrainerControl
            Returned unchanged.

        Example
        -------
        Use JSONLLoggerCallback to write eval metrics:

        - configure:
          scalars: ["metrics.eval_loss"]
        - it will resolve env["metrics"]["eval_loss"] and log it if numeric.
        """
        env = self._env(args, state, control, phase="eval", metrics=metrics, **kwargs)
        self.controller.apply(env, events=self.eval_events)
        return control

    def _env(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        phase: str,
        **extra: Any,
    ) -> Dict[str, Any]:
        """
        Construct a framework-agnostic environment dict for `hotcb`.

        Parameters
        ----------
        args:
            HF TrainingArguments.

        state:
            HF TrainerState.
            Used fields (best-effort):
              - global_step (int)
              - epoch (float | None)

        control:
            HF TrainerControl.

        phase:
            A short string describing where we are in the lifecycle.
            Values used by this adapter:
              - "train_begin"
              - "train"
              - "eval"
            You may add your own phases if you extend this adapter.

        **extra:
            Additional fields to include in env. These are merged into env
            and can override existing keys (use carefully).

            Example extras:
              - metrics: dict from evaluation
              - model: if you choose to add it from kwargs or via trainer hooks
              - logs: if using on_log hook extension

        Returns
        -------
        Dict[str, Any]
            The env dictionary passed to `HotController.apply()` and ultimately
            to each callback handle(event, env).

        Notes
        -----
        - Logging: env["log"] is set to a simple print-based logger. You can
          swap this for a structured logger by passing `log=` in **extra or by
          wrapping this adapter.
        - Epoch: HF epoch is often float. We preserve that for consistency.
        """
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