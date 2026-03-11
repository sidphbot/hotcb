from __future__ import annotations

from typing import Any, Dict, Optional, List

import pytorch_lightning as pl  # lightning>=2.x

from .. import HotController


class HotCallbackController(pl.Callback):
    """
    PyTorch Lightning adapter for `hotcb`.

    This adapter connects Lightning's callback hooks to the framework-agnostic
    `HotController`. It enables live hot-updates such as:

      - enabling/disabling diagnostics callbacks,
      - adjusting callback parameters,
      - loading new callbacks from a module or a Python file path,

    without restarting the Trainer.

    Philosophy
    ----------
    - Low modification: does not patch Lightning internals.
    - Safe-point updates: calls `HotController.apply()` only at stable boundaries.
    - No DDP assumptions: this version does not broadcast config across ranks.

    Hooks used (safe points)
    ------------------------
    - `on_fit_start`:
        One-time event to apply initial config and allow callbacks to initialize
        resources early.

    - `on_train_batch_end`:
        Main per-step dispatch point during training.

    - `on_validation_batch_end`:
        Per-step dispatch point during validation.

    Parameters
    ----------
    controller:
        `HotController` instance that manages callback registry and control plane
        polling.

        Typical setup:

        >>> controller = HotController(
        ...     config_path="runs/exp1/hotcb.yaml",
        ...     commands_path="runs/exp1/hotcb.commands.jsonl",
        ...     debounce_steps=5,
        ...     log_path="runs/exp1/hotcb.log",
        ... )

    train_events:
        List of event names to dispatch on each `on_train_batch_end`.
        Defaults to ["train_batch_end"].

        Acceptable values:
          - any list of strings

        Recommended:
          - keep them stable across your project for config-driven callbacks.

    val_events:
        List of event names to dispatch on each `on_validation_batch_end`.
        Defaults to ["val_batch_end"].

    Environment (`env`) contract
    ----------------------------
    The adapter constructs an `env` dict including:

      - "framework": "lightning"
      - "phase": "fit_start" | "train" | "val"
      - "step": int trainer.global_step
      - "epoch": int trainer.current_epoch
      - "model": pl.LightningModule (pl_module)
      - "trainer": pl.Trainer
      - "log": callable(str) -> None (uses trainer.print when available)

    Plus hook-specific fields via **extra:
      - "outputs": hook outputs
      - "batch": current batch
      - "batch_idx": int
      - "dataloader_idx": int (validation hook)

    Best practice: `loss` exposure (optional)
    ----------------------------------------
    Many diagnostics callbacks (TensorStats / JSONLLogger / AnomalyGuard) benefit
    from a consistent `env["loss"]` key. Lightning's `outputs` may be:
      - a Tensor loss,
      - a dict containing "loss",
      - or None depending on your training_step.

    This adapter does not enforce a single convention, but you can optionally
    normalize loss by setting env["loss"] if you want (see below).

    Example usage
    -------------
    >>> from hotcb.adapters.lightning import HotCallbackController
    >>> trainer = pl.Trainer(callbacks=[HotCallbackController(controller)])
    >>> trainer.fit(model, datamodule=dm)

    Live control from another terminal:
    >>> hotcb --dir runs/exp1 enable tstats
    >>> hotcb --dir runs/exp1 set tstats every=20 paths=loss,outputs.logits
    >>> hotcb --dir runs/exp1 load my_diag --file /tmp/my_diag.py --symbol MyDiag --enabled --init msg="hi"
    """

    def __init__(
        self,
        controller: HotController,
        train_events: Optional[List[str]] = None,
        val_events: Optional[List[str]] = None,
    ) -> None:
        """
        Create a Lightning adapter callback.

        Parameters
        ----------
        controller:
            HotController instance used to apply updates and dispatch events.

        train_events:
            Event names dispatched during training at `on_train_batch_end`.
            If None, defaults to ["train_batch_end"].

        val_events:
            Event names dispatched during validation at `on_validation_batch_end`.
            If None, defaults to ["val_batch_end"].
        """
        super().__init__()
        self.controller = controller
        self.train_events = train_events or ["train_batch_end"]
        self.val_events = val_events or ["val_batch_end"]

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Lightning hook: called when fit begins.

        Use cases
        ---------
        - Load/enable initial diagnostics callbacks before the first step.
        - Create artifact directories.
        - Emit a "fit_start" event to your hot callbacks.

        Parameters
        ----------
        trainer:
            Lightning Trainer instance.

        pl_module:
            LightningModule being trained.
        """
        env = self._env(trainer, pl_module, phase="fit_start")
        self.controller.apply(env, events=["fit_start"])

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        Lightning hook: called at the end of a training batch.

        This is the primary safe point for hot updates during training. It is
        invoked after the training step is computed and logged.

        Behavior
        --------
        - Builds env with phase="train" and includes outputs, batch, batch_idx.
        - Calls controller.apply(env, events=self.train_events).

        Parameters
        ----------
        trainer:
            Lightning Trainer.

        pl_module:
            LightningModule.

        outputs:
            Output from `training_step`. Common patterns:
              - loss tensor
              - dict with "loss" and other entries
              - None

        batch:
            Current batch as yielded by dataloader.

        batch_idx:
            Batch index within the epoch.

        Notes
        -----
        If you want to normalize loss for diagnostics callbacks, you can do:
          - if torch Tensor: env["loss"] = outputs
          - if dict and "loss" in outputs: env["loss"] = outputs["loss"]
        (Not required, but recommended.)
        """
        env = self._env(
            trainer,
            pl_module,
            phase="train",
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
        )
        self.controller.apply(env, events=self.train_events)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Lightning hook: called at the end of a validation batch.

        This is a safe point for diagnostics during validation, such as:
          - logging tensor stats on outputs,
          - writing artifacts,
          - monitoring evaluation stability.

        Behavior
        --------
        - Builds env with phase="val" and includes outputs/batch/batch_idx/dataloader_idx.
        - Calls controller.apply(env, events=self.val_events).

        Parameters
        ----------
        trainer:
            Lightning Trainer.

        pl_module:
            LightningModule.

        outputs:
            Output from `validation_step`.

        batch:
            Current batch.

        batch_idx:
            Validation batch index.

        dataloader_idx:
            Index of validation dataloader (for multiple val loaders).

        Notes
        -----
        Similar to training, you may normalize env["loss"] from outputs if your
        validation_step returns loss.
        """
        env = self._env(
            trainer,
            pl_module,
            phase="val",
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )
        self.controller.apply(env, events=self.val_events)

    def _env(self, trainer: pl.Trainer, pl_module: pl.LightningModule, phase: str, **extra: Any) -> Dict[str, Any]:
        """
        Construct a framework-agnostic env dict for Lightning.

        Parameters
        ----------
        trainer:
            Lightning Trainer instance.

        pl_module:
            LightningModule instance.

        phase:
            Lifecycle phase string. Values used by this adapter:
              - "fit_start"
              - "train"
              - "val"

        **extra:
            Additional fields to merge into env. Common extras (by hook):
              - outputs: Any
              - batch: Any
              - batch_idx: int
              - dataloader_idx: int

            If keys collide with base env keys, extras overwrite base keys.

        Returns
        -------
        Dict[str, Any]
            Environment dictionary for hot callbacks.

        Logging behavior
        ----------------
        env["log"] uses `trainer.print` if available (preferred in Lightning),
        falling back to built-in print otherwise.

        Notes on step/epoch
        -------------------
        - step is derived from trainer.global_step (int), defaulting to 0.
        - epoch is derived from trainer.current_epoch (int), defaulting to 0.
        """
        def _log(s: str) -> None:
            try:
                trainer.print(s)
            except Exception:
                print(s)

        env: Dict[str, Any] = {
            "framework": "lightning",
            "phase": phase,
            "step": int(getattr(trainer, "global_step", 0)),
            "epoch": int(getattr(trainer, "current_epoch", 0)),
            "model": pl_module,
            "trainer": trainer,
            "log": _log,
        }

        env.update(extra)

        # OPTIONAL (docstring-aligned QoL):
        # If training_step returns a loss tensor or dict with loss, expose env["loss"].
        # This is safe and helps generic diagnostics callbacks.
        try:
            outputs = env.get("outputs")
            if "loss" not in env:
                # loss tensor
                import torch  # optional dependency in Lightning projects anyway
                if isinstance(outputs, torch.Tensor):
                    env["loss"] = outputs
                elif isinstance(outputs, dict) and "loss" in outputs:
                    env["loss"] = outputs["loss"]
        except Exception:
            pass

        return env