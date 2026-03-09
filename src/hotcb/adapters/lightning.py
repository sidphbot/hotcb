from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytorch_lightning as pl

from hotcb.kernel import HotKernel


class HotCBLightning(pl.Callback):
    """
    PyTorch Lightning adapter for hotcb.

    Connects Lightning hooks to HotKernel, exposing optimizer, scheduler,
    and loss_state in the env dict for hotopt/hotloss modules.
    """

    def __init__(
        self,
        kernel: HotKernel,
        train_events: Optional[List[str]] = None,
        val_events: Optional[List[str]] = None,
        loss_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.train_events = train_events or ["train_batch_end"]
        self.val_events = val_events or ["val_batch_end"]
        self._loss_state = loss_state

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        env = self._env(trainer, pl_module, phase="fit_start")
        self.kernel.apply(env, events=["fit_start"])

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        env = self._env(trainer, pl_module, phase="train", outputs=outputs, batch=batch, batch_idx=batch_idx)
        self.kernel.apply(env, events=self.train_events)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        env = self._env(trainer, pl_module, phase="val", outputs=outputs, batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        self.kernel.apply(env, events=self.val_events)

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        env = self._env(trainer, pl_module, phase="val")
        self.kernel.apply(env, events=["val_epoch_end"])

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        env = self._env(trainer, pl_module, phase="fit_end")
        self.kernel.apply(env, events=["run_end"])
        self.kernel.close(env)

    def _env(self, trainer: pl.Trainer, pl_module: pl.LightningModule, phase: str, **extra: Any) -> Dict[str, Any]:
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

        env["kernel"] = self.kernel

        # Metric accessor for hottune
        def _metric(name: str, default: Any = None) -> Any:
            # Check callback metrics first
            try:
                cb_metrics = trainer.callback_metrics
                if name in cb_metrics:
                    val = cb_metrics[name]
                    try:
                        import torch
                        if isinstance(val, torch.Tensor):
                            return val.item()
                    except ImportError:
                        pass
                    return val
            except Exception:
                pass
            # Check logged metrics
            try:
                logged = trainer.logged_metrics
                if name in logged:
                    val = logged[name]
                    try:
                        import torch
                        if isinstance(val, torch.Tensor):
                            return val.item()
                    except ImportError:
                        pass
                    return val
            except Exception:
                pass
            # Normalized env fields
            if name == "lr":
                opt = env.get("optimizer")
                if opt is not None:
                    try:
                        return opt.param_groups[0]["lr"]
                    except Exception:
                        pass
            if name in ("train/loss", "loss"):
                loss = env.get("loss")
                if loss is not None:
                    try:
                        import torch
                        if isinstance(loss, torch.Tensor):
                            return loss.item()
                    except ImportError:
                        pass
                    return loss
            return default

        env["metric"] = _metric

        # Expose max_steps for phase binning
        try:
            if hasattr(trainer, "max_steps") and trainer.max_steps and trainer.max_steps > 0:
                env["max_steps"] = trainer.max_steps
            elif hasattr(trainer, "estimated_stepping_batches"):
                env["max_steps"] = int(trainer.estimated_stepping_batches)
        except Exception:
            pass

        # Expose optimizer for hotopt
        try:
            optimizers = trainer.optimizers
            if optimizers:
                env["optimizer"] = optimizers[0]
        except Exception:
            pass

        # Expose LR scheduler for hotopt
        try:
            configs = trainer.lr_scheduler_configs
            if configs:
                env["scheduler"] = configs[0].scheduler
        except Exception:
            pass

        # Expose loss_state for hotloss
        if self._loss_state is not None:
            env["loss_state"] = self._loss_state

        env.update(extra)

        # Normalize loss from outputs
        try:
            outputs = env.get("outputs")
            if "loss" not in env:
                import torch
                if isinstance(outputs, torch.Tensor):
                    env["loss"] = outputs
                elif isinstance(outputs, dict) and "loss" in outputs:
                    env["loss"] = outputs["loss"]
        except Exception:
            pass

        return env
