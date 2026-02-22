from __future__ import annotations
from typing import Any, Dict, Optional, List

import pytorch_lightning as pl  # lightning>=2.x

from hotcb import HotController


class HotCallbackController(pl.Callback):
    """
    Lightning adapter that routes safe points -> HotController.apply().

    No DDP assumptions. No touching Trainer callbacks list.
    """
    def __init__(
        self,
        controller: HotController,
        train_events: Optional[List[str]] = None,
        val_events: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.train_events = train_events or ["train_batch_end"]
        self.val_events = val_events or ["val_batch_end"]

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        env = self._env(trainer, pl_module, phase="fit_start")
        self.controller.apply(env, events=["fit_start"])

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        env = self._env(trainer, pl_module, phase="train", outputs=outputs, batch=batch, batch_idx=batch_idx)
        self.controller.apply(env, events=self.train_events)

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        env = self._env(trainer, pl_module, phase="val", outputs=outputs, batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        self.controller.apply(env, events=self.val_events)

    def _env(self, trainer: pl.Trainer, pl_module: pl.LightningModule, phase: str, **extra: Any) -> Dict[str, Any]:
        # Lightning exposes global_step; epoch via trainer.current_epoch
        def _log(s: str) -> None:
            # use trainer.print for nice formatting
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
        return env