from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytorch_lightning as pl

from hotcb.kernel import HotKernel
from hotcb.capabilities import TrainingCapabilities, validate_mutable_state


class HotCBLightning(pl.Callback):
    """
    PyTorch Lightning adapter for hotcb.

    Connects Lightning hooks to HotKernel, exposing optimizer, scheduler,
    and mutable_state in the env dict for hotopt/hotloss modules.

    Multi-optimizer support
    -----------------------
    All optimizers from ``trainer.optimizers`` are exposed as
    ``env["optimizers"]``.  ``env["optimizer"]`` always points to the
    first one for backward compatibility.  Commands can target a
    specific optimizer via ``params.opt_idx``.

    Grad accumulation
    -----------------
    When accumulation is active, micro-steps (where no optimizer step
    occurs) are tagged with ``env["is_accumulation_step"] = True``.
    The kernel skips metric collection on those steps to avoid noisy
    intermediate values.  Commands still apply immediately.
    """

    def __init__(
        self,
        kernel: HotKernel,
        train_events: Optional[List[str]] = None,
        val_events: Optional[List[str]] = None,
        mutable_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.train_events = train_events or ["train_batch_end"]
        self.val_events = val_events or ["val_batch_end"]
        self._mutable_state = mutable_state
        self._capabilities: Optional[TrainingCapabilities] = None

    # -- lifecycle hooks ---------------------------------------------------

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._capabilities = self._detect_capabilities(trainer, pl_module)
        if self.kernel.run_dir:
            self._capabilities.save(self.kernel.run_dir)
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

    # -- capabilities detection --------------------------------------------

    def _detect_capabilities(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> TrainingCapabilities:
        """Build a TrainingCapabilities snapshot from the current setup."""
        # Optimizers
        opt_names: list[str] = []
        group_counts: list[int] = []
        try:
            optimizers = trainer.optimizers or []
            for opt in optimizers:
                opt_names.append(type(opt).__name__)
                group_counts.append(len(opt.param_groups))
        except Exception:
            pass

        # Schedulers
        sched_types: list[str] = []
        has_scheduler = False
        try:
            configs = trainer.lr_scheduler_configs or []
            for cfg in configs:
                has_scheduler = True
                sched_types.append(type(cfg.scheduler).__name__)
        except Exception:
            pass

        # Grad accumulation
        accum = 1
        auto_opt = getattr(pl_module, "automatic_optimization", True)
        try:
            accum = getattr(trainer, "accumulate_grad_batches", 1) or 1
        except Exception:
            pass
        # For manual optimization, check module attribute
        if not auto_opt and accum == 1:
            accum = getattr(pl_module, "_grad_accum", 1) or 1

        # Mutable state
        ls_detected = False
        ls_keys: list[str] = []
        ls = self._resolve_mutable_state(pl_module)
        if ls is not None:
            ls_detected = True
            weights = ls.get("weights", {})
            ls_keys = list(weights.keys())

        # Grad clip
        clip_val = None
        clip_wired = False
        try:
            if hasattr(trainer, "gradient_clip_val") and trainer.gradient_clip_val:
                clip_val = float(trainer.gradient_clip_val)
                clip_wired = auto_opt  # only wired if Lightning manages the loop
        except Exception:
            pass

        return TrainingCapabilities(
            framework="lightning",
            num_optimizers=len(opt_names) or 1,
            optimizer_names=tuple(opt_names),
            num_param_groups=tuple(group_counts),
            has_scheduler=has_scheduler,
            scheduler_types=tuple(sched_types),
            grad_accumulation_steps=accum,
            automatic_optimization=auto_opt,
            mutable_state_detected=ls_detected,
            mutable_state_keys=tuple(ls_keys),
            grad_clip_value=clip_val,
            grad_clip_wired=clip_wired,
        )

    # -- env builder -------------------------------------------------------

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

        if self._capabilities is not None:
            env["capabilities"] = self._capabilities

        # -- metric accessor (for hottune and MetricsCollector strategy 1) --
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

        # -- max_steps for phase binning --
        try:
            if hasattr(trainer, "max_steps") and trainer.max_steps and trainer.max_steps > 0:
                env["max_steps"] = trainer.max_steps
            elif hasattr(trainer, "estimated_stepping_batches"):
                env["max_steps"] = int(trainer.estimated_stepping_batches)
        except Exception:
            pass

        # -- all optimizers (multi-optimizer support) --
        try:
            optimizers = trainer.optimizers
            if optimizers:
                env["optimizer"] = optimizers[0]  # backward compat default
                env["optimizers"] = list(optimizers)
        except Exception:
            pass

        # -- all schedulers --
        try:
            configs = trainer.lr_scheduler_configs
            if configs:
                env["scheduler"] = configs[0].scheduler
                env["schedulers"] = [cfg.scheduler for cfg in configs]
        except Exception:
            pass

        # -- mutable_state (explicit > auto-detected from pl_module) --
        ls = self._resolve_mutable_state(pl_module)
        if ls is not None:
            env["mutable_state"] = ls

        # -- grad accumulation detection --
        batch_idx = extra.get("batch_idx")
        accum = 1
        if self._capabilities:
            accum = self._capabilities.grad_accumulation_steps
        else:
            try:
                accum = getattr(trainer, "accumulate_grad_batches", 1) or 1
            except Exception:
                pass

        if accum > 1 and batch_idx is not None:
            is_accum_step = (batch_idx + 1) % accum != 0
            env["is_accumulation_step"] = is_accum_step
            env["grad_accumulation_steps"] = accum

        # -- all logged/callback metrics as dict (for MetricsCollector discovery) --
        try:
            all_metrics: Dict[str, float] = {}
            import torch
            cb_metrics = trainer.callback_metrics
            if cb_metrics:
                for k, v in cb_metrics.items():
                    try:
                        all_metrics[k] = v.item() if isinstance(v, torch.Tensor) else float(v)
                    except (TypeError, ValueError):
                        pass
            logged = trainer.logged_metrics
            if logged:
                for k, v in logged.items():
                    if k not in all_metrics:
                        try:
                            all_metrics[k] = v.item() if isinstance(v, torch.Tensor) else float(v)
                        except (TypeError, ValueError):
                            pass
            if all_metrics:
                env["metrics"] = all_metrics
        except Exception:
            pass

        env.update(extra)

        # -- normalize loss from outputs --
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

    # -- helpers -----------------------------------------------------------

    def _resolve_mutable_state(self, pl_module: pl.LightningModule) -> Optional[Dict[str, Any]]:
        """Resolve mutable_state: explicit > pl_module attribute."""
        if self._mutable_state is not None:
            valid, normalized = validate_mutable_state(self._mutable_state)
            if valid:
                return normalized if normalized is not self._mutable_state else self._mutable_state

        if hasattr(pl_module, "mutable_state"):
            obj = pl_module.mutable_state
            valid, _ = validate_mutable_state(obj)
            if valid:
                return obj  # return original ref so mutations are visible

        # Fallback: check loss_state (legacy name used by some integrations)
        if hasattr(pl_module, "loss_state"):
            obj = pl_module.loss_state
            valid, _ = validate_mutable_state(obj)
            if valid:
                return obj

        return None
