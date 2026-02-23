# src/hotcb/callbacks/grad_stats.py
from __future__ import annotations

from typing import Any, Dict
from .utils import get_log, safe_int


class GradStatsCallback:
    """
    Periodically compute gradient statistics for the model.

    Requirements
    ------------
    - torch must be installed
    - env["model"] must be a torch.nn.Module-like object with .parameters()
    - You must call this callback after backward has run if you want real grads.
      In Lightning/HF, map it to an event that happens after backward.

    Parameters
    ----------
    id:
        Callback identifier.

    every:
        Log stats every N steps.

    norm_type:
        Gradient norm order, passed to vector norm.
        Typical values:
          - 2.0 (L2 norm)
          - 1.0 (L1 norm)
          - float('inf') (max norm) [note: our implementation uses vector_norm]

    What it logs
    ------------
    - norm: aggregated gradient norm across parameters
    - max_abs: maximum absolute gradient value across all parameters
    - grads: how many params had gradients vs total params
    - nan_params: number of parameters whose grad contains NaN/Inf

    Example (YAML)
    -------------
    callbacks:
      gstats:
        enabled: true
        target: { kind: module, path: hotcb.callbacks.grad_stats, symbol: GradStatsCallback }
        init: { every: 200, norm_type: 2.0 }

    Example (CLI)
    -------------
    $ hotcb --dir runs/exp1 enable gstats
    $ hotcb --dir runs/exp1 set gstats every=50
    """

    def __init__(self, id: str, every: int = 200, norm_type: float = 2.0) -> None:
        self.id = id
        self.every = int(every)
        self.norm_type = float(norm_type)

    def set_params(self, **kwargs: Any) -> None:
        """
        Supported hot params
        --------------------
        every: int
        norm_type: float
        """
        if "every" in kwargs:
            self.every = int(kwargs["every"])
        if "norm_type" in kwargs:
            self.norm_type = float(kwargs["norm_type"])

    def handle(self, event: str, env: Dict[str, Any]) -> None:
        """
        Compute and log gradient stats.

        Parameters
        ----------
        event:
            Event name string.

        env:
            Environment dict. Expected keys:
              - model: torch.nn.Module
              - step: int
              - log: optional callable
        """
        step = safe_int(env.get("step", 0))
        if self.every <= 0 or step % self.every != 0:
            return

        model = env.get("model")
        if model is None:
            return

        try:
            import torch
        except Exception:
            return

        total_norm_acc = 0.0
        max_abs = 0.0
        n_params = 0
        n_with_grad = 0
        n_nan = 0

        for p in model.parameters():
            n_params += 1
            if p.grad is None:
                continue
            g = p.grad
            n_with_grad += 1
            if torch.isnan(g).any() or torch.isinf(g).any():
                n_nan += 1
            max_abs = max(max_abs, float(g.detach().abs().max().item()))
            gn = torch.linalg.vector_norm(g.detach(), ord=self.norm_type).item()
            total_norm_acc += float(gn) ** self.norm_type

        total_norm = total_norm_acc ** (1.0 / self.norm_type) if n_with_grad > 0 else 0.0

        log = get_log(env)
        log(
            f"[hotcb:{self.id}] grad event={event} step={step} "
            f"norm={total_norm:.4g} max_abs={max_abs:.4g} grads={n_with_grad}/{n_params} nan_params={n_nan}"
        )