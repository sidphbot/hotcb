"""
Writing and loading a custom hot callback.

A hot callback is any Python class that implements three things:
  - self.id: str
  - handle(event, env): called at each training event
  - set_params(**kwargs): called when `hotcb cb set <id> key=val` is run live

No base class required.

Once written, load it live while training is running:

    hotcb --dir runs/exp1 cb load grad_monitor \\
      --file /path/to/custom_callback_example.py \\
      --symbol GradMonitor \\
      --enabled \\
      --init every=25 threshold=5.0
"""

from __future__ import annotations

from typing import Any, Dict


class GradMonitor:
    """
    Monitors gradient norms and logs a warning when they exceed a threshold.

    Demonstrates:
    - Per-step throttling with `every`
    - Live parameter updates via set_params
    - Reading from the env dict
    - Auto-disable on repeated anomalies (manual guard pattern)
    """

    def __init__(self, id: str, every: int = 50, threshold: float = 10.0):
        self.id = id
        self.every = every
        self.threshold = threshold
        self._consecutive_high = 0
        self._disabled = False

    def set_params(self, **kwargs: Any) -> None:
        """Called live when: hotcb --dir runs/exp1 cb set grad_monitor every=10 threshold=2.0"""
        if "every" in kwargs:
            self.every = int(kwargs["every"])
        if "threshold" in kwargs:
            self.threshold = float(kwargs["threshold"])

    def handle(self, event: str, env: Dict[str, Any]) -> None:
        if self._disabled:
            return
        if event != "train_step_end":
            return

        step = env.get("step", 0)
        if step % self.every != 0:
            return

        log = env.get("log", print)
        model = env.get("model")

        if model is None:
            return

        # Compute total gradient norm across all parameters
        total_norm = 0.0
        n_params = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
                n_params += 1

        if n_params == 0:
            return

        total_norm = total_norm ** 0.5

        if total_norm > self.threshold:
            self._consecutive_high += 1
            log(
                f"[{self.id}] WARNING grad_norm={total_norm:.3f} "
                f"threshold={self.threshold} step={step} "
                f"(high for {self._consecutive_high} checks)"
            )
            if self._consecutive_high >= 5:
                log(f"[{self.id}] Auto-disabling after 5 consecutive high grad norms.")
                self._disabled = True
        else:
            self._consecutive_high = 0
            log(f"[{self.id}] grad_norm={total_norm:.3f} step={step}")


# ---------------------------------------------------------------------------
# Minimal training loop showing how to wire it in manually (bare PyTorch).
# In practice you load it via the CLI while training is running.
# ---------------------------------------------------------------------------

def _demo():
    import torch
    from hotcb import HotKernel

    # Initialize run directory first: hotcb --dir runs/demo init
    kernel = HotKernel(run_dir="runs/demo", debounce_steps=1)

    model = torch.nn.Linear(16, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for step in range(200):
        x = torch.randn(8, 16)
        y = torch.randn(8, 1)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        kernel.apply(
            env={
                "step": step,
                "phase": "train",
                "model": model,
                "optimizer": optimizer,
                "loss": loss,
                "log": print,
            },
            events=["train_step_end"],
        )

    # From a second terminal once this is running:
    #
    #   hotcb --dir runs/demo cb load grad_monitor \
    #     --file docs/examples/custom_callback_example.py \
    #     --symbol GradMonitor \
    #     --enabled \
    #     --init every=10 threshold=3.0
    #
    #   hotcb --dir runs/demo cb set grad_monitor threshold=1.0
    #   hotcb --dir runs/demo disable grad_monitor
    #   hotcb --dir runs/demo enable grad_monitor


if __name__ == "__main__":
    _demo()
