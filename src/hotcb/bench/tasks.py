"""
Benchmark task definitions for hotcb.

Synthetic tasks require no downloads.  Real tasks (e.g. CIFAR-10) guard their
heavy imports behind ``try/except`` so the package stays light.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn


@dataclass
class BenchmarkTask:
    """A self-contained benchmark scenario."""

    name: str                           # e.g. "synthetic_quadratic"
    description: str
    create_model: Callable[[], nn.Module]
    create_optimizer: Callable[[nn.Module], torch.optim.Optimizer]
    create_dataloader: Callable[[], object]  # iterable of (x, y) batches
    loss_fn: Callable                   # (output, target) -> loss tensor
    max_steps: int
    target_metric: Optional[float] = None      # e.g. 0.05 for val_loss
    target_metric_name: str = "val_loss"
    hp_space: Optional[dict] = None            # search space for tuning
    create_scheduler: Optional[Callable] = None  # (optimizer, max_epochs) -> scheduler
    create_val_dataloader: Optional[Callable] = None  # () -> dataloader
    batch_size: int = 128
    epochs: int = 1
    val_every_n_epochs: int = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _CycleLoader:
    """Wraps a list of (x, y) batches and cycles through them indefinitely."""

    def __init__(self, batches: list):
        self._batches = batches

    def __iter__(self):
        while True:
            yield from self._batches


def _make_quadratic_data(n_samples: int = 256, batch_size: int = 32, seed: int = 42):
    """y = x^2 + noise, 1-D input."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, 1, generator=gen)
    y = x ** 2 + 0.1 * torch.randn(n_samples, 1, generator=gen)
    batches = []
    for i in range(0, n_samples, batch_size):
        batches.append((x[i:i + batch_size], y[i:i + batch_size]))
    return batches


def _make_classification_data(
    n_samples: int = 512, n_features: int = 20, n_classes: int = 4,
    batch_size: int = 64, seed: int = 42,
):
    """Synthetic multi-class classification blobs."""
    gen = torch.Generator().manual_seed(seed)
    # Create class centres
    centres = torch.randn(n_classes, n_features, generator=gen) * 3.0
    xs, ys = [], []
    per_class = n_samples // n_classes
    for c in range(n_classes):
        x_c = centres[c].unsqueeze(0) + torch.randn(per_class, n_features, generator=gen)
        xs.append(x_c)
        ys.append(torch.full((per_class,), c, dtype=torch.long))
    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    # shuffle
    perm = torch.randperm(x_all.size(0), generator=gen)
    x_all = x_all[perm]
    y_all = y_all[perm]
    batches = []
    for i in range(0, len(x_all), batch_size):
        batches.append((x_all[i:i + batch_size], y_all[i:i + batch_size]))
    return batches


# ---------------------------------------------------------------------------
# Built-in tasks
# ---------------------------------------------------------------------------

def _quadratic_task(max_steps: int = 200) -> BenchmarkTask:
    return BenchmarkTask(
        name="synthetic_quadratic",
        description="Simple quadratic regression (y = x^2 + noise). Single linear layer.",
        create_model=lambda: nn.Linear(1, 1),
        create_optimizer=lambda model: torch.optim.SGD(model.parameters(), lr=0.01),
        create_dataloader=lambda: _CycleLoader(_make_quadratic_data()),
        loss_fn=nn.MSELoss(),
        max_steps=max_steps,
        target_metric=0.5,
        target_metric_name="val_loss",
        hp_space={"lr": (1e-4, 0.1)},
        batch_size=32,
    )


def _classification_task(max_steps: int = 300) -> BenchmarkTask:
    return BenchmarkTask(
        name="synthetic_classification",
        description="Multi-class classification on synthetic blobs. 2-layer MLP.",
        create_model=lambda: nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        ),
        create_optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
        create_dataloader=lambda: _CycleLoader(_make_classification_data()),
        loss_fn=nn.CrossEntropyLoss(),
        max_steps=max_steps,
        target_metric=0.5,
        target_metric_name="val_loss",
        hp_space={"lr": (1e-5, 1e-2)},
        batch_size=64,
    )


# ---------------------------------------------------------------------------
# CIFAR-10 ResNet-20  (He et al. 2015)
# ---------------------------------------------------------------------------

def _cifar10_resnet20_task(max_steps: int | None = None) -> BenchmarkTask:
    """ResNet-20 on CIFAR-10 — published 91.25% accuracy (He et al. 2015).

    Standard config:
    - SGD, lr=0.1, momentum=0.9, wd=1e-4
    - LR schedule: /10 at epoch 82, /10 at epoch 123
    - 164 epochs, batch_size=128
    - Data aug: random crop 32 w/ 4px pad, random horizontal flip
    """
    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError:
        raise ImportError(
            "CIFAR-10 benchmark requires torchvision: pip install torchvision"
        )

    from .models import resnet20

    _CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    _CIFAR10_STD = (0.2023, 0.1994, 0.2010)

    _BATCH_SIZE = 128
    _EPOCHS = 164
    # 50000 / 128 = 390.625 -> 391 batches per epoch
    _STEPS_PER_EPOCH = 391
    _DEFAULT_MAX_STEPS = _EPOCHS * _STEPS_PER_EPOCH  # 64124

    if max_steps is None:
        max_steps = _DEFAULT_MAX_STEPS

    def _create_train_dataloader():
        transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform,
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=_BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=False,
        )

    def _create_val_dataloader():
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform,
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=_BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=True,
        )

    def _create_optimizer(model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4,
        )

    def _create_scheduler(optimizer, max_epochs):
        """Multi-step LR: divide by 10 at epoch 82 and 123."""
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[82, 123], gamma=0.1,
        )

    return BenchmarkTask(
        name="cifar10_resnet20",
        description=(
            "ResNet-20 on CIFAR-10 — published 91.25% top-1 accuracy "
            "(He et al. 2015). SGD, lr=0.1, milestones=[82,123]."
        ),
        create_model=resnet20,
        create_optimizer=_create_optimizer,
        create_dataloader=_create_train_dataloader,
        loss_fn=nn.CrossEntropyLoss(),
        max_steps=max_steps,
        target_metric=91.25,
        target_metric_name="val_accuracy",
        hp_space={"lr": (1e-3, 0.5), "weight_decay": (1e-5, 1e-3)},
        create_scheduler=_create_scheduler,
        create_val_dataloader=_create_val_dataloader,
        batch_size=_BATCH_SIZE,
        epochs=_EPOCHS,
        val_every_n_epochs=1,
    )


BUILTIN_TASKS = {
    "synthetic_quadratic": _quadratic_task,
    "synthetic_classification": _classification_task,
    "cifar10_resnet20": _cifar10_resnet20_task,
}
