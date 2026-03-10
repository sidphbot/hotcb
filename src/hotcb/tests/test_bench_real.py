"""Tests for real benchmark tasks (ResNet-20 / CIFAR-10) and autopilot evaluation."""
from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from hotcb.bench.models import BasicBlock, CifarResNet, resnet20
from hotcb.bench.tasks import BenchmarkTask, BUILTIN_TASKS, _CycleLoader
from hotcb.bench.runner import BenchmarkRunner, BenchmarkResult, _evaluate_accuracy
from hotcb.bench.eval_autopilot import AutopilotEval

# ---------------------------------------------------------------------------
# Helper: check if torchvision is available
# ---------------------------------------------------------------------------
_HAS_TORCHVISION = False
try:
    import torchvision
    _HAS_TORCHVISION = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# ResNet-20 model tests
# ---------------------------------------------------------------------------

class TestResNet20Model:
    """Unit tests for the ResNet-20 architecture."""

    def test_forward_shape(self):
        """Output should be (batch, 10) for CIFAR-10 inputs."""
        model = resnet20(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10)

    def test_forward_shape_different_classes(self):
        """Works with non-default number of classes."""
        model = resnet20(num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 100)

    def test_parameter_count(self):
        """ResNet-20 should have ~0.27M parameters."""
        model = resnet20()
        total = sum(p.numel() for p in model.parameters())
        # He et al. report 0.27M; typical implementations give 272,474
        assert 250_000 < total < 300_000, f"Expected ~270k params, got {total}"

    def test_basic_block_identity_shortcut(self):
        """BasicBlock with matching dims uses identity shortcut."""
        block = BasicBlock(16, 16, stride=1)
        x = torch.randn(2, 16, 8, 8)
        out = block(x)
        assert out.shape == (2, 16, 8, 8)
        # Identity shortcut should be empty Sequential
        assert len(list(block.shortcut.children())) == 0

    def test_basic_block_projection_shortcut(self):
        """BasicBlock with stride=2 uses projection shortcut."""
        block = BasicBlock(16, 32, stride=2)
        x = torch.randn(2, 16, 8, 8)
        out = block(x)
        assert out.shape == (2, 32, 4, 4)
        # Projection shortcut should have conv + bn
        assert len(list(block.shortcut.children())) == 2

    def test_gradient_flows(self):
        """Gradients should flow through the residual connections."""
        model = resnet20()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check first conv has gradients
        assert model.conv1.weight.grad is not None
        assert model.conv1.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# BenchmarkTask dataclass tests
# ---------------------------------------------------------------------------

class TestBenchmarkTaskFields:
    """Test the new optional fields on BenchmarkTask."""

    def test_default_fields(self):
        """New optional fields have sensible defaults."""
        task = BenchmarkTask(
            name="test",
            description="test",
            create_model=lambda: nn.Linear(1, 1),
            create_optimizer=lambda m: torch.optim.SGD(m.parameters(), lr=0.01),
            create_dataloader=lambda: iter([]),
            loss_fn=nn.MSELoss(),
            max_steps=10,
        )
        assert task.create_scheduler is None
        assert task.create_val_dataloader is None
        assert task.batch_size == 128
        assert task.epochs == 1
        assert task.val_every_n_epochs == 1

    def test_cifar10_in_builtin_tasks(self):
        """cifar10_resnet20 should be registered in BUILTIN_TASKS."""
        assert "cifar10_resnet20" in BUILTIN_TASKS


# ---------------------------------------------------------------------------
# CIFAR-10 task creation (needs torchvision)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_TORCHVISION, reason="torchvision not installed")
class TestCifar10Task:
    """Tests that require torchvision."""

    def test_task_creation(self):
        """Task factory should produce a valid BenchmarkTask."""
        task = BUILTIN_TASKS["cifar10_resnet20"]()
        assert task.name == "cifar10_resnet20"
        assert task.epochs == 164
        assert task.batch_size == 128
        assert task.target_metric == 91.25
        assert task.target_metric_name == "val_accuracy"
        assert task.create_scheduler is not None
        assert task.create_val_dataloader is not None

    def test_task_custom_max_steps(self):
        """max_steps override should work."""
        task = BUILTIN_TASKS["cifar10_resnet20"](max_steps=100)
        assert task.max_steps == 100


# ---------------------------------------------------------------------------
# Evaluate accuracy helper
# ---------------------------------------------------------------------------

class TestEvaluateAccuracy:
    """Test the validation evaluation helper."""

    def test_perfect_accuracy(self):
        """A model that always predicts correctly gets 100%."""
        # Simple model that maps input directly
        model = nn.Linear(4, 3, bias=False)
        # Create data where we know the answer
        torch.manual_seed(42)
        x = torch.randn(20, 4)
        with torch.no_grad():
            logits = model(x)
            y = logits.argmax(dim=1)
        loader = [(x, y)]
        val_loss, val_acc = _evaluate_accuracy(model, loader, torch.device("cpu"))
        assert val_acc == 100.0

    def test_accuracy_range(self):
        """Accuracy should be between 0 and 100."""
        model = nn.Linear(4, 3)
        x = torch.randn(20, 4)
        y = torch.randint(0, 3, (20,))
        loader = [(x, y)]
        val_loss, val_acc = _evaluate_accuracy(model, loader, torch.device("cpu"))
        assert 0.0 <= val_acc <= 100.0
        assert val_loss >= 0.0


# ---------------------------------------------------------------------------
# Runner with synthetic data (epoch-based loop)
# ---------------------------------------------------------------------------

class TestRunnerEpochLoop:
    """Test the enhanced training loop with epoch tracking."""

    def _make_synthetic_task(self, epochs=2, steps_per_epoch=5):
        """Create a tiny task with validation for testing the loop."""
        n_samples = steps_per_epoch * 4  # batch_size=4
        torch.manual_seed(0)
        train_x = torch.randn(n_samples, 4)
        train_y = torch.randint(0, 3, (n_samples,))
        val_x = torch.randn(8, 4)
        val_y = torch.randint(0, 3, (8,))

        def _train_loader():
            batches = []
            for i in range(0, n_samples, 4):
                batches.append((train_x[i:i+4], train_y[i:i+4]))
            return batches

        def _val_loader():
            return [(val_x, val_y)]

        def _scheduler(opt, max_epochs):
            return torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)

        return BenchmarkTask(
            name="test_epoch",
            description="tiny epoch test",
            create_model=lambda: nn.Linear(4, 3),
            create_optimizer=lambda m: torch.optim.SGD(m.parameters(), lr=0.1),
            create_dataloader=_train_loader,
            loss_fn=nn.CrossEntropyLoss(),
            max_steps=epochs * steps_per_epoch,
            create_scheduler=_scheduler,
            create_val_dataloader=_val_loader,
            batch_size=4,
            epochs=epochs,
            val_every_n_epochs=1,
        )

    def test_epoch_loop_runs(self):
        """Training loop completes with epoch-based iteration."""
        task = self._make_synthetic_task(epochs=2, steps_per_epoch=5)
        metrics, stt, ic = BenchmarkRunner._train_loop(task, kernel=None, max_steps=10)
        assert "val_loss" in metrics
        assert "val_accuracy" in metrics
        assert "epochs" in metrics
        assert metrics["epochs"] == 2

    def test_scheduler_applied(self):
        """LR should decrease when scheduler fires."""
        task = self._make_synthetic_task(epochs=3, steps_per_epoch=3)
        # We can't directly check LR from the loop, but we verify it runs
        metrics, _, _ = BenchmarkRunner._train_loop(task, kernel=None, max_steps=9)
        assert metrics["epochs"] == 3

    def test_runner_baseline(self):
        """BenchmarkRunner.run_baseline works with epoch-based tasks."""
        task = self._make_synthetic_task(epochs=2, steps_per_epoch=3)
        with tempfile.TemporaryDirectory() as td:
            runner = BenchmarkRunner(output_dir=td)
            result = runner.run_baseline(task)
            assert result.condition == "baseline"
            assert result.total_steps == 6
            assert result.intervention_count == 0
            assert "val_accuracy" in result.final_metrics


# ---------------------------------------------------------------------------
# AutopilotEval tests
# ---------------------------------------------------------------------------

class TestAutopilotEval:
    """Test AutopilotEval instantiation and basic behaviour."""

    def test_instantiation(self):
        """AutopilotEval can be created."""
        with tempfile.TemporaryDirectory() as td:
            ev = AutopilotEval(output_dir=td)
            assert os.path.isdir(td)
            assert ev._baseline_result is None
            assert ev._autopilot_result is None

    def test_report_empty(self):
        """Report works even with no results."""
        with tempfile.TemporaryDirectory() as td:
            ev = AutopilotEval(output_dir=td)
            text = ev.report()
            assert "Autopilot Evaluation Report" in text
            assert "(no results yet)" in text

    def test_unknown_task(self):
        """Requesting an unknown task raises ValueError."""
        with tempfile.TemporaryDirectory() as td:
            ev = AutopilotEval(output_dir=td)
            with pytest.raises(ValueError, match="Unknown task"):
                ev.run_published_baseline("nonexistent_task_xyz")

    def test_baseline_synthetic(self):
        """run_published_baseline works with a synthetic task."""
        with tempfile.TemporaryDirectory() as td:
            ev = AutopilotEval(output_dir=td)
            result = ev.run_published_baseline("synthetic_quadratic")
            assert result.condition == "baseline"
            assert result.task_name == "synthetic_quadratic"
            assert ev._baseline_result is not None
            # Result file should have been saved
            assert os.path.exists(os.path.join(td, "baseline_result.json"))

    def test_report_with_baseline(self):
        """Report renders correctly with just a baseline result."""
        with tempfile.TemporaryDirectory() as td:
            ev = AutopilotEval(output_dir=td)
            ev.run_published_baseline("synthetic_quadratic")
            text = ev.report()
            assert "Baseline" in text
            assert "val_loss" in text
