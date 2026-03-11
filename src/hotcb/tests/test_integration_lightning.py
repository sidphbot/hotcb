"""Integration tests for hotcb Lightning adapter (spec §19.8)."""
from __future__ import annotations

import json
import os

import pytest
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from hotcb.kernel import HotKernel
from hotcb.adapters.lightning import HotCBLightning


class TinyModel(pl.LightningModule):
    def __init__(self, lr: float = 0.1):
        super().__init__()
        self.layer = nn.Linear(4, 2)
        self._lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        return nn.functional.mse_loss(self.layer(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self._lr)

    def train_dataloader(self):
        ds = TensorDataset(torch.randn(32, 4), torch.randn(32, 2))
        return DataLoader(ds, batch_size=4)


def _read_ledger(run_dir):
    path = os.path.join(run_dir, "hotcb.applied.jsonl")
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def _write_commands(run_dir, *cmds):
    path = os.path.join(run_dir, "hotcb.commands.jsonl")
    with open(path, "a") as f:
        for c in cmds:
            f.write(json.dumps(c) + "\n")


def _write_freeze(run_dir, **kwargs):
    path = os.path.join(run_dir, "hotcb.freeze.json")
    with open(path, "w") as f:
        f.write(json.dumps(kwargs))


class TestOptChangeViaCommand:
    def test_optimizer_lr_changed(self, tmp_path):
        run_dir = str(tmp_path / "run1")
        os.makedirs(run_dir, exist_ok=True)

        # Pre-write command to change lr
        _write_commands(run_dir, {"module": "opt", "op": "set_params", "id": "main", "params": {"lr": 0.001}})

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        adapter = HotCBLightning(kernel=kernel)
        model = TinyModel(lr=0.1)

        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[adapter],
        )
        trainer.fit(model)

        # Verify optimizer was changed
        opt = trainer.optimizers[0]
        assert opt.param_groups[0]["lr"] == pytest.approx(0.001)

        # Verify ledger has applied entry
        ledger = _read_ledger(run_dir)
        applied = [e for e in ledger if e.get("module") == "opt" and e.get("decision") == "applied"]
        assert len(applied) >= 1
        assert applied[0]["source"] == "external"


class TestReplayReproduces:
    def test_replay_same_lr_change(self, tmp_path):
        # --- Run 1: external command changes lr ---
        run1 = str(tmp_path / "run1")
        os.makedirs(run1, exist_ok=True)

        _write_commands(run1, {"module": "opt", "op": "set_params", "id": "main", "params": {"lr": 0.001}})

        kernel1 = HotKernel(run_dir=run1, debounce_steps=1)
        adapter1 = HotCBLightning(kernel=kernel1)
        model1 = TinyModel(lr=0.1)

        trainer1 = pl.Trainer(
            max_epochs=1, accelerator="cpu",
            enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
            callbacks=[adapter1],
        )
        trainer1.fit(model1)

        # Export recipe from applied ledger
        ledger1 = _read_ledger(run1)
        applied1 = [e for e in ledger1 if e.get("decision") == "applied" and e.get("module") in {"cb", "opt", "loss"}]
        assert len(applied1) >= 1

        recipe_path = os.path.join(run1, "recipe.jsonl")
        with open(recipe_path, "w") as f:
            for e in applied1:
                entry = {
                    "at": {"step": e["step"], "event": e["event"]},
                    "module": e["module"],
                    "op": e["op"],
                    "id": e["id"],
                }
                payload = e.get("payload") or {}
                for k in ("params", "target", "init", "enabled"):
                    if k in payload:
                        entry[k] = payload[k]
                # If payload has no "params" key but op is set_params,
                # the payload IS the params dict (opt/loss modules return it directly)
                if e.get("op") == "set_params" and "params" not in entry and payload:
                    entry["params"] = payload
                f.write(json.dumps(entry) + "\n")

        # --- Run 2: replay ---
        run2 = str(tmp_path / "run2")
        os.makedirs(run2, exist_ok=True)

        _write_freeze(run2, mode="replay", recipe_path=recipe_path, policy="best_effort", step_offset=0)

        kernel2 = HotKernel(run_dir=run2, debounce_steps=1)
        adapter2 = HotCBLightning(kernel=kernel2)
        model2 = TinyModel(lr=0.1)

        trainer2 = pl.Trainer(
            max_epochs=1, accelerator="cpu",
            enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
            callbacks=[adapter2],
        )
        trainer2.fit(model2)

        # Verify same lr change happened
        opt2 = trainer2.optimizers[0]
        assert opt2.param_groups[0]["lr"] == pytest.approx(0.001)

        # Verify ledger shows replay source
        ledger2 = _read_ledger(run2)
        replay_applied = [e for e in ledger2 if e.get("source") == "replay" and e.get("decision") == "applied"]
        assert len(replay_applied) >= 1


class TestLossStateMutation:
    def test_loss_state_changed(self, tmp_path):
        run_dir = str(tmp_path / "run_loss")
        os.makedirs(run_dir, exist_ok=True)

        _write_commands(run_dir, {"module": "loss", "op": "set_params", "id": "main", "params": {"distill_w": 0.5}})

        loss_state = {"weights": {}, "terms": {}, "ramps": {}}
        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        adapter = HotCBLightning(kernel=kernel, loss_state=loss_state)
        model = TinyModel()

        trainer = pl.Trainer(
            max_epochs=1, accelerator="cpu",
            enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
            callbacks=[adapter],
        )
        trainer.fit(model)

        assert loss_state["weights"]["distill"] == 0.5

        ledger = _read_ledger(run_dir)
        loss_applied = [e for e in ledger if e.get("module") == "loss" and e.get("decision") == "applied"]
        assert len(loss_applied) >= 1
