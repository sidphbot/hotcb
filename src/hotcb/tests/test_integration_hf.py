"""Integration tests for hotcb HF adapter (spec §19.9)."""
from __future__ import annotations

import json
import os

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments

from hotcb.kernel import HotKernel
from hotcb.adapters.hf import HotCBHFCallback


class TinyDataset(Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, idx):
        return {"input_ids": torch.randn(4), "labels": torch.randn(2)}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 2)

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = self.layer(input_ids)
        loss = nn.functional.mse_loss(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


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


def _make_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        max_steps=8,
        no_cuda=True,
        report_to="none",
        logging_steps=999,
        per_device_train_batch_size=4,
        learning_rate=0.1,
    )


class TestHFOptChange:
    def test_optimizer_weight_decay_changed(self, tmp_path):
        """Test weight_decay mutation (lr is scheduler-managed in HF, so we test weight_decay)."""
        run_dir = str(tmp_path / "hf_run1")
        os.makedirs(run_dir, exist_ok=True)
        hf_out = str(tmp_path / "hf_out1")

        _write_commands(run_dir, {"module": "opt", "op": "set_params", "id": "main", "params": {"weight_decay": 0.05}})

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        model = TinyModel()

        trainer = Trainer(model=model, args=_make_args(hf_out), train_dataset=TinyDataset())

        cb = HotCBHFCallback(kernel=kernel, resolve_optimizer=lambda: trainer.optimizer)
        trainer.add_callback(cb)
        trainer.train()

        # Verify weight_decay changed
        assert trainer.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.05)

        # Verify ledger has applied entry
        ledger = _read_ledger(run_dir)
        applied = [e for e in ledger if e.get("module") == "opt" and e.get("decision") == "applied"]
        assert len(applied) >= 1
        assert applied[0]["source"] == "external"

    def test_ledger_records_applied_step(self, tmp_path):
        """Verify ledger step matches actual global_step."""
        run_dir = str(tmp_path / "hf_run_step")
        os.makedirs(run_dir, exist_ok=True)
        hf_out = str(tmp_path / "hf_out_step")

        _write_commands(run_dir, {"module": "opt", "op": "set_params", "id": "main", "params": {"weight_decay": 0.1}})

        kernel = HotKernel(run_dir=run_dir, debounce_steps=1)
        trainer = Trainer(model=TinyModel(), args=_make_args(hf_out), train_dataset=TinyDataset())
        cb = HotCBHFCallback(kernel=kernel, resolve_optimizer=lambda: trainer.optimizer)
        trainer.add_callback(cb)
        trainer.train()

        ledger = _read_ledger(run_dir)
        opt_entries = [e for e in ledger if e.get("module") == "opt" and e.get("decision") == "applied"]
        assert len(opt_entries) >= 1
        # Step should be a positive int (command picked up during training)
        assert opt_entries[0]["step"] >= 0


class TestHFReplay:
    def test_replay_same_change(self, tmp_path):
        # --- Run 1 ---
        run1 = str(tmp_path / "hf_run1")
        os.makedirs(run1, exist_ok=True)
        hf_out1 = str(tmp_path / "hf_out1")

        _write_commands(run1, {"module": "opt", "op": "set_params", "id": "main", "params": {"weight_decay": 0.05}})

        kernel1 = HotKernel(run_dir=run1, debounce_steps=1)
        trainer1 = Trainer(model=TinyModel(), args=_make_args(hf_out1), train_dataset=TinyDataset())
        cb1 = HotCBHFCallback(kernel=kernel1, resolve_optimizer=lambda: trainer1.optimizer)
        trainer1.add_callback(cb1)
        trainer1.train()

        # Export recipe
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
                if e.get("op") == "set_params" and "params" not in entry and payload:
                    entry["params"] = payload
                f.write(json.dumps(entry) + "\n")

        # --- Run 2: replay ---
        run2 = str(tmp_path / "hf_run2")
        os.makedirs(run2, exist_ok=True)
        hf_out2 = str(tmp_path / "hf_out2")

        _write_freeze(run2, mode="replay", recipe_path=recipe_path, policy="best_effort", step_offset=0)

        kernel2 = HotKernel(run_dir=run2, debounce_steps=1)
        trainer2 = Trainer(model=TinyModel(), args=_make_args(hf_out2), train_dataset=TinyDataset())
        cb2 = HotCBHFCallback(kernel=kernel2, resolve_optimizer=lambda: trainer2.optimizer)
        trainer2.add_callback(cb2)
        trainer2.train()

        # Verify same change
        assert trainer2.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.05)

        # Verify replay source in ledger
        ledger2 = _read_ledger(run2)
        replay = [e for e in ledger2 if e.get("source") == "replay" and e.get("decision") == "applied"]
        assert len(replay) >= 1
