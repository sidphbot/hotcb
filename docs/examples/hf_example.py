"""
Minimal HuggingFace Trainer + hotcb example.

Run directory must be initialized first:
    hotcb --dir runs/exp-001 init
"""

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

from hotcb.kernel import HotKernel
from hotcb.adapters.hf import HotCBHFCallback


class DummyDataset(Dataset):
    def __init__(self, n=200):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 1000, (32,)),
            "attention_mask": torch.ones(32, dtype=torch.long),
            "labels": torch.tensor(idx % 2),
        }


def main():
    model = AutoModelForSequenceClassification.from_pretrained(
        "prajjwal1/bert-tiny", num_labels=2
    )
    kernel = HotKernel(run_dir="runs/exp-001", debounce_steps=1)

    # Pass resolve_optimizer so hotopt can access the optimizer
    trainer_ref = {}
    hotcb_cb = HotCBHFCallback(
        kernel,
        resolve_optimizer=lambda: trainer_ref.get("trainer") and trainer_ref["trainer"].optimizer,
        loss_state={"weights": {"ce": 1.0}, "terms": {}, "ramps": {}},
    )

    args = TrainingArguments(
        output_dir="/tmp/hf_hotcb_test",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        logging_steps=10,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=DummyDataset(), callbacks=[hotcb_cb])
    trainer_ref["trainer"] = trainer
    trainer.train()


if __name__ == "__main__":
    main()
