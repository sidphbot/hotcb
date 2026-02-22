import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hotcb import HotController
from adapters.hf import HotHFCallback


class TinyTextDS(Dataset):
    def __init__(self, tok):
        self.tok = tok
        self.texts = ["hello world"] * 2048
        self.labels = [0] * 2048

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding="max_length", max_length=16, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def main():
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    ds = TinyTextDS(tok)

    controller = HotController(
        config_path="examples/hotcb.yaml",
        debounce_steps=10,
        log_path="debug/hotcb.log",
    )
    hot = HotHFCallback(controller=controller)

    args = TrainingArguments(
        output_dir="debug/hf_out",
        per_device_train_batch_size=16,
        max_steps=200,
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        callbacks=[hot],
    )
    trainer.train()


if __name__ == "__main__":
    main()