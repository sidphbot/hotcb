import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset

from hotcb import HotController
from adapters.lightning import HotCallbackController


class ToyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.net(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main():
    # dummy data
    x = torch.randn(20_000, 32)
    y = torch.randint(0, 10, (20_000,))
    dl = DataLoader(TensorDataset(x, y), batch_size=128, shuffle=True, num_workers=0)

    controller = HotController(
        config_path="examples/hotcb.yaml",
        commands_path=None,              # optionally: "examples/hotcb.commands.jsonl"
        debounce_steps=5,                # poll every 5 steps
        poll_interval_sec=0.0,
        auto_disable_on_error=True,
        log_path="debug/hotcb.log",
    )

    hot = HotCallbackController(controller=controller)

    trainer = pl.Trainer(
        max_epochs=3,
        callbacks=[hot],
        log_every_n_steps=10,
    )
    trainer.fit(ToyModel(), train_dataloaders=dl)


if __name__ == "__main__":
    main()