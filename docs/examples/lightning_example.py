"""
Minimal Lightning + hotcb example.

Run directory must be initialized first:
    hotcb --dir runs/exp-001 init
"""

import torch
import pytorch_lightning as pl
from hotcb.kernel import HotKernel
from hotcb.adapters.lightning import HotCBLightning


class TinyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(16, 1)
        self.loss_state = {"weights": {"mse": 1.0}, "terms": {}, "ramps": {}}

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.net(x)
        w = self.loss_state["weights"].get("mse", 1.0)
        return w * torch.nn.functional.mse_loss(pred, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main():
    model = TinyModel()
    kernel = HotKernel(run_dir="runs/exp-001", debounce_steps=1)
    hotcb_cb = HotCBLightning(kernel, loss_state=model.loss_state)

    dataset = torch.utils.data.TensorDataset(
        torch.randn(200, 16), torch.randn(200, 1)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    trainer = pl.Trainer(max_epochs=5, callbacks=[hotcb_cb])
    trainer.fit(model, loader)


if __name__ == "__main__":
    main()
