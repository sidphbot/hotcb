"""
Minimal bare PyTorch training loop with hotcb.

Run directory must be initialized first:
    hotcb --dir runs/exp-001 init
"""

import torch
from hotcb.kernel import HotKernel


def main():
    model = torch.nn.Linear(16, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_state = {"weights": {"mse": 1.0}, "terms": {}, "ramps": {}}

    kernel = HotKernel(run_dir="runs/exp-001", debounce_steps=1)

    dataset = torch.utils.data.TensorDataset(
        torch.randn(200, 16), torch.randn(200, 1)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    for epoch in range(5):
        for step, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            pred = model(x)
            w = loss_state["weights"].get("mse", 1.0)
            loss = w * torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()

            global_step = epoch * len(loader) + step
            env = {
                "step": global_step,
                "epoch": epoch,
                "phase": "train",
                "model": model,
                "optimizer": optimizer,
                "loss": loss,
                "loss_state": loss_state,
                "log": print,
            }
            kernel.apply(env, events=["train_step_end"])


if __name__ == "__main__":
    main()
