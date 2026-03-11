# hotcb Integration Guide

Minimal reference for integrating hotcb into any PyTorch training project.
This file is designed to be read by AI agents working on external repos.

## Install

```bash
pip install "hotcb[dashboard]"       # core + dashboard
pip install "hotcb[dashboard,ai]"    # + AI autopilot (LLM-driven)
```

## The Contract

hotcb communicates via two JSONL files in a run directory:

| File | Direction | Purpose |
|------|-----------|---------|
| `hotcb.metrics.jsonl` | Training → hotcb | Metrics each step (loss, lr, val_loss, etc.) |
| `hotcb.commands.jsonl` | hotcb → Training | Commands to apply (lr change, loss weight, etc.) |

Training writes metrics. hotcb (dashboard/CLI/autopilot) writes commands. That's it.

## Option A: Minimal (10 lines, no dependencies)

Add to your training loop — works with any framework:

```python
import json, os

RUN_DIR = "./runs/exp1"  # or any path
os.makedirs(RUN_DIR, exist_ok=True)

# 1. Write metrics each step
def write_metrics(step, metrics_dict):
    with open(os.path.join(RUN_DIR, "hotcb.metrics.jsonl"), "a") as f:
        f.write(json.dumps({"step": step, "metrics": metrics_dict}) + "\n")

# 2. Read and apply commands (call periodically, e.g. every 10 steps)
_cmd_offset = 0
def poll_commands():
    global _cmd_offset
    path = os.path.join(RUN_DIR, "hotcb.commands.jsonl")
    if not os.path.exists(path):
        return []
    cmds = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < _cmd_offset:
                continue
            line = line.strip()
            if line:
                try:
                    cmds.append(json.loads(line))
                except Exception:
                    pass
    _cmd_offset += len(cmds)
    return cmds

# In your training loop:
for step in range(max_steps):
    loss = train_step(batch)
    write_metrics(step, {"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

    if step % 10 == 0:
        for cmd in poll_commands():
            if cmd.get("module") == "opt" and cmd.get("op") == "set_params":
                for k, v in cmd["params"].items():
                    if k == "lr":
                        for pg in optimizer.param_groups:
                            pg["lr"] = float(v)
                    elif k == "weight_decay":
                        for pg in optimizer.param_groups:
                            pg["weight_decay"] = float(v)
```

## Option B: HotKernel (full features, still framework-agnostic)

```python
from hotcb.kernel import HotKernel
from hotcb.metrics import MetricsCollector

kernel = HotKernel(run_dir="./runs/exp1", debounce_steps=10)
mc = MetricsCollector("./runs/exp1/hotcb.metrics.jsonl")

for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    mc.log(step=step, metrics={"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

    kernel.apply(
        env={
            "framework": "torch",
            "phase": "train",
            "step": step,
            "optimizer": optimizer,
            "loss_state": model.loss_state,  # optional, for loss weight control
            "log": print,
        },
        events=["train_step_end"],
    )
```

## Option C: Framework Adapters (Lightning / HuggingFace)

```python
# PyTorch Lightning
from hotcb.kernel import HotKernel
from hotcb.adapters.lightning import HotCBLightning
from hotcb.metrics import MetricsCollector

kernel = HotKernel(
    run_dir="./runs/exp1",
    metrics_collector=MetricsCollector("./runs/exp1/hotcb.metrics.jsonl"),
)
trainer = pl.Trainer(callbacks=[HotCBLightning(kernel)])
trainer.fit(model)
```

```python
# HuggingFace Trainer
from hotcb.kernel import HotKernel
from hotcb.adapters.hf import HotCBHFCallback

kernel = HotKernel(run_dir="./runs/exp1")
trainer = Trainer(..., callbacks=[HotCBHFCallback(kernel)])
trainer.train()
```

## Metrics Format

Each line in `hotcb.metrics.jsonl`:

```json
{"step": 100, "metrics": {"loss": 0.45, "lr": 0.001, "val_loss": 0.52}}
```

Log whatever metrics you have. Common ones the autopilot understands:
- `loss`, `train_loss` — training loss
- `val_loss`, `val_accuracy`, `val_acc` — validation metrics
- `lr` — current learning rate
- `grad_norm` — gradient norm (useful for divergence detection)
- Any custom metric names work — the autopilot discovers them automatically

## Command Format

Each line in `hotcb.commands.jsonl`:

```json
{"module": "opt", "op": "set_params", "params": {"lr": 0.0005}}
{"module": "opt", "op": "set_params", "params": {"weight_decay": 0.01}}
{"module": "loss", "op": "set_params", "params": {"recon_w": 0.3, "cls_w": 0.7}}
```

If using Option A, you handle these yourself. Options B/C handle them automatically.

## Exposing Loss Weights (for multi-task training)

If your model has a multi-task or weighted loss, expose it as a mutable dict:

```python
# On your model or as a standalone dict
loss_state = {
    "weights": {"cls": 1.0, "recon": 0.5, "reg": 0.1},
    "terms": {"cls": True, "recon": True, "reg": True},  # toggleable
}

# Pass via env (Option B) or set on LightningModule (Option C)
# hotcb can then adjust weights live via: hotcb set cls_w=0.8 recon_w=0.2
```

## Making Your Training Launchable

`hotcb launch` and the `launch()` API can manage the full lifecycle (training + dashboard + autopilot) if your training function follows this signature:

```python
def train_fn(run_dir: str, max_steps: int, step_delay: float, stop_event: threading.Event):
    ...
```

| Argument | Type | What it is |
|----------|------|------------|
| `run_dir` | `str` | Directory for JSONL files — write metrics here, read commands from here |
| `max_steps` | `int` | Maximum training steps (respect this limit) |
| `step_delay` | `float` | Minimum seconds between steps (for demos/pacing; can be 0.0 in real training) |
| `stop_event` | `threading.Event` | Check `stop_event.is_set()` each step — if True, exit cleanly. This is how `handle.stop()` and `--max-time` work. |

### Wrapping an existing training loop

If you already have a training script, wrap it:

```python
# my_project/train.py
import json, os, time, threading

def train(run_dir: str, max_steps: int, step_delay: float, stop_event: threading.Event):
    """hotcb-compatible training function."""
    # --- your existing setup ---
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataloader = get_dataloader()

    cmd_offset = 0

    for step, batch in enumerate(dataloader):
        if step >= max_steps or stop_event.is_set():
            break

        # --- your existing training step ---
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # --- hotcb integration (metrics out, commands in) ---
        # Write metrics
        metrics = {
            "loss": loss.item(),
            "lr": optimizer.param_groups[0]["lr"],
        }
        # Add val metrics if you run validation periodically
        if step % 100 == 0:
            val_loss = validate(model)
            metrics["val_loss"] = val_loss

        with open(os.path.join(run_dir, "hotcb.metrics.jsonl"), "a") as f:
            f.write(json.dumps({"step": step, "metrics": metrics}) + "\n")

        # Poll commands (every 10 steps to keep overhead low)
        if step % 10 == 0:
            cmd_path = os.path.join(run_dir, "hotcb.commands.jsonl")
            if os.path.exists(cmd_path):
                with open(cmd_path) as f:
                    lines = f.readlines()
                for line in lines[cmd_offset:]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cmd = json.loads(line)
                    except Exception:
                        continue
                    if cmd.get("module") == "opt" and cmd.get("op") == "set_params":
                        for k, v in cmd.get("params", {}).items():
                            if k == "lr":
                                for pg in optimizer.param_groups:
                                    pg["lr"] = float(v)
                            elif k == "weight_decay":
                                for pg in optimizer.param_groups:
                                    pg["weight_decay"] = float(v)
                cmd_offset = len(lines)

        if step_delay > 0:
            time.sleep(step_delay)
```

Key points:
- **Check `stop_event`** each step — this is how `--max-time` and `handle.stop()` terminate training gracefully
- **Respect `max_steps`** — but your loop can also end earlier (e.g. dataset exhausted)
- **`step_delay`** is mainly for demos/simulation — in real training, pass `step_delay=0` or ignore it
- **Write metrics every step** (or as often as you can) — the autopilot needs them to detect trends
- **Poll commands periodically** — every 10-50 steps is fine, doesn't need to be every step

### Using HotKernel instead of manual command polling

If you prefer the full kernel (automatic command routing, loss_state support, freeze modes):

```python
def train(run_dir, max_steps, step_delay, stop_event):
    from hotcb.kernel import HotKernel
    from hotcb.metrics import MetricsCollector

    kernel = HotKernel(run_dir=run_dir, debounce_steps=10)
    mc = MetricsCollector(os.path.join(run_dir, "hotcb.metrics.jsonl"))

    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step, batch in enumerate(dataloader):
        if step >= max_steps or stop_event.is_set():
            break

        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        mc.log(step=step, metrics={"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
        kernel.apply(
            env={"framework": "torch", "phase": "train", "step": step,
                 "optimizer": optimizer, "loss_state": getattr(model, "loss_state", None), "log": print},
            events=["train_step_end"],
        )

        if step_delay > 0:
            time.sleep(step_delay)
```

## Create `hotcb.launch.json` (recommended)

Once integration is done, create a `hotcb.launch.json` in your project root. This tells hotcb (and the Claude Code autopilot skill) how to launch your training — no questions asked.

```json
{
  "train_fn": "my_project.train:train",
  "run_dir": "./runs",
  "key_metric": "val_loss",
  "max_steps": 5000,
  "max_time": 300,
  "autopilot": "ai_suggest",
  "port": 8421
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `train_fn` | yes | Training function as `module.path:fn_name` (must follow the contract above) |
| `run_dir` | no | Run directory (default: auto-created temp dir) |
| `key_metric` | no | Primary metric to optimize (default: `val_loss`) |
| `max_steps` | no | Max training steps |
| `max_time` | no | Wall-clock time limit in seconds |
| `autopilot` | no | Autopilot mode (default: `off`) |
| `port` | no | Dashboard port (default: `8421`) |
| `step_delay` | no | Seconds between steps (default: `0` for real training) |
| `seed` | no | Random seed |

**Why this matters**: when someone opens Claude Code in your project and invokes the hotcb autopilot skill, it reads `hotcb.launch.json` and launches immediately — zero setup questions. Without it, the skill has to ask the user for every parameter.

You can also pass it directly to `hotcb launch`:

```bash
hotcb launch --config-file hotcb.launch.json
```

## Launch Dashboard + Autopilot

Once your training function follows the contract:

```bash
# CLI — references the function as module:fn
hotcb launch --train-fn my_project.train:train --autopilot ai_suggest --key-metric val_loss

# Time-bounded (5 minutes, regardless of GPU speed)
hotcb launch --train-fn my_project.train:train --max-time 300 --autopilot ai_suggest

# Steps + time (whichever limit hits first)
hotcb launch --train-fn my_project.train:train --max-steps 5000 --max-time 600
```

Or programmatically:

```python
from hotcb.launch import launch

handle = launch(
    train_fn="my_project.train:train",  # or pass the callable directly
    autopilot="ai_suggest",
    key_metric="val_loss",
    max_time=300,                       # 5-minute wall-clock limit
    serve=True,
)
```

Or attach to an already-running training (just the dashboard + autopilot):

```bash
# If training is already writing hotcb.metrics.jsonl in ./runs/exp1:
hotcb serve --dir ./runs/exp1 --autopilot ai_suggest
```

Dashboard: `http://localhost:8421`
