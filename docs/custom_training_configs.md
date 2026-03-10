# Adding Custom Training Configurations

hotcb's dashboard launcher supports pluggable training configurations. You can
register your own training loop (bare PyTorch, Lightning, HuggingFace, or any
framework) to appear in the dashboard's Training card dropdown.

## How It Works

The launcher runs your training function in a background thread, communicating
with the dashboard via JSONL files in a shared `run_dir`:

```
run_dir/
├── hotcb.metrics.jsonl     # Your loop writes metric records here
├── hotcb.commands.jsonl     # Dashboard writes commands here (lr changes, etc.)
├── hotcb.applied.jsonl      # Your loop writes applied mutations here
├── hotcb.recipe.jsonl       # Scheduled interventions (optional)
└── hotcb.features.jsonl     # Activation captures (optional)
```

## Quick Start: Register a Custom Config

```python
from hotcb.server.launcher import TrainingLauncher, TrainingConfig
import threading

def my_training(run_dir, max_steps, step_delay, stop_event):
    """Your training loop. Must accept these 4 positional args."""
    # ... training code (see framework examples below) ...
    pass

# Create config
my_config = TrainingConfig(
    config_id="my_experiment",
    name="My Custom Experiment",
    description="ResNet-50 on CIFAR-100 with cosine annealing",
    train_fn=my_training,
    defaults={"max_steps": 10000, "step_delay": 0.0},
)

# Register with the launcher (before starting the server)
launcher = TrainingLauncher(run_dir="/path/to/runs")
launcher.register_config(my_config)
```

The config will appear in the dashboard dropdown alongside the built-in configs.

## Training Function Contract

Your `train_fn` must:

1. **Accept 4 positional args**: `(run_dir, max_steps, step_delay, stop_event)`
2. **Check `stop_event`** periodically — call `stop_event.is_set()` each step
3. **Write metrics** to `{run_dir}/hotcb.metrics.jsonl` as JSON lines
4. **Read commands** from `{run_dir}/hotcb.commands.jsonl` to respond to UI knobs
5. **Write applied mutations** to `{run_dir}/hotcb.applied.jsonl`

### Metric Record Format

```json
{"step": 42, "metrics": {"train_loss": 0.523, "val_loss": 0.612, "lr": 0.001, "accuracy": 0.78}}
```

All values in `metrics` must be numbers. The dashboard auto-discovers metric
names and creates chart lines for each.

### Command Format (read from commands JSONL)

```json
{"module": "opt", "op": "set_params", "params": {"lr": 0.0005}}
{"module": "loss", "op": "set_params", "params": {"weight": 0.7}}
{"module": "cb", "op": "set_params", "params": {"backbone_frozen": false}}
```

### Applied Mutation Format (write to applied JSONL)

```json
{"step": 42, "module": "opt", "op": "set_params", "params": {"lr": 0.0005}, "status": "applied"}
```

## Framework Examples

### Bare PyTorch

```python
import json, os, time, torch, torch.nn as nn

def pytorch_training(run_dir, max_steps, step_delay, stop_event):
    metrics_path = os.path.join(run_dir, "hotcb.metrics.jsonl")
    commands_path = os.path.join(run_dir, "hotcb.commands.jsonl")
    applied_path = os.path.join(run_dir, "hotcb.applied.jsonl")

    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    cmd_offset = 0

    for step in range(1, max_steps + 1):
        if stop_event.is_set():
            break

        # --- Read commands from dashboard ---
        cmds, cmd_offset = _read_commands(commands_path, cmd_offset)
        for cmd in cmds:
            if cmd.get("module") == "opt" and cmd.get("op") == "set_params":
                params = cmd.get("params", {})
                if "lr" in params:
                    for pg in optimizer.param_groups:
                        pg["lr"] = float(params["lr"])
                _write_jsonl(applied_path, {
                    "step": step, "module": "opt", "op": "set_params",
                    "params": params, "status": "applied",
                })

        # --- Training step ---
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        out = model(x)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Write metrics ---
        _write_jsonl(metrics_path, {
            "step": step,
            "metrics": {
                "train_loss": round(loss.item(), 6),
                "lr": optimizer.param_groups[0]["lr"],
            },
        })

        if step_delay > 0:
            time.sleep(step_delay)


def _write_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

def _read_commands(path, offset):
    if not os.path.exists(path):
        return [], offset
    cmds = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            line = line.strip()
            if line:
                try: cmds.append(json.loads(line))
                except: pass
    return cmds, offset + len(cmds)
```

### PyTorch Lightning

```python
import json, os
import pytorch_lightning as pl
from hotcb.modules.cb import HotCallbackController

def lightning_training(run_dir, max_steps, step_delay, stop_event):
    # hotcb provides a Lightning callback adapter
    from hotcb.modules.cb.adapters import LightningAdapter

    model = MyLightningModule()
    adapter = LightningAdapter(run_dir=run_dir)

    # The adapter handles JSONL I/O automatically:
    # - Writes metrics on_train_batch_end / on_validation_end
    # - Reads commands and applies them to optimizer/loss
    trainer = pl.Trainer(
        max_steps=max_steps,
        callbacks=[adapter],
    )
    trainer.fit(model)
    # Note: stop_event is checked internally by the adapter each step
```

### HuggingFace Transformers

```python
import json, os
from transformers import Trainer, TrainingArguments

def hf_training(run_dir, max_steps, step_delay, stop_event):
    from hotcb.modules.cb.adapters import HuggingFaceAdapter

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    adapter = HuggingFaceAdapter(run_dir=run_dir, stop_event=stop_event)

    training_args = TrainingArguments(
        output_dir=os.path.join(run_dir, "checkpoints"),
        max_steps=max_steps,
        logging_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=my_dataset,
        callbacks=[adapter],  # The adapter is a TrainerCallback
    )
    trainer.train()
```

## Programmatic Server Launch with Custom Configs

```python
from hotcb.server.app import create_app, run_server
from hotcb.server.launcher import TrainingLauncher, TrainingConfig

run_dir = "./my_runs"
launcher = TrainingLauncher(run_dir)
launcher.register_config(TrainingConfig(
    config_id="my_custom",
    name="My Custom Training",
    description="Fine-tune BERT on sentiment analysis",
    train_fn=my_training_fn,
    defaults={"max_steps": 5000, "step_delay": 0.0},
))

# The create_app function picks up the launcher from app state
app = create_app(run_dir)
# Manually set the launcher with custom configs
app.state.training_launcher = launcher

import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8421)
```

## Built-in Training Configs

| Config ID    | Name                          | Steps | Description |
|-------------|-------------------------------|-------|-------------|
| `simple`    | Simple (Quadratic)            | 500   | Single-task synthetic. Test basic controls. |
| `multitask` | Multi-Objective (Golden Demo) | 800   | Two-task with recipe-driven lambda shifts. |
| `finetune`  | Finetune (Pretrained Backbone)| 600   | Transfer learning with freeze/unfreeze control. |
