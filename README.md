# hotcb

**Live Training Control Plane for PyTorch**

hotcb lets you modify training behavior **while your run is active** — no restart, no lost progress. Every change is recorded, exportable, and replayable.

Version 2.0 expands the original live-callback system into a full control plane: you can now swap callbacks, tune optimizer parameters, and adjust loss weights — all from another terminal while the model trains.

---

## What you get

| Module | What you can change live |
|---|---|
| **cb** | Load/unload/enable/disable/reconfigure callbacks |
| **opt** | Learning rate, weight decay, gradient clipping, per-group |
| **loss** | Loss weights, term toggles, ramp configs |

Plus:

- **Applied ledger** (`hotcb.applied.jsonl`): step-indexed, authoritative record of what actually happened
- **Recipe export + replay**: export a run's changes as a portable plan, replay in future runs deterministically
- **Freeze modes**: production lock, deterministic replay, replay-with-adjustments

---

## Installation

```bash
pip install hotcb
```

**With YAML config support:**

```bash
pip install "hotcb[yaml]"
```

**Full extras (YAML + Lightning + HuggingFace adapters):**

```bash
pip install "hotcb[all]"
```

---

## Quickstart

### 1. Initialize a run directory

```bash
hotcb --dir runs/exp1 init
```

### 2. Integrate into training

#### PyTorch Lightning

```python
from hotcb import HotKernel
from hotcb.adapters.lightning import HotCBLightning

kernel = HotKernel(run_dir="runs/exp1", debounce_steps=10)

trainer = pl.Trainer(callbacks=[HotCBLightning(kernel)])
trainer.fit(model, datamodule=dm)
trainer.fit(model, datamodule=dm)
```

#### HuggingFace Trainer

```python
from hotcb import HotKernel
from hotcb.adapters.hf import HotCBHFCallback

kernel = HotKernel(run_dir="runs/exp1", debounce_steps=10)
trainer = Trainer(..., callbacks=[HotCBHFCallback(kernel)])
trainer.train()
```

#### Bare PyTorch

```python
from hotcb import HotKernel

kernel = HotKernel(run_dir="runs/exp1", debounce_steps=10)

for step, batch in enumerate(dl):
    # ... forward, backward, optimizer step ...

    kernel.apply(
        env={
            "framework": "torch",
            "phase": "train",
            "step": step,
            "optimizer": optimizer,
            "loss_state": model.loss_state,
            "log": print,
        },
        events=["train_step_end"],
    )
```

### 3. Control training live (from another terminal)

```bash
# Load a diagnostic callback
hotcb --dir runs/exp1 cb load feat_viz \
  --file /tmp/feat_viz.py \
  --symbol FeatureVizCallback \
  --enabled --init every=50

# Change learning rate
hotcb --dir runs/exp1 opt set_params lr=1e-4 weight_decay=0.02

# Change loss weights
hotcb --dir runs/exp1 loss set_params distill_w=0.2 depth_w=1.5

# Toggle a loss term off
hotcb --dir runs/exp1 loss set_params terms.aux_depth=false
```

---

## Syntactic sugar

```bash
# enable/disable default to the cb module
hotcb --dir runs/exp1 enable timing
hotcb --dir runs/exp1 disable timing

# set auto-routes based on key patterns
hotcb --dir runs/exp1 set lr=5e-5          # → opt
hotcb --dir runs/exp1 set distill_w=0.25   # → loss
```

---

## Run artifacts

| File | Purpose |
|---|---|
| `hotcb.commands.jsonl` | What you asked for (incoming commands) |
| `hotcb.applied.jsonl` | What actually happened (step-indexed, authoritative) |
| `hotcb.recipe.jsonl` | Portable replay plan exported from the ledger |
| `hotcb.sources/` | Captured callback source files for deterministic replay |
| `hotcb.freeze.json` | Current freeze mode state |

---

## Freeze modes

| Mode | Behavior |
|---|---|
| `off` | Normal — all live commands accepted |
| `prod` | Ignore all external commands (production lock) |
| `replay` | Ignore external commands, replay recipe deterministically |
| `replay_adjusted` | Replay recipe with YAML overlay patches |

```bash
# Lock a production run
hotcb --dir runs/exp1 freeze --mode prod

# Replay a previous run exactly
hotcb --dir runs/exp1 recipe export --out runs/exp1/hotcb.recipe.jsonl
hotcb --dir runs/exp2 freeze --mode replay --recipe runs/exp1/hotcb.recipe.jsonl

# Replay with adjustments
hotcb --dir runs/exp2 freeze --mode replay_adjusted \
  --recipe runs/exp1/hotcb.recipe.jsonl \
  --adjust runs/exp2/hotcb.adjust.yaml

# Unlock
hotcb --dir runs/exp1 freeze --mode off
```

---

## Exposing optimizer and loss state

hotcb never monkeypatches the trainer. It mutates only what you pass via `env`.

### Optimizer

Pass `env["optimizer"]` (or `env["resolve_optimizer"]` as a callable). The Lightning and HF adapters handle this automatically.

### Loss state

Keep a mutable dict on your model:

```python
self.loss_state = {
    "weights": {"distill": 0.2, "depth": 1.5},
    "terms":   {"aux_depth": True, "aux_heatmap": False},
    "ramps":   {"depth": {"type": "linear", "warmup_frac": 0.2, "end": 2.0}},
}
```

Set `env["loss_state"] = self.loss_state` — the adapters do this automatically if the attribute exists on your LightningModule or HF model.

---

## Deterministic callback replay

When you load a callback from a Python file, hotcb captures its source:

```bash
hotcb --dir runs/exp1 cb load feat_viz \
  --file /tmp/feat_viz.py --symbol FeatureVizCallback
```

hotcb computes SHA-256, copies the file to `hotcb.sources/`, and records the version in the ledger. Replay mode uses the captured version — even if the original file has since changed.

---

## Status and inspection

```bash
# Show current freeze mode and recent applied entries
hotcb --dir runs/exp1 status

# Validate a recipe file
hotcb --dir runs/exp1 recipe validate --recipe runs/exp1/hotcb.recipe.jsonl

# Inspect the ledger directly
tail -n 20 runs/exp1/hotcb.applied.jsonl
```

---

## Included diagnostic callbacks

hotcb ships with ready-to-use callbacks:

| Callback | What it does |
|---|---|
| `HeartbeatCallback` | Periodic "I'm alive" log signal |
| `TimingCallback` | Step timing and throughput |
| `SystemStatsCallback` | CPU / RAM / GPU utilization |
| `TensorStatsCallback` | Tensor mean / std / min / max |
| `GradStatsCallback` | Gradient norm and stability |
| `AnomalyGuardCallback` | NaN/Inf detection with auto-disable |
| `JSONLLoggerCallback` | Structured append-only JSONL metrics log |

```bash
hotcb --dir runs/exp1 cb load heartbeat \
  --path hotcb.modules.cb.callbacks.heartbeat \
  --symbol HeartbeatCallback \
  --enabled --init every=100
```

---

## Writing a hot callback

Minimal contract — no base class required:

```python
class MyCallback:
    def __init__(self, id: str, every: int = 50):
        self.id = id
        self.every = every

    def set_params(self, **kwargs):
        if "every" in kwargs:
            self.every = int(kwargs["every"])

    def handle(self, event: str, env: dict):
        step = env.get("step", 0)
        if step % self.every == 0:
            env.get("log", print)(f"[{self.id}] step={step}")
```

---

## Safety

- No training loop mutation — hotcb never touches the trainer internals
- Safe-point updates only — changes applied at batch/step boundaries
- Fail-safe — crashing callbacks and modules auto-disable, training continues
- Full audit trail — every mutation written to the applied ledger

---

## Docs

- [Concepts](docs/concepts.md) — HotKernel, ops, ledger, recipe, freeze modes
- [CLI Reference](docs/cli.md) — all commands and sugar rules
- [Replay](docs/replay.md) — recipe export, replay modes, overlays
- [Formats](docs/formats.md) — JSONL, JSON, and YAML schemas
- Modules: [cb](docs/modules/cb.md) | [opt](docs/modules/hotopt.md) | [loss](docs/modules/hotloss.md)
- Examples: [Lightning](docs/examples/lightning_example.py) | [HF](docs/examples/hf_example.py) | [Bare PyTorch](docs/examples/bare_torch_example.py) | [Custom callback](docs/examples/custom_callback_example.py) | [Adjust overlay](docs/examples/adjust_overlay.yaml)
- [CLI Walkthrough](docs/examples/cli_walkthrough.md) — full live-control session from init to replay

---

## License

MIT License (see LICENSE file).
