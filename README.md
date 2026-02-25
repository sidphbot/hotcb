# hotcb 🔥  
Hot-swappable callbacks for PyTorch Lightning, HuggingFace Trainer, or bare PyTorch.

Enable, disable, modify, or load new callbacks **while training is running** — without restarting.

---

## ✨ Features

- ✅ Enable / disable callbacks live
- ✅ Update callback parameters at runtime
- ✅ Load callbacks from a new Python file path
- ✅ Works with:
  - PyTorch Lightning
  - HuggingFace Trainer
  - Bare PyTorch loops
- ✅ No DDP required
- ✅ CLI helper included
- ✅ Minimal and framework-agnostic core

---

## 📦 Installation

Core only:

```bash
pip install hotcb
```

With YAML support:

```bash
pip install "hotcb[yaml]"
```

With Lightning adapter:

```bash
pip install "hotcb[lightning]"
```

With HuggingFace adapter:

```bash
pip install "hotcb[hf]"
```

Install everything:

```bash
pip install "hotcb[all]"
```

---

# 🚀 Quickstart (PyTorch Lightning)

```python
from hotcb import HotController
from hotcb.adapters.lightning import HotCallbackController
import lightning.pytorch as pl

controller = HotController(
    config_path="runs/exp1/hotcb.yaml",
    commands_path="runs/exp1/hotcb.commands.jsonl",
    debounce_steps=5,
)

trainer = pl.Trainer(
    callbacks=[HotCallbackController(controller)],
)
```

---

# 🚀 Quickstart (HuggingFace Trainer)

```python
from hotcb import HotController
from hotcb.adapters.hf import HotHFCallback
from transformers import Trainer

controller = HotController(
    config_path="runs/exp1/hotcb.yaml",
    commands_path="runs/exp1/hotcb.commands.jsonl",
)

trainer = Trainer(
    ...,
    callbacks=[HotHFCallback(controller)],
)
```

---

# 🚀 Quickstart (Bare PyTorch)

```python
controller = HotController(
    config_path="runs/exp1/hotcb.yaml",
    commands_path="runs/exp1/hotcb.commands.jsonl",
)

for step, batch in enumerate(loader):
    # training logic...
    controller.apply(
        env={
            "step": step,
            "phase": "train",
            "model": model,
            "log": print,
        },
        events=["train_step_end"],
    )
```

# 🧭 CLI Control (Live, No Restart)

`hotcb` includes a lightweight CLI to control callbacks while training is running.

First, initialize a run directory:

```bash
hotcb --dir runs/exp1 init
```

This creates:

```
runs/exp1/
  hotcb.yaml
  hotcb.commands.jsonl
```

---

### 🔥 Load a callback from a new file

```bash
hotcb --dir runs/exp1 load feat_viz \
  --file /tmp/feat_viz.py \
  --symbol FeatureVizCallback \
  --enabled \
  --init every=100 out_dir=debug/features
```

It starts running immediately (at the next safe step).

---

### ⚡ Enable / Disable instantly

```bash
hotcb --dir runs/exp1 enable feat_viz
hotcb --dir runs/exp1 disable feat_viz
```

Disable = soft remove (no restart required).

---

### 🎛 Adjust parameters live

```bash
hotcb --dir runs/exp1 set feat_viz every=25
hotcb --dir runs/exp1 set feat_viz threshold=30.5 prefix=[debug]
```

Changes are applied at the next safe point.

---

### 🧹 Unload completely (optional)

```bash
hotcb --dir runs/exp1 unload feat_viz
```

This disables and drops the instance.

---

### 💡 Typical Workflow

1. Start training once.
2. Notice something odd.
3. Drop a new `.py` diagnostic file.
4. `hotcb load ...`
5. Inspect.
6. `hotcb disable ...`
7. Continue training uninterrupted.

No restarts. No trainer hacks. No killing long runs.


---

# 🧠 Writing a Hot Callback

Minimal contract:

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

That’s it.

---

### 📚 See Real Examples

For more complete examples (including file-based hot loading and artifact writing), check:

- `examples/callbacks/print_metrics.py` — minimal logging callback  
- `examples/callbacks/feat_viz.py` — writes step-based artifacts to disk  
- `examples/lightning_train.py` — Lightning integration example  
- `examples/hf_train.py` — HuggingFace Trainer integration example  

These examples are fully runnable and demonstrate live parameter updates via the CLI.

---

---

# 🧰 Included Diagnostic Callbacks

`hotcb` includes a lightweight built-in diagnostics pack so you can start instrumenting runs immediately:

- **HeartbeatCallback** — periodic “I’m alive” signal for long runs  
- **TimingCallback** — step timing & throughput tracking  
- **SystemStatsCallback** — CPU / RAM / (optional) GPU utilization  
- **TensorStatsCallback** — tensor mean/std/min/max tracking  
- **GradStatsCallback** — gradient norm & stability diagnostics  
- **AnomalyGuardCallback** — basic NaN / Inf detection & auto-disable protection  
- **JSONLLoggerCallback** — structured append-only JSONL event logging  

These are intentionally minimal, composable, and safe to enable/disable at runtime.

Example:

```bash
hotcb --dir runs/exp1 load heartbeat \
  --module hotcb.callbacks.heartbeat \
  --symbol HeartbeatCallback \
  --enabled \
  --init every=100
```

Or enable a gradient monitor mid-training:

```bash
hotcb --dir runs/exp1 load grad_stats \
  --module hotcb.callbacks.grad_stats \
  --symbol GradStatsCallback \
  --enabled \
  --init every=50
```

All included callbacks support live parameter updates:

```bash
hotcb --dir runs/exp1 set grad_stats every=10
```

No restart required.

---

# 🔎 Intelligent Logging Resolvers

hotcb callbacks run inside the same Python process as your training loop.
That means they can often discover and reuse the logging infrastructure already configured by your framework (Lightning, HuggingFace Trainer, or custom code).

To support this cleanly and safely, hotcb provides logging resolvers — utilities that attempt to discover common logging backends from the runtime env passed to callbacks.

### Individual resolvers - For Scalars, Images, Histograms etc 

Resolvers allow a callback to “plug into” existing logging backends automatically.

- Discover logger candidates (no strict contract required) - Resolvers inspect the env dictionary and attempt to extract logger-like objects from common locations (No adapter-level constraints are imposed. This is purely best-effort introspection.)

- Or, Resolve specific backends (official + heuristic detection) - Resolvers find Known official classes (when installed) + Safe attribute-based heuristics

|Supported backends| resolver function                 | Returns              |
|------------------|-----------------------------------|----------------------|
|Tensorboard| `resolve_tensorboard_writer(env)` | `writer`             |
|MLFlow| `resolve_mlflow(env)`             | `experiment, run_id` |
|Comet| `resolve_comet_experiment(env)`   | `experiment`         |

Typical sources:

- Lightning TensorBoardLogger, MLFlowLogger or CometLogger
- HF TensorBoardCallback, MLflowCallback or CometCallback (best effort)
- Direct SummaryWriter or (client, run_id) tuple passed in `env["mlflow"]` or object passed in `env["comet_experiment"]`

### Holistic logging Convenience Helper - For Scalars only

A unified helper is also provided:
```
log_scalar(env, key, value, step=None)
```
Behavior:

- Try TensorBoard (add_scalar)
- Try MLflow (log_metric)
- Try Comet (log_metric)

Returns `True` if logging succeeded to at least one backend.

Failures are swallowed — logging will never crash your training loop.

Use framework-native logging for training-critical metrics.  
Use hotcb resolvers for live instrumentation, debugging, and temporary analytics.

---

# 🔌 Import Scope in Hot-Loaded Callbacks

Callbacks run in the same interpreter as your training job.

You can import from your training repo if:
- You run from repo root, or
- The project is installed (editable or normal), or
- PYTHONPATH is configured.

Prefer absolute imports in hot-loaded `.py` files.

---


# 🧬 A Unified Callback Model

`hotcb` is a thin portability layer: it lets you write one callback once, then run it across
PyTorch Lightning, HuggingFace Trainer, or bare PyTorch by mapping framework hook arguments into a
small, normalized `env` dictionary.

`env` is intentionally small and predictable. Adapters fill it from the native framework objects:

| `env` key | Lightning (source) | HF Trainer (source) | Bare PyTorch (source) |
|---|---|---|---|
| `env["step"]` | `trainer.global_step` | `state.global_step` | loop `step` |
| `env["epoch"]` | `trainer.current_epoch` | `state.epoch` | loop `epoch` |
| `env["phase"]` | adapter sets `"train"/"val"` | adapter sets `"train"/"eval"` | you set it |
| `env["model"]` | `pl_module` | *(adapter-provided)* | your model |
| `env["batch"]` | `batch` | *(adapter-provided)* | your batch |
| `env["outputs"]` | `outputs` | *(optional)* | your outputs |
| `env["log"]` | adapter wraps `trainer.print` | adapter wraps `print` | your logger |

> `env` is a portability contract. If you want extra fields, you can always include them in `env`
> from your own loop, or extend adapters later — but the minimal set above keeps callbacks simple
> and avoids accidental retention of large tensors.


---

# 🛠 Making existing callbacks Hot-Adjustable

All four variants below do the same thing:
- print once every `every` steps
- easy to tune `every`
- the hotcb version supports runtime updates via the CLI (`hotcb set ...`)

### 1) PyTorch Lightning callback

```python
import lightning.pytorch as pl

class PrintEveryN_Lightning(pl.Callback):
    def __init__(self, every: int = 50, prefix: str = "[metrics]"):
        self.every = int(every)
        self.prefix = str(prefix)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if self.every > 0 and (step % self.every) == 0:
            trainer.print(f"{self.prefix} step={step} batch_idx={batch_idx}")
```

### 2) HuggingFace Trainer callback

```python
from transformers import TrainerCallback

class PrintEveryN_HF(TrainerCallback):
    def __init__(self, every: int = 50, prefix: str = "[metrics]"):
        self.every = int(every)
        self.prefix = str(prefix)

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if self.every > 0 and (step % self.every) == 0:
            print(f"{self.prefix} step={step}")
        return control
```

### 3) Bare PyTorch “hook style”

```python
class PrintEveryN_TorchHook:
    def __init__(self, every: int = 50, prefix: str = "[metrics]"):
        self.every = int(every)
        self.prefix = str(prefix)

    def on_step_end(self, step: int, batch_idx: int):
        if self.every > 0 and (step % self.every) == 0:
            print(f"{self.prefix} step={step} batch_idx={batch_idx}")
```

Usage:

```python
hook = PrintEveryN_TorchHook(every=50)

for step, batch in enumerate(loader):
    # forward/backward/step...
    hook.on_step_end(step=step, batch_idx=step)
```

### 4) hotcb callback (portable + hot-adjustable)

```python
class PrintEveryN_HotCB:
    def __init__(self, id: str, every: int = 50, prefix: str = "[metrics]"):
        self.id = id
        self.every = int(every)
        self.prefix = str(prefix)

    def set_params(self, **kwargs):
        if "every" in kwargs:
            self.every = int(kwargs["every"])
        if "prefix" in kwargs:
            self.prefix = str(kwargs["prefix"])

    def handle(self, event: str, env: dict):
        step = int(env.get("step", 0))
        batch_idx = env.get("batch_idx", None)
        log = env.get("log", print)

        if self.every > 0 and (step % self.every) == 0:
            log(f"{self.prefix} id={self.id} step={step} event={event} batch_idx={batch_idx}")
```

Runtime tuning (no restart):

```bash
hotcb --dir runs/exp1 set print_metrics every=5
```

---

# 📡 How It Works

Two control layers:

1. `hotcb.yaml` — desired state (optional)
2. `hotcb.commands.jsonl` — append-only command stream

Changes are applied at safe adapter-defined boundaries:
- Lightning → end of batch
- HF → end of step / eval
- Bare torch → wherever you call `apply()`

---

# ❓ Why this exists

Hot-swappable callbacks are often used for:

- Temporary diagnostics
- Feature visualization 
- Gradient/statistics inspection
- Mid-run debugging
- Experiment instrumentation

You don’t want to:

- Modify your Trainer code
- Restart a long training job

The in-built hot-reloaders and logging resolvers allow a callback to “plug into” existing training run and logging backends automatically. 

Safely - without impacting your run even when it fails.

---

# 🛡 Safety

- No training loop mutation
- No framework internals modified
- Fail-safe: crashing callbacks can auto-disable
- “Remove” = disable (optional unload supported)

---

# 🌱 Philosophy

Training shouldn’t require restarts for diagnostics.

`hotcb` treats debugging and visualization as live instrumentation — not static configuration.

---

# 📄 License

MIT License (see LICENSE file).

