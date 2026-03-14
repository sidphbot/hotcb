# hottune -- Online Hyperparameter Adaptation

`hottune` is an **optional hotcb module** that performs online, constrained, Bayesian-guided hyperparameter adaptation during training. It observes metrics, proposes bounded mutations, applies them at safe points, evaluates over a short horizon, and accepts or rolls back.

## When to use it

hottune covers the gap between:

- Static recipes (set once, hope for the best)
- Full-run sweeps (expensive, offline)
- Manual mid-run tweaking (error-prone, unrecorded)

The core loop: **observe -> propose bounded mutation -> apply at safe point -> evaluate over horizon -> accept/reject -> persist learning**

## Installation

```bash
pip install "hotcb[tune]"   # installs optuna + pyyaml
```

hottune works without optuna (falls back to random proposals), but TPE-based search requires it.

## Architecture

hottune consists of five layers:

| Layer | Responsibility |
|---|---|
| **Metric access** | Adapter-provided `env["metric"]` for framework-agnostic metric reading |
| **Actuation** | Kernel-owned actuators that can snapshot/validate/apply/restore parameters |
| **Search** | TPE or random proposal over bounded mutation space |
| **Evaluation** | Short-horizon scoring with accept/reject/rollback |
| **Recipe evolution** | Cross-run learning persisted as evolved priors |

## Quick start

### Lightning

```python
from hotcb import HotKernel
from hotcb.adapters.lightning import HotCBLightning
from hotcb.actuators import OptimizerActuator, MutableStateActuator

kernel = HotKernel(
    run_dir="runs/exp1",
    tune_recipe_path="tune_recipe.yaml",  # optional
)

# Register actuators so tune knows what it can mutate
kernel.register_actuator("opt", OptimizerActuator())
kernel.register_actuator("loss", MutableStateActuator())

trainer = pl.Trainer(callbacks=[HotCBLightning(kernel, mutable_state=model.mutable_state)])
trainer.fit(model, datamodule=dm)
```

Then enable tuning from another terminal:

```bash
hotcb --dir runs/exp1 tune enable --mode active
```

### Bare PyTorch

```python
from hotcb import HotKernel
from hotcb.actuators import OptimizerActuator

kernel = HotKernel(run_dir="runs/exp1")
kernel.register_actuator("opt", OptimizerActuator())

for epoch in range(num_epochs):
    for step, batch in enumerate(dl):
        # ... train step ...
        kernel.apply(env={...}, events=["train_batch_end"])

    # Decision point for tune
    kernel.apply(
        env={"step": step, "epoch": epoch, "optimizer": optimizer,
             "metric": lambda name, default=None: metrics.get(name, default)},
        events=["val_epoch_end"],
    )

kernel.close(env={"step": step, "epoch": epoch})
```

## Runtime modes

| Mode | Behavior |
|---|---|
| `off` | No overhead beyond tiny module existence |
| `observe` | No mutations; collects windows, evaluates pending segments |
| `suggest` | Writes proposals to logs but does not apply |
| `active` | Proposes and applies bounded mutations |
| `replay` | Replays prior tune mutations from a mutations log |

```bash
hotcb --dir runs/exp1 tune enable --mode active
hotcb --dir runs/exp1 tune enable --mode observe
hotcb --dir runs/exp1 tune disable
```

## Actuators

Actuators are the bridge between the tuner and live training state.

### Built-in actuators

**OptimizerActuator** -- mutates optimizer param groups:

| Op | Description |
|---|---|
| `lr_mult` | Multiplicative LR change |
| `lr_set` | Absolute LR set |
| `wd_mult` | Multiplicative weight decay change |
| `wd_set` | Absolute weight decay set |
| `betas_set` | Set Adam betas |

**MutableStateActuator** -- mutates loss weights:

| Op | Description |
|---|---|
| `set` | Set weight to absolute value |
| `mult` | Multiply weight by value |
| `delta` | Add value to weight |

### Custom actuators

Implement the `BaseActuator` protocol:

```python
class MyActuator:
    name = "my_knob"

    def snapshot(self, env: dict) -> dict: ...
    def validate(self, patch: dict, env: dict) -> ValidationResult: ...
    def apply(self, patch: dict, env: dict) -> ApplyResult: ...
    def restore(self, snapshot: dict, env: dict) -> ApplyResult: ...
    def describe_space(self) -> dict: ...

kernel.register_actuator("my_knob", MyActuator())
```

## Tune recipe

The recipe defines the search space, objective, phases, acceptance criteria, and safety constraints.

```yaml
version: 1
objective:
  primary: val/loss
  mode: min
  backup_metrics:
    - grad/norm
phases:
  early: {start_frac: 0.0, end_frac: 0.2}
  mid:   {start_frac: 0.2, end_frac: 0.7}
  late:  {start_frac: 0.7, end_frac: 1.0}
actuators:
  opt:
    enabled: true
    mutations:
      lr_mult:
        bounds: [0.7, 1.2]
        prior_center: 0.95
        cooldown: 2
        risk: low
  loss:
    enabled: true
    keys:
      sp_mse_w:
        mode: mult
        bounds: [0.5, 2.0]
        cooldown: 1
search:
  algorithm: tpe
  startup_trials: 8
acceptance:
  epsilon: 0.001
  horizon: next_val_epoch_end
  rollback_on_reject: true
safety:
  block_on_nan: true
  block_on_anomaly: true
  max_global_reject_streak: 4
```

Place at `hotcb.tune.recipe.yaml` in the run directory, or pass via `tune_recipe_path` to the kernel.

## Safety model

Every mutation must pass:

- **Bounds check**: value within recipe-declared bounds
- **Cooldown**: per-mutation-family cooldown prevents thrashing
- **Risk class**: v1 supports `low` and `medium` only
- **Stability blockers**: NaN/inf loss, anomaly flags block all mutations
- **Reject streak limit**: too many consecutive rejections pauses tuning

Kernel freeze/replay semantics apply to tune the same way as cb/opt/loss -- `prod` mode blocks all tune commands.

## Evaluation and acceptance

At each `val_epoch_end`:

1. If there's an active segment, evaluate it (read post-metrics, score, accept/reject)
2. If accepted: reset reject streak, keep mutation applied
3. If rejected: rollback to snapshot, increment reject streak, enter cooldown
4. Propose next mutation if feasible

Scoring: `score_delta = primary_metric_gain - instability_penalty`

Accept if `score_delta > epsilon` and no stability blockers.

## Run artifacts

| File | Purpose |
|---|---|
| `hotcb.tune.recipe.yaml` | Tune recipe (search space, objective, constraints) |
| `hotcb.tune.mutations.jsonl` | Log of all proposed/applied/rejected mutations |
| `hotcb.tune.segments.jsonl` | Evaluation segments with pre/post metrics and decisions |
| `hotcb.tune.summary.json` | Run summary with accept rate, win rates per family |
| `hotcb.tune.study.sqlite` | Optional Optuna study state for TPE |

## CLI commands

```bash
hotcb --dir runs/exp1 tune enable                     # Enable active tuning
hotcb --dir runs/exp1 tune enable --mode observe      # Observe-only mode
hotcb --dir runs/exp1 tune disable                    # Disable tuning
hotcb --dir runs/exp1 tune status                     # Show tune status/summary
hotcb --dir runs/exp1 tune set acceptance.epsilon=0.002
hotcb --dir runs/exp1 tune export-recipe --out recipe_evolved.json
```

## YAML config

Add a `tune` section to `hotcb.yaml`:

```yaml
tune:
  enabled: true
  mode: active   # active, observe, suggest
```

## Recipe evolution

After multiple runs, evolve the recipe to incorporate learned priors:

```python
from hotcb.modules.tune.recipe import evolve_recipe
from hotcb.modules.tune.schemas import TuneRecipe

base = TuneRecipe.from_dict(load_yaml("tune_recipe.yaml"))
summaries = [load_json(f"run{i}/hotcb.tune.summary.json") for i in range(5)]
evolved = evolve_recipe(base, summaries, alpha=0.3)
save_yaml("tune_recipe_evolved.yaml", evolved.to_dict())
```

## Adapter contract

Every supported adapter must expose:

- `env["metric"]`: `(name: str, default=None) -> Any` -- framework-agnostic metric accessor
- `env["kernel"]`: reference to the HotKernel instance
- `env["max_steps"]`: total training steps for phase binning

The Lightning and HF adapters provide all three automatically.

### Standard metric names

Adapters normalize toward these where practical:

- `train/loss`, `val/loss`, `val/score`
- `lr`, `grad/norm`
- `time/step_sec`, `system/gpu_mem_mb`

## Error handling

- If tune deps missing: module self-disables, logs install hint
- If no metric accessor: observe-only with warning
- If no actuators registered: no mutations proposed
- If mutation apply fails: logged, training continues
- If rollback fails: logged, training continues
- Respects `auto_disable_on_error` kernel setting
