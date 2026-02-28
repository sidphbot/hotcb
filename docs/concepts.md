# hotcb Concepts

## What hotcb Is

hotcb is a live control plane for **live training mutation** in PyTorch. It lets you change optimizer parameters, loss weights, and instrumentation callbacks during a running training job -- without restarting, without monkeypatching, and without losing reproducibility.

hotcb ships three live-mutation modules (`cb`, `opt`, `loss`) plus infrastructure for deterministic replay and freeze protection, all driven through a single `HotKernel` runtime.

## HotKernel

`HotKernel` is the shared runtime at the center of hotcb. It:

1. **Tails the control plane** -- reads new commands from `hotcb.commands.jsonl` using an incremental byte cursor, and optionally reconciles `hotcb.yaml` on mtime changes.
2. **Routes ops by module** -- each command includes a `module` field (`cb`, `opt`, `loss`, or `core`) and is dispatched to the appropriate controller.
3. **Enforces freeze modes** -- checks `hotcb.freeze.json` on each poll and applies/ignores ops accordingly.
4. **Injects replay ops** -- in replay modes, the `RecipePlayer` injects saved ops at matching `(step, event)` pairs.
5. **Writes the applied ledger** -- every processed op produces an entry in `hotcb.applied.jsonl` with its decision (`applied`, `ignored_freeze`, `failed`, etc.).
6. **Captures sources** -- when a callback is loaded from a Python file, the kernel copies the source to `hotcb.sources/` with a SHA-256 filename for deterministic replay.

```python
from hotcb.kernel import HotKernel

kernel = HotKernel(
    run_dir="runs/exp-001",
    debounce_steps=5,
    poll_interval_sec=1.0,
    auto_disable_on_error=True,
)
```

### Polling and Debounce

The kernel does not poll on every `apply()` call. Two gates control polling frequency:

- `debounce_steps` -- poll only every N calls to `apply()`.
- `poll_interval_sec` -- enforce a minimum wall-clock interval between polls.

This keeps overhead near zero during fast inner loops.

### Module Routing

Ops are routed by their `module` field:

| Module | Controller | Purpose |
|--------|-----------|---------|
| `cb` | `CallbackModule` | Instrumentation callbacks |
| `opt` | `HotOptController` | Optimizer/scheduler control |
| `loss` | `HotLossController` | Loss weight/term/ramp control |
| `core` | Kernel itself | Freeze, unfreeze, status |

## The Three Streams

hotcb maintains three distinct JSONL streams in the run directory.

### 1. Commands Stream (`hotcb.commands.jsonl`)

Written by the CLI or external tools. Records **intent**, not truth. The CLI does not know the current training step -- it just appends requests.

```json
{"module":"opt","op":"set_params","id":"main","params":{"lr":3e-5}}
```

### 2. Applied Ledger (`hotcb.applied.jsonl`)

Written **only** by the training process (HotKernel). Every processed op produces a ledger entry with step, event, source, decision, and payload. This is the canonical ground truth for what happened during training.

```json
{"seq":42,"step":1200,"event":"train_step_end","module":"opt","op":"set_params","id":"main","source":"external","decision":"applied","payload":{"lr":3e-5}}
```

### 3. Recipe Stream (`hotcb.recipe.jsonl`)

Exported from the applied ledger. Contains only `decision=="applied"` entries, normalized into step-indexed directives. Used for replay.

```json
{"at":{"step":1200,"event":"train_step_end"},"module":"opt","op":"set_params","id":"main","params":{"lr":3e-5}}
```

**Rule:** Replay is always based on the applied ledger (via recipe export), never on the raw command stream, because the command stream may contain ops that were ignored or failed.

## Freeze Modes

Freeze modes are kernel-level state that controls which ops are allowed.

### `off` (default)

All external and YAML ops are processed normally.

### `prod` (production lock)

External ops targeting `cb`, `opt`, or `loss` are **ignored** and logged with `decision="ignored_freeze"`. The kernel still reads commands (for audit) but does not apply them. Core ops like `unfreeze` still work.

### `replay`

External ops are ignored. The kernel replays a saved recipe, injecting ops at their recorded `(step, event)`. Ledger entries from replay have `source="replay"`.

### `replay_adjusted`

Same as `replay`, but the recipe is first transformed by an adjustment overlay (`hotcb.adjust.yaml`). This allows controlled variations: replay everything but with a different LR at step 1200, for example.

Freeze state is stored in `hotcb.freeze.json` and checked by the kernel on each poll via mtime comparison.

## Failure Isolation

hotcb never kills training on a module error. When an op fails:

1. The failure is recorded in the ledger with `decision="failed"` and the error message.
2. If `auto_disable_on_error` is enabled, the offending handle is disabled.
3. Training continues uninterrupted.

This applies uniformly to all modules. A bad callback, an invalid LR value, or a missing loss_state reference will be logged and isolated -- not propagated.

## Safe-Point Updates Only

All mutations happen at **safe points** -- stable boundaries in the training loop where the model is not mid-forward or mid-backward. Framework adapters define these:

- **Lightning:** `on_train_batch_end`, `on_validation_batch_end`
- **HuggingFace:** `on_step_end`, `on_evaluate`
- **Bare torch:** wherever you call `kernel.apply()`

The kernel never reaches into the optimizer or loss state at an unsafe moment. Your adapter controls when `kernel.apply()` is called, and the kernel applies all pending ops at that instant.
