# hotcb CLI Reference

The `hotcb` CLI provides a single entrypoint for controlling all hotcb modules. It writes commands to `hotcb.commands.jsonl` (or directly to state files for freeze management).

All commands accept `--dir <run_dir>` to specify the run directory (defaults to `.`).

## `hotcb status`

Show the current state of a run: freeze mode, and last applied entries per handle.

```bash
hotcb --dir runs/exp-001 status
```

Output shows freeze mode, recipe/adjust paths if set, and a summary of the latest applied ledger entry for each module:handle combination.

## `hotcb init`

Bootstrap a run directory with all required files.

```bash
hotcb --dir runs/exp-001 init
```

Creates:
- `hotcb.yaml` (with `version: 1` if not present)
- `hotcb.commands.jsonl`
- `hotcb.applied.jsonl`
- `hotcb.recipe.jsonl`
- `hotcb.freeze.json`

## Callback Commands (`hotcb cb`)

### `hotcb cb enable <id>`

Enable a callback by ID.

```bash
hotcb --dir runs/exp-001 cb enable timing
```

### `hotcb cb disable <id>`

Disable a callback by ID.

```bash
hotcb --dir runs/exp-001 cb disable feat_viz
```

### `hotcb cb load <id>`

Load a callback dynamically from a Python file or module.

```bash
# From a Python file
hotcb --dir runs/exp-001 cb load feat_viz \
    --file /tmp/feat_viz.py \
    --symbol FeatureVizCallback \
    --enabled \
    --init every=50 layer=conv3

# From an importable module
hotcb --dir runs/exp-001 cb load timing \
    --path hotcb.modules.cb.callbacks.timing \
    --symbol TimingCallback \
    --enabled
```

Options:
- `--file <path>` -- Python file path (sets `target.kind=python_file`)
- `--path <module>` -- importable module path (sets `target.kind=module`)
- `--symbol <name>` -- class name inside the file/module (required)
- `--enabled` / `--no-enabled` -- initial enabled state
- `--init k=v ...` -- constructor keyword arguments

### `hotcb cb set_params <id> [k=v ...]`

Update callback parameters at runtime.

```bash
hotcb --dir runs/exp-001 cb set_params timing every=100
```

### `hotcb cb unload <id>`

Unload a callback.

```bash
hotcb --dir runs/exp-001 cb unload feat_viz
```

## Optimizer Commands (`hotcb opt`)

### `hotcb opt enable [--id <id>]`

Enable the optimizer controller handle (default ID: `main`).

```bash
hotcb --dir runs/exp-001 opt enable
```

### `hotcb opt disable [--id <id>]`

Disable the optimizer controller handle.

```bash
hotcb --dir runs/exp-001 opt disable
```

### `hotcb opt set_params [--id <id>] k=v ...`

Update optimizer parameters.

```bash
# Set global learning rate and weight decay
hotcb --dir runs/exp-001 opt set_params lr=3e-5 weight_decay=0.01

# Set clip norm
hotcb --dir runs/exp-001 opt set_params clip_norm=1.0

# Scheduler scale (multiplicative)
hotcb --dir runs/exp-001 opt set_params scheduler_scale=0.5

# Scheduler one-shot drop
hotcb --dir runs/exp-001 opt set_params scheduler_drop=0.1
```

## Loss Commands (`hotcb loss`)

### `hotcb loss enable [--id <id>]`

Enable the loss controller handle (default ID: `main`).

```bash
hotcb --dir runs/exp-001 loss enable
```

### `hotcb loss disable [--id <id>]`

Disable the loss controller handle.

```bash
hotcb --dir runs/exp-001 loss disable
```

### `hotcb loss set_params [--id <id>] k=v ...`

Update loss parameters.

```bash
# Set loss weights (suffix _w maps to weights dict)
hotcb --dir runs/exp-001 loss set_params distill_w=0.2 depth_w=1.5

# Toggle loss terms
hotcb --dir runs/exp-001 loss set_params terms.aux_depth=false terms.aux_heatmap=true

# Set ramp config (JSON value)
hotcb --dir runs/exp-001 loss set_params \
    ramps.depth='{"type":"linear","warmup_frac":0.2,"end":2.0}'
```

## Freeze Management (`hotcb freeze`)

Write freeze state directly to `hotcb.freeze.json`. The running kernel picks it up on the next poll.

```bash
# Production lock -- block all external mutations
hotcb --dir runs/exp-001 freeze --mode prod

# Replay mode -- replay a saved recipe
hotcb --dir runs/exp-001 freeze --mode replay \
    --recipe hotcb.recipe.jsonl \
    --policy best_effort

# Replay with adjustments
hotcb --dir runs/exp-001 freeze --mode replay_adjusted \
    --recipe hotcb.recipe.jsonl \
    --adjust hotcb.adjust.yaml \
    --policy strict \
    --step-offset 100

# Unfreeze
hotcb --dir runs/exp-001 freeze --mode off
```

Options:
- `--mode {off,prod,replay,replay_adjusted}` -- freeze mode (required)
- `--recipe <path>` -- recipe file for replay modes
- `--adjust <path>` -- adjustment overlay for `replay_adjusted`
- `--policy {best_effort,strict}` -- replay policy (default: `best_effort`)
- `--step-offset <int>` -- global step offset for replay (default: `0`)

## Recipe Commands (`hotcb recipe`)

### `hotcb recipe export`

Export a recipe from the applied ledger. Includes only entries with `decision=="applied"` from modules `cb`, `opt`, `loss`.

```bash
hotcb --dir runs/exp-001 recipe export
hotcb --dir runs/exp-001 recipe export --out /tmp/my_recipe.jsonl
```

### `hotcb recipe validate`

Validate a recipe file for schema correctness.

```bash
hotcb --dir runs/exp-001 recipe validate --recipe hotcb.recipe.jsonl
```

Checks each line for valid JSON, required fields (`at`, `module`, `op`), valid module names, and `at.step` presence.

### `hotcb recipe patch-template`

Generate a YAML adjustment overlay template from a recipe file. Produces one stub patch entry per unique `(module, op, id)` combination found in the recipe.

```bash
hotcb --dir runs/exp-001 recipe patch-template \
    --recipe hotcb.recipe.jsonl \
    --output hotcb.adjust.yaml
```

Options:
- `--recipe <path>` -- recipe file to read (default: `hotcb.recipe.jsonl`)
- `--output <path>` -- output path for the generated template (default: `hotcb.adjust.yaml`)

The generated file contains commented-out patch fields for each unique operation, ready to fill in:

```yaml
# Generated from hotcb.recipe.jsonl
patches:
  - match:
      module: opt
      op: set_params
      id: main
    # replace_params: {}
    # shift_step: 0
    # drop: false
```

## Syntactic Sugar

Shortcut commands that auto-route to the right module.

### `hotcb enable <id>`

Defaults to `cb enable`. Queues a callback enable command.

```bash
hotcb --dir runs/exp-001 enable timing
```

### `hotcb disable <id>`

Defaults to `cb disable`. Queues a callback disable command.

```bash
hotcb --dir runs/exp-001 disable timing
```

### `hotcb set [--id <id>] k=v ...`

Auto-routes to `opt` or `loss` based on key patterns:

- Keys `lr`, `weight_decay`, `clip_norm`, `scheduler_scale`, `scheduler_drop`, `group`, `groups` route to **opt**
- Keys ending in `_w`, or starting with `terms.` or `ramps.` route to **loss**
- Ambiguous keys produce an error — use explicit subcommands instead

```bash
hotcb --dir runs/exp-001 set lr=5e-5           # -> opt set_params
hotcb --dir runs/exp-001 set distill_w=0.25    # -> loss set_params
```

## Key-Value Parsing

All `k=v` arguments support automatic type inference:

| Input | Parsed as |
|-------|-----------|
| `lr=3e-5` | `float` |
| `every=50` | `int` |
| `enabled=true` | `bool` |
| `ramps.depth={"type":"linear"}` | `dict` (JSON) |
| `name=foo` | `str` |
