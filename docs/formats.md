# hotcb File Formats

All hotcb state lives in a single run directory. This document specifies the schema for each file.

## Run Directory Layout

```
runs/<run_id>/
  hotcb.yaml                   # desired-state config (optional)
  hotcb.commands.jsonl         # external command stream (append-only)
  hotcb.applied.jsonl          # applied ledger (append-only, kernel-written)
  hotcb.recipe.jsonl           # exported recipe (portable replay plan)
  hotcb.freeze.json            # freeze state (written by CLI)
  hotcb.adjust.yaml            # adjustment overlay for replay_adjusted (optional)
  hotcb.sources/               # captured source files for python_file loads
  hotcb.log                    # kernel log sink (optional)
```

## `hotcb.commands.jsonl`

Append-only JSONL file written by the CLI. Each line is a JSON object.

### Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `module` | string | yes | `"cb"`, `"opt"`, `"loss"`, or `"core"` |
| `op` | string | yes | Operation name (`enable`, `disable`, `set_params`, `load`, `unload`, `freeze`) |
| `id` | string | conditional | Handle identifier (required for module ops) |
| `params` | object | no | Parameters for `set_params` |
| `target` | object | no | `{kind, path, symbol}` for `cb load` |
| `init` | object | no | Constructor kwargs for `cb load` |
| `enabled` | bool | no | Initial enabled state for `cb load` |
| `mode` | string | no | Freeze mode for `core freeze` |

### Examples

```jsonl
{"module":"cb","op":"enable","id":"timing"}
{"module":"cb","op":"load","id":"feat_viz","target":{"kind":"python_file","path":"/tmp/feat_viz.py","symbol":"FeatureVizCallback"},"init":{"every":50},"enabled":true}
{"module":"opt","op":"set_params","id":"main","params":{"lr":3e-5,"weight_decay":0.01}}
{"module":"loss","op":"set_params","id":"main","params":{"distill_w":0.2,"depth_w":1.5}}
{"module":"core","op":"freeze","mode":"prod"}
```

## `hotcb.applied.jsonl`

Append-only JSONL file written exclusively by the training process (HotKernel). Every processed op produces one entry.

### Schema

| Field | Type | Description |
|-------|------|-------------|
| `seq` | int | Monotonically increasing sequence number |
| `wall_time` | float | Epoch time (optional, kernel may omit) |
| `step` | int | Training step when op was processed |
| `epoch` | float/int/null | Training epoch if available |
| `event` | string | Event being processed (e.g. `"train_step_end"`) |
| `phase` | string/null | Training phase (`"train"`, `"val"`, etc.) |
| `module` | string | Module that handled the op |
| `op` | string | Operation name |
| `id` | string/null | Handle identifier |
| `source` | string | `"external"`, `"yaml"`, or `"replay"` |
| `decision` | string | `"applied"`, `"ignored_freeze"`, `"ignored_replay"`, `"skipped_noop"`, or `"failed"` |
| `payload` | object/null | Relevant data (params, target, capture info) |
| `error` | string/null | Error message if `decision=="failed"` |
| `notes` | string/null | Additional context |

### Example

```json
{
  "seq": 431,
  "step": 9800,
  "epoch": 3.0,
  "event": "train_step_end",
  "phase": "train",
  "module": "opt",
  "op": "set_params",
  "id": "main",
  "source": "external",
  "decision": "applied",
  "payload": {"lr": 3e-5},
  "error": null,
  "notes": null
}
```

## `hotcb.recipe.jsonl`

JSONL file containing step-indexed replay directives. Exported from the applied ledger.

### Schema

| Field | Type | Description |
|-------|------|-------------|
| `at` | object | `{step: int, event: string}` -- when to apply |
| `module` | string | Target module |
| `op` | string | Operation name |
| `id` | string/null | Handle identifier |
| `params` | object | Parameters (for `set_params`) |
| `target` | object | `{kind, path, symbol}` (for `cb load`) |
| `init` | object | Constructor kwargs (for `cb load`) |
| `enabled` | bool | Enabled state (for `cb load`) |
| `source_capture` | object | `{sha256, captured_path}` (for `python_file` loads) |

### Example

```json
{
  "at": {"step": 300, "event": "train_step_end"},
  "module": "cb",
  "op": "load",
  "id": "feat_viz",
  "target": {"kind": "python_file", "path": "/tmp/feat_viz.py", "symbol": "FeatureVizCallback"},
  "init": {"every": 50},
  "enabled": true,
  "source_capture": {"sha256": "a1b2c3...", "captured_path": "hotcb.sources/a1b2c3....py"}
}
```

## `hotcb.freeze.json`

Single JSON object written by the CLI. The kernel checks this file's mtime on each poll.

### Schema

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | string | `"off"` | `"off"`, `"prod"`, `"replay"`, or `"replay_adjusted"` |
| `recipe_path` | string/null | null | Path to recipe file (for replay modes) |
| `adjust_path` | string/null | null | Path to adjustment overlay (for `replay_adjusted`) |
| `policy` | string | `"best_effort"` | `"best_effort"` or `"strict"` |
| `step_offset` | int | 0 | Global step offset for replay |

### Example

```json
{
  "mode": "replay_adjusted",
  "recipe_path": "hotcb.recipe.jsonl",
  "adjust_path": "hotcb.adjust.yaml",
  "policy": "best_effort",
  "step_offset": 0
}
```

## `hotcb.adjust.yaml`

YAML (or JSON) file defining adjustment patches for `replay_adjusted` mode. See [replay.md](replay.md) for detailed patch semantics.

### Schema

| Field | Type | Description |
|-------|------|-------------|
| `version` | int | Schema version (currently `1`) |
| `patches` | list | List of patch rules |

Each patch rule contains:
- `match` (object) -- criteria to select recipe entries
- One or more actions: `replace_params`, `shift_step`, `drop`, `transform_params`
- Or `insert` (object) -- a new recipe entry to add (no `match` needed)

### Example

```yaml
version: 1
patches:
  - match: {module: "opt", op: "set_params", at_step: 1200}
    replace_params: {lr: 2e-5}
  - match: {module: "loss", id: "main"}
    transform_params:
      scale: {distill_w: 1.1}
  - match: {module: "cb", id: "feat_viz"}
    drop: true
  - insert:
      at: {step: 1000, event: "train_step_end"}
      module: "cb"
      op: "enable"
      id: "sys"
```

## `hotcb.yaml`

Optional desired-state config file. The kernel reconciles it on mtime changes, generating idempotent ops.

### Schema (v1)

```yaml
version: 1

core:
  freeze_mode: "off"    # off | prod | replay | replay_adjusted
  replay:
    recipe_path: "hotcb.recipe.jsonl"
    adjust_path: "hotcb.adjust.yaml"
    policy: "best_effort"
    step_offset: 0

cb:
  callbacks:
    timing:
      enabled: true
      target:
        kind: module
        path: hotcb.modules.cb.callbacks.timing
        symbol: TimingCallback
      init: {every: 50}
      params: {}

opt:
  enabled: true
  id: "main"
  params:
    lr: 3e-5
    weight_decay: 0.01
    clip_norm: 1.0

loss:
  enabled: true
  id: "main"
  params:
    distill_w: 0.2
    depth_w: 1.5
    terms:
      aux_depth: false
    ramps:
      depth:
        type: linear
        warmup_frac: 0.2
        end: 2.0
```

YAML-derived ops are recorded in the ledger with `source="yaml"`.
