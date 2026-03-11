# Replay and Recipes

Replay is hotcb's replay mechanism for deterministically reapplying the same sequence of training mutations in future runs. It is built on three components: the applied ledger, exported recipes, and the `RecipePlayer`.

## Exporting Recipes from the Applied Ledger

A recipe is derived from `hotcb.applied.jsonl` by filtering to only `decision=="applied"` entries from modules `cb`, `opt`, and `loss`, then normalizing into step-indexed directives.

```bash
hotcb --dir runs/exp-001 recipe export
hotcb --dir runs/exp-001 recipe export --out /tmp/recipe.jsonl
```

The export logic:
1. Reads every line from `hotcb.applied.jsonl`.
2. Skips entries where `decision != "applied"`.
3. Skips entries where `module` is not `cb`, `opt`, or `loss`.
4. Writes a recipe entry with `at.step`, `at.event`, `module`, `op`, `id`, and relevant payload keys (`params`, `target`, `init`, `enabled`).
5. Preserves ordering by `seq`.

Example recipe entry:

```json
{
  "at": {"step": 300, "event": "train_step_end"},
  "module": "opt",
  "op": "set_params",
  "id": "main",
  "params": {"lr": 3e-5}
}
```

## Replay Modes

Enable replay by setting freeze mode to `replay` or `replay_adjusted`:

```bash
hotcb --dir runs/exp-001 freeze --mode replay \
    --recipe hotcb.recipe.jsonl \
    --policy best_effort
```

During replay:
- External ops targeting `cb`/`opt`/`loss` are **ignored** (logged as `ignored_replay`).
- The `RecipePlayer` injects recipe ops at matching `(step, event)` pairs.
- Injected ops are recorded in the ledger with `source="replay"`.

### Matching Semantics

A recipe entry triggers when:
- `env.step == entry.at.step` (adjusted by step offset), **AND**
- the current event matches `entry.at.event`.

If multiple entries share the same `(step, event)`, they are applied in recipe order (which is original ledger `seq` order).

## Replay Policies

### `best_effort` (default)

If the run ends before all recipe entries are reached, leftover entries are silently skipped. Use this for runs that may be shorter than the original.

### `strict`

All scheduled entries must be applied by end of run. Missed entries are flagged. Use this when exact reproduction is required.

## Step Offsets

Apply the recipe with a global step shift to align runs with different warmup phases:

```bash
hotcb --dir runs/exp-001 freeze --mode replay \
    --recipe hotcb.recipe.jsonl \
    --step-offset 100
```

With `step_offset=100`, a recipe entry at `step=300` triggers at actual training step `400`. The formula is:

```
target_step = current_env_step - step_offset
```

So if `step_offset=100` and recipe says `at.step=300`, it fires when `env.step=400`.

## Adjusted Overlays

The `replay_adjusted` mode replays a recipe with systematic modifications defined in an overlay file (`hotcb.adjust.yaml` or JSON).

```bash
hotcb --dir runs/exp-001 freeze --mode replay_adjusted \
    --recipe hotcb.recipe.jsonl \
    --adjust hotcb.adjust.yaml
```

### The 5 Patch Types

#### 1. `replace_params` -- Surgical parameter replacement

```yaml
patches:
  - match: {module: "opt", op: "set_params", at_step: 1200}
    replace_params: {lr: 2e-5}
```

Replaces specific keys in the matched entry's params. Other keys are preserved.

#### 2. `shift_step` -- Delay or advance an action

```yaml
patches:
  - match: {module: "cb", id: "feat_viz"}
    shift_step: 50
```

Shifts the matched entry's `at.step` by the given delta. Positive delays, negative advances.

#### 3. `drop` -- Remove entries

```yaml
patches:
  - match: {module: "cb", id: "feat_viz"}
    drop: true
```

Removes all matching entries from the effective recipe.

#### 4. `insert` -- Add new entries

```yaml
patches:
  - insert:
      at: {step: 1000, event: "train_step_end"}
      module: "cb"
      op: "enable"
      id: "sys"
```

Inserts a new entry into the recipe. No `match` needed.

#### 5. `transform_params` -- Bulk transforms

```yaml
patches:
  - match: {module: "loss", op: "set_params", id: "main"}
    transform_params:
      scale:
        distill_w: 1.1
      add:
        depth_w: 0.5
```

Applies arithmetic transforms to matching params. `scale` multiplies, `add` adds.

### Match Criteria

Patches match recipe entries by any combination of:

| Field | Meaning |
|-------|---------|
| `module` | Module name |
| `op` | Operation name |
| `id` | Handle ID |
| `at_step` | Exact step match |
| `at_event` | Exact event match |
| `step_min` / `step_max` | Step range |
| `has_param` | Params contains this key |
| `nth` | Only the Nth match (0-indexed) |

### Effective Recipe Snapshots

When using `replay_adjusted`, the kernel can write the post-overlay recipe to `hotcb.recipe.effective.jsonl` for inspection. This shows exactly what will be replayed after all patches are applied.

## Source Capture for `python_file` Loads

When a callback is loaded via `target.kind=python_file`, the kernel captures the source for deterministic replay:

1. Reads the file bytes from the target path.
2. Computes SHA-256 hash.
3. Copies the file to `hotcb.sources/<sha256>.py`.
4. Records `sha256` and `captured_path` in the ledger entry.

During replay, the `RecipePlayer` includes `source_capture` metadata. The callback module prefers loading from the captured path. If the captured file is missing, it falls back to the original path and records a `capture_missing_fallback` note in the ledger.

This ensures that even if the original Python file is modified between runs, replay uses the exact version that was active during the original run.

Example ledger entry with source capture:

```json
{
  "seq": 5,
  "step": 100,
  "event": "train_step_end",
  "module": "cb",
  "op": "load",
  "id": "feat_viz",
  "source": "external",
  "decision": "applied",
  "payload": {
    "source_capture": {
      "sha256": "a1b2c3d4...",
      "captured_path": "hotcb.sources/a1b2c3d4....py"
    }
  }
}
```
