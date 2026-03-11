# CLI

The `hotcb` CLI is a lightweight control-plane writer. It **does not** talk to the training process directly.
Instead, it appends commands to an **append-only JSONL file** (`hotcb.commands.jsonl`)
inside your run directory. Your training process runs a `HotKernel` pointed at the same directory,
and will apply new commands on its next polling tick.

---

## CLI module reference

::: hotcb.cli

---

## Typical workflow

### 1) Initialize a run directory

```bash
hotcb --dir runs/exp1 init
```

This creates:
```
runs/exp1/hotcb.yaml (desired-state config template; optional)
runs/exp1/hotcb.commands.jsonl (imperative command stream)
runs/exp1/hotcb.applied.jsonl (kernel-written ledger)
runs/exp1/hotcb.recipe.jsonl (portable replay plan)
runs/exp1/hotcb.freeze.json (freeze state)
```

### 2) Enable/disable callbacks live

```
hotcb --dir runs/exp1 enable timing
hotcb --dir runs/exp1 disable timing
```

### 3) Hot-update parameters

```
hotcb --dir runs/exp1 cb set_params timing every=10 window=200
hotcb --dir runs/exp1 opt set_params lr=1e-4 weight_decay=0.02
hotcb --dir runs/exp1 loss set_params distill_w=0.2 depth_w=1.5
```

### 4) Hot-load a callback from a file

```
hotcb --dir runs/exp1 cb load my_diag \
  --file /tmp/my_diag.py \
  --symbol MyDiag \
  --enabled \
  --init every=25
```

### 5) Unload (free resources)

```
hotcb --dir runs/exp1 cb unload my_diag
```

### Operational notes

Commands are append-only: this preserves audit history and makes it safe to write from multiple terminals.

The kernel reads new commands incrementally using a byte-offset cursor.

If you issue multiple commands quickly, the training run will apply them on its next poll boundary
(`debounce_steps` and/or `poll_interval_sec`).
