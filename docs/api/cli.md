# CLI

The `hotcb` CLI is a lightweight control-plane writer. It **does not** talk to the training process directly.
Instead, it appends commands to an **append-only JSONL file** (by default: `hotcb.commands.jsonl`)
inside your run directory. Your training process runs a `HotController` pointed at the same file,
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
```


### 2) Enable/disable callbacks live

```
hotcb --dir runs/exp1 enable timing
hotcb --dir runs/exp1 disable timing 
```

### 3) Hot-update parameters

```
hotcb --dir runs/exp1 set timing every=10 window=200
hotcb --dir runs/exp1 set guard raise_on_trigger=true
```

### 5) Hot-load a callback from a file

```
hotcb --dir runs/exp1 load my_diag \
  --file /tmp/my_diag.py \
  --symbol MyDiag \
  --enabled \
  --init every=25 prefix="[diag]"
```

### 6) Unload (free resources)

```
hotcb --dir runs/exp1 unload my_diag
```

### Operational notes

Commands are append-only: this preserves audit history and makes it safe to write from multiple terminals.

The controller reads new commands incrementally using a byte-offset cursor.

If you issue multiple commands quickly, the training run will apply them on its next poll boundary
(debounce_steps and/or poll_interval_sec).