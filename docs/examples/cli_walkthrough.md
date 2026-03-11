# CLI Walkthrough — Live Control Session

This walkthrough shows a complete hotcb session from initialization to replay.
All commands run from a **second terminal** while training is active.

---

## Setup

```bash
pip install "hotcb[yaml]"
```

Initialize a run directory before starting training:

```bash
hotcb --dir runs/exp1 init
```

This creates:

```
runs/exp1/
  hotcb.yaml            # desired-state config (callbacks, opt, loss)
  hotcb.commands.jsonl  # command queue (append-only, polled by kernel)
```

Start your training script (see the integration examples for Lightning, HF, and bare PyTorch).

---

## 1. Check status

```bash
hotcb --dir runs/exp1 status
```

Shows current freeze mode, active callbacks, and the last few applied ledger entries.

---

## 2. Load a built-in callback

```bash
hotcb --dir runs/exp1 cb load timing \
  --path hotcb.modules.cb.callbacks.timing \
  --symbol TimingCallback \
  --enabled \
  --init every=50 window=200
```

The kernel picks this up at the next safe point (end of next batch) and instantiates
`TimingCallback(id="timing", every=50, window=200)`.

Load a few more:

```bash
hotcb --dir runs/exp1 cb load sys \
  --path hotcb.modules.cb.callbacks.system_stats \
  --symbol SystemStatsCallback \
  --enabled --init every=100

hotcb --dir runs/exp1 cb load anomaly \
  --path hotcb.modules.cb.callbacks.anomaly_guard \
  --symbol AnomalyGuardCallback \
  --enabled --init raise_on_trigger=false
```

---

## 3. Enable / disable callbacks

```bash
# Disable timing to reduce log noise
hotcb --dir runs/exp1 disable timing

# Re-enable it later
hotcb --dir runs/exp1 enable timing
```

---

## 4. Reconfigure a running callback

```bash
# Halve the timing window
hotcb --dir runs/exp1 cb set timing every=25 window=100
```

Or use the shorthand `set` (auto-routes by key pattern):

```bash
hotcb --dir runs/exp1 set every=25   # won't auto-route — use cb set for callback params
```

---

## 5. Load a custom callback from a local file

```python
# /tmp/my_diag.py
class MyDiag:
    def __init__(self, id, prefix="step"):
        self.id = id
        self.prefix = prefix

    def set_params(self, **kwargs):
        if "prefix" in kwargs:
            self.prefix = kwargs["prefix"]

    def handle(self, event, env):
        if event == "train_step_end":
            print(f"[{self.id}] {self.prefix}={env.get('step')}")
```

```bash
hotcb --dir runs/exp1 cb load my_diag \
  --file /tmp/my_diag.py \
  --symbol MyDiag \
  --enabled \
  --init prefix=batch
```

hotcb captures the source file (SHA-256, stored in `hotcb.sources/`) so the exact
version is available for deterministic replay later.

---

## 6. Tune the optimizer live

```bash
# Drop learning rate
hotcb --dir runs/exp1 opt set_params lr=5e-5

# Add weight decay
hotcb --dir runs/exp1 opt set_params lr=5e-5 weight_decay=0.01

# Add gradient clipping
hotcb --dir runs/exp1 opt set_params clip_norm=1.0

# Shorthand (set auto-routes lr, weight_decay, clip_norm → opt)
hotcb --dir runs/exp1 set lr=5e-5 weight_decay=0.01
```

---

## 7. Adjust loss weights live

```bash
# Scale up the depth loss weight
hotcb --dir runs/exp1 loss set_params depth_w=2.0

# Turn off an auxiliary loss term
hotcb --dir runs/exp1 loss set_params terms.aux_heatmap=false

# Turn it back on
hotcb --dir runs/exp1 loss set_params terms.aux_heatmap=true

# Shorthand (set auto-routes *_w, *_loss keys → loss)
hotcb --dir runs/exp1 set distill_w=0.3
```

---

## 8. Unload a callback

```bash
hotcb --dir runs/exp1 cb unload my_diag
```

The callback is torn down cleanly at the next safe point.

---

## 9. Inspect the applied ledger

Every mutation is written to `hotcb.applied.jsonl` with the exact step it took effect:

```bash
tail -n 20 runs/exp1/hotcb.applied.jsonl
```

Example entries:

```jsonl
{"step": 150, "event": "train_step_end", "module": "cb", "op": "load", "id": "timing", ...}
{"step": 200, "event": "train_step_end", "module": "opt", "op": "set_params", "params": {"lr": 5e-5}}
{"step": 350, "event": "train_step_end", "module": "loss", "op": "set_params", "params": {"depth_w": 2.0}}
```

---

## 10. Export a recipe

At any point, export the run's mutations as a portable replay plan:

```bash
hotcb --dir runs/exp1 recipe export --out runs/exp1/hotcb.recipe.jsonl
```

---

## 11. Validate the recipe

```bash
hotcb --dir runs/exp1 recipe validate --recipe runs/exp1/hotcb.recipe.jsonl
```

---

## 12. Replay a previous run exactly

```bash
# Initialize a new run directory
hotcb --dir runs/exp2 init

# Freeze in replay mode with the exported recipe
hotcb --dir runs/exp2 freeze --mode replay \
  --recipe runs/exp1/hotcb.recipe.jsonl
```

Training in `runs/exp2` will replay every cb/opt/loss change at the exact
same steps — using the captured callback source files, not your current disk versions.

---

## 13. Replay with adjustments

Generate a patch template from the recipe to see what's adjustable:

```bash
hotcb --dir runs/exp1 recipe patch-template \
  --recipe runs/exp1/hotcb.recipe.jsonl \
  --output runs/exp2/hotcb.adjust.yaml
```

Edit the YAML to add your overrides, then replay:

```bash
hotcb --dir runs/exp2 freeze --mode replay_adjusted \
  --recipe runs/exp1/hotcb.recipe.jsonl \
  --adjust runs/exp2/hotcb.adjust.yaml
```

See [adjust_overlay.yaml](adjust_overlay.yaml) for the full list of patch types
(`replace_params`, `shift_step`, `drop`, `insert`, `transform_params`).

---

## 14. Lock a production run

```bash
hotcb --dir runs/prod freeze --mode prod
```

All incoming commands are silently ignored. The run produces the same output
regardless of what anyone types in the control terminal.

Unlock when done:

```bash
hotcb --dir runs/prod freeze --mode off
```

---

## Full artifact reference

| File | Written by | Purpose |
|---|---|---|
| `hotcb.yaml` | You (init) | Desired-state config, polled by kernel |
| `hotcb.commands.jsonl` | CLI commands | Command queue, append-only |
| `hotcb.applied.jsonl` | Kernel | Authoritative mutation timeline |
| `hotcb.recipe.jsonl` | `recipe export` | Portable replay plan |
| `hotcb.sources/` | Kernel (on cb load) | Captured callback source files |
| `hotcb.freeze.json` | `freeze` command | Current freeze mode and recipe path |
