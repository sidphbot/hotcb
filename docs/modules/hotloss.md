# hotloss -- Live Loss Composition Control

`hotloss` mutates loss configuration during live training: scalar weights, term toggles, and ramp settings. It does not hook into autograd -- it mutates a **loss config object** that your loss function reads.

## Supported Mutations

### Weights (`_w` suffix)

Parameters ending in `_w` map to `mutable_state["weights"]` with the suffix stripped:

```bash
hotcb loss set_params distill_w=0.2 depth_w=1.5
# -> mutable_state["weights"]["distill"] = 0.2
# -> mutable_state["weights"]["depth"] = 1.5
```

### Term Toggles (`terms.` prefix)

Parameters starting with `terms.` toggle individual loss terms:

```bash
hotcb loss set_params terms.aux_depth=false terms.aux_heatmap=true
# -> mutable_state["terms"]["aux_depth"] = False
# -> mutable_state["terms"]["aux_heatmap"] = True
```

A `terms` dict value also works: `terms='{"aux_depth":false}'`.

### Ramps (`ramps.` prefix)

Parameters starting with `ramps.` configure ramp schedules:

```bash
hotcb loss set_params ramps.depth='{"type":"linear","warmup_frac":0.2,"end":2.0}'
# -> mutable_state["ramps"]["depth"] = {"type": "linear", "warmup_frac": 0.2, "end": 2.0}
```

### Fallback

Any parameter that does not match the above patterns is placed directly in `mutable_state["weights"]`.

## Expected `mutable_state` Shape

```python
mutable_state = {
    "weights": {"distill": 0.2, "depth": 1.5},
    "terms": {"aux_depth": True, "aux_heatmap": False},
    "ramps": {"depth": {"type": "linear", "warmup_frac": 0.2, "end": 2.0}},
}
```

Your loss function reads from this dict. hotcb mutates it; you consume it.

## Env Requirements

The loss state must be accessible via `env`:

1. `env["mutable_state"]` -- direct mutable dict reference.
2. `env["resolve_mutable_state"]` -- callable that returns the dict.

If neither is available, the op fails with `missing_mutable_state`.

Framework adapters accept a `mutable_state` dict at construction time and inject it into every `env`.

## Error Handling

Same as hotopt: auto-disable on error, failure logged to ledger, training continues.
