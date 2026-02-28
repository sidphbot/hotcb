# hotopt -- Live Optimizer Control

`hotopt` mutates optimizer behavior during live training: learning rate, weight decay, gradient clipping thresholds, and scheduler nudges.

## Supported Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `lr` | float | Learning rate (applied to all param groups) |
| `weight_decay` | float | Weight decay |
| `clip_norm` | float | Gradient clipping threshold (stored as `hotcb_clip_norm` on param group) |
| `scheduler_scale` | float | Multiplicative factor applied to current LR |
| `scheduler_drop` | float | One-shot multiplicative drop applied to current LR |
| `group` | int | Target a specific param group index |
| `groups` | dict | Map of group index to params, e.g. `{"0": {"lr": 1e-4}, "1": {"lr": 1e-5}}` |

## Per-Group Support

By default, `set_params` applies to all optimizer param groups. To target a specific group:

```bash
# Set LR for group 1 only
hotcb opt set_params group=1 lr=1e-5

# Set different LR per group (JSON value)
hotcb opt set_params groups='{"0":{"lr":1e-4},"1":{"lr":1e-5}}'
```

## Ops

| Op | Effect |
|----|--------|
| `enable` | Enable the handle (ops will be applied) |
| `disable` | Disable the handle (ops are skipped with `skipped_noop`) |
| `set_params` | Apply parameter changes to the optimizer |

## Env Requirements

The optimizer must be accessible via the `env` dict. The controller checks in order:

1. `env["optimizer"]` -- direct optimizer reference.
2. `env["resolve_optimizer"]` -- callable that returns the optimizer.

If neither is available, the op fails with `missing_optimizer`.

Framework adapters populate these automatically:
- **Lightning:** `trainer.optimizers[0]`
- **HuggingFace:** via `resolve_optimizer` callback passed to the adapter.

## Error Handling

On failure, the handle is auto-disabled (if `auto_disable_on_error=True`) and the error is recorded in the ledger. Subsequent `set_params` ops are skipped until the handle is re-enabled.
