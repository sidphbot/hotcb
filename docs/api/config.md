# Config

`hotcb` supports a **desired-state** config file (YAML) that describes which callbacks should exist,
their enabled/disabled state, how they should be loaded, and what parameters to apply.

This is complementary to the **imperative command stream** (JSONL) written by the CLI:
- YAML describes what you want the world to look like (desired-state).
- JSONL describes actions to apply (imperative).

---

## Config module reference

::: hotcb.config

---

## YAML schema (v1)

```yaml
version: 1
callbacks:
  <id>:
    enabled: true|false
    target:
      kind: module|python_file
      path: hotcb.callbacks.timing | /abs/path/to/file.py
      symbol: TimingCallback
    init:   # constructor kwargs applied only when the instance is created
      any_kw: any_value
    params: # applied repeatedly via set_params(**params)
      any_param: any_value
```

## Example config

```
version: 1
callbacks:
  timing:
    enabled: true
    target: { kind: module, path: hotcb.callbacks.timing, symbol: TimingCallback }
    init: { every: 50, window: 200 }

  guard:
    enabled: true
    target: { kind: module, path: hotcb.callbacks.anomaly_guard, symbol: AnomalyGuardCallback }
    init:
      every: 1
      paths: ["loss", "outputs.logits"]
      raise_on_trigger: false

```

## Behavior in the controller

- The controller watches config file mtime.
- On changes, it parses YAML into internal ops:
    - load
    - set_params (for params)
    - enable / disable

- Applying these ops repeatedly converges to the desired state.

## Best practices

- Use init for constructor-only values that should not change frequently (paths, output dirs).
- Use params for values you want to tweak live without re-instantiation (every, thresholds, toggles).
- For large teams/runs, treat YAML as the reproducible config and JSONL as ad-hoc experiments.