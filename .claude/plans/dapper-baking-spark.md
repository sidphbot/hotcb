# Holistic Dashboard Stabilization — MutableState Redesign

## Context

The dashboard has drifted from hotcb's core principle: **the training framework is
sacred — hook onto it, never alter it.** The split into opt/loss/cb modules creates
hardcoded assumptions at every layer. External projects can't add custom controls
without fitting into one of 3 buckets. Ramps, terms, and custom signals get filtered
out at `kernel.py:263` because they don't match any module.

**The fix:** Unify all mutable params into a single `MutableState` container with
per-param `HotcbActuator` instances. Each actuator has a user-provided `apply_fn`,
a metrics link for verification, and a state machine (INIT→UNTOUCHED→UNVERIFIED→
VERIFIED→DISABLED). The user declares this on their model; the kernel discovers it
at first step; the dashboard generates controls from it.

**Key decisions:**
- Existing opt/loss/cb modules become **thin wrappers** delegating to MutableState (backward compat)
- Verification window is **configurable per-actuator** (default 5 steps)
- Built-in `apply_fn` helpers provided for common cases (optimizer lr, wd, etc.)

---

## Phase A: Foundation Bug Fixes

Fix known bugs before any architectural changes.

### A1: FileCursor deduplication
- `src/hotcb/modules/cb/util.py:12-35` — stale copy missing `last_size`/`truncated`
- Replace with re-exports from `hotcb.util`

### A2: `_applied_cache` stale after reset
- `app.py:659` — add size-0 early return + `_clear_applied_cache()` function
- `launcher.py` reset endpoint calls cache clear

### A3: `_resolve_active_run_dir` unreliable mtime
- `app.py:727` — sort by `hotcb.metrics.jsonl` mtime, not dir mtime

### A4: Forecast gated on pinned metrics
- `charts.js:1040` — auto-select first 3 metrics if none pinned

**Tests first** (`src/hotcb/tests/test_server_stability.py` — NEW):
```
test_cb_util_reexports_canonical_filecursor
test_read_new_jsonl_detects_truncation
test_applied_summary_empty_file_returns_empty
test_applied_summary_cache_invalidated
test_resolve_uses_metrics_file_mtime
test_resolve_skips_backup_dirs
test_resolve_direct_metrics_returns_root
test_forecast_endpoint_returns_data
```

**Files:** `modules/cb/util.py`, `server/app.py`, `server/launcher.py`, `static/js/charts.js`

---

## Phase B: `MutableState` + `HotcbActuator` Core

The heart of the redesign. Defines the unified control plane data model.

### `HotcbActuator` — 1 param ↔ 1 actuator

```python
# src/hotcb/mutable_state.py (NEW)

class ActuatorState(str, Enum):
    INIT = "init"               # declared but kernel hasn't seen it
    UNTOUCHED = "untouched"     # kernel discovered, populated current_value
    UNVERIFIED = "unverified"   # mutation applied, waiting for metrics verification
    VERIFIED = "verified"       # metrics confirmed change took effect
    DISABLED = "disabled"       # verification failed — grayed out in UI

@dataclass
class Mutation:
    step: int
    old_value: Any
    new_value: Any
    verified: bool = False
    verify_deadline: int = 0    # step by which verification must happen

@dataclass
class HotcbActuator:
    param_key: str              # unique name: "lr", "weight_a", "temperature"
    type: str                   # "float", "int", "bool", "choice"
    apply_fn: Callable          # user-provided: actually mutates the value
    metrics_dict_name: str      # key in env["metrics"] for verification
    current_value: Any = None   # INIT or last known
    default_value: Any = None   # initial/reset value
    last_changed: int = -1      # step of last verified change (-1 = never)
    state: ActuatorState = ActuatorState.INIT
    mutations: List[Mutation] = field(default_factory=list)
    verification_window: int = 5  # steps to wait before DISABLED
    bounds: tuple = None        # (min, max) for float/int
    choices: list = None        # for "choice" type
    scale: str = "linear"       # "linear" or "log10" for UI slider
    label: str = ""             # display label (defaults to param_key)
    group: str = ""             # visual grouping hint

    def to_spec(self) -> dict:
        """Serialize for /api/controls — dashboard reads this."""
        return {
            "param_key": self.param_key,
            "label": self.label or self.param_key,
            "type": self.type,
            "scale": self.scale,
            "bounds": list(self.bounds) if self.bounds else None,
            "choices": self.choices,
            "current_value": self.current_value,
            "default_value": self.default_value,
            "state": self.state.value,
            "group": self.group,
        }
```

### `MutableState` container

```python
class MutableState:
    actuators: Dict[str, HotcbActuator]  # keyed by param_key

    def init_datastructures(self, metrics: dict):
        """Called by kernel at first step. Verify metrics links, populate values."""
        for act in self.actuators.values():
            if act.metrics_dict_name in metrics:
                act.current_value = metrics[act.metrics_dict_name]
                act.state = ActuatorState.UNTOUCHED
            # If metrics_dict_name not found, stay INIT (warn but don't disable yet)

    def change(self, param_key: str, new_value: Any, at_step: int):
        """Apply mutation via actuator's apply_fn. Mark UNVERIFIED."""
        act = self.actuators[param_key]
        if act.state == ActuatorState.DISABLED:
            raise ValueError(f"Actuator {param_key} is disabled")
        old = act.current_value
        act.apply_fn(new_value)
        act.mutations.append(Mutation(
            step=at_step, old_value=old, new_value=new_value,
            verify_deadline=at_step + act.verification_window
        ))
        act.current_value = new_value
        act.state = ActuatorState.UNVERIFIED

    def verify_pending(self, metrics: dict, current_step: int):
        """Called each step by kernel. Check metrics for verification."""
        for act in self.actuators.values():
            if act.state != ActuatorState.UNVERIFIED:
                continue
            pending = [m for m in act.mutations if not m.verified]
            if not pending:
                continue
            latest = pending[-1]
            metric_val = metrics.get(act.metrics_dict_name)
            if metric_val is not None and metric_val != latest.old_value:
                latest.verified = True
                act.state = ActuatorState.VERIFIED
                act.last_changed = current_step
            elif current_step >= latest.verify_deadline:
                act.state = ActuatorState.DISABLED

    def snapshot(self) -> dict:
        """For rollback / recipe export."""
        return {k: act.current_value for k, act in self.actuators.items()}

    def get_specs(self) -> list:
        """For dashboard /api/controls."""
        return [act.to_spec() for act in self.actuators.values()]
```

### Built-in `apply_fn` helpers

```python
# src/hotcb/mutable_state.py — convenience functions

def apply_optimizer_lr(optimizer, param_group_idx=0):
    """Returns an apply_fn for optimizer learning rate."""
    def _apply(value):
        for pg in optimizer.param_groups:
            pg["lr"] = value
    return _apply

def apply_optimizer_wd(optimizer):
    def _apply(value):
        for pg in optimizer.param_groups:
            pg["weight_decay"] = value
    return _apply

def apply_dict_key(target_dict, key):
    """Returns an apply_fn for a dict key (e.g., loss weights)."""
    def _apply(value):
        target_dict[key] = value
    return _apply

def apply_attr(obj, attr_name):
    """Returns an apply_fn for an object attribute."""
    def _apply(value):
        setattr(obj, attr_name, value)
    return _apply
```

### User-facing factory

```python
def mutable_state(actuators: List[HotcbActuator]) -> MutableState:
    """Create a MutableState from a list of actuators."""
    ms = MutableState()
    ms.actuators = {a.param_key: a for a in actuators}
    return ms
```

### User integration example

```python
class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # ... model layers ...
        self.mutable_state = hotcb.mutable_state([
            HotcbActuator(
                param_key="lr", type="float", scale="log10",
                apply_fn=hotcb.apply_optimizer_lr(self.optimizer),
                metrics_dict_name="lr",
                bounds=(1e-7, 1.0), default_value=3e-4,
            ),
            HotcbActuator(
                param_key="temperature", type="float",
                apply_fn=hotcb.apply_attr(self, "temperature"),
                metrics_dict_name="temperature",
                bounds=(0.1, 10.0), default_value=1.0,
                verification_window=10,  # slow metric, give more time
            ),
        ])
```

**Tests first** (`src/hotcb/tests/test_mutable_state.py` — NEW):
```
test_actuator_state_machine_init_to_untouched
test_actuator_state_machine_untouched_to_unverified
test_actuator_state_machine_verified_on_metric_change
test_actuator_state_machine_disabled_on_timeout
test_mutable_state_init_from_metrics
test_mutable_state_change_calls_apply_fn
test_mutable_state_verify_pending_marks_verified
test_mutable_state_verify_pending_marks_disabled
test_mutable_state_snapshot
test_mutable_state_get_specs
test_builtin_apply_optimizer_lr
test_builtin_apply_optimizer_wd
test_builtin_apply_dict_key
test_builtin_apply_attr
test_mutable_state_disabled_rejects_change
test_actuator_custom_verification_window
test_mutable_state_multiple_mutations_same_param
```

**Files:** `src/hotcb/mutable_state.py` (NEW)

---

## Phase C: Kernel Integration + Module Wrappers

Wire `MutableState` into the kernel and make opt/loss/cb thin wrappers.

### Kernel changes (`kernel.py`)

1. **Discovery at first step:** In `apply()`, after first `_should_poll()`, check
   `env` for `mutable_state` attribute on model/trainer objects. If found, store
   reference and call `mutable_state.init_datastructures(env.get("metrics", {}))`.

2. **Default module for MutableState commands:** At `kernel.py:263`, instead of
   `unknown_module` error, route to MutableState if param_key matches a known actuator:
   ```python
   # After existing module lookup fails:
   if self._mutable_state and op.params:
       for key in op.params:
           if key in self._mutable_state.actuators:
               # Route through MutableState
               self._mutable_state.change(key, op.params[key], current_step)
               self._write_ledger(op, event, step, decision="applied", ...)
               return
   ```

3. **Verification each step:** After applying ops, call
   `self._mutable_state.verify_pending(env.get("metrics", {}), current_step)`

4. **Controls endpoint data source:** Add `kernel.get_control_specs()` that returns
   `self._mutable_state.get_specs()` for the dashboard.

### Thin wrappers for backward compat

**opt module** (`modules/opt.py`):
- `apply_op()` still works for `{"module": "opt", "op": "set_params", "params": {"lr": 0.001}}`
- Internally delegates: if MutableState has an actuator for "lr", calls
  `mutable_state.change("lr", 0.001, step)` instead of direct optimizer mutation
- If no MutableState, falls back to current direct optimizer mutation (legacy path)

**loss module** (`modules/loss.py`):
- Same pattern: delegates to MutableState for known keys, falls back to direct
  mutable_state dict mutation for legacy projects

**cb module** (`modules/cb/`):
- Unchanged — callbacks are a different beast (load/unload/enable/disable).
  But `set_params` can optionally delegate to MutableState if the callback
  registered its params there.

### Command format

Dashboard sends: `POST /api/controls/apply {"changes": {"lr": 0.001, "temperature": 0.5}}`

Server writes to `hotcb.commands.jsonl`:
```json
{"module": "mutable", "op": "set_params", "params": {"lr": 0.001, "temperature": 0.5}}
```

Kernel routes to MutableState default module handler.

**Tests first** (`src/hotcb/tests/test_kernel_mutable.py` — NEW):
```
test_kernel_discovers_mutable_state_at_first_step
test_kernel_routes_mutable_command
test_kernel_verifies_pending_each_step
test_kernel_opt_wrapper_delegates_to_mutable_state
test_kernel_loss_wrapper_delegates_to_mutable_state
test_kernel_legacy_opt_without_mutable_state
test_kernel_legacy_loss_without_mutable_state
test_kernel_unknown_param_key_fails_gracefully
test_kernel_disabled_actuator_rejects_command
test_kernel_get_control_specs
```

**Files:** `kernel.py`, `modules/opt.py`, `modules/loss.py`

---

## Phase D: DashboardConfig + `/api/config` + `/api/controls`

### DashboardConfig (`src/hotcb/server/config.py` — NEW)

```python
@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8421
    poll_interval: float = 0.5
    ws_initial_burst: int = 500

@dataclass
class ChartConfig:
    max_render_points: int = 2000
    line_tension: float = 0.15

@dataclass
class UIConfig:
    state_save_interval: int = 5000
    alert_poll_interval: int = 15000
    forecast_poll_interval: int = 10000
    forecast_step_cadence: int = 20
    staged_change_threshold: float = 0.005

@dataclass
class DashboardConfig:
    server: ServerConfig
    chart: ChartConfig
    ui: UIConfig
    run_dir: str = ""
    demo_mode: bool = False
```

### API endpoints

- `GET /api/config` — returns `DashboardConfig` as JSON (frontend stores in `S.config`)
- `GET /api/controls` — returns `mutable_state.get_specs()` (actuator list with types, bounds, state)
- `POST /api/controls/apply` — accepts `{"changes": {key: value}}`, writes to commands.jsonl

**Tests first** (`src/hotcb/tests/test_dashboard_config.py` — NEW):
```
test_dashboard_config_defaults
test_config_endpoint_returns_full
test_controls_endpoint_returns_actuator_specs
test_controls_endpoint_without_mutable_state (returns empty/defaults)
test_controls_apply_writes_command
test_controls_apply_diff_only (0.5% threshold)
```

**Files:** `server/config.py` (NEW), `server/app.py`, `server/api.py`

---

## Phase E: Dynamic Control Generation (Frontend)

Replace hardcoded knob HTML with type-based templates generated from `/api/controls`.

### Templates per actuator type

| Actuator type | UI Template |
|---------------|-------------|
| `float + log10` | Range slider with log transform + exponential display |
| `float + linear` | Range slider with direct value + decimal display |
| `int` | Range slider with integer step |
| `bool` | Toggle switch |
| `choice` | Select dropdown |

### Changes

**`index.html`:**
- Remove hardcoded knob rows (~lines 360-408)
- Keep: `<div class="card-body" id="knobPanel"></div>` + Apply/Schedule buttons
- Remove CSS classes: `single-loss-only`, `multitask-only`, `finetune-only`

**`controls.js`:**
- Add `buildControlsFromSpecs(specs)` — iterates specs, calls `buildKnobRow(spec)`
- Add `buildKnobRow(spec)` — creates slider/toggle/dropdown based on `spec.type`
- Each row shows actuator state: normal=VERIFIED, spinner=UNVERIFIED, grayed=DISABLED
- Apply handler: read all `[data-param]` inputs, diff against applied values,
  POST to `/api/controls/apply` with changed params only
- Remove `_trainConfigDefaults`, `_updateConfigControls()`, all module-specific routing
- `demo_mode === false` → hide Training config dropdown row only (not entire card)

**`init.js`:**
- Fetch `S.config = await api('GET', '/api/config')` at startup
- Fetch controls: `var specs = await api('GET', '/api/controls'); buildControlsFromSpecs(specs);`

**`state.js`:**
- Add `S.config = null`

**Actuator state in UI:**
- INIT: not shown (controls buffered until discovered)
- UNTOUCHED: normal slider, label shows "ready"
- UNVERIFIED: subtle pulse/spinner on the control row
- VERIFIED: solid, normal appearance
- DISABLED: grayed out, tooltip "Verification failed — param may not be mutable"

**Tests (manual verification checklist):**
```
hotcb demo              → lr, wd, loss_w generated dynamically, all VERIFIED
hotcb demo --golden     → lr, wd, weight_a, weight_b generated
hotcb serve --dir <ext> → controls from MutableState (or defaults)
Apply lr change         → slider shows UNVERIFIED → VERIFIED after metric confirms
Apply broken param      → shows UNVERIFIED → DISABLED after timeout
```

**Files:** `static/index.html`, `static/js/controls.js`, `static/js/init.js`,
`static/js/state.js`, `static/css/dashboard.css`

---

## Phase F: Launcher Simplification

**Principle:** Train function accepts minimal args. `max_steps`/`step_delay` are demo concerns.

Use `inspect.signature` to detect arity:
- 0 args: `fn()` — fully external
- 1 arg: `fn(stop_event)` — respects stop signal
- 2 args: `fn(run_dir, stop_event)` — needs IPC dir
- 4 args: `fn(run_dir, max_steps, step_delay, stop_event)` — demo contract

**Tests first:**
```
test_launch_zero_arg_fn
test_launch_one_arg_fn
test_launch_two_arg_fn
test_launch_four_arg_backward_compat
```

**Files:** `launch.py`, `server/launcher.py`

---

## Phase G: Metric UI Fixes

### G1: Tooltip = non-interactive color sphere
Chart tooltip shows filled circle with `pointer-events: none`. Read-only color ref.

### G2: Explicit pin button in metrics dropdown
Each dropdown item: `[dot] [name] [...] [pin-btn]`
- Dot: click toggles visibility (filled/hollow)
- Pin button: always visible, outline pushpin when unpinned, filled when pinned
- Click pin → `toggleMetricCard(name)`

### G3: Persistent end-of-run summary pane
- Add "Summary" tab to left column tabs (index.html)
- On run complete: auto-save summary to `hotcb.run.summary.json`, populate Summary tab
- Tab persists — always accessible, not a dismissible popup
- `GET /api/run/summary` endpoint

**Tests first:**
```
test_run_summary_auto_saved
test_run_summary_endpoint
```

**Files:** `charts.js`, `panels.js`, `index.html`, `dashboard.css`, `app.py`, `launcher.py`

---

## Phase H: External Golden Demo + Comprehensive Tests

### External golden demo (`src/hotcb/tests/external_golden_demo.py` — NEW)

Self-contained script that behaves exactly like an external project:
1. Isolated temp dir
2. Creates model with `mutable_state = hotcb.mutable_state([...])`
3. Registers custom actuators: `"alpha"`, `"beta"`, `"temperature"` (not standard names)
4. Uses built-in `apply_fn` helpers where applicable
5. Runs 50 steps with HotKernel
6. Returns run_dir for test verification

### Integration tests (`src/hotcb/tests/test_external_integration.py` — NEW)

```
# Discovery
test_external_controls_discovered           — /api/controls returns custom actuators
test_external_controls_have_bounds          — specs have correct bounds
test_external_controls_show_state           — UNTOUCHED initially, VERIFIED after change

# Command flow
test_external_apply_custom_control          — POST apply with alpha=0.5
test_external_apply_triggers_verification   — state goes UNVERIFIED → VERIFIED
test_external_apply_broken_param_disables   — bad metrics_dict_name → DISABLED

# Dashboard
test_external_demo_mode_false               — Training config hidden, controls visible
test_external_capabilities_detected         — capabilities endpoint works

# Randomized
test_controls_random_float_names            — random names
test_controls_random_types                  — mix of float, int, bool, choice
test_controls_random_bounds                 — random bound ranges

# Both demo variants
test_builtin_demo_controls_match            — built-in demo has expected controls
test_external_demo_controls_match           — external demo has expected controls
test_both_demos_same_kernel_behavior        — same apply/verify flow
```

---

## Phase I: Magic Number Centralization

Replace all hardcoded constants with `DashboardConfig`/`S.config` references.

**Frontend → `S.config.*`:**
- `_maxRenderPoints`, `tension`, dash patterns, poll intervals, thresholds, batch sizes

**Backend → `config.*`:**
- History limits, WS burst, poll intervals, record estimates, cadence thresholds

**Files:** All JS files, `app.py`, `tailer.py`, `ai_engine.py`

---

## Phase Dependencies

```
A ─── Foundation fixes
│
B ─── MutableState + HotcbActuator core (the heart)
│
C ─── Kernel integration + module wrappers (depends on B)
│
├── D ─── DashboardConfig + /api/config + /api/controls (depends on C)
│   │
│   └── E ─── Dynamic frontend (depends on D)
│
├── F ─── Launcher simplification (parallel with D/E)
│
├── G ─── Metric UI fixes (parallel with D/E/F)
│
└── H ─── External golden demo + integration tests (validates A-G)
    │
    I ─── Magic number centralization (last)
```

## Files Summary

| File | Phases | Action |
|------|--------|--------|
| `src/hotcb/mutable_state.py` | B | NEW — HotcbActuator, MutableState, apply_fn helpers |
| `src/hotcb/kernel.py` | C | MutableState discovery, routing, verification |
| `src/hotcb/modules/opt.py` | C | Thin wrapper delegating to MutableState |
| `src/hotcb/modules/loss.py` | C | Thin wrapper delegating to MutableState |
| `src/hotcb/modules/cb/util.py` | A | Remove dupe FileCursor |
| `src/hotcb/server/config.py` | D | NEW — DashboardConfig |
| `src/hotcb/server/app.py` | A,D,G | Fix cache/mtime, add /api/config, /api/controls, /api/run/summary |
| `src/hotcb/server/api.py` | D | Add /api/controls/apply |
| `src/hotcb/server/launcher.py` | A,F,G | Cache clear, signature detect, summary save |
| `src/hotcb/launch.py` | F | Signature detection |
| `static/index.html` | E,G | Remove hardcoded knobs, add Summary tab |
| `static/js/controls.js` | E | buildControlsFromSpecs(), type templates |
| `static/js/charts.js` | A,G | Forecast fix, pin button, tooltip sphere |
| `static/js/init.js` | D,E | Fetch config + controls |
| `static/js/state.js` | D | Add S.config |
| `static/js/panels.js` | G | Summary tab, replace overlay |
| `static/css/dashboard.css` | E,G | Control templates, pin button, summary tab |
| `tests/test_server_stability.py` | A | NEW — bug fix tests |
| `tests/test_mutable_state.py` | B | NEW — state machine + actuator tests |
| `tests/test_kernel_mutable.py` | C | NEW — kernel integration tests |
| `tests/test_dashboard_config.py` | D | NEW — config + controls endpoint tests |
| `tests/external_golden_demo.py` | H | NEW — external project simulation |
| `tests/test_external_integration.py` | H | NEW — comprehensive integration tests |

## Verification

```bash
# Every phase:
pytest src/hotcb/tests/ -x -q --no-cov

# After Phase C:
# Existing opt/loss tests still pass (backward compat wrappers)

# After Phase E (manual):
hotcb demo              → controls generated from MutableState, state indicators work
hotcb serve --dir <ext> → custom controls discovered, verification visible

# After Phase H:
pytest src/hotcb/tests/test_external_integration.py -v  # full external flow
```
