# Unified Actuator Model + Config Refactor

## Context

Two planned efforts — the **MutableState/HotcbActuator redesign** and the
**principled config refactor** (`dev-notes/principled_config_refactor.md`) — share a
critical overlap at the controls layer. If done separately, config refactor Phase 4
(dynamic controls from `describe_space()`) would be immediately rewritten when the
actuator redesign changes the data source. Merging them eliminates throwaway work and
produces a cleaner result.

**No backward compatibility needed** — opt/loss module wire format is not in production.

---

## What We're Fixing

### Problem 1: Three-module duplication

`kernel.py:66-75` hard-codes `{cb, opt, loss, tune}`. `register_actuator()` (lines 89-102)
has `if name == "opt"` / `elif name == "loss"` wiring. `_apply_single()` at line 263
rejects anything that isn't one of these four with `unknown_module` error.

`HotOptController` and `HotLossController` duplicate the same pattern:
resolve target → optional actuator validation → apply params → handle errors →
enable/disable state. Each has its own `_resolve_*()`, `set_actuator()`,
`_actuator_patches()` translation layer.

### Problem 2: Filtering / masquerading

`MutableStateActuator.snapshot()` only backs up `weights` — terms and ramps have no
rollback. `HotLossController._actuator_weight_patches()` explicitly skips `terms.*`
and `ramps.*`, so they bypass actuator validation entirely. `describe_space()` doesn't
include them, so tune can't optimize them.

Any external control that isn't a float weight gets shoved into an unvalidated bucket.
The user's additional ramps/terms are "masqueraded as loss state and getting filtered
out at some connections."

### Problem 3: Dashboard hardcoded controls

The UI has hardcoded slider names (`knobLr`, `knobWd`, `knobLossW`, etc.) and CSS
visibility classes (`single-loss-only`, `multitask-only`, `finetune-only`). Meanwhile
the core already has `TrainingCapabilities` and `describe_space()` — the dashboard
ignores both and hardcodes its own schema.

### Problem 4: Magic numbers everywhere

Poll intervals, history limits, thresholds, pixel sizes, batch sizes — bare literals
scattered across JS and Python files. No central source of truth.

---

## Design

### A. `HotcbActuator` — The Unified Parameter Handle

Every controllable scalar/toggle/choice becomes one `HotcbActuator`. Replaces the
current split between `OptimizerActuator` (all optimizer params in one),
`MutableStateActuator` (all loss weights in one), and the module controllers
(`HotOptController`, `HotLossController`).

```python
# src/hotcb/actuators/actuator.py

class ActuatorType(Enum):
    BOOL = "bool"
    FLOAT = "float"
    INT = "int"
    CHOICE = "choice"       # discrete set of allowed values
    LOG_FLOAT = "log_float" # float on log scale (lr, wd)
    TUPLE = "tuple"         # e.g. betas

class ActuatorState(Enum):
    INIT = "init"               # registered but not yet observed
    UNTOUCHED = "untouched"     # observed initial value, no mutations applied
    UNVERIFIED = "unverified"   # mutation applied, not yet confirmed via metrics
    VERIFIED = "verified"       # mutation confirmed via metrics_dict_name
    DISABLED = "disabled"       # user-disabled or auto-disabled on error

@dataclass
class Mutation:
    step: int
    old_value: Any
    new_value: Any
    verified: bool = False

@dataclass
class HotcbActuator:
    """Single controllable parameter."""
    param_key: str                          # unique key, e.g. "lr", "recon_w", "use_augment"
    type: ActuatorType                      # drives UI control type
    apply_fn: Callable[[Any, dict], ApplyResult]  # (value, env) -> result
    metrics_dict_name: str = ""             # metric name for verification (empty = no verification)
    label: str = ""                         # display label, defaults to param_key
    group: str = ""                         # UI grouping hint ("optimizer", "loss", "custom")

    # Bounds (for FLOAT, LOG_FLOAT, INT)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step_size: Optional[float] = None
    log_base: float = 10.0                  # for LOG_FLOAT

    # For CHOICE type
    choices: Optional[list] = None

    # Mutable runtime state
    current_value: Any = field(default=_INIT_SENTINEL)
    state: ActuatorState = field(default=ActuatorState.INIT)
    last_changed_step: int = -1
    mutations: list[Mutation] = field(default_factory=list)

    def validate(self, value: Any) -> ValidationResult:
        """Type-check and bounds-check a proposed value."""
        ...

    def snapshot(self) -> dict:
        """Return state for rollback."""
        return {"value": self.current_value, "state": self.state}

    def restore(self, snapshot: dict, env: dict) -> ApplyResult:
        """Rollback to snapshot."""
        return self.apply_fn(snapshot["value"], env)

    def describe_space(self) -> dict:
        """Return schema for tune search + UI generation."""
        return {
            "param_key": self.param_key,
            "type": self.type.value,
            "label": self.label or self.param_key,
            "group": self.group,
            "min": self.min_value,
            "max": self.max_value,
            "step": self.step_size,
            "log_base": self.log_base,
            "choices": self.choices,
            "current": self.current_value,
            "state": self.state.value,
        }
```

### B. `MutableState` — The Container

```python
# src/hotcb/actuators/state.py

class MutableState:
    """Container of HotcbActuator instances. This is the user-facing API."""

    def __init__(self, actuators: list[HotcbActuator]):
        self._actuators: dict[str, HotcbActuator] = {a.param_key: a for a in actuators}

    def get(self, key: str) -> Optional[HotcbActuator]:
        return self._actuators.get(key)

    def keys(self) -> list[str]:
        return list(self._actuators.keys())

    def apply(self, key: str, value: Any, env: dict, step: int) -> ApplyResult:
        """Validate, apply, record mutation, transition state."""
        act = self._actuators.get(key)
        if act is None:
            return ApplyResult(success=False, error=f"unknown_param:{key}")
        vr = act.validate(value)
        if not vr.valid:
            return ApplyResult(success=False, error="; ".join(vr.errors))
        old = act.current_value
        result = act.apply_fn(value, env)
        if result.success:
            act.current_value = value
            act.mutations.append(Mutation(step=step, old_value=old, new_value=value))
            act.last_changed_step = step
            act.state = ActuatorState.UNVERIFIED
        return result

    def initialize(self, env: dict) -> None:
        """Read current values from live objects at first step.
        Transitions all actuators INIT → UNTOUCHED."""
        ...

    def verify(self, key: str, metrics: dict) -> bool:
        """Check metrics_dict_name in latest metrics. UNVERIFIED → VERIFIED if match."""
        ...

    def snapshot_all(self) -> dict:
        """Snapshot all actuators for rollback."""
        ...

    def describe_all(self) -> list[dict]:
        """Return describe_space() for all actuators. Used by config endpoint + tune."""
        ...
```

### C. Convenience Constructors

```python
# src/hotcb/actuators/__init__.py

def optimizer_actuators(optimizer, lr_bounds=(1e-7, 1.0), wd_bounds=(0, 1.0)) -> list[HotcbActuator]:
    """Auto-create actuators for lr, weight_decay, betas, eps from a torch optimizer."""
    ...

def loss_actuators(loss_weights: dict, global_bounds=(0.0, 100.0)) -> list[HotcbActuator]:
    """Auto-create actuators from a dict of loss weight names → values."""
    ...

def mutable_state(actuators: list[HotcbActuator]) -> MutableState:
    """User-facing constructor."""
    return MutableState(actuators)
```

### D. Kernel Default Stream

`kernel.py:263-266` becomes:

```python
def _apply_single(self, op, env, event, step):
    # ... freeze enforcement ...

    if op.module == "core":
        ...  # freeze/recipe ops
        return

    if op.module == "cb":
        ...  # cb stays special — code lifecycle
        return

    if op.module == "tune":
        ...  # tune stays special — search orchestrator
        return

    # DEFAULT STREAM: opt, loss, or any custom param_key
    # Route through MutableState
    if self._mutable_state is not None:
        key = self._resolve_param_key(op)  # from op.params.key, op.target, or op.id
        result = self._mutable_state.apply(key, op.params.get("value"), env, step)
        self._write_ledger(op, event, step,
                          decision=result.decision, error=result.error,
                          payload=op.to_dict(), env=env)
    else:
        self._write_ledger(op, event, step,
                          decision="failed", error="no_mutable_state",
                          payload=op.to_dict(), env=env)
```

The `op.module` field is preserved in commands/ledger for grouping and UI display,
but it no longer determines which code path executes. Everything goes through the
same `MutableState.apply()` → `HotcbActuator.apply_fn()` pipeline.

### E. `DashboardConfig` — Centralized Configuration

```python
# src/hotcb/server/config.py

@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8421
    poll_interval: float = 0.5
    history_limit_metrics: int = 500
    history_limit_applied: int = 200
    ws_initial_burst: int = 200
    ws_max_retries: int = 20
    ws_retry_base: float = 3.0
    ws_retry_cap: float = 30.0

@dataclass(frozen=True)
class ChartConfig:
    max_render_points: int = 2000
    line_tension: float = 0.15
    forecast_dash: tuple = (6, 3)
    mutation_dash: tuple = (3, 4)
    annotation_stagger_rows: int = 10
    annotation_min_distance: int = 70

@dataclass(frozen=True)
class AutopilotConfig:
    divergence_threshold: float = 2.0
    ratio_threshold: float = 0.5
    ai_min_interval: int = 10
    ai_max_wait: int = 200
    ai_default_cadence: int = 50

@dataclass(frozen=True)
class UIConfig:
    state_save_interval: int = 5000
    alert_poll_interval: int = 15000
    manifold_refresh_interval: int = 10000
    recipe_refresh_interval: int = 5000
    forecast_poll_interval: int = 5000
    forecast_step_cadence: int = 10
    forecast_batch_size: int = 8
    staged_change_threshold: float = 0.005
    health_ema_alpha: float = 0.1

@dataclass
class DashboardConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    chart: ChartConfig = field(default_factory=ChartConfig)
    autopilot: AutopilotConfig = field(default_factory=AutopilotConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    run_dir: str = ""           # IMMUTABLE after startup
    controls: list[dict] = field(default_factory=list)  # populated from MutableState.describe_all()

    @classmethod
    def load(cls, run_dir: str, yaml_path: Optional[str] = None, **cli_overrides) -> "DashboardConfig":
        """Resolve: defaults → YAML → CLI → env vars → actuator discovery."""
        ...

    def to_dict(self) -> dict:
        """Serialize for /api/config endpoint."""
        ...
```

The `controls` field is populated at startup from `MutableState.describe_all()` when
capabilities are available, or left empty for observe-only dashboards.

### F. Immutable `run_dir`

`run_dir` is set once at `create_app()` and never changes. Remove:
- `app.state.run_dir` mutation in launcher
- `_rewire_dir()` helper
- `tailer.rewire()` method

Launcher writes to `config.run_dir` directly. For multi-run, launcher returns the new
path and the user restarts `hotcb serve` pointing at it. Compare tab reads sibling
dirs read-only via `/api/runs/discover`.

### G. Dynamic Frontend Controls

```javascript
// controls.js — replaces hardcoded HTML
function buildControls(controlSpecs) {
  var panel = $('#knobPanel');
  panel.innerHTML = '';
  controlSpecs.forEach(function(spec) {
    // spec = {param_key, type, label, group, min, max, step, current, state}
    panel.appendChild(buildKnobRow(spec));
  });
}

function buildKnobRow(spec) {
  // Generate slider/toggle/dropdown based on spec.type
  // "log_float" → log-scale slider
  // "float" → linear slider
  // "bool" → toggle switch
  // "choice" → dropdown
  // "int" → integer stepper
}
```

No more `single-loss-only`, `multitask-only`, `finetune-only` CSS classes.

---

## What Gets Deleted

| File/Code | Reason |
|-----------|--------|
| `src/hotcb/modules/opt.py` | Absorbed into `HotcbActuator` + `optimizer_actuators()` |
| `src/hotcb/modules/loss.py` | Absorbed into `HotcbActuator` + `loss_actuators()` |
| `src/hotcb/actuators/optimizer.py` | Replaced by per-param `HotcbActuator` instances |
| `src/hotcb/actuators/mutable_state.py` | Replaced by per-param `HotcbActuator` instances |
| `kernel.py` hard-coded `opt`/`loss` module init | Default stream handles all |
| `kernel.py` `register_actuator()` `if name == "opt"` wiring | No more module↔actuator coupling |
| `capabilities.py` `validate_mutable_state()` | `MutableState` replaces the raw dict |
| `tailer.py` `rewire()` method | `run_dir` immutable |
| `app.py` `_rewire_dir()` helper | `run_dir` immutable |
| `app.py` `app.state.run_dir` mutation | `run_dir` immutable |
| `index.html` hardcoded knob rows | Dynamic generation from config |
| `controls.js` hardcoded slider names | Dynamic generation from config |
| CSS `single-loss-only`, `multitask-only`, `finetune-only` | Gone entirely |

---

## What Stays

| Component | Why |
|-----------|-----|
| `modules/cb/` | Code lifecycle management is fundamentally different from scalar params |
| `modules/tune/` | Search orchestrator — consumes actuators, doesn't become one |
| `actuators/base.py` `BaseActuator` Protocol | `HotcbActuator` implements a superset of it |
| `HotOp` and `command_to_hotop()` | Command format is orthogonal; `op.module` becomes metadata |
| `TrainingCapabilities` | Still useful for framework-level info (num_optimizers, has_scheduler, etc.) |
| JSONL filesystem IPC | Untouched |
| Adapter layer | Still populates capabilities; now also creates `MutableState` |
| `modules/result.py` `ModuleResult` | Still used by cb and tune modules |

---

## Implementation Phases

### Phase 1: `DashboardConfig` Foundation
**Goal:** Config dataclass exists, loads from YAML/env, serves at `/api/config`.
Frontend fetches once at startup.

**New files:**
- `src/hotcb/server/config.py`
- `src/hotcb/tests/test_dashboard_config.py`

**Modified files:**
- `src/hotcb/server/app.py` — add `/api/config` endpoint, construct config at startup
- `src/hotcb/server/static/js/state.js` — add `S.config`
- `src/hotcb/server/static/js/init.js` — fetch config before other init

**Tests (write first):**
```
test_config_defaults()
    ServerConfig(), ChartConfig(), etc. have documented defaults.
    DashboardConfig() is valid with all defaults.

test_config_from_yaml()
    Write a YAML file with server.port=9000, chart.line_tension=0.3.
    Load → verify overrides applied, other defaults preserved.

test_config_from_yaml_missing_file()
    Load with nonexistent YAML → all defaults, no error.

test_config_from_env()
    Set HOTCB_PORT=9000, HOTCB_POLL_INTERVAL=1.0 in env.
    Load → verify env overrides applied.

test_config_env_overrides_yaml()
    YAML sets port=9000, env sets HOTCB_PORT=8000.
    Load → port is 8000 (env wins).

test_config_cli_overrides_all()
    YAML + env + cli_overrides={port: 7000}.
    Load → port is 7000 (CLI wins).

test_config_to_dict_roundtrip()
    config.to_dict() → JSON serializable.
    All nested sub-configs appear.

test_config_run_dir_in_dict()
    config = DashboardConfig(run_dir="/tmp/x")
    d = config.to_dict()
    assert d["run_dir"] == "/tmp/x"

test_config_endpoint_returns_full(client)
    GET /api/config → 200, body has server, chart, autopilot, ui, run_dir keys.

test_config_endpoint_reflects_overrides(client)
    App created with yaml overriding port.
    GET /api/config → server.port matches override.
```

**Phase 1 does NOT touch:** controls, actuators, kernel, modules.

---

### Phase 2: `HotcbActuator` + `MutableState`
**Goal:** New actuator types exist, state machine works, convenience constructors
produce correct actuators from optimizer/loss dicts.

**New files:**
- `src/hotcb/actuators/actuator.py` — `HotcbActuator`, `ActuatorType`, `ActuatorState`, `Mutation`
- `src/hotcb/actuators/state.py` — `MutableState` container
- `src/hotcb/tests/test_actuator_unified.py`

**Modified files:**
- `src/hotcb/actuators/__init__.py` — export `mutable_state()`, `optimizer_actuators()`, `loss_actuators()`

**Tests (write first):**
```
--- ActuatorType & validation ---

test_float_actuator_validate_in_bounds()
    HotcbActuator(type=FLOAT, min=0, max=1). validate(0.5) → valid.

test_float_actuator_validate_out_of_bounds()
    validate(1.5) → invalid, error mentions bounds.

test_log_float_actuator_validate()
    HotcbActuator(type=LOG_FLOAT, min=1e-7, max=1.0). validate(1e-4) → valid.
    validate(-1.0) → invalid.

test_bool_actuator_validate()
    HotcbActuator(type=BOOL). validate(True) → valid. validate("yes") → invalid.

test_int_actuator_validate()
    HotcbActuator(type=INT, min=0, max=100). validate(50) → valid.
    validate(50.5) → invalid (not int).

test_choice_actuator_validate()
    HotcbActuator(type=CHOICE, choices=["adam", "sgd", "adamw"]).
    validate("adam") → valid. validate("rmsprop") → invalid.

test_tuple_actuator_validate()
    HotcbActuator(type=TUPLE). validate((0.9, 0.999)) → valid.
    validate("not a tuple") → invalid.

--- State machine ---

test_initial_state_is_init()
    Fresh actuator → state == INIT.

test_initialize_transitions_to_untouched()
    MutableState with lr actuator.
    ms.initialize(env) → lr.state == UNTOUCHED, lr.current_value == actual lr.

test_apply_transitions_to_unverified()
    After initialize, ms.apply("lr", 1e-3, env, step=10).
    lr.state == UNVERIFIED.

test_verify_transitions_to_verified()
    After apply, ms.verify("lr", {"lr": 1e-3}).
    lr.state == VERIFIED.

test_apply_after_verified_goes_back_to_unverified()
    VERIFIED → apply new value → UNVERIFIED again.

test_disabled_actuator_rejects_apply()
    act.state = DISABLED. ms.apply("lr", ...) → fails with "actuator_disabled".

test_disable_actuator()
    ms.disable("lr") → lr.state == DISABLED.

--- Mutation tracking ---

test_mutation_recorded_on_apply()
    ms.apply("lr", 1e-3, env, step=10) → lr.mutations has 1 entry.
    mutation.step == 10, old_value == original, new_value == 1e-3.

test_multiple_mutations_accumulated()
    3 applies → 3 mutations in list.

test_last_changed_step_updated()
    ms.apply("lr", ..., step=50) → lr.last_changed_step == 50.

--- apply_fn ---

test_apply_fn_receives_value_and_env()
    Mock apply_fn. ms.apply("lr", 1e-3, env, step=1).
    apply_fn called with (1e-3, env).

test_apply_fn_failure_does_not_mutate_state()
    apply_fn returns ApplyResult(success=False).
    current_value unchanged, no mutation recorded, state unchanged.

test_apply_fn_exception_caught()
    apply_fn raises RuntimeError.
    ms.apply() returns ApplyResult(success=False, error=...).
    State not corrupted.

--- Snapshot / restore ---

test_snapshot_all()
    MutableState with lr + wd.
    snapshot = ms.snapshot_all()
    Has entries for both keys with value + state.

test_restore_from_snapshot()
    Apply mutations, snapshot, apply more, restore.
    Values back to snapshot state.

--- describe_space ---

test_describe_space_includes_all_fields()
    act.describe_space() → dict with param_key, type, label, group,
    min, max, step, choices, current, state.

test_describe_all()
    MutableState with 3 actuators.
    ms.describe_all() → list of 3 dicts.

--- Convenience constructors ---

test_optimizer_actuators_from_torch_optimizer()
    opt = MockOptimizer(lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999)).
    acts = optimizer_actuators(opt)
    → 3 actuators: lr (LOG_FLOAT), weight_decay (LOG_FLOAT), betas (TUPLE).
    Each has correct current_value from optimizer.

test_optimizer_actuators_bounds()
    acts = optimizer_actuators(opt, lr_bounds=(1e-6, 0.1))
    → lr actuator has min=1e-6, max=0.1.

test_optimizer_actuators_apply_fn_sets_param_groups()
    acts = optimizer_actuators(opt)
    lr_act = [a for a in acts if a.param_key == "lr"][0]
    lr_act.apply_fn(5e-4, {"optimizer": opt})
    → all param groups now have lr=5e-4.

test_optimizer_actuators_apply_fn_coordinates_scheduler()
    Optimizer + scheduler with base_lrs.
    lr_act.apply_fn(5e-4, {"optimizer": opt, "scheduler": sched})
    → scheduler.base_lrs updated too.

test_loss_actuators_from_dict()
    weights = {"recon": 1.0, "kl": 0.5, "perceptual": 0.3}
    acts = loss_actuators(weights)
    → 3 actuators, all FLOAT, correct current_values.

test_loss_actuators_apply_fn_mutates_dict()
    acts = loss_actuators(weights)
    acts[0].apply_fn(2.0, {}) → weights["recon"] == 2.0.

test_loss_actuators_bounds()
    acts = loss_actuators(weights, global_bounds=(0, 10))
    → all have min=0, max=10.

test_mutable_state_constructor()
    ms = hotcb.mutable_state([a1, a2, a3])
    → isinstance(ms, MutableState), ms.keys() == ["lr", "wd", "recon_w"].
```

**Phase 2 does NOT touch:** kernel, modules, server, frontend.

---

### Phase 3: Kernel Unification (Default Stream)
**Goal:** Kernel routes opt/loss/custom ops through `MutableState`. Delete
`HotOptController`, `HotLossController`, `OptimizerActuator`, `MutableStateActuator`.

**Deleted files:**
- `src/hotcb/modules/opt.py`
- `src/hotcb/modules/loss.py`
- `src/hotcb/actuators/optimizer.py`
- `src/hotcb/actuators/mutable_state.py`

**Modified files:**
- `src/hotcb/kernel.py` — accept `MutableState`, remove opt/loss from `self.modules`,
  default stream routing
- `src/hotcb/actuators/__init__.py` — remove old exports
- `src/hotcb/modules/__init__.py` — remove opt/loss re-exports if any
- `src/hotcb/ops.py` — `command_to_hotop()` default_module stays "cb" (unchanged)

**Tests (write first):**

```
--- Kernel accepts MutableState ---

test_kernel_init_with_mutable_state()
    k = HotKernel(run_dir=..., mutable_state=ms)
    k._mutable_state is ms.

test_kernel_init_without_mutable_state()
    k = HotKernel(run_dir=...)
    k._mutable_state is None. No error.

--- Default stream routing ---

test_opt_set_params_routes_to_mutable_state()
    op = HotOp(module="opt", op="set_params", params={"key": "lr", "value": 1e-3})
    Kernel with MutableState containing lr actuator.
    kernel._apply_single(op, env, "train_step", 10)
    → Ledger record has decision="applied".
    → MutableState lr actuator now has current_value=1e-3.

test_loss_set_params_routes_to_mutable_state()
    op = HotOp(module="loss", op="set_params", params={"key": "recon_w", "value": 2.0})
    → Same routing, applied via MutableState.

test_custom_module_routes_to_mutable_state()
    op = HotOp(module="custom", op="set_params", params={"key": "dropout", "value": 0.3})
    MutableState has "dropout" actuator.
    → Routes through default stream, applied successfully.

test_unknown_param_key_fails_gracefully()
    op = HotOp(module="opt", op="set_params", params={"key": "nonexistent", "value": 1.0})
    → Ledger: decision="failed", error="unknown_param:nonexistent".

test_no_mutable_state_fails_gracefully()
    Kernel with mutable_state=None.
    op with module="opt" → decision="failed", error="no_mutable_state".

test_cb_still_routes_to_cb_module()
    op = HotOp(module="cb", op="enable", ...)
    → Routes to CallbackModule, not MutableState.

test_tune_still_routes_to_tune_module()
    op = HotOp(module="tune", op="observe", ...)
    → Routes to HotTuneController.

test_core_still_routes_to_core()
    op = HotOp(module="core", op="freeze", ...)
    → Routes to _apply_core_op.

--- Freeze enforcement still works ---

test_freeze_blocks_default_stream()
    Kernel in freeze mode="prod".
    op with module="opt" → decision="ignored_freeze".

test_freeze_blocks_custom_module()
    op with module="custom" → decision="ignored_freeze".

--- Param key resolution ---

test_param_key_from_params_key()
    op.params = {"key": "lr", "value": 1e-3}
    _resolve_param_key(op) → "lr".

test_param_key_from_legacy_opt_format()
    op.module = "opt", op.params = {"lr": 1e-3}  (no "key" field)
    _resolve_param_key(op) → extracts "lr" from params dict.

test_param_key_from_legacy_loss_format()
    op.module = "loss", op.params = {"recon_w": 2.0}  (suffix convention)
    _resolve_param_key(op) → "recon".

--- register_actuator compatibility with tune ---

test_register_mutable_state_propagates_to_tune()
    k = HotKernel(..., mutable_state=ms)
    tune module can see all actuators via ms.describe_all().

--- Enable/disable via default stream ---

test_enable_disable_actuator_via_op()
    op = HotOp(module="opt", op="disable", params={"key": "lr"})
    → lr actuator state becomes DISABLED.
    op = HotOp(module="opt", op="enable", params={"key": "lr"})
    → lr actuator state becomes UNTOUCHED (or last known good state).

--- Ledger format ---

test_ledger_preserves_module_field()
    op with module="opt" applied via default stream.
    Ledger record has module="opt" (not "default" or "mutable_state").

test_ledger_records_mutation_detail()
    Applied lr change.
    Ledger payload includes old_value, new_value, param_key.
```

**Migration of existing tests:**
- `test_hotopt.py` (12 tests): Rewrite to test through kernel default stream
  instead of `HotOptController` directly. Same behaviors, different entry point.
- `test_hotloss.py` (14 tests): Same — rewrite to test through kernel.
- `test_kernel_core.py` `test_route_to_opt`, `test_route_to_loss`: Update to verify
  default stream routing instead of module lookup.
- `test_hottune.py` actuator tests: Update `OptimizerActuator` → `optimizer_actuators()`,
  `MutableStateActuator` → `loss_actuators()`. The tune controller's
  `register_actuator()` interface may need updating to work with `MutableState`.
- `test_server_api.py` opt/loss endpoint tests: Still work — API writes same JSONL
  commands, kernel routes differently.
- `test_new_features.py` opt/loss traceback tests: Rewrite for new error path.

---

### Phase 4: Dynamic Controls from Actuators
**Goal:** `/api/config` `controls` field populated from `MutableState.describe_all()`.
Frontend generates controls dynamically. Remove all hardcoded slider HTML.

**Modified files:**
- `src/hotcb/server/config.py` — `controls` populated from actuator metadata
- `src/hotcb/server/app.py` — wire `MutableState.describe_all()` into config
- `src/hotcb/server/static/index.html` — remove hardcoded knob rows, keep `<div id="knobPanel">`
- `src/hotcb/server/static/js/controls.js` — `buildControls(specs)`, `buildKnobRow(spec)`
- `src/hotcb/server/static/css/dashboard.css` — remove `single-loss-only` etc.
- `src/hotcb/capabilities.py` — remove `validate_mutable_state()`, keep `TrainingCapabilities`
  but remove `mutable_state_detected`/`mutable_state_keys` (now in `MutableState`)

**Tests (write first):**
```
test_config_controls_from_mutable_state()
    App with MutableState(lr, wd, recon_w).
    GET /api/config → controls has 3 entries.
    Each has param_key, type, label, min, max, current.

test_config_controls_empty_when_no_mutable_state()
    App without MutableState.
    GET /api/config → controls is [].

test_config_controls_types_match_actuators()
    MutableState with LOG_FLOAT lr, FLOAT recon_w, BOOL use_augment.
    Controls: [{type: "log_float"}, {type: "float"}, {type: "bool"}].

test_config_controls_groups_present()
    optimizer_actuators → group="optimizer".
    loss_actuators → group="loss".
    Custom → group="custom".

test_config_controls_reflect_live_state()
    Apply mutation to lr.
    GET /api/config → lr control has updated current + state="unverified".

--- Server API endpoints adapt ---

test_opt_set_endpoint_still_works()
    POST /api/opt/set with {params: {lr: 1e-3}}
    → command written to JSONL with module="opt".

test_loss_set_endpoint_still_works()
    POST /api/loss/set with {params: {recon_w: 2.0}}
    → command written to JSONL with module="loss".

test_control_state_endpoint_uses_mutable_state()
    GET /api/state/controls
    → Returns live values from MutableState, not hardcoded schema.
```

---

### Phase 5: Immutable `run_dir`
**Goal:** `run_dir` set once, never mutated. Remove rewire infrastructure.

**Modified files:**
- `src/hotcb/server/app.py` — remove `app.state.run_dir` mutation, use `config.run_dir`
- `src/hotcb/server/launcher.py` — write to `config.run_dir` directly, no subdirs
- `src/hotcb/server/tailer.py` — remove `rewire()` method

**Tests (write first):**
```
test_run_dir_set_once()
    config = DashboardConfig(run_dir="/tmp/x")
    All endpoints use "/tmp/x". No mutation observed.

test_launcher_writes_to_config_run_dir()
    Launcher.start() → JSONL files appear in config.run_dir, not subdirs.

test_launcher_truncates_on_restart()
    Launcher.start() twice → JSONL files truncated (like reset), no subdirs.

test_tailer_no_rewire_method()
    JsonlTailer has no rewire() attribute.

test_endpoints_use_immutable_run_dir()
    Start app. Simulate launcher.
    GET /api/metrics/history → reads from original run_dir.

test_compare_reads_siblings_readonly()
    GET /api/runs/discover → scans parent dir.
    Original monitored dir unchanged.
```

---

### Phase 6: Magic Number Extraction
**Goal:** All bare literals replaced with config reads.

**Modified files:**
- `src/hotcb/server/app.py` — history limits, WS burst from config
- `src/hotcb/server/tailer.py` — poll_interval from config
- `src/hotcb/server/autopilot.py` — thresholds from config
- `src/hotcb/server/ai_engine.py` — cadence/thresholds from config
- `src/hotcb/server/static/js/charts.js` — read from `S.config.chart.*`
- `src/hotcb/server/static/js/controls.js` — threshold from `S.config.ui.*`
- `src/hotcb/server/static/js/websocket.js` — retries from `S.config.server.*`
- `src/hotcb/server/static/js/init.js` — intervals from `S.config.ui.*`
- `src/hotcb/server/static/js/panels.js` — intervals from `S.config.ui.*`

**Tests (write first):**
```
test_tailer_uses_config_poll_interval()
    Config with poll_interval=2.0.
    Tailer constructed with it → internal interval is 2.0.

test_history_limits_from_config()
    Config with history_limit_metrics=100.
    GET /api/metrics/history → returns at most 100 records.

test_ws_burst_from_config()
    Config with ws_initial_burst=50.
    WS connect → at most 50 records in initial burst.

test_autopilot_thresholds_from_config()
    Config with divergence_threshold=5.0.
    Autopilot engine uses 5.0, not hardcoded 2.0.

test_ai_cadence_from_config()
    Config with ai_default_cadence=100.
    AI engine uses 100, not hardcoded 50.
```

Frontend magic number replacement is verified manually + by existing test suite
(backend serves correct config, frontend reads it).

---

## Adapter Integration

Adapters (`lightning.py`, `hf.py`) currently populate `TrainingCapabilities` and
put `optimizer`/`mutable_state` in `env`. After this change:

1. Adapters create `MutableState` from the optimizer + any user-registered actuators
2. Pass `MutableState` to kernel constructor (or register after init)
3. `env` still carries `optimizer`, `scheduler`, etc. for `apply_fn` closures
4. `TrainingCapabilities` still written to `hotcb.capabilities.json` for
   framework-level info, but `mutable_state_detected`/`mutable_state_keys`
   are replaced by `MutableState.describe_all()`

---

## Demo Updates

All 3 demos (`demo.py`, `golden_demo.py`, `finetune_demo.py`) currently create
`_OptProxy` + register `OptimizerActuator` and `MutableStateActuator`. After this:

```python
# Instead of:
k.register_actuator("opt", OptimizerActuator())
k.register_actuator("loss", MutableStateActuator())

# Becomes:
from hotcb.actuators import optimizer_actuators, loss_actuators, mutable_state
ms = mutable_state(
    optimizer_actuators(opt_proxy) + loss_actuators(loss_weights)
)
k = HotKernel(run_dir=..., mutable_state=ms)
```

---

## Impact on Existing Test Suites

| Test file | Tests | Impact |
|-----------|-------|--------|
| `test_hotopt.py` | 12 | **Rewrite**: Test via kernel default stream |
| `test_hotloss.py` | 14 | **Rewrite**: Test via kernel default stream |
| `test_kernel_core.py` | 19 | **Update 4**: route_to_opt/loss tests change; rest unchanged |
| `test_hottune.py` | 104+ | **Update ~30**: Replace OptimizerActuator/MutableStateActuator with new types |
| `test_server_api.py` | 40+ | **Unchanged**: API writes same JSONL, routing is kernel-internal |
| `test_server_app.py` | 15 | **Update ~5**: Config endpoint, status endpoint |
| `test_new_features.py` | 49 | **Update 2**: opt/loss traceback tests |
| `test_launch.py` | 21 | **Update ~5**: Actuator registration changes |
| `test_backend_gaps.py` | varies | **Update**: Actuator-related tests |

New test files:
- `test_dashboard_config.py` — ~15 tests (Phase 1)
- `test_actuator_unified.py` — ~35 tests (Phase 2)
- Phase 3 kernel tests integrated into `test_kernel_core.py`

---

## Phase Ordering & Dependencies

```
Phase 1 (Config)  ──────────────────────────────────────→ Phase 5 (Immutable run_dir)
                                                        → Phase 6 (Magic numbers)
Phase 2 (Actuator types) → Phase 3 (Kernel) → Phase 4 (Dynamic controls)
```

Phases 1 and 2 are independent — can be done in parallel.
Phase 3 depends on Phase 2.
Phase 4 depends on Phases 1 + 3 (needs config endpoint + actuator metadata).
Phase 5 depends on Phase 1 (uses config).
Phase 6 depends on Phases 1 + 5 (config exists + run_dir stable).

**Recommended order:** 1 → 2 → 3 → 4 → 5 → 6
(Phases 1+2 could run in parallel if two sessions available.)

---

## Verification

After each phase:
```bash
pytest src/hotcb/tests/ -x -q --no-cov   # full suite passes
```

After Phase 3 (the big one):
```bash
hotcb demo                                 # all 3 demo configs work
hotcb demo --golden                        # golden demo metrics flow
```

After Phase 4:
```bash
hotcb serve --dir <project-with-5-loss-weights>
# Dashboard shows 5 loss weight sliders (not 2 hardcoded ones)
```

After Phase 6:
```bash
# Create hotcb.dashboard.yaml with custom intervals
hotcb serve --dir runs/exp1
# Verify custom values appear in browser console: S.config
```
