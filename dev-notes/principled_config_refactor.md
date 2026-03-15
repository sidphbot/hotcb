# Principled Config Refactor — Design Document

## Problem Statement

The dashboard layer has drifted from hotcb's core design principle: **treat the training
framework as sacred, hook onto it, never alter it.**

Three violations:

1. **`run_dir` mutation** — The launcher creates subdirs under `run_dir` and rewires
   the tailer/endpoints mid-flight via `_ctx["run_dir"]`. This turns the dashboard from
   a monitor into an orchestrator. The original contract: `hotcb serve --dir X` monitors
   `X`. Period.

2. **Hardcoded control schemas** — The UI has hardcoded slider names (`knobLr`, `knobWd`,
   `knobLossW`, `knobWeightA`, `knobWeightB`), CSS visibility classes
   (`single-loss-only`, `multitask-only`, `finetune-only`), and boolean switches for
   what to show/hide. Meanwhile, the core already discovers what's mutable via
   `TrainingCapabilities` and actuator `describe_space()`. The dashboard ignores this
   and hardcodes its own schema.

3. **Magic numbers scattered everywhere** — Poll intervals, history limits, retry params,
   bounds, thresholds, tension values, pixel sizes, batch sizes. Each lives as a bare
   literal in the file that uses it. No central source of truth. Changing any requires
   hunting through JS and Python files.

---

## Design Principles

1. **`run_dir` is immutable after startup.** The dashboard attaches to a directory and
   monitors it. If the launcher needs a new dir, it tells the *user*, who restarts with
   a new `--dir`. The dashboard never rewires itself.

2. **Controls are data-driven.** The set of tunable parameters (sliders, toggles, bounds)
   comes from a config object built at startup from `TrainingCapabilities` +
   actuator `describe_space()` + user overrides. The UI renders from this config; no
   hardcoded slider HTML.

3. **All tunables live in a config.** Every magic number, interval, limit, threshold, and
   feature flag lives in a typed config object. Defaults are explicit. Users can override
   via a `hotcb.dashboard.yaml` file, CLI flags, or env vars.

4. **TDD first.** The UI is volatile. Every config path, every endpoint contract, every
   default value has a test before the implementation changes. Tests define the contract;
   code fulfills it.

---

## Architecture

### A. `DashboardConfig` — The Central Config Object

A single Python dataclass that holds every tunable for the server and UI.

```python
# src/hotcb/server/config.py

@dataclass
class ServerConfig:
    """Server-level tunables."""
    host: str = "0.0.0.0"
    port: int = 8421
    poll_interval: float = 0.5          # tailer JSONL poll frequency (s)
    history_limit_metrics: int = 500    # /api/metrics/history last_n
    history_limit_applied: int = 200    # /api/applied/history last_n
    ws_initial_burst: int = 200         # records sent on WS connect
    ws_max_retries: int = 20            # client reconnect attempts
    ws_retry_base: float = 3.0          # retry backoff base (s)
    ws_retry_cap: float = 30.0          # retry backoff ceiling (s)

@dataclass
class ChartConfig:
    """Chart rendering tunables."""
    max_render_points: int = 2000       # LTTB downsample target per dataset
    line_tension: float = 0.15          # bezier curve smoothing
    forecast_dash: tuple = (6, 3)       # forecast line pattern
    mutation_dash: tuple = (3, 4)       # mutation overlay pattern
    annotation_stagger_rows: int = 10   # mutation label vertical slots
    annotation_min_distance: int = 70   # px between mutation labels
    comparison_dash_patterns: list      # per-run dash styles

@dataclass
class ControlConfig:
    """What the user can mutate and within what bounds."""
    # Populated from TrainingCapabilities + actuator describe_space()
    opt_params: list[ParamSpec]         # [{name, type, min, max, step, default, label}]
    loss_params: list[ParamSpec]        # [{name, type, min, max, step, default, label}]
    cb_enabled: bool                    # whether cb module is active
    tune_enabled: bool                  # whether tune module is active

@dataclass
class ParamSpec:
    """Schema for a single tunable parameter."""
    name: str                           # internal key (e.g. "lr", "weight_a")
    label: str                          # display label (e.g. "Learning Rate")
    type: str                           # "log_range" | "linear_range" | "toggle" | "select"
    min: float = 0.0
    max: float = 1.0
    step: float = 0.01
    default: float = 0.0
    log_base: float = 10.0             # for log_range type
    group: str = "opt"                  # which module owns this

@dataclass
class AutopilotConfig:
    """Autopilot tunables."""
    divergence_threshold: float = 2.0
    ratio_threshold: float = 0.5
    ai_min_interval: int = 10           # min steps between AI checks
    ai_max_wait: int = 200              # max steps before forced AI check
    ai_default_cadence: int = 50        # default periodic interval

@dataclass
class UIConfig:
    """UI behavior tunables."""
    state_save_interval: int = 5000     # ms between localStorage saves
    alert_poll_interval: int = 15000    # ms between alert fetches
    manifold_refresh_interval: int = 10000  # ms for 3D view auto-refresh
    recipe_refresh_interval: int = 5000 # ms for recipe auto-refresh
    forecast_poll_interval: int = 5000  # ms for forecast refresh
    forecast_step_cadence: int = 10     # min steps between forecast updates
    forecast_batch_size: int = 8        # concurrent forecast requests
    staged_change_threshold: float = 0.005  # 0.5% relative diff for highlight
    health_ema_alpha: float = 0.1       # health score smoothing

@dataclass
class DashboardConfig:
    """Top-level config aggregating all sub-configs."""
    server: ServerConfig = field(default_factory=ServerConfig)
    chart: ChartConfig = field(default_factory=ChartConfig)
    controls: ControlConfig = field(default_factory=ControlConfig)
    autopilot: AutopilotConfig = field(default_factory=AutopilotConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    run_dir: str = ""                   # IMMUTABLE after startup
```

### B. Config Resolution Order

```
1. Built-in defaults (dataclass defaults above)
2. hotcb.dashboard.yaml in run_dir (if exists)
3. CLI flags (--port, --poll-interval, etc.)
4. Environment variables (HOTCB_PORT, HOTCB_POLL_INTERVAL, etc.)
5. TrainingCapabilities auto-discovery (for ControlConfig only)
```

Later sources override earlier ones. `TrainingCapabilities` **merges** into
`ControlConfig` — it doesn't replace user overrides.

### C. Config Serving

Single endpoint: `GET /api/config` returns the full `DashboardConfig` as JSON.

The frontend fetches this **once at startup** (`init.js`) and stores it in `S.config`.
All JS code reads from `S.config.*` instead of hardcoded literals.

```javascript
// init.js
var config = await api('GET', '/api/config');
S.config = config;
```

### D. `run_dir` is Immutable

**Current (broken):**
```
create_app(run_dir)
  → _ctx = {"run_dir": resolved}
  → launcher.start() creates subdir, mutates _ctx["run_dir"]
  → all endpoints see new dir
```

**Proposed:**
```
create_app(run_dir)
  → config.run_dir = run_dir   # NEVER CHANGES
  → tailer watches run_dir     # NEVER REWIRES
  → launcher.start() writes to run_dir directly
  → if user wants a new subdir, launcher returns the path
    and the user restarts `hotcb serve --dir new_path`
```

**For the built-in demo launcher:** The launcher writes JSONL files into `config.run_dir`
directly (not into subdirs). Each `start()` call truncates the existing files (as
`reset()` already does). This matches the original design where `run_dir` is a single
flat directory.

**For multi-run comparison:** The Compare tab uses `GET /api/runs/discover` which
scans the parent directory read-only. It doesn't rewire anything — it just reads
old JSONL files from sibling dirs. The active monitored dir never changes.

### E. Dynamic Controls Generation

**Current (broken):**
```html
<!-- Hardcoded in index.html -->
<div class="knob-row" data-param="lr">
  <span class="knob-label">lr</span>
  <input type="range" min="-6" max="0" step="0.01" ...>
</div>
<div class="knob-row single-loss-only" data-param="loss_w">...</div>
<div class="knob-row multitask-only" data-param="weight_a">...</div>
```

**Proposed:**
```html
<!-- index.html just has the container -->
<div class="card-body" id="knobPanel"></div>
```

```javascript
// controls.js — generates from config
function buildControls(config) {
  var panel = $('#knobPanel');
  panel.innerHTML = '';
  config.controls.opt_params.forEach(function(p) {
    panel.appendChild(buildKnobRow(p));
  });
  config.controls.loss_params.forEach(function(p) {
    panel.appendChild(buildKnobRow(p));
  });
}

function buildKnobRow(spec) {
  // spec = {name: "lr", label: "Learning Rate", type: "log_range",
  //         min: 1e-7, max: 1.0, step: 0.01, default: 3.16e-4}
  var row = document.createElement('div');
  row.className = 'knob-row';
  row.dataset.param = spec.name;
  // ... build slider/input based on spec.type
  return row;
}
```

No more `single-loss-only`, `multitask-only`, `finetune-only` CSS classes. If a param
isn't in `config.controls`, it doesn't get rendered.

---

## Implementation Plan (TDD)

### Phase 1: Config Object + Tests

**Goal:** `DashboardConfig` exists, is testable, serializes to JSON, loads from YAML.

```
Tests to write FIRST:
  test_config_defaults()          — verify all defaults match documented values
  test_config_from_yaml()         — load hotcb.dashboard.yaml, verify override
  test_config_from_env()          — HOTCB_PORT=9000 overrides port
  test_config_to_json()           — round-trip serialization
  test_config_merge_capabilities()— TrainingCapabilities merges into ControlConfig
  test_param_spec_schema()        — ParamSpec validates types and bounds
  test_config_immutable_run_dir() — run_dir cannot be changed after construction
```

**Files:**
- `src/hotcb/server/config.py` — config dataclasses + loader
- `src/hotcb/tests/test_dashboard_config.py` — tests

### Phase 2: Config Endpoint + Frontend Fetch

**Goal:** `/api/config` serves the config. Frontend reads it at startup.

```
Tests to write FIRST:
  test_config_endpoint_returns_full()  — GET /api/config returns all sections
  test_config_endpoint_includes_caps() — includes discovered ParamSpecs
  test_config_endpoint_json_schema()   — response matches expected shape
```

**Files:**
- `src/hotcb/server/app.py` — add `/api/config` endpoint
- `src/hotcb/server/static/js/state.js` — add `S.config`
- `src/hotcb/server/static/js/init.js` — fetch config at startup

### Phase 3: Immutable `run_dir`

**Goal:** Remove `_ctx` pattern. `run_dir` is set once and never changes.

```
Tests to write FIRST:
  test_run_dir_immutable()             — config.run_dir cannot be reassigned
  test_launcher_writes_to_run_dir()    — start() writes to config.run_dir, not subdirs
  test_launcher_truncates_on_start()   — start() clears JSONL files (like reset)
  test_endpoints_use_config_run_dir()  — all endpoints read config.run_dir
  test_no_tailer_rewatch()             — tailer.rewatch() is never called
  test_compare_reads_siblings()        — /api/runs/discover reads parent dir read-only
```

**Files:**
- `src/hotcb/server/app.py` — replace all `_ctx["run_dir"]` with `config.run_dir`
- `src/hotcb/server/launcher.py` — remove subdir creation, write to `run_dir` directly
- `src/hotcb/server/tailer.py` — remove `rewatch()` method

### Phase 4: Dynamic Controls from Config

**Goal:** Controls are generated from `config.controls`, not from hardcoded HTML.

```
Tests to write FIRST:
  test_control_config_from_capabilities() — TrainingCapabilities → ParamSpec list
  test_control_config_simple_demo()       — simple demo produces lr+wd params
  test_control_config_multitask_demo()    — golden demo adds weight_a, weight_b
  test_control_config_external_project()  — external project discovers from caps.json
  test_control_config_no_caps()           — fallback: lr+wd only
  test_opt_params_have_bounds()           — bounds from actuator describe_space()
  test_loss_params_have_bounds()          — bounds from mutable_state actuator
```

**Files:**
- `src/hotcb/server/config.py` — add `ControlConfig.from_capabilities()`
- `src/hotcb/server/static/js/controls.js` — `buildControls(config)` replaces hardcoded HTML
- `src/hotcb/server/static/index.html` — remove hardcoded knob rows, keep container only

### Phase 5: Replace Frontend Magic Numbers

**Goal:** All JS magic numbers read from `S.config`.

```
Tests to write FIRST (backend contract tests):
  test_config_chart_defaults()      — chart config has expected defaults
  test_config_ui_defaults()         — UI config has expected defaults
  test_config_ws_defaults()         — WebSocket config has expected defaults
```

**Frontend changes (verified manually + by existing backend tests):**

| Current literal | Replaced with | File |
|----------------|---------------|------|
| `_maxRenderPoints = 2000` | `S.config.chart.max_render_points` | charts.js |
| `tension: 0.15` | `S.config.chart.line_tension` | charts.js |
| `[6, 3]` (forecast dash) | `S.config.chart.forecast_dash` | charts.js |
| `[3, 4]` (mutation dash) | `S.config.chart.mutation_dash` | charts.js |
| `0.005` (staged threshold) | `S.config.ui.staged_change_threshold` | controls.js |
| `setInterval(..., 5000)` (state save) | `S.config.ui.state_save_interval` | init.js |
| `setInterval(..., 15000)` (alerts) | `S.config.ui.alert_poll_interval` | init.js |
| `setInterval(..., 10000)` (manifold) | `S.config.ui.manifold_refresh_interval` | panels.js |
| `setInterval(..., 5000)` (recipe) | `S.config.ui.recipe_refresh_interval` | panels.js |
| `setInterval(..., 5000)` (forecast) | `S.config.ui.forecast_poll_interval` | charts.js |
| `>= 10` (forecast cadence) | `S.config.ui.forecast_step_cadence` | charts.js |
| `batchSize = 8` | `S.config.ui.forecast_batch_size` | charts.js |
| `_wsMaxRetries = 20` | `S.config.server.ws_max_retries` | websocket.js |
| `3000 * 1.5^n, max 30000` | `S.config.server.ws_retry_base/cap` | websocket.js |
| `last_n=500` (metrics) | `S.config.server.history_limit_metrics` | app.py |
| `last_n=200` (applied) | `S.config.server.history_limit_applied` | app.py |
| `last_n=200` (WS burst) | `S.config.server.ws_initial_burst` | app.py |

### Phase 6: Replace Backend Magic Numbers

**Goal:** All Python magic numbers read from `DashboardConfig`.

```
Tests to write FIRST:
  test_poll_interval_from_config()     — tailer uses config.server.poll_interval
  test_history_limits_from_config()    — endpoints use config.server.history_limit_*
  test_autopilot_thresholds_config()   — autopilot reads config.autopilot.*
  test_ai_cadence_from_config()        — AI engine reads config.autopilot.ai_*
```

**Files:**
- `src/hotcb/server/app.py` — all endpoints read from config
- `src/hotcb/server/tailer.py` — poll_interval from config
- `src/hotcb/server/ai_engine.py` — cadence/thresholds from config
- `src/hotcb/server/autopilot.py` — thresholds from config

---

## What This Does NOT Change

- **Core kernel** — untouched. `kernel.apply(env, events)` contract stays.
- **Actuator protocol** — untouched. `snapshot/validate/apply/restore/describe_space`.
- **JSONL filesystem IPC** — untouched. Commands, applied, metrics stay as files.
- **Adapter layer** — untouched. Lightning/HF adapters still write capabilities.json.
- **Module system** — untouched. opt/loss/cb/tune modules stay.
- **CLI subcommands** — untouched (except `serve` gets new flags for config overrides).

---

## Config File Format

```yaml
# hotcb.dashboard.yaml (optional, placed in run_dir)

server:
  port: 8421
  poll_interval: 0.5
  history_limit_metrics: 500
  history_limit_applied: 200
  ws_initial_burst: 200

chart:
  max_render_points: 2000
  line_tension: 0.15

controls:
  # Override or extend discovered params
  opt_params:
    - name: lr
      label: Learning Rate
      type: log_range
      min: 1.0e-7
      max: 1.0
      step: 0.01
      default: 3.16e-4
    - name: weight_decay
      label: Weight Decay
      type: log_range
      min: 1.0e-6
      max: 1.0
      step: 0.01
      default: 1.0e-4
  # loss_params: auto-discovered from TrainingCapabilities if not specified

autopilot:
  divergence_threshold: 2.0
  ratio_threshold: 0.5
  ai_min_interval: 10
  ai_default_cadence: 50

ui:
  state_save_interval: 5000
  alert_poll_interval: 15000
  staged_change_threshold: 0.005
```

---

## Migration Path

1. **Phase 1-2** are additive — no breaking changes. Config object + endpoint exist
   alongside current code.
2. **Phase 3** removes `_ctx` and subdirectory creation — this is the breaking change.
   Existing runs in subdirs still work (serve points at the subdir directly).
3. **Phase 4** removes hardcoded HTML controls — the UI looks the same but is generated.
4. **Phase 5-6** are pure refactors — same behavior, values come from config instead
   of literals.

Each phase has its own test suite that passes before moving on. No phase depends on
a later phase. Any phase can be shipped independently.

---

## Verification

After each phase:
```bash
pytest src/hotcb/tests/ -x -q --no-cov   # full suite passes
hotcb demo                                 # dashboard works with demo
hotcb serve --dir <external-project>       # dashboard works with real project
```

After Phase 3 specifically:
```bash
# Verify run_dir is truly immutable
hotcb serve --dir runs/exp1
# Start training from dashboard → writes to runs/exp1/ directly
# Stop, start again → same dir, files truncated, no subdirs created
```

After Phase 4 specifically:
```bash
# Verify controls are discovered, not hardcoded
hotcb serve --dir <project-with-3-loss-weights>
# Dashboard shows 3 loss weight sliders (not 2 hardcoded ones)
```
