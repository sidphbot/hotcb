# Dashboard Fixes v2 — Immediate + Structural

Supersedes the ad-hoc Fix A-G patches. Informed by `principled_config_refactor.md`
(the long-term config architecture) but scoped to what's broken *right now*.

---

## Issues (Diagnosed)

### Issue 1: Training Panel Leaks into External Mode

**What:** The Training card (Start/Stop/Reset, config dropdown, steps/delay/seed) shows
for external projects. The `is_external` flag hides the dropdown but the entire card
with launcher controls is still visible — confusing for users who attached to an
external training run and don't control the process.

**Root cause:** No concept of "demo mode" vs "monitor mode". The launcher panel is
always rendered in HTML; only the dropdown gets `display:none`.

**Fix:** Gate the *entire* Training card behind a `_HOTCB_DEMO` internal flag.
- `hotcb demo` and `hotcb demo --golden` set `_HOTCB_DEMO=1` internally
- `hotcb serve --dir X` does NOT set it (monitor-only mode)
- Frontend: `/api/config` response includes `demo_mode: true/false`
- If `demo_mode === false`, hide the entire Training card (`#trainPanel`'s parent `.card`)
- Keep the card in HTML (not removed) so demo mode still works
- No public CLI flag for now — this is an internal switch for the demo codepath

### Issue 2: Tooltip Color Box Looks Like Checkbox

**What:** When hovering over a data point on the main metrics chart, Chart.js renders
a small colored **rectangle** next to the metric name in the tooltip. This default
"legend box" style looks like a checkbox/tickbox.

**Root cause:** Chart.js tooltip config has no `usePointStyle` setting.
Default behavior draws a `boxWidth x boxHeight` rectangle swatch per label item.

**Fix:** Add `usePointStyle: true` to tooltip config. This renders a small filled
circle (matching `elements.point` style) instead of a rectangle.

```javascript
// charts.js — main chart tooltip config
tooltip: {
  ...existing,
  usePointStyle: true,        // circle instead of rectangle
  boxWidth: 8,                // small solid dot
  boxHeight: 8,
}
```

Same fix for the compare chart tooltip in `panels.js`.

### Issue 3: Pin Indicator Not Visible in Metric Dropdown

**What:** The metric dropdown shows filled/hollow dots for visibility toggle, but
the "pinned" state is only indicated by a subtle `box-shadow` glow — too easy to miss.
User wants an explicit pin symbol.

**Root cause:** Fix E (from the prior plan) replaced checkbox+swatch+pin-emoji with
dot-only, removing the pin indicator entirely and replacing it with a CSS glow.

**Fix:** Add a small pin icon (Unicode `\u{1F4CC}` or text "pin") next to the label
for pinned metrics. Keep the dot for visibility toggle. The pin icon is a separate
element, not overloaded onto the dot.

```javascript
// After the label element in _renderMetricDropdown:
if (isPinned) {
  var pinIcon = document.createElement('span');
  pinIcon.className = 'metric-pin-icon';
  pinIcon.textContent = '\u{1F4CC}';   // or use CSS ::after with a pin glyph
  pinIcon.title = 'Pinned — double-click dot to unpin';
  row.appendChild(pinIcon);
}
```

```css
.metric-pin-icon {
  font-size: 10px; flex-shrink: 0; margin-left: 2px;
  opacity: 0.8; cursor: pointer;
}
```

### Issue 4: Stale Mutations Create Ugly Connector Lines (CRITICAL)

**What:** When opening `hotcb serve --dir <external-project>`, old mutations from a
previous training run load from `hotcb.applied.jsonl`. The mutation annotation plugin
draws vertical dashed lines at those old steps. If the user starts new training, the
forecast/mutation overlays create connector lines from the old step values to the new
curve's starting point — a long diagonal line across the chart.

**Root cause — two layers:**

**Layer A: Old applied records load on connect.**
`initialLoad()` (init.js:55-62) calls `GET /api/applied/history?last_n=200` which
reads `hotcb.applied.jsonl` — this file persists across runs if the user reuses the
same `run_dir`. The records have `step` fields from a previous run, which don't
correspond to current metric data.

**Layer B: Mutation overlay lines anchor at wrong coordinates.**
`updateChart()` (charts.js:450-464) creates mutation overlay datasets starting at
`{x: mu.fromStep, y: mu.fromVal}`. If `mu.fromStep` is from a previous run (e.g.
step 347) but the current run has only reached step 50, Chart.js draws a line from
step 347 back to step 50 — the ugly diagonal connector.

Similarly, forecast overlays start at `{x: lastStep, y: lastVal}` — if the forecast
cache has stale entries from a previous run, this creates orphaned dotted lines.

**Fix (multi-part):**

**4a. Scope applied records to current metric range.**
Before rendering mutation annotations, filter out records whose `step` is outside
the range of actual metric data:

```javascript
// In mutationAnnotationPlugin.afterDraw():
var dataMinStep = _getMinStep();
var dataMaxStep = _getMaxStep();
// Skip annotations outside data range
if (rec.step < dataMinStep || rec.step > dataMaxStep) continue;
```

**4b. Filter mutation overlays to valid step range.**
In `updateChart()`, skip mutation/forecast overlays where the anchor step doesn't
exist in the current metric data:

```javascript
// Before rendering mutation overlay:
if (cache.mutation && cache.mutation.fromStep !== undefined) {
  var mu = cache.mutation;
  // Only render if fromStep is within current data range
  var inRange = pts.length > 0 && mu.fromStep >= pts[0].step && mu.fromStep <= pts[pts.length-1].step;
  if (showOverlays && inRange && mu.values && mu.values.length) {
    // ... render
  }
}
```

**4c. Invalidate forecast cache when data range changes dramatically.**
In the WebSocket metrics handler, detect if the incoming step is much lower than the
last known step (indicating a new run started). If so, clear forecast cache:

```javascript
// websocket.js, inside metrics handler:
if (step < maxStep - 10) {
  // Steps went backwards — likely a new run. Clear stale caches.
  _forecastCache = {};
  _highlightedMutationStep = null;
}
```

**4d. Clear `_metricToggleState` on state reset.**
`_clearTrainingState()` currently does NOT clear `_metricToggleState`, so old metric
names persist in the toggle state from a previous run:

```javascript
// controls.js _clearTrainingState():
if (typeof _metricToggleState !== 'undefined') _metricToggleState = {};
if (typeof _metricDropdownShowAll !== 'undefined') _metricDropdownShowAll = false;
if (typeof _lastMetricCount !== 'undefined') _lastMetricCount = 0;
```

### Issue 5: run_dir/subdir Business

**What:** The launcher creates timestamped subdirs (`{config_id}_{timestamp}/`) under
`run_dir`, then rewires the tailer and all REST endpoint closures via `_ctx["run_dir"]`.
This violates the sacred principle: `hotcb serve --dir X` should monitor X, period.

**Root cause:** Fix B (TensorBoard-style run discovery) from the prior plan added
subdir creation and tailer rewiring.

**Fix (for now — not MVP, gated behind demo flag):**
- When `demo_mode=true`: launcher writes directly to `run_dir` (flat), truncates
  JSONL files on new start (same as `reset()` does). No subdirs, no rewiring.
- When `demo_mode=false`: the entire Training card is hidden (Issue 1), so the
  launcher code is unreachable. `run_dir` is immutable.
- Remove `_ctx` dict pattern. Use `run_dir` as a plain closure variable (immutable).
- Remove `tailer.rewatch()` — tailer watches `run_dir` once at startup, forever.
- The `/api/runs/discover` endpoint stays (read-only scan of sibling dirs for Compare
  tab) but never rewires anything.

---

## Implementation Order

```
1. Issue 4  — Stale mutations (critical UX bug, data correctness)
2. Issue 2  — Tooltip color box (quick visual fix)
3. Issue 3  — Pin indicator (quick visual fix)
4. Issue 1  — Demo mode gate (structural, enables Issue 5)
5. Issue 5  — Immutable run_dir (structural cleanup)
```

---

## Detailed Implementation

### Step 1: Fix Stale Mutations (Issue 4)

**Files:** `charts.js`, `websocket.js`, `controls.js`

**1a. Filter mutation annotations to data range** (`charts.js`, `mutationAnnotationPlugin`):
- In `afterDraw()`, compute `dataMinStep` and `dataMaxStep` from chart datasets
- Skip drawing annotation lines for records outside `[dataMinStep, dataMaxStep]`

**1b. Guard mutation/forecast overlay rendering** (`charts.js`, `updateChart()`):
- Before building `muPts`, check `mu.fromStep` is within `[pts[0].step, pts[last].step]`
- Before building `fcPts`, check `lastStep > 0` and forecast steps are contiguous

**1c. Detect run reset in WS handler** (`websocket.js`):
- Track `_lastSeenStep` per metric
- If incoming `step < _lastSeenStep - 10`, treat as new run: clear `S.metricsData`,
  `S.appliedData`, `_forecastCache`, timeline

**1d. Clear toggle state on reset** (`controls.js`, `_clearTrainingState()`):
- Add `_metricToggleState = {}`, `_metricDropdownShowAll = false`, `_lastMetricCount = 0`

**Tests (backend contract):**
```
test_applied_history_returns_records()     — existing test, verify shape
test_metrics_history_returns_records()     — existing test, verify shape
```
Frontend fixes are verified manually + by existing test suite not regressing.

### Step 2: Fix Tooltip Color Box (Issue 2)

**Files:** `charts.js`, `panels.js`

**2a. Main chart tooltip** (`charts.js:365-368`):
- Add `usePointStyle: true, boxWidth: 8, boxHeight: 8`

**2b. Compare chart tooltip** (`panels.js:1186-1216`):
- Add `usePointStyle: true, boxWidth: 8, boxHeight: 8`

**2c. Mini metric card charts** (`charts.js`, `createMetricCard()`):
- Same tooltip fix if applicable

### Step 3: Restore Pin Indicator (Issue 3)

**Files:** `charts.js`, `dashboard.css`

**3a. Add pin icon element** (`charts.js`, `_renderMetricDropdown()`):
- After label element, if `isPinned`, append a `span.metric-pin-icon` with pin glyph
- Click on pin icon triggers `toggleMetricCard(name)`

**3b. CSS for pin icon** (`dashboard.css`):
- `.metric-pin-icon { font-size: 10px; opacity: 0.8; cursor: pointer; margin-left: 2px; }`

### Step 4: Demo Mode Gate (Issue 1)

**Files:** `app.py`, `launcher.py`, `controls.js`, `cli.py`

**4a. Add `demo_mode` to app state** (`app.py`):
- `create_app()` accepts `demo_mode: bool = False`
- Stored in `app.state.demo_mode`
- Returned in existing `/api/state/controls` response as `demo_mode: false`

**4b. CLI sets demo_mode** (`cli.py`):
- `hotcb demo` → calls `create_app(run_dir, demo_mode=True)`
- `hotcb serve --dir X` → calls `create_app(run_dir, demo_mode=False)`

**4c. Frontend hides Training card** (`controls.js`, `hydrateControlsFromServer()`):
- If `state.demo_mode === false`, hide the Training card's parent `.card` element
- Also hide the config dropdown area

**4d. Launcher writes flat when demo_mode** (`launcher.py`):
- When `demo_mode=True`: `start()` writes to `self._run_dir` directly (no subdirs),
  truncates JSONL files first
- When `demo_mode=False`: `start()` is unreachable (Training card hidden)

### Step 5: Immutable run_dir (Issue 5)

**Files:** `app.py`, `launcher.py`, `tailer.py`

**5a. Remove `_ctx` pattern** (`app.py`):
- Replace `_ctx = {"run_dir": active_run_dir}` with plain `run_dir = active_run_dir`
- All endpoint closures capture `run_dir` directly (it never changes)
- Remove `app.state._ctx`

**5b. Remove `rewatch()`** (`tailer.py`):
- Delete the `rewatch()` method
- Tailer watches files once at startup in `lifespan()`

**5c. Remove rewiring from launcher** (`launcher.py`):
- Remove the `start_training` tailer rewatch block
- Remove `_ctx` update
- In demo mode, launcher truncates existing files and writes in-place

**5d. Keep discover endpoint read-only** (`app.py`):
- `/api/runs/discover` still scans parent dir
- Compare tab still works (reads old run dirs)
- But nothing rewires the active monitoring

---

## What NOT to Do

- Do NOT implement the full `DashboardConfig` refactor yet (that's the long-term plan
  in `principled_config_refactor.md`). This plan fixes what's broken now.
- Do NOT remove `initialLoad()` or the WS initial burst — those are needed for
  page-refresh recovery. Just filter stale data at render time.
- Do NOT change the core kernel, actuator protocol, or JSONL format.
- Do NOT change the CLI interface (no new public flags).

---

## Verification

After each step:
```bash
pytest src/hotcb/tests/ -x -q --no-cov
```

After Step 1:
```bash
hotcb serve --dir <external-project-with-old-applied>
# Verify: no ugly diagonal connector lines
# Verify: old mutation annotations don't appear outside data range
# Verify: starting new training clears stale overlays
```

After Step 4:
```bash
hotcb serve --dir <any-dir>
# Verify: Training card is NOT visible
hotcb demo
# Verify: Training card IS visible, Start/Stop/Reset work
```

After Step 5:
```bash
hotcb demo
# Start training → JSONL files written to run_dir (not subdir)
# Stop, start again → files truncated, fresh run in same dir
# Compare tab → discovers sibling dirs if parent has them
```
