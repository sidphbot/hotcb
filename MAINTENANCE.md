# MAINTENANCE.md — hotcb Release Readiness Audit

Generated: 2026-03-12

## Priority Legend
- **P0** — Must fix before release (broken functionality)
- **P1** — Should fix (poor UX, data loss risk)
- **P2** — Nice to fix (code quality, polish)
- **P3** — Backlog (low-impact improvements)

---

## 1. User-Reported Issues (P0)

### 1.1 Claude Skill PYTHONPATH Issue
**File:** `.claude/skills/hotcb-autopilot/SKILL.md:78-79`
**Problem:** Skill uses `python3 -c "import hotcb; ..."` which fails when hotcb is installed in editable mode or not on `sys.path`. Need `PYTHONPATH=src` prefix or `python3 -m hotcb` pattern.
**Fix:** Update all `python3 -c` invocations to use `PYTHONPATH=src python3 -c` or `python3 -m` equivalents.

### 1.2 Claude Skill Cadence Cap
**File:** `.claude/skills/hotcb-autopilot/SKILL.md:309-314`
**Problem:** Phase 3.6 cadence section has no wall-clock time limit. If training is slow (1 step/sec), "check every 100 steps" means 100 seconds of silence. Need max 60-second wall-clock cap.
**Fix:** Add wall-clock time limit to cadence rules.

### 1.3 Tooltip Position — Show on Side, Not Top
**File:** `src/hotcb/server/static/js/charts.js:312-314`
**Problem:** Chart.js tooltip appears above the cursor, blocking view of data points. Should appear to the side.
**Fix:** Register a custom Chart.js tooltip positioner that places tooltip to the right of the cursor.

### 1.4 Controls Don't Reflect State for Non-Demo Projects
**Files:** `src/hotcb/server/static/js/controls.js:579-627`, `src/hotcb/server/app.py:398-468`
**Problem:** `/api/train/status` returns empty config when training wasn't started via the dashboard's `TrainingLauncher`. External training (via adapters, `hotcb launch`, or direct kernel usage) writes `hotcb.run.json` and `hotcb.applied.jsonl` but the frontend only syncs from launcher status.
**Fix:** On dashboard load, call `/api/state/controls` to hydrate sliders from last applied opt/loss params and run config from `hotcb.run.json`. Add frontend init that reads this endpoint.

### 1.5 Mutation Capsules Don't Render for Non-Demo Runs
**File:** `src/hotcb/server/static/js/panels.js` (addTimelineItem function)
**Problem:** Applied JSONL records from external training use `payload` field for params, not `params`. The capsule rendering code only checks `rec.params`. Similarly, chart annotation code in `charts.js:76` only checks `rec.params`.
**Fix:** Check both `rec.params` and `rec.payload` in timeline rendering and chart annotations.

### 1.6 Autopilot Alert Tooltips
**File:** `src/hotcb/server/static/js/controls.js:795-845`
**Problem:** Autopilot action items show truncated `condition_met` (80 chars). No hover tooltip with full detail.
**Fix:** Add `title` attribute with full condition text, and show rule parameters on hover.

---

## 2. Frontend Issues (P1)

### 2.1 API Error Handling — Missing HTTP Status Check
**File:** `src/hotcb/server/static/js/utils.js:8-15`
**Problem:** `api()` function doesn't check `r.ok` before calling `r.json()`. 404/500 responses cause silent failures or JSON parse errors.
**Fix:** Add `if (!r.ok)` check, return structured error.

### 2.2 WebSocket Reconnection — No Backoff
**File:** `src/hotcb/server/static/js/websocket.js:16-25`
**Problem:** Reconnects every 3s with no exponential backoff, no max retries, no cleanup of old WS instance.
**Fix:** Add exponential backoff (3s → 6s → 12s → 30s cap), max 20 retries, cleanup old `S.ws`.

### 2.3 Event Listener Leaks in Metric Dropdown
**File:** `src/hotcb/server/static/js/charts.js:648-653`
**Problem:** `document.addEventListener('click', ...)` is re-registered on every `_renderMetricDropdown()` call. Old listeners reference stale `wrapper` closures.
**Fix:** Use a single delegated listener or remove old listener before adding new one.

### 2.4 Three.js Memory Leaks
**File:** `src/hotcb/server/static/js/manifold3d.js:70-131`
**Problem:** Scene children removed but geometry/material not disposed. GPU memory accumulates.
**Fix:** Call `.geometry.dispose()` and `.material.dispose()` before removing from scene.

### 2.5 Chart Tooltip Hardcoded Colors
**File:** `src/hotcb/server/static/js/charts.js:312-313`
**Problem:** Tooltip background/border colors are hardcoded midnight theme values. Don't update on theme switch.
**Fix:** Read from CSS variables on tooltip render, or update chart options in `setTheme()`.

### 2.6 Forecast Polling Floods
**File:** `src/hotcb/server/static/js/charts.js:824-829`
**Problem:** `fetchAllForecasts()` spawns one request per metric name. With 50+ metrics, this is 50+ concurrent requests every 5s.
**Fix:** Batch forecast API or limit concurrent requests to 5-10.

### 2.7 Interval Accumulation
**Files:** `src/hotcb/server/static/js/init.js:127,204`, `src/hotcb/server/static/js/controls.js:194,317`
**Problem:** `setInterval` calls not stored in variables or cleared on reset. Multiple intervals can accumulate.
**Fix:** Store interval IDs, clear on reset.

---

## 3. Backend Issues (P1)

### 3.1 Malformed JSON Crash in read_new_jsonl
**File:** `src/hotcb/util.py:58`
**Problem:** `json.loads(s)` without try-catch. Malformed JSONL lines crash the kernel's command loading.
**Fix:** Wrap in try-except, log warning, skip malformed lines.

### 3.2 JSONL Append Race Condition
**File:** `src/hotcb/util.py:66`
**Problem:** `append_jsonl()` has no file locking. Concurrent writes from dashboard API + training thread can interleave lines.
**Fix:** Add `fcntl.flock()` around write.

### 3.3 FreezeState Missing Mode Validation
**File:** `src/hotcb/freeze.py:22-40`
**Problem:** `FreezeState.load()` accepts any string for mode without validation.
**Fix:** Validate mode is in `{"off", "prod", "replay", "replay_adjusted"}`.

### 3.4 Duplicate Imports
**File:** `src/hotcb/modules/cb/controller.py:7,15-16`
**Problem:** `dataclasses` and `util` imported twice.
**Fix:** Remove duplicate imports.

### 3.5 Incomplete `all` Optional Dependencies
**File:** `pyproject.toml:38-43`
**Problem:** `all` extras missing `slack_sdk>=3.0` (notifications), `matplotlib>=3.5`, `pandas>=1.5` (bench).
**Fix:** Add missing deps to `all`.

---

## 4. Packaging Issues (P2)

### 4.1 Missing MANIFEST.in
**Problem:** No `MANIFEST.in` for sdist. `pyproject.toml` package-data globs may not work with all setuptools versions for source distributions.
**Fix:** Create `MANIFEST.in` with `recursive-include` for static files, guidelines, prompt YAML.

### 4.2 Missing py.typed Marker
**Problem:** No PEP 561 `py.typed` marker. Type checkers don't recognize hotcb as typed.
**Fix:** Create `src/hotcb/py.typed` (empty file).

---

## 5. Accessibility Issues (P2)

### 5.1 Focus Styles
**File:** `src/hotcb/server/static/css/dashboard.css`
**Problem:** Multiple inputs use `outline: none` without adequate replacement. Violates WCAG 2.4.7.
**Fix:** Use `outline: 2px solid var(--accent); outline-offset: 2px;` or `:focus-visible`.

### 5.2 Missing ARIA Labels
**File:** `src/hotcb/server/static/index.html`
**Problem:** Icon buttons (pin, close, tour) lack `aria-label`. Inputs lack associated `<label>` elements.
**Fix:** Add `aria-label` to icon buttons, associate labels with inputs.

---

## 6. Code Quality (P3)

### 6.1 Bare `except Exception: pass` Patterns
**Files:** Multiple (cli.py, kernel.py, config.py, recipe.py)
**Problem:** Silently swallows all errors including programming bugs.
**Fix:** Use specific exception types or log warnings.

### 6.2 API Response Format Inconsistency
**File:** `src/hotcb/server/api.py`
**Problem:** Responses use different schemas: `{status, command}`, `{status, path, config}`, `{valid, errors}`.
**Fix:** Standardize to `{status, data?, error?}` pattern (lower priority — breaking change for consumers).

### 6.3 Stale Docstring in app.py
**File:** `src/hotcb/server/app.py:8`
**Problem:** Says "React SPA (when built)" but frontend is vanilla JS, not React.
**Fix:** Update docstring.

---

## 7. Audit-Discovered Issues (P1)

### 7.1 Missing `/api/cb/set_params` Endpoint
**File:** `src/hotcb/server/api.py`
**Problem:** `controls.js:121` calls `POST /api/cb/set_params` for finetune backbone freeze toggle, but endpoint didn't exist.
**Fix:** Added `CbSetParamsRequest` model and `/cb/set_params` endpoint to api.py.

### 7.2 Autopilot History Not Reset on New Training Start
**File:** `src/hotcb/server/launcher.py` (start_training endpoint)
**Problem:** `/api/train/start` resets projection engine but not autopilot engine. Old autopilot actions persist across runs. `/api/train/reset` does reset autopilot, creating inconsistent behavior.
**Fix:** Added `autopilot.reset()` call to start_training endpoint.

### 7.3 Recipe Loading Broad Exception Catch
**File:** `src/hotcb/recipe.py:229`
**Problem:** `except Exception: self._entries = []` silently swallows all errors including malformed JSON, permission errors.
**Status:** Deferred — low-impact since recipe load failures are recoverable.

### 7.4 Adapter Imports Not Gated as Optional
**Files:** `src/hotcb/adapters/lightning.py:5`, `src/hotcb/adapters/hf.py:5`
**Problem:** Unconditional `import pytorch_lightning` / `from transformers import ...` at module level. If deps not installed, import fails.
**Status:** Deferred — by design (adapters are explicitly optional, users import them knowing the dep is needed).

---

## Status Tracker

### Fixed (this session)
- [x] 1.1 SKILL.md PYTHONPATH
- [x] 1.2 Skill cadence cap
- [x] 1.3 Tooltip position
- [x] 1.4 Controls state hydration
- [x] 1.5 Capsule rendering (payload/params)
- [x] 1.6 Autopilot alert tooltips
- [x] 2.1 API error handling
- [x] 2.2 WebSocket backoff
- [x] 3.1 Malformed JSON handling
- [x] 3.2 JSONL append locking
- [x] 3.3 FreezeState validation
- [x] 3.4 Duplicate imports
- [x] 3.5 Incomplete `all` deps
- [x] 4.1 MANIFEST.in
- [x] 4.2 py.typed marker
- [x] 6.3 Stale docstring
- [x] 7.1 Missing cb/set_params endpoint
- [x] 7.2 Autopilot reset on start

### Fixed (second pass)
- [x] 2.3 Event listener leaks in dropdown
- [x] 2.4 Three.js memory leaks
- [x] 2.5 Chart tooltip hardcoded colors
- [x] 2.6 Forecast polling floods
- [x] 2.7 Interval accumulation
- [x] 5.1 Focus styles
- [x] 5.2 ARIA labels
- [x] 7.3 Recipe loading exception handling

### Fixed (third pass — demo restructuring + integration cleanup)
- [x] NaN/inf crash: `sanitize_floats()` at all JSON serialization boundaries (WebSocket, JSONL, forecast API)
- [x] `[object Object]` in mutation timeline: `JSON.stringify()` for object-typed params in panels.js/charts.js
- [x] Dashboard startup buffer: chart waiting overlay with spinner until first metrics arrive
- [x] Staged knob highlights: visual indicator for changed-but-not-applied slider values
- [x] `sys` import missing in launch.py (caused NameError in test)
- [x] Kernel ledger now writes `params` alongside `payload` for consistent slider hydration
- [x] `app.py` `get_control_state()` reads `params`-first for slider sync
- [x] Demo restructuring: all 3 demos rewritten to use HotKernel + MetricsCollector + actuators (no hand-rolled JSONL)
- [x] INTEGRATION.md: fixed non-existent `mc.log()` API, updated Option B to show kernel-managed metrics
- [x] Legacy `examples/` directory deleted (broken v1 imports); `docs/examples/` has correct v2 patterns
- [x] `docs/concepts.md`: fixed `mc.log()` to show proper MetricsCollector + HotKernel pattern
- [x] `docs/custom_training_configs.md`: updated all framework examples to use HotKernel/adapters, removed `_write_jsonl`/`_read_commands` helpers

### Deferred (by design / breaking change)
- [ ] 6.1 Bare except patterns (multi-file, low risk — silently catches rare edge cases)
- [ ] 6.2 API response format consistency (breaking change for consumers)
- [ ] 7.4 Adapter optional import gating (by design — adapters are explicitly optional)
