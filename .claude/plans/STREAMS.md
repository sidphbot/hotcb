# Streams

> **Protocol:** claim a stream (`status → active`, note your branch), work,
> update checkboxes + log, release when done (`→ done`) or paused (`→ planned`).
> New stream: add a section + row to table. Use `/stream` to browse/attach.
> Use `/stream branch <name>` to import an existing git branch as a stream.

| ID | Type | Pri | Status | Branch | Summary |
|----|------|-----|--------|--------|---------|
| v2-stabilization | chore | p0 | active | claude_skill | Post-2.0 stabilization: maintenance fixes, demo restructure, dashboard UX, docs, coordination |
| fix-error-handling | fix | p2 | planned | — | Replace bare `except: pass` with specific types + logging |
| fix-api-consistency | refactor | p3 | planned | — | Standardize REST responses to `{status, data?, error?}` |
| fix-adapter-imports | fix | p3 | planned | — | Gate lightning/hf imports with friendly ImportError |
| feature-test-coverage | test | p1 | planned | — | Integration tests for demos, launcher, dashboard E2E |
| docs-examples-refresh | docs | p2 | planned | — | Verify examples, add notebooks |
| chore-release-prep | chore | p1 | planned | — | sdist/wheel validation, changelog, PyPI publish |

Dependencies: `chore-release-prep` blocks on `v2-stabilization` + `feature-test-coverage`

---

## v2-stabilization
**Goal:** Full post-2.0 stabilization pass — audit, fix, restructure, document. This is the `claude_skill` branch.
**Branch:** `claude_skill` (diverged from `master`, 1 commit + large uncommitted working tree)
**Scope:**
- MAINTENANCE.md P0-P2 audit and fixes (28 items fixed)
- Demo restructuring to HotKernel integration path
- Dashboard UX (controls hydration, stale data, run dir backup)
- Docs cleanup (INTEGRATION.md, concepts.md, custom_training_configs.md)
- Claude Code skill (`.claude/skills/hotcb-autopilot/`)
- Multi-agent coordination system (`.claude/plans/`, `/stream` command)

**Done:**
- [x] MAINTENANCE.md audit — all P0 user-reported (1.1-1.6)
- [x] Frontend P1 fixes (API error handling, WS backoff, listener leaks, Three.js, tooltip colors, forecast polling, intervals)
- [x] Backend P1 fixes (malformed JSON, JSONL locking, FreezeState validation, duplicate imports, deps)
- [x] Packaging P2 (MANIFEST.in, py.typed)
- [x] Accessibility P2 (focus styles, ARIA labels)
- [x] Demo restructuring — 3 demos rewritten to HotKernel + MetricsCollector + actuators
- [x] Docs — fixed mc.log() refs, updated framework examples, deleted legacy examples/
- [x] NaN/inf, [object Object], chart waiting, staged knob highlights, sys import
- [x] Claude Code skill (SKILL.md with 5-phase autopilot protocol)
- [x] Dashboard slider sync from WS metrics, `_slidersInitialized`
- [x] Launcher run dir backup (`_backup_run_dir_if_needed`)
- [x] Launcher JSONL truncation on start (was skip-if-exists)
- [x] `weight_decay` added to demo metrics
- [x] Multi-agent coordination (STREAMS.md + /stream command)

**Remaining:**
- [ ] Manual verify: sliders sync on each demo config
- [ ] Manual verify: no stale timeline on restart
- [ ] Manual verify: backup dir created on re-run
- [ ] Commit all working tree changes
- [ ] PR to main

**Log:**
- 2026-03-12: MAINTENANCE.md audit, P0-P2 fixes, demo restructuring, docs cleanup
- 2026-03-13: Dashboard UX fixes (slider sync, stale data, backup). Plans system created. 754 tests pass.

---

## fix-error-handling
**Goal:** Replace silent `except Exception: pass` with specific types + `log.warning()`.
**Files:** `cli.py`, `kernel.py`, `recipe.py` (audit for others)
- [ ] Audit all bare-except sites
- [ ] Replace with specific exceptions + logging
- [ ] Verify tests pass

---

## fix-api-consistency
**Goal:** Unify REST responses to `{status, data?, error?}`. Breaking change — needs frontend + SKILL.md updates.
**Files:** `api.py`, `utils.js`, `controls.js`, `init.js`, `panels.js`, `SKILL.md`, `INTEGRATION.md`
- [ ] Catalog current response shapes
- [ ] Design envelope schema
- [ ] Update backend + frontend + docs

---

## fix-adapter-imports
**Goal:** Friendly error when `pytorch_lightning` / `transformers` not installed.
**Files:** `adapters/lightning.py`, `adapters/hf.py`
- [ ] Wrap imports in try/except with install instructions
- [ ] Add test for the friendly error message

---

## feature-test-coverage
**Goal:** Integration tests for demo→launcher→dashboard→stop cycle. Currently 754 unit tests, zero integration.
**Files:** `src/hotcb/tests/`
- [ ] Test demo functions: run 10 steps, check metrics JSONL fields
- [ ] Test launcher lifecycle: start → status → stop → reset
- [ ] Test run dir backup: existing data → start → verify backup
- [ ] Test `/api/state/controls` returns correct values
- [ ] Test WS initial data burst

---

## docs-examples-refresh
**Goal:** Verify `docs/examples/` match v2.0, add Jupyter notebooks.
**Files:** `docs/examples/*.py`
- [ ] Verify 3 existing examples run
- [ ] Create notebook using `launch()` API
- [ ] Create notebook for autopilot comparison

---

## chore-release-prep
**Goal:** PyPI 2.0.0 publish. Depends on `v2-stabilization` + `feature-test-coverage`.
- [ ] Merge fix branches to main
- [ ] Build sdist + wheel, verify static files included
- [ ] Test install in fresh venv, run `hotcb demo`
- [ ] Write CHANGELOG.md
- [ ] Tag + publish
