# HotOps Phased Implementation Plan

**Date:** 2026-02-27
**Based on:** `docs/upgrade_plan.md` (Golden Plan)
**Starting point:** hotops src is ~95% implemented, zero tests, no adapters, no docs

---

## Current State Summary

The `src/hotops/` directory already contains working implementations of:
- HotKernel (ingestion, routing, freeze modes, replay, debounce/poll, ledger writing)
- HotOp data model + command-to-op conversion
- Module controllers: CallbackModule (with source capture), HotOptController, HotLossController
- RecipePlayer (step/event matching, step offset, cursor-based streaming)
- FreezeState management (prod/replay/replay_adjusted)
- Ledger writer (append-only JSONL)
- Unified CLI (`hotcb`) with cb/opt/loss/freeze/recipe subcommands
- YAML config parser for all modules
- Shared utilities (FileCursor, read_new_jsonl, etc.)

**What's missing:**
1. Recipe overlay/adjustment patch system (stubbed in `recipe.py`)
2. HotOps-level adapters (Lightning + HF)
3. Full test suite (~60+ tests across 7 categories)
4. Documentation
5. Verification of backward compatibility and edge cases

---

## Phase 1: Core Completion + Unit Tests (Session 1)

**Goal:** Fill the one code gap and establish a solid test foundation.

### 1A. Implement recipe overlay/adjustment system
- **File:** `src/hotops/recipe.py`
- Implement `_apply_overlay(ops, adjust_data)` method in RecipePlayer
- Support all 5 patch rule types from spec §11.1:
  - Replace params
  - Shift step
  - Drop entry
  - Insert entry
  - Bulk transform (scale factors)
- Matching criteria from spec §11.2 (module, op, id, step/range, event, payload key)
- Optional: write effective recipe snapshot (`hotops.recipe.effective.jsonl`)

### 1B. Unit tests — Core ingestion and routing (spec §19.2)
- JSONL tailing (incremental reads, cursor correctness)
- Debounce and poll interval behavior
- Op routing to correct module controllers
- Ledger record correctness (every op → one record, fields populated)

### 1C. Unit tests — Freeze modes (spec §19.3)
- Freeze prod: external ops ignored, ledger shows `ignored_freeze`
- Freeze replay: recipe ops applied, external ops ignored with `ignored_replay`
- Freeze replay adjusted: overlay patches applied correctly

### 1D. Unit tests — Recipe export + replay (spec §19.4)
- Export filters to `decision=applied` only, preserves seq order
- Replay matching at (step, event), order preserved for same step
- Policy strict vs best_effort behavior
- Step offset (+N shifts)

### 1E. Unit tests — Source capture/versioning (spec §19.5)
- Capture on python_file load (sha256, captured_path)
- Replay uses captured version
- Fallback when captured file missing

**Estimated scope:** ~500-700 lines of test code, ~100 lines of overlay implementation.
**Parallelism:** Overlay (1A) first, then tests (1B-1E) can be written in parallel by subagents.

---

## Phase 2: Module Tests + Adapters (Session 2)

**Goal:** Module-level tests and framework integration.

### 2A. Unit tests — hotopt controller (spec §19.6)
- Global lr update across param groups
- Group-specific lr
- Weight decay update
- Missing optimizer handling
- Auto-disable on error

### 2B. Unit tests — hotloss controller (spec §19.7)
- Mutate weights/toggles in mutable_state
- Mapping rules (distill_w → weights.distill, etc.)
- Missing mutable_state handling
- Error/auto-disable behavior

### 2C. HotOps Lightning adapter
- **New file:** `src/hotops/adapters/lightning.py`
- Build env with optimizer/scheduler references
- Build env with mutable_state hook (user-provided)
- Call `kernel.apply(env, events)` at safe points
- Pattern from existing `src/hotcb/adapters/lightning.py`

### 2D. HotOps HF adapter
- **New file:** `src/hotops/adapters/hf.py`
- Similar pattern from `src/hotcb/adapters/hf.py`
- Expose optimizer via env

### 2E. Integration tests — Lightning (spec §19.8)
- Tiny model, CPU trainer
- External command applied at step K → optimizer changed
- Ledger step matches
- Replay test: export recipe → new run → same change at same step

### 2F. Integration tests — HF (spec §19.9)
- Minimal Trainer with tiny model/dataset
- External command at global_step K
- Replay reproduces

**Estimated scope:** ~600-800 lines of test code, ~300 lines of adapter code.

---

## Phase 3: Robustness, CLI Polish, Docs (Session 3)

**Goal:** Harden, document, and prepare for release.

### 3A. Robustness tests (spec §19.10)
- JSONL with blank lines
- Partial line writes (simulated crash)
- Large burst (20k commands, max_lines cap behavior)

### 3B. CLI verification and polish
- Verify syntactic sugar routing (auto-detect opt vs loss from keys)
- Verify `hotcb init` creates correct directory layout
- `hotcb recipe export` end-to-end
- `hotcb recipe validate` (if not yet implemented)
- `hotcb recipe patch-template` (if not yet implemented)

### 3C. Backward compatibility verification
- hotcb standalone mode still works (existing tests pass)
- HotController can still be used directly without kernel
- Existing hotcb CLI unchanged

### 3D. Documentation
- `docs/concepts.md` — HotKernel, ops, ledger, recipe, freeze modes
- `docs/cli.md` — all commands + sugar rules
- `docs/replay.md` — exporting, replay modes, step offsets, adjusted overlays
- `docs/modules/hotcb.md`, `hotopt.md`, `hotloss.md`
- `docs/formats.md` — all JSONL/JSON/YAML schemas
- `docs/examples/` — Lightning, HF, bare torch, sample adjust overlay

### 3E. Package structure finalization
- Verify `pyproject.toml` supports separate installability narrative
- Entry points for module discovery (`hotops.modules` group)
- Verify `hotcb` standalone install still works

**Estimated scope:** ~300 lines of tests, ~200 lines of CLI, ~2000 lines of docs.

---

## Subagent Strategy Per Phase

### Phase 1 (this session or next)
```
Agent A — Implement overlay patch system in recipe.py
Agent B — Write core unit tests (ingestion, routing, ledger) [after A completes]
Agent C — Write freeze mode tests [after A completes]
Agent D — Write recipe/replay tests [after A completes]
Agent E — Write source capture tests [parallel with B-D]
```

### Phase 2
```
Agent A — Write hotopt unit tests
Agent B — Write hotloss unit tests
Agent C — Write Lightning adapter + integration tests
Agent D — Write HF adapter + integration tests
```

### Phase 3
```
Agent A — Robustness tests + CLI polish
Agent B — All documentation
Agent C — Backward compat verification + package finalization
```

---

## Risk Notes

1. **Integration tests need torch/lightning/transformers installed.** Verify dev environment has these before Phase 2.
2. **The overlay patch system (Phase 1A) is the only real code gap.** Everything else is testing/adapters/docs around working code.
3. **hotcb backward compat** is critical — any changes to hotcb internals for kernel integration must not break existing hotcb-only users.
4. **Recipe overlay is the most complex new code** — the matching criteria (§11.2) with step ranges, payload key existence, and nth occurrence matching need careful design.

---

## Decision: What to tackle first?

Recommended: **Start with Phase 1 now.** The overlay system is the only real code gap, and the core unit tests will validate the ~95% of code that's already written but untested.
