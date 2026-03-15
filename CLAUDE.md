# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**hotcb** — a live training control plane for PyTorch. Lets you swap callbacks, tune optimizer params, adjust loss weights, and run online HPO while a model trains. Changes are recorded in a step-indexed JSONL ledger for replay.

## Build & Install

```bash
pip install -e ".[dev,all]"          # editable install with all extras + dev deps
pip install -e ".[dev,dashboard]"    # lighter: just dashboard extras
```

## Running Tests

```bash
pytest                              # run full suite (testpaths configured in pyproject.toml)
pytest src/hotcb/tests/test_kernel_core.py           # single file
pytest src/hotcb/tests/test_kernel_core.py::test_name -x  # single test, stop on first failure
pytest -k "hotopt"                  # keyword filter
```

pytest config is in `pyproject.toml` — defaults: `-q --disable-warnings --maxfail=1 --cov=src/hotcb`. Test paths: `src/hotcb/tests/` and `src/hotcb/tests/cb/`.

## Running the Dashboard & Demo

```bash
hotcb serve --dir runs/exp1         # start dashboard server (port 8421)
hotcb demo                          # synthetic training + dashboard
hotcb demo --golden                 # multi-task golden demo with rich metrics
```

## Architecture

### Core flow

1. **CLI/API** writes commands to `hotcb.commands.jsonl` in the run directory
2. **HotKernel** (`kernel.py`) tails the commands file each training step, parses into `HotOp` objects (`ops.py`), routes to the correct module, and writes results to `hotcb.applied.jsonl` via the ledger (`ledger.py`)
3. **Modules** execute the operations — each module owns one domain of control

The kernel and training process communicate through the filesystem (JSONL files), so the CLI/dashboard run in a separate process.

### Module system (`src/hotcb/modules/`)

| Module | Path | Controls |
|--------|------|----------|
| **cb** | `modules/cb/` | Callback load/unload/enable/disable/reconfigure. Has its own controller, loader, protocol, adapters |
| **tune** | `modules/tune/` | Online constrained HPO via Optuna (optional `hotcb[tune]`) |
| **opt/loss/custom** | Default stream → `MutableState` | All scalar parameter control (lr, weights, custom knobs) via unified actuator system |

### Key types

- **`HotOp`** (`ops.py`): Normalized operation dataclass — every command becomes one. Fields: `module`, `op`, `id`, `params`, `target`, etc.
- **`CallbackTarget`** (`ops.py`): Specifies a callback to load (kind, path, symbol).
- **`HotKernel`** (`kernel.py`): Central coordinator. Holds module instances, `MutableState`, optional `metrics_collector`. Called via `kernel.apply(env=..., events=...)` each training step. Ops for `cb`/`tune` route to their modules; all others (opt/loss/custom) go through the default stream to `MutableState`.
- **`HotcbActuator`** (`actuators/actuator.py`): Single controllable parameter — 1:1 mapping (param_key ↔ actuator). Has type (BOOL/FLOAT/INT/CHOICE/LOG_FLOAT/TUPLE), `apply_fn`, bounds, state machine (INIT→UNTOUCHED→UNVERIFIED→VERIFIED→DISABLED).
- **`MutableState`** (`actuators/state.py`): Container of `HotcbActuator` instances. Provides `apply()`, `initialize()`, `verify()`, `describe_all()`.
- **`FreezeState`** (`freeze.py`): Freeze mode manager (off/prod/replay/replay_adjusted).
- **`RecipePlayer`** (`recipe.py`): Deterministic replay of exported recipes.

### Actuator system (`src/hotcb/actuators/`)

Unified per-parameter actuator model. Convenience constructors:
- `optimizer_actuators(optimizer)` — creates lr, wd, betas actuators from a torch optimizer
- `loss_actuators(weights_dict)` — creates FLOAT actuators that mutate the original dict
- `mutable_state(actuators)` — wraps a list of `HotcbActuator` instances into a `MutableState`

Adapters auto-discover optimizer actuators from the framework (Lightning/HF). Users register custom actuators via `mutable_state()`.

### Dashboard config (`src/hotcb/server/config.py`)

`DashboardConfig` centralizes all tunables (poll intervals, history limits, chart settings, UI timers). Loaded from defaults → YAML → env vars → CLI. Served at `/api/config`, fetched once by frontend into `S.config`. Controls are generated dynamically from `MutableState.describe_all()` — no hardcoded slider HTML.

### Server / Dashboard (`src/hotcb/server/`)

FastAPI app (`app.py`) served via `hotcb serve`. Architecture:
- **`tailer.py`**: `JsonlTailer` polls JSONL files and pushes to WebSocket subscribers via `ConnectionManager`
- **`api.py`**: REST router — command endpoints append to `hotcb.commands.jsonl`
- **`projections.py`**, **`manifolds.py`**, **`autopilot.py`**, **`recipe_editor.py`**, **`notifications.py`**: Feature routers using **closure-based factory pattern** (not Request injection) — each has a `create_*_router(deps)` function
- **`ai_engine.py`**: `LLMAutopilotEngine` — LLM decision engine with `AIConfig`, `AIState`, cost tracking, multi-run state persistence (`hotcb.ai.state.json`)
- **`ai_prompts.py`**: `TrendCompressor`, `build_context()`, `parse_ai_response()`, `ACTION_SCHEMA` — prompt assembly and response parsing for AI autopilot
- **`launcher.py`**: Training launch/stop/reset from the dashboard
- Static frontend: `server/static/` — vanilla JS (charts.js, controls.js, panels.js, websocket.js, state.js, init.js)

### Demos (`src/hotcb/demo.py`, `golden_demo.py`, `finetune_demo.py`)

Synthetic training loops that use HotKernel + MetricsCollector + actuators — the same integration path as real projects. Demos use a lightweight `_OptProxy` (dict with `param_groups`) instead of a real torch optimizer. Recipe-driven changes are injected as commands to `hotcb.commands.jsonl` at scheduled steps (not freeze/replay mode), so interactive dashboard control works simultaneously.

### Launch API (`src/hotcb/launch.py`)

Programmatic API for starting training + dashboard + autopilot in one call. Returns `LaunchHandle` with methods for metrics access, live commands, and AI state inspection. Used by `hotcb launch` CLI and notebook workflows.

### Adapters (`src/hotcb/adapters/`)

Top-level adapters (`lightning.py`, `hf.py`) wrap HotKernel for PyTorch Lightning and HuggingFace Trainer. The `modules/cb/adapters/` has callback-specific framework adapters.

### Metrics (`src/hotcb/metrics/`)

- `collector.py`: `MetricsCollector` — writes `hotcb.metrics.jsonl`
- `features.py`: `FeatureCapture` — activation hook capture to `hotcb.features.jsonl`

### Bench (`src/hotcb/bench/`)

Synthetic benchmarks and CIFAR-10 autopilot evaluation. `tasks.py` defines tasks, `runner.py` runs them, `report.py` generates outputs, `eval_autopilot.py` compares baseline vs autopilot.

## Multi-Agent Coordination

`.claude/plans/STREAMS.md` is the shared roadmap for parallel Claude Code sessions.
One file, all streams. Use `/stream` to browse, attach, create, or release streams.
Claim a stream (`status → active`), update checkboxes + log as you work, release when done.

## Conventions

- **Filesystem as IPC**: Training ↔ dashboard communication is via JSONL files, not sockets or shared memory.
- **Factory pattern for FastAPI routers**: Server feature routers use `create_*_router()` closures to avoid `from __future__ import annotations` issues with FastAPI/Pydantic.
- **`CallbackTarget` lives in `hotcb.ops`**, not in `modules/cb/`.
- **No base class required for callbacks**: Duck-typed protocol — implement `handle(event, env)` and optionally `set_params(**kwargs)`.
- **Source layout**: `src/hotcb/` is the single package. All imports use `hotcb.*`.
- **Autopilot modes**: `off`, `suggest`, `auto` (rule-based); `ai_suggest`, `ai_auto` (LLM-driven). Rules act as sensor layer for AI modes.
- **AI autopilot uses OpenAI-compatible API**: configured via `HOTCB_AI_KEY` env var and `AIConfig`. Works with OpenAI, ollama, vLLM.


Always use skills /python-runtime-patterns /python-project-setup /python-dev-practices when working with this project.