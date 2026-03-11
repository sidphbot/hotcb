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
| **opt** | `modules/opt.py` | Live optimizer param changes (lr, weight_decay, clip) |
| **loss** | `modules/loss.py` | Loss weights, term toggles, ramp configs |
| **tune** | `modules/tune/` | Online constrained HPO via Optuna (optional `hotcb[tune]`) |

### Key types

- **`HotOp`** (`ops.py`): Normalized operation dataclass — every command becomes one. Fields: `module`, `op`, `id`, `params`, `target`, etc.
- **`CallbackTarget`** (`ops.py`): Specifies a callback to load (kind, path, symbol).
- **`HotKernel`** (`kernel.py`): Central coordinator. Holds module instances, actuator registry, optional `metrics_collector`. Called via `kernel.apply(env=..., events=...)` each training step.
- **`FreezeState`** (`freeze.py`): Freeze mode manager (off/prod/replay/replay_adjusted).
- **`RecipePlayer`** (`recipe.py`): Deterministic replay of exported recipes.

### Actuator system (`src/hotcb/actuators/`)

Protocol-based (`BaseActuator`) — optimizer and loss_state actuators register with the kernel and are auto-propagated to the tune controller.

### Server / Dashboard (`src/hotcb/server/`)

FastAPI app (`app.py`) served via `hotcb serve`. Architecture:
- **`tailer.py`**: `JsonlTailer` polls JSONL files and pushes to WebSocket subscribers via `ConnectionManager`
- **`api.py`**: REST router — command endpoints append to `hotcb.commands.jsonl`
- **`projections.py`**, **`manifolds.py`**, **`autopilot.py`**, **`recipe_editor.py`**, **`notifications.py`**: Feature routers using **closure-based factory pattern** (not Request injection) — each has a `create_*_router(deps)` function
- **`ai_engine.py`**: `LLMAutopilotEngine` — LLM decision engine with `AIConfig`, `AIState`, cost tracking, multi-run state persistence (`hotcb.ai.state.json`)
- **`ai_prompts.py`**: `TrendCompressor`, `build_context()`, `parse_ai_response()`, `ACTION_SCHEMA` — prompt assembly and response parsing for AI autopilot
- **`launcher.py`**: Training launch/stop/reset from the dashboard
- Static frontend: `server/static/` — vanilla JS (charts.js, controls.js, panels.js, websocket.js, state.js, init.js)

### Launch API (`src/hotcb/launch.py`)

Programmatic API for starting training + dashboard + autopilot in one call. Returns `LaunchHandle` with methods for metrics access, live commands, and AI state inspection. Used by `hotcb launch` CLI and notebook workflows.

### Adapters (`src/hotcb/adapters/`)

Top-level adapters (`lightning.py`, `hf.py`) wrap HotKernel for PyTorch Lightning and HuggingFace Trainer. The `modules/cb/adapters/` has callback-specific framework adapters.

### Metrics (`src/hotcb/metrics/`)

- `collector.py`: `MetricsCollector` — writes `hotcb.metrics.jsonl`
- `features.py`: `FeatureCapture` — activation hook capture to `hotcb.features.jsonl`

### Bench (`src/hotcb/bench/`)

Synthetic benchmarks and CIFAR-10 autopilot evaluation. `tasks.py` defines tasks, `runner.py` runs them, `report.py` generates outputs, `eval_autopilot.py` compares baseline vs autopilot.

## Conventions

- **Filesystem as IPC**: Training ↔ dashboard communication is via JSONL files, not sockets or shared memory.
- **Factory pattern for FastAPI routers**: Server feature routers use `create_*_router()` closures to avoid `from __future__ import annotations` issues with FastAPI/Pydantic.
- **`CallbackTarget` lives in `hotcb.ops`**, not in `modules/cb/`.
- **No base class required for callbacks**: Duck-typed protocol — implement `handle(event, env)` and optionally `set_params(**kwargs)`.
- **Source layout**: `src/hotcb/` is the single package. All imports use `hotcb.*`.
- **Autopilot modes**: `off`, `suggest`, `auto` (rule-based); `ai_suggest`, `ai_auto` (LLM-driven). Rules act as sensor layer for AI modes.
- **AI autopilot uses OpenAI-compatible API**: configured via `HOTCB_AI_KEY` env var and `AIConfig`. Works with OpenAI, ollama, vLLM.
