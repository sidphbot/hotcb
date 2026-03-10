# hotcb 2.0 — Live Training Control Plane: Dashboard & Interactive System Plan

## Realism Assessment: The Paradigm Shift

**The core insight is sound.** Training today is "fire and forget" — you set hyperparameters, walk away, and check back hours/days later. The feedback loop is glacially slow. Making it interactive (3-10 interventions/hour instead of 1/3-hours) is genuinely valuable. Here's why:

**What makes this realistic:**
- You already have the hard part built — the kernel, actuators, freeze/replay, tune engine. Most "interactive training" projects die trying to build safe hot-mutation infrastructure. You have it.
- The JSONL streaming architecture is inherently dashboard-friendly — just tail the files.
- Recipe replay means every human intervention is reproducible, which is the killer feature for papers and production.
- XGBoost projections on loss curves are computationally cheap and surprisingly accurate for short-horizon forecasting.

**What to be honest about:**
- The LinkedIn traction issue isn't the concept — it's that callback hot-swap alone is a power-user niche. The dashboard + interactive tuning story is what makes it "paradigm shift" material. **Lead with the UI, not the plumbing.**
- At-scale (100B+ params, multi-node) the human-in-the-loop model breaks down — but your freeze/recipe modes already handle this correctly. Position it as: "interactive development → freeze recipe → production replay."
- Competition: W&B Sweeps, Optuna Dashboard, Ray Tune — but none of them do **live hot-swap with human override + replay**. That's your moat.

**Marketing pivot suggestion:** Don't call it "callback management." Call it something like **"hotcb: Live Training Cockpit"** or **"Interactive Training Control Plane."** The simulation/cockpit metaphor maps perfectly to your knobs + graphs + projections vision.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Training Process                     │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ Adapter   │→│ HotKernel │→│ Actuators/Modules  │  │
│  │(LT/HF)   │  │          │  │(opt/loss/cb/tune)  │  │
│  └──────────┘  └────┬─────┘  └───────────────────┘  │
│                     │ JSONL streams + metrics          │
│                     ▼                                  │
│  ┌──────────────────────────────────────────┐         │
│  │  MetricsCollector (new)                   │         │
│  │  - intercepts env metrics at each step    │         │
│  │  - writes hotcb.metrics.jsonl             │         │
│  │  - ring buffer for feature snapshots      │         │
│  └──────────────┬───────────────────────────┘         │
└─────────────────┼─────────────────────────────────────┘
                  │ filesystem (JSONL/YAML)
                  ▼
┌─────────────────────────────────────────────────────┐
│  hotcb-server (new, separate process)                │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ File Tailer  │  │ Projection   │  │ WebSocket  │ │
│  │ (metrics,    │  │ Engine       │  │ Server     │ │
│  │  applied,    │  │ (XGBoost,    │  │ (FastAPI)  │ │
│  │  mutations,  │  │  manifold,   │  │            │ │
│  │  segments)   │  │  feature-PCA)│  │            │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Notification │  │ Recipe       │  │ REST API   │ │
│  │ Engine       │  │ Editor       │  │ (commands, │ │
│  │ (email/slack)│  │ (trim/edit/  │  │  status)   │ │
│  │              │  │  replay)     │  │            │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP + WebSocket
                      ▼
┌─────────────────────────────────────────────────────┐
│  Dashboard (React SPA, served by hotcb-server)       │
│                                                       │
│  ┌─ Control Bar ──────────────────────────────────┐  │
│  │ [Mode: Engineer ▾] [Freeze ▾] [Notifications]  │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌─ Live Metrics Panel ───────────────────────────┐  │
│  │  streaming loss/metric charts (multi-run)       │  │
│  │  + projection overlays (dashed lines)           │  │
│  │  + intervention markers (vertical lines)        │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌─ Knob Panel ──────┐  ┌─ Projection Panel ──────┐ │
│  │ lr: [====●===] 3e-4│  │ XGBoost forecast        │ │
│  │ wd: [==●=====] 1e-2│  │ manifold plot           │ │
│  │ loss_w: [●===] 0.5 │  │ feature PCA (3D)        │ │
│  │ [Apply] [Schedule] │  │ [Lock metrics ▾]        │ │
│  └────────────────────┘  └─────────────────────────┘ │
│  ┌─ Mutation Timeline ─────────────────────────────┐ │
│  │ step 100: lr 3e-4→1e-3 ✓  step 200: wd +0.01 ✗│ │
│  │ [Edit Recipe] [Export] [Replay Preview]          │ │
│  └─────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────┘
```

**Key design decision:** The server is a **separate process** that communicates with the training process only through the filesystem (JSONL streams). This means:
- Zero coupling to training code — no new imports in training loop
- Works with any framework adapter (Lightning, HF, raw PyTorch)
- Dashboard can attach/detach without affecting training
- Multiple dashboards can observe the same run

---

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| **Server** | FastAPI + uvicorn | Async WebSocket native, lightweight, Python ecosystem |
| **File tailing** | watchdog + incremental JSONL read | Reuse existing `FileCursor` pattern from `util.py` |
| **Frontend** | React + TypeScript | Professional look, rich ecosystem, SSR not needed |
| **Charts** | Plotly.js (via react-plotly) | 3D support, streaming updates, publication-quality |
| **Knobs/Controls** | Custom React + headless UI | Sliders, toggles, scheduling modals |
| **Projections** | XGBoost (server-side) | Multivariate forecast, cheap, well-understood |
| **Manifolds** | UMAP/t-SNE (server-side) | For metric manifolds and feature space |
| **Feature extraction** | PyTorch hooks (opt-in) | Register forward hooks on selected layers |
| **Notifications** | slack_sdk + smtplib | Threshold-based alerts, anomaly projections |
| **Build** | Vite for frontend, bundled as static assets | `hotcb serve` serves the SPA |
| **State sync** | WebSocket (server→client), REST POST (client→server→JSONL) | Unidirectional data flow |

---

## Phase Plan

### Phase 1: Foundation — Metrics Streaming + Server Skeleton (Week 1-2)

**Goal:** Get a live dashboard showing streaming metrics from a running training.

**New modules:**
- `hotcb.server` — FastAPI app
  - `hotcb.server.app` — main app, mount static, WebSocket endpoints
  - `hotcb.server.tailer` — background task tailing JSONL files, pushing to WebSocket
  - `hotcb.server.api` — REST endpoints for commands, status, config
- `hotcb.metrics` — MetricsCollector
  - Hooks into kernel.apply() to capture `env["metric"]` values
  - Writes `hotcb.metrics.jsonl` with step, epoch, timestamp, metric_name, value
  - Configurable metric whitelist/blacklist

**CLI addition:**
```bash
hotcb serve --dir <run_dir> --port 8421 --host 0.0.0.0
```

**Frontend (minimal):**
- Single page with streaming line charts (Plotly)
- Multi-metric overlay (select which metrics to show)
- Intervention markers from `hotcb.applied.jsonl`
- Basic status bar (freeze mode, tune mode, active mutations)

**Kernel changes:**
- Add `MetricsCollector` to kernel (opt-in, zero overhead when unused)
- Emit structured metric events: `{"step": N, "metrics": {"train_loss": 0.5, "val_loss": 0.6, ...}}`

### Phase 2: Interactive Controls — Knobs + Commands (Week 2-3)

**Goal:** Control training from the dashboard.

**Server additions:**
- REST endpoints that write to `hotcb.commands.jsonl`:
  - `POST /api/opt/set` — `{lr: 0.001, wd: 0.01}`
  - `POST /api/loss/set` — `{recon_w: 0.5}`
  - `POST /api/tune/mode` — `{mode: "active"}`
  - `POST /api/cb/{id}/enable|disable`
  - `POST /api/freeze` — `{mode: "prod"}`
  - `POST /api/schedule` — `{at_step: 500, module: "opt", op: "set_params", params: {...}}`
- Validation endpoint: `POST /api/validate` — dry-run a mutation against actuator bounds

**Frontend additions:**
- **Knob panel:** Sliders for each actuator parameter with:
  - Current value (live from applied ledger)
  - Bounds from `actuator.describe_space()` / tune recipe
  - "Apply" button → writes command
  - "Schedule" button → deferred application at step N
- **Quick actions:** Enable/disable tune, freeze mode toggle
- **Command history:** Live feed of applied operations with status badges

### Phase 3: Projections — XGBoost + Metric Forecasting (Week 3-4)

**Goal:** Show where training is heading.

**Server additions:**
- `hotcb.server.projections` module:
  - **Univariate forecast:** XGBoost trained on recent N steps of a single metric, projects K steps ahead
  - **Multivariate forecast:** Given a proposed HP change, predict impact on all tracked metrics
    - Train XGBoost on (step, hp_values, metric_values) → next_metric_values
    - Show "what-if" overlays: "if you change lr to X, projected loss trajectory is..."
  - **Confidence bands:** Bootstrap or quantile regression for uncertainty
- WebSocket channel for projection updates (recomputed on new data or HP change preview)

**Frontend additions:**
- Dashed projection lines on metric charts with confidence bands
- "What-if" mode: drag a knob, see projected impact before committing
- "Lock metrics" selector: choose a set of metrics to project together
- Projection horizon slider (how far ahead to forecast)

### Phase 4: Manifolds + Feature Space (Week 4-5)

**Goal:** Visualize the loss landscape and feature space dynamics.

**Server additions:**
- `hotcb.server.manifolds`:
  - **Metric manifold:** UMAP/t-SNE on the vector of (all tracked metrics) across steps
    - Shows trajectory through metric space, colored by time
    - Intervention points highlighted
  - **Feature space projection (opt-in):**
    - Training process registers forward hooks on selected layers
    - Writes activation snapshots to `hotcb.features.bin` (memory-mapped, ring buffer)
    - Server reads snapshots, runs PCA→3D
    - Shows how representation space evolves

**Frontend additions:**
- 3D Plotly scatter for metric manifold (rotatable, zoomable)
- 3D feature space viewer (toggled on-demand to avoid overhead)
- Color coding: step progression, intervention markers, segment boundaries
- Side-by-side: metric manifold + loss curve, linked brushing

**Kernel changes:**
- Optional `FeatureCapture` hook:
  ```python
  kernel.enable_feature_capture(model, layer_names=["encoder.layer.4"], every_n_steps=50, max_samples=256)
  ```
- Writes compressed activations (PCA pre-reduced to 64 dims in-process to save I/O)

### Phase 5: Management — Notifications + Alerts (Week 5-6)

**Goal:** The dashboard works for you when you're away.

**Server additions:**
- `hotcb.server.notifications`:
  - **Threshold alerts:** "Notify me if val_loss > X" or "if projection shows divergence"
  - **Anomaly detection:** Z-score on recent metric windows, flag spikes
  - **Channels:** Slack webhook, email (SMTP), desktop notification (WebSocket push)
  - **Suggestion toggles:** "Pause and suggest" mode — when anomaly detected, pause tune and suggest human review
- **Scheduling:** Cron-like for recurring checks

**Frontend additions:**
- Notification panel with alert history
- Alert configuration UI (metric, threshold, channel, action)
- "Call for help" button → sends formatted Slack/email with current state snapshot + charts

### Phase 6: Recipe Editor + Replay Dashboard (Week 6-7)

**Goal:** Edit and replay training recipes with the same visual quality.

**Server additions:**
- `hotcb.server.recipe_editor`:
  - Load recipe JSONL, parse into timeline
  - CRUD operations on recipe entries (add, remove, modify, reorder)
  - Apply adjustment overlays (shift_step, replace_params, etc.)
  - Validate recipe against actuator bounds
  - Export edited recipe

**Frontend additions:**
- **Timeline editor:** Visual timeline of all recipe entries
  - Drag to reorder, click to edit params, right-click to delete
  - "Insert intervention" at any step
- **Replay preview:** Show what the recipe would do at each step (dry-run visualization)
- **Diff view:** Compare two recipes side-by-side
- **Replay dashboard:** Same streaming charts, but replaying a previous run's recipe
  - Overlay: original run metrics vs. replay run metrics

### Phase 7: Multi-Mode UI — Engineer / Education / Vibe-Coder (Week 7-8)

**Goal:** Three audience modes.

**Mode definitions:**
- **Engineer mode** (default): All knobs exposed, raw metric names, actuator details, full recipe editor, CLI integration
- **Education mode:** Simplified knobs with explanations ("Learning Rate: how fast the model learns"), tooltips, guided tutorials, "what does this do?" on every control, limited to safe mutations
- **Vibe-coder mode:** AI-assisted suggestions, natural language commands ("make it learn faster"), auto-bounds from tune recipe, simplified dashboard with just key metrics + a "health score"

**Implementation:**
- Mode stored in dashboard state + persisted to `hotcb.ui.json`
- Components conditionally render based on mode
- Education mode: wrap controls in `<Explainer>` components with tooltip text from a knowledge base
- Vibe-coder mode: add a chat/command bar that translates NL → hotcb commands (could use a local LLM or rule-based for v1)

### Phase 8: Self-Mode + Community Guidelines (Week 8-9)

**Goal:** Autonomous operation with guardrails.

**Server additions:**
- `hotcb.server.autopilot`:
  - **Rule engine:** Load community guidelines YAML (published best practices)
    ```yaml
    rules:
      - if: "val_loss plateau > 5 epochs"
        then: "reduce lr by 0.5x"
        confidence: high
      - if: "train_loss < val_loss * 0.5"
        then: "increase weight_decay by 2x"
        confidence: medium
    ```
  - **Action loop:** Monitor metrics → match rules → propose or auto-apply based on confidence
  - **Human-in-the-loop:** Low confidence → notify + wait for approval. High confidence → apply + notify.
  - **Community guideline sources:** Built-in defaults + user-contributed YAML files (future: community repo)

**Integration with tune module:**
- Self-mode uses the existing tune controller but with rule-based proposals instead of TPE
- Mutations go through the same safety checks (constraints, cooldowns, risk levels)

### Phase 9: Benchmarking + Paper Eval (Week 9-10)

**Goal:** Reproducible benchmarks for publication.

**Components:**
- `hotcb.bench` module:
  - **Benchmark suite:** Standard tasks (CIFAR-10, MNIST, synthetic) with defined HP search spaces
  - **Comparison modes:**
    - Baseline: fixed HP, no intervention
    - Auto-tune: hotcb tune in active mode (no human)
    - Human-interactive: hotcb with dashboard (track human decision times, quality)
    - Recipe replay: reproduce best interactive run
  - **Metrics collected:** final metric, time-to-target, human intervention count, compute cost
  - **Export:** LaTeX tables, matplotlib figures, raw CSV

**Production recipe-replay benchmark:**
- Compare: original training (hours of tuning) vs. recipe replay (deterministic, no search overhead)
- Show: same final quality, fraction of the compute

---

## Multi-Run Support

One thing your vision implies but isn't explicit: **multi-run comparison.**

- Dashboard should support attaching to multiple `run_dir`s simultaneously
- Overlay metrics from different runs (different HP configs, different recipes)
- Compare: "Run A (lr=1e-3) vs Run B (lr=1e-4)" live
- This is what makes the "simulation with knobs" metaphor really land

Implementation: server takes `--dirs run1,run2,run3` or discovers runs in a parent directory.

---

## Critical Path & Dependencies

```
Phase 1 (metrics + server) ← everything depends on this
  ├── Phase 2 (knobs) ← needs server + WebSocket
  ├── Phase 3 (projections) ← needs metrics stream
  │     └── Phase 5 (notifications) ← needs projections for anomaly
  ├── Phase 4 (manifolds) ← needs metrics, independent of knobs
  ├── Phase 6 (recipe editor) ← needs server, independent of projections
  └── Phase 7 (multi-mode) ← needs all UI components to exist
        └── Phase 8 (self-mode) ← needs multi-mode + projections
              └── Phase 9 (benchmarks) ← needs everything working
```

Phases 2, 3, 4, 6 can be parallelized after Phase 1.

---

## Package Structure (proposed)

```
src/hotcb/
├── ... (existing)
├── metrics/
│   ├── __init__.py
│   ├── collector.py      # MetricsCollector, hooks into kernel
│   └── features.py       # FeatureCapture (opt-in forward hooks)
├── server/
│   ├── __init__.py
│   ├── app.py            # FastAPI app, mount everything
│   ├── tailer.py         # Background JSONL tailers → WebSocket
│   ├── api.py            # REST endpoints (commands, status, config)
│   ├── projections.py    # XGBoost forecasting, what-if
│   ├── manifolds.py      # UMAP/t-SNE computation
│   ├── notifications.py  # Slack/email alerts
│   ├── recipe_editor.py  # Recipe CRUD + validation
│   ├── autopilot.py      # Self-mode rule engine
│   └── static/           # Built React SPA assets
├── dashboard/             # React source (separate build)
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── MetricsChart.tsx
│   │   │   ├── KnobPanel.tsx
│   │   │   ├── ProjectionOverlay.tsx
│   │   │   ├── ManifoldViewer.tsx
│   │   │   ├── RecipeTimeline.tsx
│   │   │   ├── NotificationPanel.tsx
│   │   │   └── ModeSelector.tsx
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts
│   │   │   └── useMetrics.ts
│   │   └── stores/
│   │       └── dashboardStore.ts  # zustand
│   └── vite.config.ts
└── bench/
    ├── __init__.py
    ├── tasks.py           # Benchmark task definitions
    ├── runner.py           # Benchmark execution
    └── report.py           # LaTeX/CSV export
```

**Optional deps update for pyproject.toml:**
```toml
[project.optional-dependencies]
tune = ["optuna>=3.0", "pyyaml>=6.0"]
dashboard = ["fastapi>=0.100", "uvicorn>=0.20", "websockets>=11.0", "xgboost>=1.7", "umap-learn>=0.5"]
bench = ["matplotlib>=3.5", "pandas>=1.5"]
all = ["hotcb[tune,dashboard,bench]"]
```

---

## What Would Make This a Paper

**Title idea:** *"From Passive to Active: Human-in-the-Loop Training Control with Live Hyperparameter Steering"*

**Key claims to benchmark:**
1. Human-interactive tuning reaches target metric in fewer GPU-hours than grid/random/Bayesian search alone
2. Recipe replay achieves deterministic reproduction with zero search overhead
3. XGBoost projections give actionable 80%+ accuracy on short-horizon metric forecasting
4. The combined human+auto system (self-mode with human override) outperforms either alone

**Eval plan:**
- Tasks: CIFAR-10 ResNet, GPT-2 small fine-tune, simple GAN
- Baselines: fixed HP, Optuna standalone, W&B Sweeps
- Conditions: hotcb auto-only, hotcb human-only, hotcb human+auto
- Metrics: time-to-target, final quality, total interventions, compute cost

---

## Summary

The vision is ambitious but **architecturally grounded** — the hard infrastructure (kernel, actuators, replay) already exists. The dashboard/server layer is a natural extension that reads the same JSONL streams your CLI already produces. The paradigm shift narrative is credible if you lead with the interactive experience rather than the plumbing.

**Immediate next step when ready to implement:** Phase 1 — MetricsCollector + FastAPI server + basic streaming charts. That alone is a demo-able product.
