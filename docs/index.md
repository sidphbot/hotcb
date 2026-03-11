# hotcb

**Live Training Control Plane for PyTorch**

A control plane for modifying training behavior while your run is active — no restart, no lost progress. Every change is recorded, exportable, and replayable.

- Works with bare PyTorch, PyTorch Lightning, and HuggingFace Trainer
- **Live mutation**: swap callbacks, tune optimizer params, adjust loss weights mid-run
- **Dashboard**: real-time metric charts, command panel, recipe editor (`hotcb serve`)
- **Autopilot**: rule-based (plateau/divergence/overfitting detection) and AI-driven (LLM reads trends, proposes actions)
- **Programmatic API**: `launch()` starts training + dashboard + autopilot from notebooks/scripts
- **Applied ledger**: step-indexed JSONL record of every mutation; fully replayable
- **Freeze modes**: production lock, deterministic replay, replay with adjustments

## Quick links

- [Getting Started](getting-started.md) — install, integrate, launch
- [Concepts](concepts.md) — HotKernel, ops, ledger, dashboard, autopilot
- [CLI Reference](cli.md) — all commands including `serve`, `demo`, `launch`
- [File Formats](formats.md) — JSONL, JSON, and YAML schemas
- Modules: [cb](modules/cb.md) | [opt](modules/hotopt.md) | [loss](modules/hotloss.md) | [tune](modules/hottune.md)
