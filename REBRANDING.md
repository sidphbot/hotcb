# Rebranding: hotcb -> TrainPilot

## Decision

**TrainPilot** is the clear winner. The product is a live training control plane with
autopilot, manual overrides, recipe systems, and real-time intervention — that is
literally what a pilot does. "TrainLens" undersells the control capabilities, and
"TrainForge" sounds like a model-building framework, not a live operations tool.
TrainPilot immediately communicates co-pilot semantics that ML engineers already
understand from GitHub Copilot and similar tools.

## Brand Identity

**Tagline:** Live copilot for PyTorch training runs.

**Positioning:** TrainPilot is the real-time control plane that watches your training,
intervenes when it matters, and lets you steer from a dashboard or let autopilot handle it.

## Migration Checklist

- [ ] PyPI package name: `hotcb` -> `trainpilot`
- [ ] CLI entrypoint: `hotcb` -> `trainpilot` (alias `tp` for power users)
- [ ] Python import namespace: `hotcb.*` -> `trainpilot.*`
- [ ] Package directory: `src/hotcb/` -> `src/trainpilot/`
- [ ] pyproject.toml: name, version (3.0.0), scripts, all references
- [ ] Runtime artifact filenames: `hotcb.*.jsonl` -> `trainpilot.*.jsonl`
- [ ] Config files: `hotcb.freeze.json`, `hotcb.adjust.yaml`, `hotcb.ui.json` -> `trainpilot.*`
- [ ] Tune artifacts: `hotcb.tune.*` -> `trainpilot.tune.*`
- [ ] Server default port: keep 8421 (no reason to change)
- [ ] All internal string references, log messages, error text
- [ ] README, docs, and docstrings
- [ ] GitHub repo name: `TorchHotSwapCallbacks` -> `trainpilot`
- [ ] GitHub topics, description, social preview image
- [ ] Test files: update all imports and artifact references
- [ ] Optional dependency groups: `trainpilot[dashboard]`, `trainpilot[all]`, etc.

## Timeline

| Phase | Scope | Duration |
|-------|-------|----------|
| 1. Internal rename | Source tree, imports, tests green | 1 week |
| 2. Compatibility shim | `hotcb` package becomes thin redirect to `trainpilot` | 2 days |
| 3. PyPI publish | `trainpilot` 3.0.0 + `hotcb` 2.1.0 (shim-only) | 1 day |
| 4. Public cutover | Docs, repo rename, announcements | 1 day |
| 5. Deprecation | `hotcb` shim prints warning, removed after 6 months | ongoing |

## Compatibility

During transition, `hotcb` will remain fully functional:

- `import hotcb` will re-export everything from `trainpilot` with a deprecation warning.
- The `hotcb` CLI command will proxy to `trainpilot` with a one-time migration notice.
- The `hotcb` PyPI package (v2.1.0+) will depend on `trainpilot` and act as a shim.
- All old artifact filenames (`hotcb.*.jsonl`) will be auto-detected and read seamlessly.
- After 6 months, `hotcb` 3.0.0 will be a tombstone package pointing users to `trainpilot`.
