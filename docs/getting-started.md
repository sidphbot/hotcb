# Getting started

## Install

```bash
pip install hotcb                        # core only
pip install "hotcb[dashboard]"           # with dashboard server
pip install "hotcb[dashboard,ai]"        # with dashboard + AI autopilot
pip install "hotcb[all]"                 # everything
```

## Minimal Lightning usage

```python
from hotcb import HotKernel
from hotcb.adapters.lightning import HotCBLightning
import pytorch_lightning as pl

kernel = HotKernel(
    run_dir="runs/exp1",
    debounce_steps=5,
)

trainer = pl.Trainer(callbacks=[HotCBLightning(kernel)])
trainer.fit(model)
```

## Hot commands (from another terminal)

```bash
hotcb --dir runs/exp1 init
hotcb --dir runs/exp1 enable timing
hotcb --dir runs/exp1 set lr=5e-5
hotcb --dir runs/exp1 cb set_params timing every=10 window=200
```

## Launch the dashboard

```bash
hotcb serve --dir runs/exp1              # attach to existing run
hotcb demo                               # synthetic training + dashboard
hotcb demo --golden                      # multi-task demo with rich metrics
```

Open `http://localhost:8421` to view live charts, send commands, and control autopilot.

## Autopilot modes

```bash
# Rule-based autopilot
hotcb demo --autopilot suggest           # proposes actions in dashboard
hotcb demo --autopilot auto              # auto-applies corrective actions

# AI-driven autopilot (requires HOTCB_AI_KEY env var or hotcb[ai])
hotcb demo --autopilot ai_suggest        # LLM proposes, human reviews
hotcb demo --autopilot ai_auto           # LLM proposes and applies
hotcb demo --autopilot ai_suggest --key-metric val_loss
```

## Programmatic launch (notebooks / scripts)

```python
from hotcb.launch import launch

handle = launch(
    train_fn="my_module:train",          # or pass a callable
    autopilot="ai_suggest",
    key_metric="val_loss",
    max_steps=1000,
    serve=True,
)

# Monitor and control
handle.metrics()                         # read latest metrics
handle.latest_metrics()                  # flat dict of most recent step
handle.metric_history("loss")            # values for a single metric
handle.set_param(lr=0.0005)              # send optimizer command
handle.ai_status()                       # AI autopilot state
handle.wait()                            # block until done
handle.stop()                            # stop early
```

## One-command launch (CLI)

```bash
hotcb launch --config multitask --autopilot ai_suggest --key-metric val_loss
hotcb launch --config multitask --max-time 300 --autopilot ai_suggest  # 5-minute run
hotcb launch --train-fn my_module:train --autopilot ai_auto --ai-budget 2.0
```
