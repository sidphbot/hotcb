# Getting started

## Install

```bash
pip install hotcb
# or with extras:
pip install "hotcb[yaml,lightning,hf]"
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
hotcb --dir runs/exp1 cb set_params timing every=10 window=200
```
