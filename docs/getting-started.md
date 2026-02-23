# Getting started

## Install

```bash
pip install hotcb
# or with extras:
pip install "hotcb[yaml,lightning,hf]"
```

Minimal Lightning usage
```python
from hotcb import HotController
from hotcb.adapters.lightning import HotCallbackController
import pytorch_lightning as pl

controller = HotController(
  config_path="runs/exp1/hotcb.yaml",
  commands_path="runs/exp1/hotcb.commands.jsonl",
  debounce_steps=5,
)

trainer = pl.Trainer(callbacks=[HotCallbackController(controller)])
trainer.fit(model)
Hot commands (from another terminal)
hotcb --dir runs/exp1 init
hotcb --dir runs/exp1 enable timing
hotcb --dir runs/exp1 set timing every=10 window=200
```
