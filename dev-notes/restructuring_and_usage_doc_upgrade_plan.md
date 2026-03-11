 Below is a **complete README \+ repo restructuring plan** for a new **HotOps** repo that:

* Presents HotOps as the umbrella ecosystem

* Lets users install **hotcb / hotopt / hotloss** independently *or* together

* Ships a **shared CLI (`hotcb`)** with module subcommands (and sugar)

* Preserves **hotcb** as “stable \+ already battle-tested”

* Makes packaging \+ docs \+ examples coherent

This is written as an implementable plan (agents can follow it directly).

---

# **HotOps Repo: README \+ Restructure Plan**

## **1\) Goals for the new repo \+ README**

### **What the README must achieve**

1. Explain **HotOps** as the product (control plane \+ kernel \+ modules).

2. Make installation options obvious:

   * “I only want hotcb”

   * “I only want hotopt”

   * “I only want hotloss”

   * “I want all of HotOps”

3. Show users how to **use** each module:

   * Quickstart commands

   * Minimal code snippet integration (Lightning / HF / bare torch)

4. Explain the **new files** and workflow:

   * run dir layout (commands, applied ledger, recipe, freeze)

   * replay/freeze semantics at a high level

5. Be explicit about stability:

   * hotcb stable

   * hotopt/hotloss impact training (and are recorded/replayable)

6. Avoid confusion:

   * single canonical CLI is `hotcb`

   * `hotcb` CLI may remain for backwards compatibility but README should steer to `hotcb` in the new repo context

---

## **2\) Repository structure (recommended)**

Use a **monorepo with multiple distributable packages**.

### `src/` per package (monorepo)**

hotops/  
 README.md  
 LICENSE  
 pyproject.toml                 \# workspace meta \+ tooling  
 packages/  
   hotops-core/  
     pyproject.toml  
     src/hotops\_core/...  
     tests/...  
     README.md (optional, short)  
   hotcb/  
     pyproject.toml  
     src/hotcb/...  
     tests/...  
     README.md (optional)  
   hotopt/  
     pyproject.toml  
     src/hotopt/...  
     tests/...  
   hotloss/  
     pyproject.toml  
     src/hotloss/...  
     tests/...  
   hotops/  
     pyproject.toml  
     src/hotops/...  
     tests/...  
 docs/  
 examples/  
 .github/workflows/

**Why this is good**

* Each package is independently installable from PyPI.

* Tests are isolated per package.

* You can still have one repo, one docs site, one examples folder.

---

## **3\) Packaging strategy (what users can install)**

You want users to install any combination.

### **Packages (distribution names)**

* `hotops-core` (kernel \+ shared CLI `hotcb` \+ recipe/ledger/freeze machinery)

* `hotcb` (stable callbacks module; can be used standalone or via hotops-core)

* `hotopt` (optimizer control)

* `hotloss` (loss control)

* `hotcb` (meta-package that depends on all three modules \+ core)

### **CLI ownership**

* The **canonical CLI `hotcb` ships with `hotops-core`** (so even if users only install `hotcb` \+ `hotops-core`, they have `hotcb cb ...`).

* `hotcb` may keep its legacy `hotcb` CLI for backward compatibility, but README will primarily teach `hotcb`.

### **Extras**

* `hotops-core[yaml]` enables YAML desired-state support (optional dependency like today).

---

## **4\) README plan (final sections \+ what they contain)**

Below is a suggested **final README structure** with specifics.

---

### **4.1 Header \+ positioning**

* Project name: **HotOps**

* Tagline: *A live control plane for ML training: callbacks (hotcb), optimizer control (hotopt), loss control (hotloss) — with ledger \+ replay.*

Include a 5–8 line “Why this exists” that clearly separates:

* instrumentation vs behavior mutation

* auditability / replay

---

### **4.2 “What you get” section (bullets)**

* ✅ hotcb (stable): enable/disable/load callbacks live

* ✅ hotopt: change LR/WD/clip mid-run, step-indexed

* ✅ hotloss: change loss weights/terms/ramps mid-run, step-indexed

* ✅ applied ledger (`hotcb.applied.jsonl`)

* ✅ recipe export \+ replay

* ✅ freeze modes: prod lock, replay, replay-adjusted

Keep it crisp but concrete.

---

### **4.3 Installation matrix (must be extremely explicit)**

A table (or bullet list) like:

**Install only callbacks**

pip install hotcb  
\# optional shared CLI:  
pip install hotops-core hotcb

**Install only optimizer control**

pip install hotops-core hotopt

**Install only loss control**

pip install hotops-core hotloss

**Install everything**

pip install hotops

**Optional YAML support**

pip install "hotops-core\[yaml\]"   \# enables hotcb.yaml desired-state

Also add: “If you install only `hotcb` you can use the legacy `hotcb` CLI; recommended is `hotcb`.”

---

### **4.4 Quickstart (the minimal happy path)**

Include the run dir init \+ training integration \+ live command example.

**Run directory**

hotops \--dir runs/exp1 init

**Training integration**  
 Provide 3 minimal snippets:

#### **Lightning**

from hotcb import HotKernel  
from hotcb.adapters.lightning import HotOpsCallback

kernel \= HotKernel(  
   run\_dir="runs/exp1",  
   debounce\_steps=10,  
)

trainer \= pl.Trainer(callbacks=\[HotOpsCallback(kernel)\])  
trainer.fit(model, datamodule=dm)

#### **HuggingFace Trainer**

kernel \= HotKernel(run\_dir="runs/exp1", debounce\_steps=10)  
trainer \= Trainer(..., callbacks=\[HotOpsHFCallback(kernel)\])  
trainer.train()

#### **Bare PyTorch**

kernel \= HotKernel(run\_dir="runs/exp1", debounce\_steps=10)

for step, batch in enumerate(dl):  
   ...  
   loss.backward()  
   opt.step()

   env \= {"framework":"torch", "phase":"train", "step":step, "optimizer":opt, "loss\_state":loss\_state}  
   kernel.apply(env, events=\["train\_step\_end"\])

(README shouldn’t be perfect API-wise, but should show where optimizer/loss\_state go.)

---

### **4.5 Live control examples (show the CLI ergonomics)**

This section is the “wow”.

**hotcb**

hotops \--dir runs/exp1 cb load feat\_viz \--file /tmp/feat\_viz.py \--symbol FeatureVizCallback \--enabled \--init every=50  
hotops \--dir runs/exp1 cb set feat\_viz every=10  
hotops \--dir runs/exp1 cb disable feat\_viz

**hotopt**

hotops \--dir runs/exp1 opt set lr=3e-5 weight\_decay=0.01  
hotops \--dir runs/exp1 opt set group=1 lr=1e-6

**hotloss**

hotops \--dir runs/exp1 loss set distill\_w=0.2 depth\_w=1.5  
hotops \--dir runs/exp1 loss set terms.aux\_depth=false  
hotops \--dir runs/exp1 loss set ramps.depth.warmup\_frac=0.2 ramps.depth.end=2.0

**Syntactic sugar** (optional)

hotops \--dir runs/exp1 enable timing  
hotops \--dir runs/exp1 set lr=2e-5  
hotops \--dir runs/exp1 set distill\_w=0.25

Include one sentence: “`hotops set` routes keys to `opt` or `loss` when unambiguous; otherwise use explicit subcommands.”

---

### **4.6 Run artifacts: commands vs applied ledger**

Explain the four core files (short but clear):

* `hotcb.commands.jsonl`: what you asked for

* `hotcb.applied.jsonl`: what actually happened (step-indexed)

* `hotcb.recipe.jsonl`: portable replay plan

* `hotcb.sources/`: captured callback source for deterministic replay

Users should immediately understand the difference.

---

### **4.7 Freeze modes (must be in README, but not super long)**

Give a small table:

* `freeze prod`: ignore external cb/opt/loss changes

* `freeze replay`: ignore external, replay recipe

* `freeze replay-adjusted`: replay recipe \+ apply overlay patches

Commands:

hotops \--dir runs/exp1 freeze \--mode prod  
hotops \--dir runs/exp1 freeze \--mode replay \--recipe runs/exp1/hotcb.recipe.jsonl  
hotops \--dir runs/exp2 freeze \--mode replay-adjusted \--recipe runs/exp1/hotcb.recipe.jsonl \--adjust runs/exp2/hotcb.adjust.yaml  
hotops \--dir runs/exp1 freeze \--mode off  
---

### **4.8 Recipe export \+ replay (high value)**

Show:

hotops \--dir runs/exp1 recipe export \--out runs/exp1/hotcb.recipe.jsonl

And mention:

* Export derives from applied ledger (truth).

* Replay uses step+event matching.

---

### **4.9 Module availability (important for partial installs)**

Explain how CLI behaves if a module isn’t installed:

* `hotcb cb ...` requires `hotcb`

* `hotcb opt ...` requires `hotopt`

* `hotcb loss ...` requires `hotloss`

If missing, CLI should print:

* “Module not installed. Install with: pip install hotopt” etc.

---

### **4.10 Compatibility section**

* `hotcb` can still be used standalone with existing scripts and legacy CLI.

* HotOps is “new repo” and the recommended unified workflow is `hotcb`.

---

### **4.11 Docs \+ Examples \+ Contributing**

* Link to docs pages (concepts, replay, formats)

* Mention examples folder

* Mention how to run tests

---

## **5\) README updates required in code & tooling**

### **5.1 New top-level README.md**

* Owned by the `hotcb` repo

* Should not be the same as old hotcb README (but can reuse concepts)

### **5.2 Package READMEs (optional but recommended)**

Inside each package folder, keep a short README:

* `packages/hotcb/README.md`: “hotcb module overview \+ link to main docs”

* `packages/hotopt/README.md`

* `packages/hotloss/README.md`

* `packages/hotops-core/README.md`

These are for PyPI package pages if you want them per-distribution.

### **5.3 Docs structure**

In docs:

* `index.md` uses the same story as top README but longer

* dedicated pages for:

  * formats (jsonl schemas)

  * replay & freeze modes

  * module specifics

---

## **6\) Restructuring work items (agent checklist)**

### **A) Repo layout migration**

* Create `packages/` monorepo layout

* Move current hotcb code into `packages/hotcb/src/hotcb`

* Adjust imports accordingly

### **B) Create hotops-core**

* Add `hotcb` CLI entrypoint here

* Add module discovery via entrypoints:

  * group: `hotops.modules`

  * each module registers a controller factory

### **C) Create hotops meta-package**

* `packages/hotops/pyproject.toml` depends on:

  * hotops-core

  * hotcb

  * hotopt

  * hotloss

### **D) Publishing approach**

* Publish each package separately to PyPI

* Publish hotops meta-package last

### **E) Backward compatibility**

* hotcb package still exposes:

  * `hotcb` CLI (legacy)

  * `HotController` API as before

* HotOps path introduces HotKernel and unified CLI

---

## **7\) README “copy deck” plan (what to actually write)**

If you want agents to implement exactly, hand them this outline:

1. Title, tagline, one-paragraph pitch

2. “Modules” section with quick bullets

3. Install section (4 install paths \+ YAML extra)

4. Quickstart:

   * init run dir

   * integrate into Lightning/HF/Bare torch (3 snippets)

5. Live control commands:

   * cb load/enable/set/disable

   * opt set LR/WD/group

   * loss set weights/toggles/ramps

6. Run artifacts explanation (commands vs applied ledger vs recipe vs sources)

7. Freeze modes \+ commands

8. Recipe export \+ replay \+ adjusted replay (very short)

9. Partial installs behavior

10. Links to docs/examples \+ testing \+ contribution note

---

## **8\) Suggested “new structure” callouts inside README**

Include a section:

### **Repo layout**

* `packages/hotops-core/` – HotKernel \+ CLI

* `packages/hotcb/` – callbacks (stable)

* `packages/hotopt/` – optimizer ops

* `packages/hotloss/` – loss ops

* `packages/hotops/` – meta-package

* `examples/` – runnable examples per framework

* `docs/` – detailed docs

Users love knowing where things live.

## **Handling Custom Optimizers and Custom Loss Weights (hotopt / hotloss)**

`hotopt` and `hotloss` can only modify what the training process exposes to the HotOps runtime `env`. This is deliberate: HotOps never monkeypatches the trainer internals, and applies changes only at safe boundaries.

The general rule:

* **hotopt** needs access to the optimizer (and optionally scheduler)

* **hotloss** needs access to a mutable loss configuration object (“loss\_state”) that your loss computation reads each step

You can provide these in one of two ways:

1. directly on `env` (`env["optimizer"]`, `env["loss_state"]`), or

2. via resolver functions (`env["resolve_optimizer"]`, `env["resolve_loss_state"]`) when direct access is awkward.

### **1\) PyTorch Lightning**

Lightning is great because you typically have one module that “owns” both optimizers and losses.

#### **Optimizers (hotopt)**

In the HotOps Lightning adapter, the recommended approach is to resolve the optimizer from the Trainer:

* For standard trainers with a single optimizer: `trainer.optimizers[0]`

* For multiple optimizers: expose either the full list or pick the one you want to control

**Recommended convention (single optimizer):**

* `env["optimizer"] = trainer.optimizers[0]`

* Optional: `env["scheduler"] = trainer.lr_scheduler_configs[0].scheduler` (when present)

If you have a non-standard Lightning setup (manual optimization, multiple optimizers, custom scheduler wiring), you can provide a resolver instead:

* `env["resolve_optimizer"] = lambda: ...`

This keeps HotOps generic while letting you do anything advanced inside Lightning.

#### **Custom loss weights (hotloss)**

The easiest, most reliable pattern is to store your loss weights/flags in a **mutable dict** on the LightningModule, and have your loss computation read from it.

Example convention:

* `self.loss_state = {"weights": {...}, "terms": {...}, "ramps": {...}}`

* Your `training_step` (or loss function) reads from `self.loss_state`

Then the adapter sets:

* `env["loss_state"] = pl_module.loss_state`

This ensures hotloss updates take effect immediately on the next step, without any trainer patching.

**Tip:** Prefer a single authoritative loss\_state dict that your loss code reads every step. Avoid copying it into local variables that never refresh.

---

### **2\) HuggingFace `transformers.Trainer`**

HF Trainer is more restrictive because the optimizer/scheduler are owned by the Trainer, and the training step is internal.

#### **Optimizers (hotopt)**

HF’s Trainer typically has `trainer.optimizer` and `trainer.lr_scheduler` after initialization.

Recommended convention for your HotOps HF adapter:

* `env["optimizer"] = trainer.optimizer` (when available)

* `env["scheduler"] = trainer.lr_scheduler` (optional)

If those aren’t available at the hook time (depends on HF version / initialization order), use a resolver:

* `env["resolve_optimizer"] = lambda: trainer.optimizer`

#### **Custom loss weights (hotloss)**

HF’s Trainer loss is commonly computed inside the model’s `forward()` or inside a custom `compute_loss()` override.

Recommended approach:

* Keep a mutable `loss_state` on the model instance, and have `compute_loss` / forward read from it.

Example:

* `model.loss_state = {...}`

* In `compute_loss`, use `model.loss_state["weights"]["distill"]` etc.

Then the adapter sets:

* `env["loss_state"] = model.loss_state`  
   or provides:

* `env["resolve_loss_state"] = lambda: model.loss_state`

This is the cleanest cross-version approach because it doesn’t require patching Trainer internals.

---

### **3\) Bare PyTorch (manual training loop)**

Bare torch is the simplest because you own everything.

#### **Optimizers (hotopt)**

Just pass the optimizer to env at your safe point:

* `env["optimizer"] = optimizer`

* Optional: `env["scheduler"] = scheduler`

If you have multiple optimizers:

* `env["optimizers"] = [opt1, opt2]` and use a stable naming/indexing convention

* Or: `env["optimizer"] = opt1` and control only one

(HotOps can support either, but README should recommend one clear convention.)

#### **Custom loss weights (hotloss)**

Use a mutable dict and read it inside your loss computation each step:

* `loss_state = {"weights": {...}, "terms": {...}, "ramps": {...}}`

* Compute loss using values from `loss_state`

Then at safe point:

* `env["loss_state"] = loss_state`

This gives you fully deterministic, step-indexed changes that can be exported to a recipe and replayed later.

---

### **Recommended `loss_state` structure (portable across frameworks)**

To keep configs consistent across Lightning/HF/bare torch, HotOps recommends a single shape:

loss\_state \= {  
 "weights": {  
   "distill": 0.2,  
   "depth": 1.5,  
 },  
 "terms": {  
   "aux\_depth": True,  
   "aux\_heatmap": False,  
 },  
 "ramps": {  
   "depth": {"type": "linear", "warmup\_frac": 0.2, "end": 2.0},  
 },  
}

Then hotloss CLI can map cleanly:

hotcb loss set distill\_w=0.2 depth\_w=1.5  
hotcb loss set terms.aux\_depth=false  
hotcb loss set ramps.depth.warmup\_frac=0.2 ramps.depth.end=2.0  
---

### **What HotOps will *not* do for you (by design)**

* It will not guess your optimizer or loss weights if you don’t expose them.

* It will not rewrite your training loop or patch Trainer internals.

* It will not mutate mid-backward.

This is what keeps HotOps safe, debuggable, and replayable.

## **Concrete snippets: exposing optimizer \+ loss\_state per framework**

### **1\) PyTorch Lightning**

**Pattern:** store a mutable `loss_state` on your `LightningModule`, and let the HotOps Lightning adapter pull the optimizer from `trainer.optimizers`.

#### **LightningModule with `loss_state`**

import pytorch\_lightning as pl  
import torch  
import torch.nn.functional as F

class MyModel(pl.LightningModule):  
   def \_\_init\_\_(self):  
       super().\_\_init\_\_()  
       self.net \= torch.nn.Linear(10, 1\)

       \# Mutable, runtime-updatable loss config (hotloss mutates this).  
       self.loss\_state \= {  
           "weights": {"main": 1.0, "aux": 0.2},  
           "terms": {"aux": True},  
           "ramps": {},  \# optional  
       }

   def forward(self, x):  
       return self.net(x)

   def training\_step(self, batch, batch\_idx):  
       x, y \= batch  
       pred \= self(x)  
       main \= F.mse\_loss(pred, y)

       w\_main \= float(self.loss\_state\["weights"\].get("main", 1.0))  
       loss \= w\_main \* main

       if self.loss\_state\["terms"\].get("aux", False):  
           aux \= (pred.abs().mean())  
           w\_aux \= float(self.loss\_state\["weights"\].get("aux", 0.0))  
           loss \= loss \+ w\_aux \* aux

       self.log("train\_loss", loss)  
       return loss

   def configure\_optimizers(self):  
       opt \= torch.optim.AdamW(self.parameters(), lr=3e-4, weight\_decay=0.01)  
       return opt

#### **HotOps Lightning adapter (minimal)**

This is the key: add optimizer \+ loss\_state to `env` before calling `kernel.apply`.

import pytorch\_lightning as pl  
from typing import Any, Dict, Optional, List

class HotOpsCallback(pl.Callback):  
   def \_\_init\_\_(self, kernel, train\_events: Optional\[List\[str\]\] \= None):  
       super().\_\_init\_\_()  
       self.kernel \= kernel  
       self.train\_events \= train\_events or \["train\_batch\_end"\]

   def on\_train\_batch\_end(  
       self,  
       trainer: pl.Trainer,  
       pl\_module: pl.LightningModule,  
       outputs: Any,  
       batch: Any,  
       batch\_idx: int,  
   ) \-\> None:  
       \# Optimizer exposure (hotopt)  
       optimizer \= None  
       try:  
           optimizer \= trainer.optimizers\[0\] if getattr(trainer, "optimizers", None) else None  
       except Exception:  
           optimizer \= None

       \# Optional scheduler exposure (if you want hotopt to nudge it)  
       scheduler \= None  
       try:  
           cfgs \= getattr(trainer, "lr\_scheduler\_configs", None) or \[\]  
           if cfgs:  
               scheduler \= cfgs\[0\].scheduler  
       except Exception:  
           scheduler \= None

       \# Loss exposure (hotloss)  
       loss\_state \= getattr(pl\_module, "loss\_state", None)

       env: Dict\[str, Any\] \= {  
           "framework": "lightning",  
           "phase": "train",  
           "step": int(getattr(trainer, "global\_step", 0)),  
           "epoch": int(getattr(trainer, "current\_epoch", 0)),  
           "trainer": trainer,  
           "model": pl\_module,  
           "outputs": outputs,  
           "batch": batch,  
           "batch\_idx": batch\_idx,  
           "optimizer": optimizer,  
           "scheduler": scheduler,  
           "loss\_state": loss\_state,  
           "log": lambda s: trainer.print(s),  
       }

       self.kernel.apply(env, events=self.train\_events)

**Usage:**

kernel \= HotKernel(run\_dir="runs/exp1", debounce\_steps=10)  
trainer \= pl.Trainer(callbacks=\[HotOpsCallback(kernel)\])  
trainer.fit(MyModel(), datamodule=dm)

Now you can:

hotops \--dir runs/exp1 opt set lr=1e-4 weight\_decay=0.02  
hotops \--dir runs/exp1 loss set weights.main=1.0 weights.aux=0.1 terms.aux=false  
---

### **2\) HuggingFace `transformers.Trainer`**

**Pattern:** put a mutable `loss_state` on the model, and compute loss using it. Expose `trainer.optimizer` / `trainer.lr_scheduler` to env in the adapter hook.

#### **Model with `loss_state` \+ custom `compute_loss`**

import torch  
from torch import nn  
from transformers import Trainer

class MyHFModel(nn.Module):  
   def \_\_init\_\_(self):  
       super().\_\_init\_\_()  
       self.net \= nn.Linear(10, 1\)

       \# Mutable loss config (hotloss mutates this)  
       self.loss\_state \= {  
           "weights": {"main": 1.0, "aux": 0.2},  
           "terms": {"aux": True},  
           "ramps": {},  
       }

   def forward(self, input\_ids=None, labels=None, \*\*kwargs):  
       x \= input\_ids.float()  
       pred \= self.net(x)  
       \# Return raw preds; loss computed in Trainer.compute\_loss override (recommended)  
       return {"pred": pred, "labels": labels}  
import torch.nn.functional as F  
from transformers import Trainer

class MyTrainer(Trainer):  
   def compute\_loss(self, model, inputs, return\_outputs=False):  
       out \= model(\*\*inputs)  
       pred \= out\["pred"\]  
       labels \= out\["labels"\]

       main \= F.mse\_loss(pred, labels)  
       w\_main \= float(model.loss\_state\["weights"\].get("main", 1.0))  
       loss \= w\_main \* main

       if model.loss\_state\["terms"\].get("aux", False):  
           aux \= pred.abs().mean()  
           w\_aux \= float(model.loss\_state\["weights"\].get("aux", 0.0))  
           loss \= loss \+ w\_aux \* aux

       return (loss, out) if return\_outputs else loss

#### **HotOps HF adapter (minimal)**

Hook at `on_step_end` (safe point) like your hotcb HF adapter does

hf

.

from typing import Any, Dict, Optional, List  
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

class HotOpsHFCallback(TrainerCallback):  
   def \_\_init\_\_(self, kernel, train\_events: Optional\[List\[str\]\] \= None):  
       self.kernel \= kernel  
       self.train\_events \= train\_events or \["train\_step\_end"\]

   def on\_step\_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, \*\*kwargs: Any):  
       trainer \= kwargs.get("trainer", None)  \# may or may not be present depending on HF version/setup  
       model \= kwargs.get("model", None)

       optimizer \= None  
       scheduler \= None  
       loss\_state \= None

       \# If you can pass trainer/model into kwargs when constructing callbacks, do it.  
       if trainer is not None:  
           optimizer \= getattr(trainer, "optimizer", None)  
           scheduler \= getattr(trainer, "lr\_scheduler", None)  
       if model is not None:  
           loss\_state \= getattr(model, "loss\_state", None)

       env: Dict\[str, Any\] \= {  
           "framework": "hf",  
           "phase": "train",  
           "step": int(getattr(state, "global\_step", 0)),  
           "epoch": float(getattr(state, "epoch", 0.0) or 0.0),  
           "args": args,  
           "state": state,  
           "control": control,  
           "optimizer": optimizer,  
           "scheduler": scheduler,  
           "loss\_state": loss\_state,  
           "log": print,  
       }

       self.kernel.apply(env, events=self.train\_events)  
       return control

**Important practical note for HF:** getting `trainer` and `model` into callback kwargs varies by HF versions and hooks. The robust pattern is:

* Construct the callback with explicit references:  
  * `HotOpsHFCallback(kernel, trainer=trainer, model=model)`  
    and store them on the callback instance.

So a more robust init pattern is:

class HotOpsHFCallback(TrainerCallback):  
   def \_\_init\_\_(self, kernel, trainer=None, model=None, train\_events=None):  
       self.kernel \= kernel  
       self.trainer \= trainer  
       self.model\_ref \= model  
       self.train\_events \= train\_events or \["train\_step\_end"\]

   def on\_step\_end(self, args, state, control, \*\*kwargs):  
       optimizer \= getattr(self.trainer, "optimizer", None) if self.trainer else None  
       scheduler \= getattr(self.trainer, "lr\_scheduler", None) if self.trainer else None  
       loss\_state \= getattr(self.model\_ref, "loss\_state", None) if self.model\_ref else None  
       ...

Then usage:

kernel \= HotKernel(run\_dir="runs/exp1", debounce\_steps=10)  
model \= MyHFModel()  
trainer \= MyTrainer(model=model, args=training\_args, train\_dataset=ds, ...)  
trainer.add\_callback(HotOpsHFCallback(kernel, trainer=trainer, model=model))  
trainer.train()  
---

### **3\) Bare PyTorch loop**

**Pattern:** pass optimizer and loss\_state directly into `env` right after your step, at your chosen safe point.

import torch  
import torch.nn.functional as F

model \= torch.nn.Linear(10, 1).to(device)  
optimizer \= torch.optim.AdamW(model.parameters(), lr=3e-4, weight\_decay=0.01)

loss\_state \= {  
   "weights": {"main": 1.0, "aux": 0.2},  
   "terms": {"aux": True},  
   "ramps": {},  
}

kernel \= HotKernel(run\_dir="runs/exp1", debounce\_steps=10)

global\_step \= 0  
for epoch in range(num\_epochs):  
   for batch in dl:  
       x, y \= batch  
       x \= x.to(device)  
       y \= y.to(device)

       pred \= model(x)  
       main \= F.mse\_loss(pred, y)

       loss \= float(loss\_state\["weights"\].get("main", 1.0)) \* main  
       if loss\_state\["terms"\].get("aux", False):  
           aux \= pred.abs().mean()  
           loss \= loss \+ float(loss\_state\["weights"\].get("aux", 0.0)) \* aux

       optimizer.zero\_grad(set\_to\_none=True)  
       loss.backward()  
       optimizer.step()

       \# Safe point (after optimizer.step)  
       env \= {  
           "framework": "torch",  
           "phase": "train",  
           "step": global\_step,  
           "epoch": epoch,  
           "model": model,  
           "optimizer": optimizer,  
           "loss\_state": loss\_state,  
           "loss": loss.detach(),  
           "log": print,  
       }  
       kernel.apply(env, events=\["train\_step\_end"\])

       global\_step \+= 1

Now run live:

hotops \--dir runs/exp1 opt set lr=1e-4  
hotops \--dir runs/exp1 loss set weights.aux=0.05 terms.aux=true  
---

### **Notes on conventions (so users don’t get confused)**

* **Prefer passing `optimizer` directly** on env when possible.  
* If you have multiple optimizers/schedulers, define your own stable convention:  
  * expose `env["optimizers"] = {...}` and make hotopt accept `name=` selectors, *or*  
  * expose only the primary optimizer as `env["optimizer"]` and keep it simple.  
* For loss, keep a single mutable `loss_state` dict and always read from it inside your loss computation each step.

# **Live CLI Examples**

Assume your training is running with:

hotops \--dir runs/exp1 init

and your Lightning / HF / Torch loop is calling:

kernel \= HotKernel(run\_dir="runs/exp1")  
---

## **🔍 hotcb — Live Callback Control**

### **Load a callback from a Python file**

hotops \--dir runs/exp1 cb load feat\_viz \\  
 \--file /tmp/feat\_viz.py \\  
 \--symbol FeatureVizCallback \\  
 \--enabled \\  
 \--init every=50

This:

* Dynamically loads `FeatureVizCallback`

* Captures its source into `runs/exp1/hotcb.sources/`

* Records the load in `hotcb.applied.jsonl`

---

### **Change callback parameters live**

hotops \--dir runs/exp1 cb set feat\_viz every=10  
---

### **Disable it**

hotops \--dir runs/exp1 cb disable feat\_viz  
---

### **Syntactic sugar**

hotops \--dir runs/exp1 enable timing  
hotops \--dir runs/exp1 disable timing

Default module is `cb` if not specified.

---

## **⚙️ hotopt — Live Optimizer Control**

### **Change global learning rate**

hotops \--dir runs/exp1 opt set lr=1e-4  
---

### **Change weight decay**

hotops \--dir runs/exp1 opt set weight\_decay=0.02  
---

### **Change only one parameter group**

hotops \--dir runs/exp1 opt set group=1 lr=1e-6  
---

### **Scale scheduler output**

hotops \--dir runs/exp1 opt set scheduler\_scale=0.5  
---

### **Sugar form**

If unambiguous:

hotops \--dir runs/exp1 set lr=5e-5  
---

## **🧮 hotloss — Live Loss Mutation**

### **Change scalar weights**

hotops \--dir runs/exp1 loss set weights.main=1.0 weights.aux=0.1

or shorthand:

hotops \--dir runs/exp1 loss set main\_w=1.0 aux\_w=0.1  
---

### **Toggle loss terms**

hotops \--dir runs/exp1 loss set terms.aux=false  
---

### **Modify ramp configuration**

hotops \--dir runs/exp1 loss set ramps.depth.warmup\_frac=0.2 ramps.depth.end=2.0  
---

### **Sugar form**

hotops \--dir runs/exp1 set distill\_w=0.25

(Automatically routes to `loss` when keys are clearly loss-related.)

---

# **📂 Inspecting What Happened**

## **Show current state**

hotops \--dir runs/exp1 status

Displays:

* freeze mode

* loaded callbacks

* current optimizer params

* current loss weights

---

## **Inspect applied ledger**

tail \-n 20 runs/exp1/hotcb.applied.jsonl

Each record includes:

* step

* event

* module

* decision (applied / ignored / failed)

* payload

This is the authoritative mutation timeline.

---

# **🔒 Freeze Modes**

## **Production lock (ignore external mutations)**

hotops \--dir runs/exp1 freeze \--mode prod

* External cb/opt/loss commands are ignored.

* Ledger records `ignored_freeze`.

Disable:

hotops \--dir runs/exp1 freeze \--mode off  
---

## **Replay a previous run**

First export recipe:

hotops \--dir runs/exp1 recipe export \--out runs/exp1/hotcb.recipe.jsonl

Then in a new run:

hotops \--dir runs/exp2 freeze \\  
 \--mode replay \\  
 \--recipe runs/exp1/hotcb.recipe.jsonl

* External commands ignored.

* Recipe applied deterministically at same step \+ event.

---

## **Replay with adjustments**

Create adjustment file:

version: 1  
patches:  
 \- match: {module: "opt", op: "set\_params", at\_step: 1200}  
   replace\_params: {lr: 2e-5}

 \- match: {module: "loss", id: "main"}  
   transform\_params:  
     scale:  
       distill\_w: 1.1

Then:

hotops \--dir runs/exp2 freeze \\  
 \--mode replay-adjusted \\  
 \--recipe runs/exp1/hotcb.recipe.jsonl \\  
 \--adjust runs/exp2/hotcb.adjust.yaml

The kernel:

* Builds effective recipe

* Applies modified steps

* Records patched entries in ledger

---

# **🔁 Deterministic Callback Replay**

If you loaded a callback from a file:

hotops \--dir runs/exp1 cb load feat\_viz \\  
 \--file /tmp/feat\_viz.py \\  
 \--symbol FeatureVizCallback

HotOps:

* Computes sha256

* Copies file into `hotcb.sources/`

* Records version in ledger

Replay mode will use the captured version — even if `/tmp/feat_viz.py` changes later.

---

# **🧪 Combined Example (Realistic Workflow)**

Training starts.

Mid-run:

hotops \--dir runs/exp1 opt set lr=2e-5  
hotops \--dir runs/exp1 loss set distill\_w=0.3  
hotops \--dir runs/exp1 cb load grad\_stats \--file ./grad\_stats.py \--symbol GradStatsCallback \--enabled

Later:

hotops \--dir runs/exp1 recipe export

New run:

hotops \--dir runs/exp2 freeze \--mode replay \--recipe runs/exp1/hotcb.recipe.jsonl

Result:

* Same optimizer and loss changes

* Same callbacks loaded

* Same step alignment

* Fully recorded in new ledger

---

# **🧠 Mental Model**

* `commands.jsonl` → what you *asked* for

* `applied.jsonl` → what *actually happened*

* `recipe.jsonl` → portable plan

* `freeze` → control who’s allowed to mutate training

* `sources/` → reproducible callback versions

