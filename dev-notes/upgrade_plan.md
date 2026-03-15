# **HotOps Golden Plan**

## **Full Ecosystem Spec \+ Upgrade Plan (from current hotcb standalone)**

**Date:** 2026-02-27 (Europe/Berlin)  
**Status:** Canonical (“freeze”) document for implementation delegation

---

## **0\) Context: Starting Point (Current State)**

You currently have **hotcb** implemented as a standalone module with:

* **Control plane (imperative):** CLI appends JSON objects to an append-only `commands.jsonl`  
   util  
* **Control plane (declarative):** optional YAML desired-state reconciliation (`hotcb.yaml`) that generates idempotent ops (load → set\_params → enable/disable)  
   config  
* **HotController:** polls config/commands at safe points via `apply(env, events)` and dispatches to callbacks; has debounce and poll interval; has auto-disable on callback failure  
   controller  
* **Op dataclass:** `load/enable/disable/set_params/unload`  
   ops  
* **Callback protocol:** `handle(event, env)` and `set_params(**kwargs)` (plus optional `on_attach`, `close`) with a framework-agnostic `env` contract  
   protocol  
* **Dynamic loading:** callback can be loaded from module path or Python file path  
   loader  
* **Adapters:** Lightning adapter and HF adapter build `env` and call controller at safe points  
   lightning

   hf  
* **FileCursor \+ incremental JSONL tail:** `read_new_jsonl(FileCursor)` reads only appended records and returns updated cursor  
   util

This is the **foundation**.

---

## **1\) Why HotOps Exists (New Requirements)**

hotcb mostly affects **instrumentation**. It does not “change training behavior” beyond overhead.

HotOps introduces two new modules that **mutate training**:

* **hotopt:** live optimizer/scheduler/clipping control  
* **hotloss:** live loss weighting/term toggles/ramp controls

Once training is mutated live, we must add:

1. **Step-indexed change recording** (what changed *at which step/event*)  
2. **Deterministic replay** of those changes in future runs  
3. **Freeze modes** to guarantee production integrity  
4. **Replay-adjusted** mode (replay with systematic edits)  
5. **Version capture** for dynamic callback file loads to make replay reproducible

All of this must be integrated under a **single shared CLI** and a shared “kernel” that abstracts the common logic.

---

## **2\) Target Product: “HotOps Ecosystem”**

### **2.1 Components**

* **hotops-core** (new)  
  * HotKernel runtime (control plane ingestion, routing, freeze/replay, ledger/recipe)  
  * Shared CLI `hotcb`  
  * Common data schemas and utilities  
* **hotcb** (existing, stable)  
  * Becomes a module plug-in under HotKernel  
  * Must remain backward compatible as standalone (important)  
* **hotopt** (new module)  
* **hotloss** (new module)  
* **hotops** (meta-package)  
  * Installs core \+ all modules (hotcb \+ hotopt \+ hotloss)

### **2.2 Primary Design Principle (carried from hotcb)**

* **No trainer monkeypatching**  
* **Safe-point updates only**: apply changes only at stable boundaries (step end, batch end, eval end) as hotcb already does  
   controller  
* **File-based control plane** (works everywhere): JSONL append-only stream (already proven)  
   util  
* **Idempotent ops**: load \+ set\_params \+ enable/disable pattern from YAML reconciliation  
   config  
* **Failure isolation**: exceptions do not kill training; module/callback can be auto-disabled  
   controller

---

## **3\) Canonical Run Directory Layout (HotOps)**

All modes and modules share a single run dir.

runs/\<run\_id\>/  
 hotcb.yaml                    \# optional desired-state config (YAML)  
 hotcb.commands.jsonl          \# external, append-only command stream (all modules)  
 hotcb.applied.jsonl           \# authoritative applied ledger (written by training process)  
 hotcb.recipe.jsonl            \# exported recipe (portable step-indexed plan)  
 hotcb.adjust.yaml|json        \# optional overlay/patch rules for replay-adjusted  
 hotcb.freeze.json             \# freeze state file written by CLI  
 hotcb.sources/                \# captured source versions for python\_file loads  
 hotops.log                     \# optional kernel log sink

**Important:** `hotcb.applied.jsonl` and `hotcb.sources/` are the key new artifacts that enable determinism.

---

## **4\) The Three Streams (Critical Concept)**

### **4.1 External Commands Stream (requested ops)**

* File: `hotcb.commands.jsonl`  
* Written by CLI (and optionally humans)  
* Append-only  
* Records intent, not truth  
* Does **not** know step number at write-time (CLI is external)

### **4.2 Applied Ledger Stream (ground truth timeline)**

* File: `hotcb.applied.jsonl`  
* Written only by training process (HotKernel)  
* Every processed op produces a ledger entry with:  
  * step, event, phase  
  * source (external/replay/yaml)  
  * decision (applied/ignored/failed/noop)  
  * payload  
  * error/traceback if failure  
* This is the canonical record for replay/export/debug

### **4.3 Recipe Stream (portable replay plan)**

* File: `hotcb.recipe.jsonl`  
* Exported from ledger (filter \+ normalize)  
* Used by replay modes  
* Can be edited manually (advanced users), but primary source is export

**Rule:** **Replay must be based on the applied ledger → recipe, not the external command stream**, because the external stream may include ops ignored due to freeze, bad params, etc.

---

## **5\) HotKernel: Shared Abstraction and Responsibilities**

HotKernel is the generalized “controller” that replaces duplicate logic currently inside hotcb’s `HotController` for polling and file cursor management.

HotKernel responsibilities:

1. **Ingest control plane**  
   * Tail `hotcb.commands.jsonl` using an incremental byte cursor (like hotcb’s FileCursor approach)  
      util  
   * Optionally reconcile `hotcb.yaml` on mtime changes (like hotcb YAML reconciliation)  
      config  
   * Read freeze state from `hotcb.freeze.json` (fast mtime check)  
2. **Route ops by module**  
   * `module="cb"` → hotcb module controller  
   * `module="opt"` → hotopt controller  
   * `module="loss"` → hotloss controller  
   * `module="core"` → kernel-level ops (freeze/unfreeze, snapshot, recipe export triggers, etc.)  
3. **Safe-point apply**  
   * `kernel.apply(env, events=[...])`  
   * Called at safe points by framework adapters (Lightning/HF/bare torch)  
4. **Freeze modes enforcement**  
   * Decide per op whether to apply/ignore based on freeze/replay mode  
5. **Ledger writing**  
   * Write a ledger record for every processed op (applied/ignored/failed/etc.)  
6. **Recipe replay**  
   * In replay modes, inject ops from recipe at matching (step,event)  
7. **Source capture/versioning for python\_file loads**  
   * When `cb load` uses `python_file`, capture source version so replay is deterministic  
8. **Unified status**  
   * Merge status from all modules \+ kernel state  
9. **Failure isolation**  
   * If module fails applying an op, handle per policy:  
     * record failure  
     * optional auto-disable the module/handle  
     * continue training

---

## **6\) Shared Data Model: HotOps Op Schema**

### **6.1 External command record schema (written to hotcb.commands.jsonl)**

All external records MUST include:

* `module`: `"cb" | "opt" | "loss" | "core"`  
* `op`: operation name  
* `id`: identifier within module (required for all but some core ops)  
* optional payload depending on op

Examples:

**Callback enable/disable:**

{"module":"cb","op":"enable","id":"timing"}  
{"module":"cb","op":"disable","id":"feat\_viz"}

**Callback load (like hotcb today)**

controller

loader

**:**

{"module":"cb","op":"load","id":"feat\_viz",  
"target":{"kind":"python\_file","path":"/tmp/feat\_viz.py","symbol":"FeatureVizCallback"},  
"init":{"every":50},  
"enabled":true}

**Optimizer params:**

{"module":"opt","op":"set\_params","id":"main","params":{"lr":3e-5,"weight\_decay":0.01}}

**Loss params:**

{"module":"loss","op":"set\_params","id":"main","params":{"distill\_w":0.2,"depth\_w":1.5}}

**Freeze mode:**

{"module":"core","op":"freeze","mode":"prod"}

Note: We keep `op` as a string consistent with hotcb’s `Op`

ops

but extend with `module` and a few `core` ops.

### **6.2 Kernel-internal normalized op**

HotKernel should normalize external JSON into an internal `HotOp` structure that includes:

* module  
* op  
* id  
* params / target / init / enabled / mode / recipe\_path / adjust\_path, etc.  
* source (`external` | `yaml` | `replay`)  
* raw\_record (optional) for debugging

---

## **7\) Applied Ledger Schema (hotcb.applied.jsonl)**

Every processed op creates a ledger entry.

Minimum fields:

* `seq`: monotonically increasing int sequence number (kernel-side)  
* `wall_time`: float epoch time  
* `step`: int env step  
* `epoch`: optional env epoch  
* `event`: the current event being processed when op considered  
* `phase`: env phase if present  
* `module`, `op`, `id`  
* `source`: `external | yaml | replay`  
* `decision`:  
  * `applied`  
  * `ignored_freeze`  
  * `ignored_replay` (external op ignored because replay mode is on)  
  * `skipped_noop` (idempotent/no state change)  
  * `failed`  
* `payload`: the relevant parts (params/target/init/enabled/etc.)  
* `error`: string if failed  
* `traceback`: optional string if failed  
* `notes`: optional

Example:

{  
 "seq": 431,  
 "wall\_time": 1762154312.55,  
 "step": 9800,  
 "epoch": 3.0,  
 "event": "train\_step\_end",  
 "phase": "train",  
 "module": "opt",  
 "op": "set\_params",  
 "id": "main",  
 "source": "external",  
 "decision": "applied",  
 "payload": {"params":{"lr":3e-5}},  
 "error": null  
}

**Rule:** Ledger is append-only and is the authoritative mutation timeline.

---

## **8\) Recipe Schema (hotcb.recipe.jsonl)**

Recipes are step-indexed “apply this at step X on event Y” directives.

Each recipe entry:

* `at.step`: int  
* `at.event`: string  
* `module`, `op`, `id`  
* payload: params/target/init/enabled/etc.  
* optional `constraints`: e.g., `min_kernel_version`, `requires_modules`, etc.  
* optional `source_capture`: for python\_file callback loads

Example:

{  
 "at": {"step": 300, "event": "train\_step\_end"},  
 "module":"cb",  
 "op":"load",  
 "id":"feat\_viz",  
 "target":{"kind":"python\_file","path":"/tmp/feat\_viz.py","symbol":"FeatureVizCallback"},  
 "init":{"every":50},  
 "enabled":true,  
 "source\_capture":{"sha256":"...","captured\_path":"hotcb.sources/\<sha\>.py"}  
}

**Recipe export logic (default):**

* Derived from `hotcb.applied.jsonl`  
* Include only entries where `decision == "applied"`  
* Preserve order by `seq`  
* Normalize `at.event` to the event recorded in ledger

---

## **9\) Replay Engine (RecipePlayer)**

### **9.1 Purpose**

Reapply the exact same sequence of applied changes, at the same relative “safe points,” to reproduce runs or compare behavior.

### **9.2 Matching semantics**

Recipe entry triggers when:

* current `env.step` equals `entry.at.step`, AND  
* the currently-dispatched event equals `entry.at.event`

This is why recipe entries include event names (Lightning vs HF differences).

lightning

hf

### **9.3 Ordering semantics**

If multiple entries share the same (step,event), apply them in recipe order (which is ledger `seq` order).

### **9.4 Replay policies (must support all)**

* `best_effort` (default):  
  * If step not reached (short run), leftover entries are marked “missed” at end (write to ledger as `decision=failed` with reason `missed_step` or `missed_event` OR separate “replay\_summary” record)  
* `strict`:  
  * If any scheduled entry isn’t applied by end of run → raise (or mark run failed) depending on config  
* `step_offset`:  
  * Apply recipe with global step shift (+/- N)  
  * Used for aligning runs with different warmup phases

### **9.5 Replay visibility**

* All replay-injected ops must be written to ledger with `source="replay"` and `decision=applied/failed`.

---

## **10\) The Three Freeze Modes (Fully Defined)**

Freeze modes are kernel-level state enforced uniformly across `cb`, `opt`, `loss`.

Freeze state is stored in `hotcb.freeze.json` and/or passed into kernel configuration.

### **10.1 Mode A: `freeze` (Production Lock)**

**Definition:**  
Ban external changes for `cb`, `opt`, `loss`. Training proceeds without hot modifications from outside.

**Behavior:**

* Kernel continues to read `hotcb.commands.jsonl` (so we don’t lose audit trail)  
* For each external op targeting module in `{cb,opt,loss}`:  
  * ignore it  
  * write ledger entry:  
    * `decision="ignored_freeze"`  
    * `source="external"`  
* Allowlist `core` ops may still apply:  
  * `status`  
  * `snapshot`  
  * `export_recipe` (optional)  
  * `unfreeze` (must always work)

**Rationale:**  
Production run must not be “poked” live.

### **10.2 Mode B: `freeze replay`**

**Definition:**  
Ban external changes AND replay a saved recipe deterministically.

**Behavior:**

* External ops to `{cb,opt,loss}` are ignored with `decision="ignored_replay"` (distinguish from prod freeze if you want; either is fine as long as explicit)  
* Kernel runs RecipePlayer:  
  * injects recipe ops at (step,event)  
  * applies them as if they were internal ops  
  * ledger `source="replay"`  
* `core` ops allowlist:  
  * `status`  
  * `snapshot`  
  * `unfreeze` (optional, but generally you’d keep it)  
  * `set_replay_policy` (optional)

**Rationale:**  
Ensures “repeat the same interventions” while preventing accidental external changes.

### **10.3 Mode C: `freeze replay adjusted`**

**Definition:**  
Same as replay mode, but replays a **modified recipe** based on an adjustment overlay, without allowing external ops.

**Behavior:**

* External ops to `{cb,opt,loss}` ignored  
* RecipePlayer loads recipe \+ adjustment overlay rules:  
  * generates an “effective recipe” in memory  
  * applies at (step,event)  
* Writes ledger with:  
  * `source="replay"`  
  * include `notes` indicating what patch was applied if a patch modifies a record (recommended for traceability)

**Rationale:**  
Allows deterministic runs with controlled variations: e.g., replay everything but adjust LR at step 1200\.

---

## **11\) Replay-Adjusted: Adjustment Overlay Rules (Full Spec)**

Adjustment overlay file: `hotcb.adjust.yaml` (or JSON)

It applies transformations to recipe entries.

### **11.1 Patch rule types (MUST support)**

1. **Replace params** (surgical change)  
2. **Shift step** (delay/advance an action)  
3. **Drop entry** (remove one or more recipe actions)  
4. **Insert entry** (add new action)  
5. **Bulk transform** (apply to all matching entries, e.g., scale all LRs by factor)

### **11.2 Matching criteria**

A patch matches recipe entries by any combination of:

* module  
* op  
* id  
* at.step (exact) or step range  
* at.event  
* payload key existence (e.g. params contains “lr”)  
* nth occurrence (optional)

### **11.3 Example overlay**

version: 1  
patches:  
 \- match: {module: "opt", op: "set\_params", at\_step: 1200}  
   replace\_params: {lr: 2e-5}

 \- match: {module: "loss", op: "set\_params", id: "main"}  
   transform\_params:  
     scale:  
       distill\_w: 1.1

 \- match: {module: "cb", id: "feat\_viz"}  
   drop: true

 \- insert:  
     at: {step: 1000, event: "train\_step\_end"}  
     module: "cb"  
     op: "enable"  
     id: "sys"

### **11.4 Deterministic output**

Kernel should optionally write an “effective recipe snapshot”:

* `hotops.recipe.effective.jsonl`  
* so the adjusted replay is reproducible and inspectable

---

## **12\) Versioning \+ File Reload Details (Dynamic Callback Source Capture)**

This is the piece you explicitly called out.

hotcb can load callbacks from a Python file path via `_load_class_from_file()`.

loader

Those files can change, so replay could otherwise load a different version.

### **12.1 Source capture requirement**

When a callback `target.kind == "python_file"` is loaded (from external/yaml/replay), kernel MUST support **source capture**:

On load:

1. Read file bytes from target.path  
2. Compute `sha256`  
3. Copy exact bytes to:  
   * `runs/<run_id>/hotcb.sources/<sha256>.py`  
4. In the *ledger entry* for that load:  
   * record `sha256`  
   * record `captured_path` (relative path)  
   * record whether capture succeeded (file missing, permission, etc.)

### **12.2 Replay behavior with captured sources**

When replaying a `cb load` entry that has `source_capture`:

* Prefer loading from `captured_path` (deterministic)  
* If missing, fallback to original `target.path` and record ledger note `capture_missing_fallback`

### **12.3 File auto-reload (future or optional)**

Your current loader notes that explicit reload semantics aren’t included in baseline.

loader

If/when you add callback “reload-on-mtime”, version capture becomes even more important:

* Each reload event becomes:  
  * ledger entry with step/event \+ new sha256  
* Recipe export includes those versions so replay replays the same version timeline

**This doc treats source capture as mandatory for determinism in replay.**

---

## **13\) Module Specs (Deep)**

### **13.1 Module: hotcb (stable, already implemented)**

**Purpose:** instrumentation, diagnostics, logging, runtime-loaded utilities.

**What stays:**

* Callback protocol  
   protocol  
* Dynamic loader  
   loader  
* Callback enable/disable \+ set\_params semantics  
   controller  
* Failure isolation semantics  
   controller  
* CLI features conceptually (but integrated into unified CLI)

**What changes (integration under kernel):**

* hotcb standalone continues to function (backward compatibility)  
* In HotOps mode:  
  * hotcb will no longer tail its own `hotcb.commands.jsonl`  
  * instead kernel routes `module="cb"` ops to hotcb controller API  
  * **OR** if you want minimal refactor: kernel can write-through to hotcb’s command stream  
    * but canonical design is routing (single source stream \+ single cursor)

**Idempotent desired-state reconciliation (YAML) stays as a pattern**:

config

HotOps YAML will reuse that reconcile approach across modules.

---

### **13.2 Module: hotopt (new)**

**Purpose:** mutate optimizer behavior during live training.

#### **13.2.1 Scope**

Hotopt MUST support:

* global learning rate  
* weight decay  
* per-parameter-group learning rate (by group index; optionally by name)  
* gradient clipping thresholds (if clipping is performed by adapter or by a shared helper)  
* scheduler nudges (minimal but real): scale factor or one-shot drop

#### **13.2.2 Safety boundaries**

Hotopt ops apply only at safe points. It must never mutate optimizer mid-backward.

**Where applied:**

* At `train_step_end` / `train_batch_end` after optimizer step, before next forward (adapter-defined safe point)

#### **13.2.3 Env requirements**

HotKernel standardizes env (as hotcb does).

protocol

Hotopt requires either:

* `env["optimizer"]` present, or  
* `env["resolve_optimizer"]` callable, which returns optimizer

Optional:

* `env["scheduler"]` or `resolve_scheduler`

If missing:

* op is recorded as `failed` or `skipped_noop` with reason `missing_optimizer` (configurable)

#### **13.2.4 Internal state (per id)**

For hotopt, you likely only need `id="main"` for most runs, but keep a handle registry like hotcb does.

Store:

* enabled flag  
* last\_params (like CallbackHandle.last\_params)  
   controller  
* last\_applied\_step  
* last\_error

#### **13.2.5 Parameters schema**

* `lr`: float  
* `weight_decay`: float  
* `clip_norm`: float  
* `group`: int (optional, for per-group ops)  
* `groups`: mapping group\_idx → params (optional)  
* `scheduler_scale`: float (multiply computed lr)  
* `scheduler_drop`: float (one-shot multiply by factor)

#### **13.2.6 Ops supported**

* `enable` / `disable`  
* `set_params`  
* `status` (via kernel unified status)  
* (optional) `reset` (dangerous; not required but can exist)

#### **13.2.7 Error handling**

* Any failure logs ledger `decision=failed`, includes error  
* If `auto_disable_on_error` is enabled at kernel/module:  
  * disable hotopt handle after failure  
  * record ledger note `auto_disabled`

---

### **13.3 Module: hotloss (new)**

**Purpose:** mutate loss composition and weights during live training.

#### **13.3.1 Scope**

Hotloss MUST support:

* scalar loss weights: `distill_w`, `depth_w`, etc.  
* term toggles: enable/disable individual loss terms  
* ramp configuration: warmup fraction, end weight, ramp type

#### **13.3.2 Key design constraint**

Hotloss should not require hooking into autograd internals. It should mutate a **loss config object** used by the user’s loss function.

#### **13.3.3 Env requirements**

* `env["mutable_state"]`: mutable mapping used by loss computation  
  OR  
* `env["resolve_mutable_state"]`: callable

If missing:

* record `failed` or `skipped_noop`

#### **13.3.4 Suggested mutable\_state shape**

A conventional structure:

mutable\_state \= {  
 "weights": {"distill": 0.2, "depth": 1.5},  
 "terms": {"aux\_depth": True, "aux\_heatmap": False},  
 "ramps": {"depth": {"type":"linear","warmup\_frac":0.2,"end":2.0}},  
}

Hotloss `set_params` maps:

* `distill_w` → `weights.distill`  
* `terms.aux_depth=false` toggles  
* `ramps.depth.end=2.0` etc.

#### **13.3.5 Ops supported**

* `enable` / `disable`  
* `set_params`  
* status

#### **13.3.6 Deterministic behavior**

All mutations are recorded step-indexed in ledger and exported in recipe.

---

## **14\) HotOps YAML Desired-State (Unified, Optional)**

hotcb already has YAML reconcile schema producing ops

config

. HotOps extends it.

### **14.1 hotcb.yaml schema (v1)**

version: 1

core:  
 freeze\_mode: "off" | "prod" | "replay" | "replay\_adjusted"  
 replay:  
   recipe\_path: "hotcb.recipe.jsonl"  
   adjust\_path: "hotcb.adjust.yaml"  
   policy: "best\_effort" | "strict"  
   step\_offset: 0

cb:  
 callbacks:  
   timing:  
     enabled: true  
     target:  
       kind: module|python\_file  
       path: ...  
       symbol: ...  
     init: {}  
     params: {}

opt:  
 enabled: true  
 id: "main"  
 params:  
   lr: 3e-5  
   weight\_decay: 0.01  
   clip\_norm: 1.0  
   scheduler\_scale: 1.0

loss:  
 enabled: true  
 id: "main"  
 params:  
   distill\_w: 0.2  
   depth\_w: 1.5  
   terms:  
     aux\_depth: false  
   ramps:  
     depth:  
       type: linear  
       warmup\_frac: 0.2  
       end: 2.0

### **14.2 Reconciliation semantics**

* On mtime change, parse YAML and generate idempotent ops:  
  * For cb: same as current hotcb YAML pattern  
     config  
  * For opt/loss: `set_params` \+ enable/disable  
* YAML-derived ops are recorded in ledger with `source="yaml"`

**Important:** YAML is optional (stdlib-only remains possible, like hotcb).

config

---

## **15\) Unified CLI (`hotcb`) with Syntactic Sugar**

### **15.1 CLI goals**

* One entrypoint for all modules  
* Writes to **one** `hotcb.commands.jsonl`  
* Includes sugar commands to reduce cognitive overhead  
* Supports freeze/replay mode management and recipe export

### **15.2 Core commands**

hotops \--dir \<run\> init  
hotops \--dir \<run\> status  
hotops \--dir \<run\> freeze \--mode prod|replay|replay\_adjusted|off \[--recipe ...\] \[--adjust ...\]  
hotops \--dir \<run\> recipe export \[--out ...\] \[--from applied\]  
hotops \--dir \<run\> recipe validate \--recipe ...  
hotops \--dir \<run\> recipe patch-template \--recipe ...

### **15.3 Module commands**

hotops \--dir \<run\> cb enable \<id\>  
hotops \--dir \<run\> cb disable \<id\>  
hotops \--dir \<run\> cb set \<id\> k=v ...  
hotops \--dir \<run\> cb load \<id\> \--file ... \--symbol ... \[--enabled\] \[--init k=v ...\]  
hotops \--dir \<run\> cb unload \<id\>

hotops \--dir \<run\> opt enable \[--id main\]  
hotops \--dir \<run\> opt disable \[--id main\]  
hotops \--dir \<run\> opt set k=v ... \[--id main\]

hotops \--dir \<run\> loss enable \[--id main\]  
hotops \--dir \<run\> loss disable \[--id main\]  
hotops \--dir \<run\> loss set k=v ... \[--id main\]

### **15.4 Syntactic sugar rules**

Provide shortcuts:

* `hotops enable <id>` → defaults to `cb enable <id>`  
* `hotops disable <id>` → defaults to cb  
* `hotcb set ...` → auto-route based on keys:  
  * if keys include `lr|weight_decay|clip_norm|scheduler_*` → opt  
  * if keys include `*_w` or prefix `terms.` or `ramps.` → loss  
  * otherwise: error and require explicit module

### **15.5 Freeze management via CLI**

CLI writes `hotcb.freeze.json` so the running process observes it at next poll.

`hotcb.freeze.json` schema:

{  
 "mode": "off|prod|replay|replay\_adjusted",  
 "recipe\_path": "hotcb.recipe.jsonl",  
 "adjust\_path": "hotcb.adjust.yaml",  
 "policy": "best\_effort|strict",  
 "step\_offset": 0  
}

**Why separate file?**  
So freeze state is “out-of-band” and doesn’t require injecting a command at exactly the right time. Kernel can check this file’s mtime cheaply like it checks YAML mtime today.

util

(You can also support freeze via command ops, but freeze-file is cleaner and immediate.)

---

## **16\) Backward Compatibility and Migration Plan**

### **16.1 hotcb standalone must continue to work**

* Existing `hotcb` CLI that writes `hotcb.commands.jsonl` remains (optional)  
* Existing scripts using `HotController(config_path, commands_path=...)` remain valid  
   controller

### **16.2 HotOps mode integration**

When using HotOps:

* preferred: training integrates `HotKernel` (not hotcb HotController directly)  
* hotcb becomes a sub-controller under kernel, receiving routed ops

### **16.3 Adapter strategy**

* Provide new adapters:  
  * `hotops.adapters.lightning.HotOpsLightning`  
  * `hotops.adapters.hf.HotOpsHF`  
* These adapters construct env similarly to existing hotcb adapters  
   lightning

   hf  
   but also expose:  
  * optimizer/scheduler references if available  
  * mutable\_state hook if user provides it  
* Keep hotcb adapters for users who only install hotcb

---

## **17\) Installation Matrix (Users can install any combination)**

### **17.1 Package layout**

* `hotops-core`  
* `hotcb`  
* `hotopt`  
* `hotloss`  
* `hotcb` meta-package

### **17.2 Install options**

* Only callbacks:  
  * `pip install hotcb`  
* Only optimizer control:  
  * `pip install hotopt`  
* Only loss control:  
  * `pip install hotloss`  
* Full ecosystem:  
  * `pip install hotops`

### **17.3 CLI availability rules**

* `hotcb` CLI comes from `hotops-core`  
* If user installs only hotcb, they can still use `hotcb` CLI; if they install hotops-core too, they get `hotcb` CLI but only `cb` subcommands are available.

### **17.4 Module discovery**

Use Python entrypoints:

* `hotops.modules` entrypoint group  
* Modules register themselves with kernel

This allows partial installs without import errors.

---

## **18\) Detailed Upgrade Implementation Plan (from current code)**

This is the actual “do it” plan agents can implement.

### **18.1 Extract shared utilities (from hotcb to hotops-core)**

Move or duplicate (prefer move but keep hotcb importing from core for compatibility):

* FileCursor, read\_new\_jsonl, safe\_mtime  
   util  
* Op-like dataclass base, expanded with module field (or keep hotcb.Op and create HotOpsOp)  
* Logging helpers  
* Common type inference for CLI kv parsing (hotcb CLI does this already)  
   util

### **18.2 Implement HotKernel (new)**

Key internal fields:

* `commands_path` \+ FileCursor  
* `applied_ledger_path`  
* `recipe_path` \+ recipe cursor/pointer (replay)  
* `freeze_state_path` \+ last mtime cache  
* `yaml_path` \+ last mtime cache (optional)  
* module registry: `{"cb": CBController, "opt": OptController, "loss": LossController}`  
* step counter \+ debounce \+ poll interval (carry from HotController)  
   controller

Core method:

* `apply(env, events)`  
  * same semantics as hotcb: increment step counter; poll on debounce boundary; dispatch events

Polling:

* Read freeze file if mtime changed  
* Read YAML if enabled and mtime changed  
* Tail JSONL command file  
* Build combined op list:  
  * replay ops (if replay mode) at current step/event  
  * yaml ops  
  * external ops (unless freeze/replay says ignore)  
* Apply ops via module controllers  
* For each op, write ledger entry

### **18.3 Integrate hotcb under kernel**

Two approaches:

**Approach : direct routing**

* Add an interface to hotcb controller: `apply_op(op, env)` and `dispatch(event, env)`  
* Kernel routes cb ops to hotcb controller without hotcb reading files itself

This spec requires Approach A as the end state.

### **18.4 Implement hotopt controller**

* Mirror hotcb’s handle pattern from `CallbackHandle`  
   controller  
   but adapted:  
  * `OptHandle(id="main", enabled=True, last_params={}, last_error=...)`  
* Provide:  
  * `apply_op(op, env)` (enable/disable/set\_params)  
  * `status()`  
* Implement param validation and mutations:  
  * For optimizer param\_groups update `lr`, `weight_decay`  
  * For group-specific updates, apply only to specified group index  
  * For scheduler, apply scale or one-shot drop as configured  
* No event dispatch required (unless you want periodic checks); ops-driven only

### **18.5 Implement hotloss controller**

* Similar to hotopt but mutates `mutable_state`

### **18.6 Ledger writer**

* A dedicated helper in hotops-core:  
  * `append_ledger(entry: dict)`  
* Must be robust (best effort like hotcb logging)  
   controller

### **18.7 Recipe export tool**

* CLI command:  
  * reads `hotcb.applied.jsonl`  
  * filters applied entries from modules cb/opt/loss  
  * writes recipe jsonl  
* Validate recipe:  
  * schema checks  
  * required module availability info  
  * optional version constraints

### **18.8 Replay engine \+ overlay patcher**

* `RecipePlayer` loads recipe entries into memory (or streams if huge)  
* Maintains pointer  
* Applies overlay patch rules to create effective recipe  
* Optionally writes effective recipe snapshot

### **18.9 Source capture integration for cb python\_file load**

* During cb load op processing, kernel tries to capture file bytes and record sha  
* On replay, use captured version

---

## **19\) Testing Strategy (Extensive, No MVP cuts)**

This is a full test plan. Assignable to agents as separate workstreams.

### **19.1 Test categories**

1. **Unit tests — hotops-core**  
2. **Unit tests — module controllers (cb routing, opt, loss)**  
3. **Integration tests — adapters (Lightning, HF)**  
4. **Determinism/replay tests**  
5. **Freeze mode tests**  
6. **Source capture/version tests**  
7. **Robustness/fuzz-ish tests (bad JSONL, partial writes, big bursts)**

---

### **19.2 Unit tests — Core ingestion and routing**

**JSONL tailing**

* Given a file, write 3 commands, poll, ensure only those 3 read  
* Append 2 more, poll, ensure only 2 new read  
* Validate offset cursor increments (same semantics as hotcb `read_new_jsonl`)  
   util

**Debounce and poll interval**

* With debounce\_steps=5, call apply() 4 times → no poll  
* 5th → poll happens  
* With poll\_interval\_sec, ensure time gate prevents frequent polls (pattern from HotController)  
   controller

**Routing**

* Send ops with modules `cb/opt/loss/core`  
* Ensure they are routed to correct controller handlers in correct order

**Ledger correctness**

* Every op produces exactly one ledger record  
* Ledger record contains step/event from env and source/decision fields  
* Failures include error text

---

### **19.3 Unit tests — Freeze modes (full semantics)**

**Freeze prod**

* Set freeze file mode=prod  
* Append external op for opt set\_params  
* Apply at safe point  
* Assert:  
  * opt state unchanged  
  * ledger decision \== ignored\_freeze

**Freeze replay**

* Set mode=replay with recipe containing opt change at step=3  
* Append external conflicting opt change  
* Run steps 1..5:  
  * step 3: recipe applied (ledger source=replay)  
  * external ops ignored (ledger ignored\_replay)

**Freeze replay adjusted**

* Same recipe, overlay changes lr from 3e-5 → 2e-5 at step 3  
* Confirm applied lr is adjusted value  
* Confirm ledger notes patch applied (recommended)

---

### **19.4 Unit tests — Recipe export \+ replay**

**Export**

* Create synthetic applied ledger with applied/ignored/failed entries  
* Export recipe  
* Ensure only applied entries included  
* Preserve ordering by seq  
* Verify at.step/at.event mapping correct

**Replay matching**

* Create recipe with:  
  * two ops at same step/event  
  * ops across steps  
* Run apply loop, ensure order preserved

**Policy strict**

* Recipe includes op at step 10 but run ends at step 5  
* strict → raises or marks run failed (define which; whichever you implement, test it)  
* best\_effort → logs missed op summary

**Step offset**

* recipe step=3 with offset \+2 → applied at step 5

---

### **19.5 Unit tests — Source capture/versioning (the “nook” you asked for)**

**Capture on load**

* Create temp python file with callback class  
* Issue cb load op with python\_file target  
* Ensure:  
  * `hotcb.sources/<sha>.py` exists  
  * ledger entry records sha \+ captured\_path

**Replay uses captured**

* Modify original file to change behavior (different output)  
* Replay run using recipe with captured\_path  
* Ensure callback loaded from captured version (behavior matches original)

**Fallback if captured missing**

* Delete captured file  
* Replay  
* Ensure it falls back to original path and logs ledger note `capture_missing_fallback`

---

### **19.6 Unit tests — hotopt controller**

**Global lr update**

* Create torch optimizer with 2 param groups  
* Apply set\_params lr=...  
* Verify both groups updated

**Group-specific lr**

* Apply set\_params group=1 lr=...  
* Verify only group 1 changed

**Weight decay update**

* Similar

**Missing optimizer**

* env lacks optimizer  
* op results in ledger failed/skipped with reason

**Auto-disable on error**

* Force an error (e.g., invalid lr type)  
* Confirm handle disabled if configured  
* Subsequent ops ignored until re-enabled (define exact policy and test it)

---

### **19.7 Unit tests — hotloss controller**

* Mutate weights/toggles in mutable\_state dict  
* Verify mapping rules (distill\_w → weights.distill, etc.)  
* Missing mutable\_state handling  
* Error/auto-disable behavior

---

### **19.8 Integration tests — Lightning adapter**

Use a tiny Lightning model and trainer (CPU is fine).

* The adapter already normalizes env and loss exposure in the hotcb Lightning adapter  
   lightning

  HotOps adapter must:  
* expose optimizer or resolver  
* expose mutable\_state if user supplies

Test:

* Run training for N steps  
* At step K apply opt change via external commands file  
* Assert:  
  * optimizer changed at expected step boundary  
  * ledger step matches

Replay test:

* Use ledger → export recipe  
* New run in replay mode  
* Assert same lr change step

---

### **19.9 Integration tests — HF adapter**

Similar pattern using a minimal Trainer (tiny model/dataset).

HF adapter constructs env similarly to your existing one

hf

Test:

* external command applied at global\_step K  
* replay reproduces

---

### **19.10 Robustness tests (messy real-world conditions)**

* JSONL with blank lines  
* JSONL with partial line write (simulate producer crash mid-write)  
  * kernel should handle gracefully:  
    * either skip until complete line  
    * or mark failed and continue (define behavior)  
* Large burst: 20k commands appended  
  * ensure max\_lines cap behavior (hotcb `read_new_jsonl` has max\_lines)  
     util  
  * kernel should process in chunks without starving training

---

## **20\) Documentation Deliverables (as part of “full plan”)**

This spec implies docs pages (at minimum):

1. `concepts.md`  
   * HotKernel, ops, ledger, recipe, freeze modes  
2. `cli.md`  
   * all commands \+ sugar rules  
3. `replay.md`  
   * exporting recipe, replay strict vs best\_effort, step offsets, adjusted overlays  
4. `modules/`  
   * `hotcb.md` (existing behavior \+ how it plugs into hotops)  
   * `hotopt.md`  
   * `hotloss.md`  
5. `formats.md`  
   * JSONL schema, ledger schema, recipe schema, freeze file schema, adjust overlay schema  
6. `examples/`  
   * lightning run using hotops  
   * HF run using hotops  
   * bare torch example  
   * sample adjusted replay overlay

---

## **21\) Non-negotiable Guarantees (The “contract”)**

When fully implemented, HotOps guarantees:

1. **Every mutation is recorded** with (step,event,source,decision) in `hotcb.applied.jsonl`.  
2. **Recipe export is reproducible**: exporting from ledger yields a deterministic sequence.  
3. **Replay modes are deterministic** within the limitations of training nondeterminism:  
   * same ops at same step/event  
   * python\_file callback loads use captured versions when available  
4. **Freeze mode protects production**: external ops cannot mutate cb/opt/loss.  
5. **Replay mode blocks external interference** by design.  
6. **Modules are independently installable** and discoverable by CLI.  
7. **hotcb standalone remains supported** and unchanged for existing users.

## **22\) Final “Agent Delegation Map” (How to split work)**

You said you’ll delegate agents. Here’s a clean breakdown:

**Agent A — hotops-core kernel**

* command ingestion \+ cursor

* freeze file handling

* op routing

* unified apply/events loop

* ledger writer

**Agent B — recipe/replay subsystem**

* applied→recipe export

* recipe validation

* RecipePlayer \+ policies

* overlay patch system \+ effective recipe snapshot

**Agent C — hotcb integration**

* turn hotcb into a module controller compatible with kernel routing

* preserve standalone mode

* integrate source capture hooks for python\_file loads

**Agent D — hotopt module**

* optimizer resolver/env contract

* param group mutation

* scheduler scale/drop

* status, error policy

**Agent E — hotloss module**

* mutable\_state contract \+ mapping

* weights/toggles/ramps

* status, error policy

**Agent F — unified CLI**

* `hotcb` CLI with module subcommands

* syntactic sugar routing rules

* freeze and recipe commands

* init command and run dir layout

**Agent G — integration tests**

* Lightning integration tests

* HF integration tests

* replay tests end-to-end
