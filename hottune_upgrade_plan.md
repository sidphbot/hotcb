Your current architecture already supports the key runtime pieces: the Lightning adapter injects `optimizer`, `scheduler`, and optional `mutable_state` into `env`, and the kernel already owns run paths, polling, recipe replay, ledger writing, and module dispatch for `cb`, `opt`, and `loss`. That means `hottune` can fit naturally as another kernel module rather than a separate repo.

# **1\. Objective**

`hottune` is an **optional hotcb module** that performs **online, constrained, Bayesian-guided hyperparameter adaptation during training**, with:

* safe-point application only

* bounded search spaces

* recipe persistence across runs

* rollback or conservative acceptance logic

* framework-agnostic operation through adapter-provided metric access and kernel-owned actuators

This is **not** intended to replace offline HPO entirely. It is meant to cover the gap between:

* static recipes

* full-run sweeps

* manual mid-run tweaking

The core value proposition is:

**observe → propose bounded mutation → apply at safe point → evaluate over horizon → accept/reject → persist learning into next recipe**

# **2\. Non-goals**

Version 1 should **not** try to do all of the following:

* full RL-based controller training

* arbitrary optimizer replacement mid-run

* automatic model architecture changes

* unbounded mutation of user code

* opaque “AI decides everything” behavior

* required dependencies in base install

# **3\. Position in the existing hotcb architecture**

## **3.1 Existing state**

Your kernel currently:

* manages `run_dir`

* tails command and yaml control planes

* writes an applied ledger

* owns recipe/freeze/replay behavior

* dispatches ops to modules `cb`, `opt`, and `loss` through `apply_op`

* dispatches callback events afterward.

Your Lightning adapter currently:

* builds `env`

* exposes framework, phase, step, epoch, model, trainer, log

* exposes `optimizer`, `scheduler`, and optional `mutable_state`

* normalizes `loss` from outputs

* calls `kernel.apply(env, events=...)`.

That is already enough to support `hottune` with **minimal architectural change**.

## **3.2 New module placement**

Add a new module under the kernel:

* module key: `"tune"`

* implementation: `HotTuneController`

Kernel modules become:

* `cb`

* `opt`

* `loss`

* `tune`

`hottune` should be **optional at runtime** and **cheap when unused**.

# **4\. High-level architecture**

`hottune` consists of five layers.

## **4.1 Metric access layer**

A standardized way for the adapter to expose metrics to the tuner.

Required adapter addition:

* `env["metric"] : Callable[[str, Any], Any]`

This accessor should abstract away framework-specific storage.

## **4.2 Actuation layer**

A stable interface for applying mutations to live training state.

Recommended ownership:

* actuators are **registered into the kernel**

* tuner interacts via kernel, not raw optimizer objects

## **4.3 Policy/search layer**

Responsible for choosing the next mutation.

Initial strategy:

* constrained Bayesian optimization / TPE-style proposal over mutation space

* phase-aware priors

* cooldowns and guardrails

## **4.4 Evaluation layer**

Measures whether a mutation helped over a short horizon.

Outputs:

* accepted

* rejected

* rolled back

* ignored

* blocked by safety

## **4.5 Recipe evolution layer**

Persists cross-run learning:

* mutation success rates

* phase-specific winning ranges

* instability histories

* default priors for future runs

# **5\. Packaging and install model**

Base package remains lightweight.

Suggested extras:

* `hotcb[tune]` → installs `optuna` and any tiny numerical deps

* no hard requirement for tune deps in default install

Suggested package structure:

src/hotcb/  
 adapters/  
 modules/  
   callback/  
   hotopt/  
   hotloss/  
   hottune/  
     \_\_init\_\_.py  
     controller.py  
     policy.py  
     search.py  
     constraints.py  
     evaluator.py  
     recipe.py  
     storage.py  
     state.py  
     schemas.py  
     events.py  
 actuators/  
   \_\_init\_\_.py  
   base.py  
   optimizer.py  
   mutable\_state.py  
 kernel.py

# **6\. Adapter contract**

## **6.1 Required additions**

Every supported adapter should expose:

* `env["metric"]`

* `env["kernel"]`

The current Lightning adapter already exposes raw objects that are useful for actuation.

## **6.2 Metric accessor behavior**

Signature:

metric(name: str, default: Any \= None) \-\> Any

Expected behavior:

* first check framework-native callback metrics

* then logged metrics

* then normalized env fields like `loss`

* convert tensor scalars to plain Python numbers when possible

* return `default` if missing

## **6.3 Recommended standard metric names**

Adapters should normalize toward a shared namespace where practical:

* `train/loss`

* `val/loss`

* `val/score`

* `lr`

* `grad/norm`

* `time/step_sec`

* `system/gpu_mem_mb`

* `system/cpu_mem_mb`

Not every framework must supply every metric.

# **7\. Kernel integration**

## **7.1 New kernel responsibilities**

The kernel should become the home for **actuator registration and safe mutation application**.

Add methods:

register\_actuator(name: str, actuator: BaseActuator) \-\> None  
get\_actuator(name: str) \-\> BaseActuator | None  
list\_actuators() \-\> dict\[str, BaseActuator\]  
apply\_patch(name: str, patch: dict, \*, source: str, validate: bool \= True) \-\> ApplyResult  
begin\_transaction(label: str | None \= None) \-\> MutationTransaction  
write\_tune\_record(kind: str, payload: dict) \-\> None

## **7.2 Why kernel ownership is preferred**

Because the kernel already owns:

* application procedures

* run directory

* recipe pathing

* ledgers

* policy/freeze context

That makes it the correct control plane for tuned mutations as well. The existing kernel already centralizes module dispatch and applied logging, so this is a natural extension rather than a new subsystem.

# **8\. Actuator system**

## **8.1 Base actuator interface**

class BaseActuator(Protocol):  
   name: str

   def snapshot(self) \-\> dict: ...  
   def validate(self, patch: dict) \-\> ValidationResult: ...  
   def apply(self, patch: dict) \-\> ApplyResult: ...  
   def restore(self, snapshot: dict) \-\> ApplyResult: ...  
   def describe\_space(self) \-\> dict: ...

## **8.2 Required semantics**

### **snapshot**

Returns minimal state needed for rollback of actuator-managed parameters.

### **validate**

Checks:

* types

* bounds

* allowed phase

* cooldown

* reversibility

* user-declared constraints

### **apply**

Applies the mutation to the live object.

### **restore**

Best-effort rollback to prior actuator state.

### **describe\_space**

Returns the legal mutation schema for search and documentation.

## **8.3 v1 reference actuators**

### **Optimizer actuator**

File:

* `actuators/optimizer.py`

Supports:

* LR multiplicative change

* LR absolute set

* weight decay multiplicative change

* beta1/beta2 small bounded changes if optimizer supports them

* optional scheduler scale factor if supported

Patch examples:

{"op": "lr\_mult", "value": 0.85}  
{"op": "lr\_set", "value": 0.0003}  
{"op": "wd\_mult", "value": 1.15}  
{"op": "betas\_set", "value": \[0.9, 0.98\]}

### **Mutable-state actuator**

File:

* `actuators/mutable_state.py`

Assumes mutable `mutable_state` dict-like structure.

Supports:

* scalar weight adjustments

* bounded deltas

* multiplicative changes

* freezing a weight temporarily if user allows it

Patch examples:

{"op": "set", "key": "sp\_mse\_w", "value": 1.2}  
{"op": "mult", "key": "grad\_w", "value": 1.1}  
{"op": "delta", "key": "hf\_w", "value": \-0.05}

## **8.4 User-defined actuators**

Users may register custom actuators for:

* augmentation knobs

* EMA decay

* teacher/student blend

* freeze-unfreeze schedules

* sampling curriculum

# **9\. Tuning model**

## **9.1 Tuning unit**

The atomic unit is a **mutation**.

A mutation is not “the whole config”.  
 It is a **bounded change** to one or a few live knobs.

Example:

* `opt.lr_mult = 0.85`

* `loss.sp_mse_w += 0.1`

* `loss.grad_w *= 1.1`

## **9.2 Decision cycle**

At a decision event:

1. read current state and recent metrics

2. determine if tuning is allowed now

3. ask policy/search layer for next mutation

4. validate mutation

5. snapshot affected actuator state

6. apply mutation

7. create an evaluation segment

8. after horizon, score outcome

9. accept or restore snapshot

10. write records

11. update in-run recipe stats

# **10\. Policy model**

## **10.1 Initial policy choice**

Use **Bayesian/TPE-style proposal over mutation space**, because that fits your stated preference and works well with sparse, expensive evaluations.

Version 1 can implement this through Optuna-backed sampling internally.

## **10.2 Context features**

The proposal is conditioned on a compact context:

* phase bin: early / mid / late

* recent train loss slope

* recent val metric slope

* recent instability flags

* recent gradient norm volatility

* last accepted mutation type

* cooldown state

* mutation budget consumed

## **10.3 Search dimensions**

The search space should be over:

* actuator choice

* mutation op type

* mutation magnitude

* optional key within actuator

This is better than searching raw full config vectors.

## **10.4 Phase bins**

Default bins:

* early: first 20%

* mid: 20–70%

* late: final 30%

Can be step-based or epoch-based.

## **10.5 Cooldowns**

Each mutation family has a cooldown to prevent thrashing.

Example:

* LR mutation cooldown: 2 decision windows

* same loss key cooldown: 1 decision window

* any mutation after reject: 1 window global cooldown

# **11\. Constraints and safety**

This is the most important part.

## **11.1 Hard constraints**

Each mutation candidate must pass:

* key exists

* actuator exists

* patch type valid

* bounds valid

* phase allowed

* cooldown satisfied

* max cumulative drift not exceeded

* risk class allowed in current mode

## **11.2 Risk classes**

Every mutation type gets a risk level:

* `low`: LR small mult, loss scalar small delta

* `medium`: beta changes, larger LR drops

* `high`: optimizer family swap, freeze/unfreeze

Version 1 supports only `low` and optionally some `medium`.

## **11.3 Max drift constraints**

Prevent recipe wandering too far from base config.

Examples:

* LR cannot move outside base\_lr × \[0.2, 3.0\]

* weight decay outside base\_wd × \[0.25, 4.0\] forbidden

* loss weights cannot exceed user bounds

## **11.4 Stability blockers**

Mutation blocked if:

* NaN or inf detected recently

* anomaly callback raised critical flag

* gradient norm above emergency threshold

* validation metric unavailable for too long

* run in replay/freeze mode forbids tune mutations

Kernel freeze/replay semantics already exist, so `tune` should respect them exactly like `cb`, `opt`, and `loss`.

# **12\. Evaluation logic**

## **12.1 Segment**

A segment is the evaluation window following one mutation.

Stored fields:

* segment id

* start step

* end step

* decision event

* mutation id

* pre metrics

* post metrics

* smoothed deltas

* stability flags

* accept/reject result

## **12.2 Horizon**

A horizon can be defined by:

* fixed steps

* fixed eval events

* first validation epoch end after mutation

Recommended v1:

* evaluate on next validation epoch end

* optionally require minimum train steps elapsed too

## **12.3 Scoring**

Each recipe defines an objective score.

Generic default:

score\_delta \=  
   primary\_metric\_gain  
 \- instability\_penalty  
 \- excessive\_train\_regression\_penalty

For loss-minimization:

* gain \= old best minus new best

For maximize metric:

* gain \= new minus old

## **12.4 Acceptance**

Accept if:

* score\_delta \> epsilon

* no blocker triggered

* no severe regression in backup metric

Otherwise:

* reject and restore if rollback available

* else mark rejected and enter cooldown

## **12.5 Rollback modes**

### **Full rollback**

Actuator restore supported.

### **Soft rollback**

Only revert actuator-managed scalars.

### **No rollback**

Allowed only for low-risk mutations and only when explicitly configured.

# **13\. Recipe system**

## **13.1 Recipe levels**

There are two recipe layers.

### **Base recipe**

User-authored defaults and legal bounds.

### **Evolved recipe**

Auto-updated priors and learned preferences from prior runs.

## **13.2 Files**

Add to run dir:

* `hotcb.tune.recipe.yaml`

* `hotcb.tune.mutations.jsonl`

* `hotcb.tune.segments.jsonl`

* `hotcb.tune.study.sqlite` or equivalent optional sampler state

* `hotcb.tune.summary.json`

## **13.3 Recipe contents**

version: 1  
objective:  
 primary: val/alignment\_score  
 mode: max  
 backup\_metrics:  
   \- val/loss  
   \- grad/norm  
phases:  
 early: {start\_frac: 0.0, end\_frac: 0.2}  
 mid:   {start\_frac: 0.2, end\_frac: 0.7}  
 late:  {start\_frac: 0.7, end\_frac: 1.0}  
actuators:  
 opt:  
   enabled: true  
   mutations:  
     lr\_mult:  
       bounds: \[0.7, 1.2\]  
       prior\_center: 0.95  
       cooldown: 2  
       risk: low  
     wd\_mult:  
       bounds: \[0.8, 1.25\]  
       prior\_center: 1.0  
       cooldown: 2  
       risk: low  
 loss:  
   enabled: true  
   keys:  
     sp\_mse\_w:  
       mode: mult  
       bounds: \[0.5, 2.0\]  
       max\_step\_mult: 1.15  
       cooldown: 1  
     grad\_w:  
       mode: mult  
       bounds: \[0.3, 3.0\]  
       max\_step\_mult: 1.15  
       cooldown: 1  
search:  
 algorithm: tpe  
 startup\_trials: 8  
 candidate\_count: 24  
 phase\_conditioned: true  
acceptance:  
 epsilon: 0.001  
 horizon: next\_val\_epoch\_end  
 rollback\_on\_reject: true  
safety:  
 block\_on\_nan: true  
 block\_on\_anomaly: true  
 max\_global\_reject\_streak: 4

## **13.4 Evolution logic**

After each run, update:

* win rate per mutation family

* mean accepted magnitude by phase

* reject causes

* instability correlation

* priors for next run

Keep it simple:

* exponential moving averages

* count-based confidence

* no black-box magic in v1

# **14\. Storage formats**

## **14.1 Mutations log**

`hotcb.tune.mutations.jsonl`

Each line:

{  
 "mutation\_id": "m\_00017",  
 "step": 14800,  
 "epoch": 4,  
 "phase\_bin": "mid",  
 "event": "val\_epoch\_end",  
 "actuator": "loss",  
 "patch": {"op": "mult", "key": "grad\_w", "value": 1.1},  
 "proposal\_source": "tpe",  
 "context": {  
   "train\_loss\_slope": \-0.004,  
   "val\_score\_slope": 0.0003  
 },  
 "snapshot\_ref": "snap\_00017",  
 "status": "applied"  
}

## **14.2 Segments log**

`hotcb.tune.segments.jsonl`

{  
 "segment\_id": "s\_00017",  
 "mutation\_id": "m\_00017",  
 "start\_step": 14800,  
 "end\_step": 15640,  
 "horizon\_type": "next\_val\_epoch\_end",  
 "pre": {"val/alignment\_score": 0.621, "val/loss": 0.842},  
 "post": {"val/alignment\_score": 0.629, "val/loss": 0.835},  
 "delta": {"val/alignment\_score": 0.008, "val/loss": \-0.007},  
 "stability": {"nan": false, "anomaly": false, "grad\_spike": false},  
 "decision": "accepted",  
 "score\_delta": 0.0074  
}

## **14.3 Summary file**

`hotcb.tune.summary.json`  
 Contains compact run summary for downstream tooling.

# **15\. Events**

## **15.1 New tuning-friendly events**

You should add coarser events to adapters, especially Lightning:

* `fit_start`

* `train_batch_end`

* `val_batch_end`

* `val_epoch_end`

* `run_end`

Right now the Lightning adapter defaults to batch-end train and val events; for tuning, `val_epoch_end` is the key decision point.

## **15.2 Tuner event usage**

Recommended:

* observe continuously if needed

* propose/apply only at `val_epoch_end`

* finalize/flush at `run_end`

# **16\. CLI and control-plane integration**

## **16.1 Philosophy**

`hottune` should fit the same control-plane story as hotcb.

## **16.2 New commands**

Examples:

hotcb tune enable  
hotcb tune disable  
hotcb tune status  
hotcb tune set objective.primary=val/alignment\_score  
hotcb tune set acceptance.epsilon=0.002  
hotcb tune set actuators.loss.keys.grad\_w.bounds=\[0.3,2.5\]  
hotcb tune export-recipe \--out run\_dir/hotcb.tune.recipe.yaml  
hotcb tune evolve-recipe \--from runs/\*/hotcb.tune.summary.json \--out recipe\_next.yaml

## **16.3 YAML support**

Allow a tune section in `hotcb.yaml` or a dedicated `hotcb.tune.yaml`.

# **17\. Runtime modes**

## **17.1 Off**

No overhead beyond tiny module existence.

## **17.2 Observe-only**

No mutations; just collect windows and estimate what would have been proposed.

Very useful for debugging.

## **17.3 Suggest-only**

Writes proposals to logs or control plane, but does not apply automatically.

## **17.4 Active**

Applies bounded mutations.

## **17.5 Replay**

Replays prior tune mutations from recipe, subject to existing kernel replay rules.

# **18\. Module API**

## **18.1 Controller class**

`HotTuneController`

Responsibilities:

* maintain tune state

* respond to control ops

* handle event-driven proposal/evaluation

* write tune storage artifacts

## **18.2 Kernel interaction**

Either:

* `kernel.modules["tune"] = HotTuneController(...)`  
   or equivalent modular registration

## **18.3 Public methods**

class HotTuneController:  
   def apply\_op(self, op: HotOp, env: dict) \-\> ModuleResult: ...  
   def on\_event(self, event: str, env: dict) \-\> None: ...  
   def close(self, env: dict | None \= None) \-\> None: ...

`on_event` can be called by kernel alongside callback dispatch, or the tune logic can be driven through `apply_op` plus an explicit event dispatch path. I would recommend explicit event dispatch support for tune, not only command application.

# **19\. Search engine details**

## **19.1 Default engine**

Optuna TPE under optional dependency.

## **19.2 Study organization**

One study per run, optionally resumed from prior recipe stats.

Optionally persist study database under run dir.

## **19.3 Candidate generation**

At each decision event:

* generate several candidates

* score feasibility through constraints

* choose top candidate by sampler utility and novelty penalty

## **19.4 Novelty penalty**

Discourage repeating the same rejected mutation too quickly.

# **20\. Failure behavior**

## **20.1 If tune deps missing**

Module self-disables and logs:

* tuning unavailable

* install `hotcb[tune]`

## **20.2 If no metric accessor**

Self-disable or fall back to observe-only with warning.

## **20.3 If no actuators registered**

Observe-only mode only.

## **20.4 If mutation apply fails**

* write failed mutation record

* do not crash training

* respect auto-disable-on-error if configured

This mirrors hotcb’s existing defensive style around module apply and ledger logging.

# **21\. Testing strategy**

## **21.1 Unit tests**

Test:

* metric accessor normalization

* actuator validate/apply/restore

* phase binning

* cooldown logic

* acceptance logic

* recipe evolve logic

* serialization of logs

## **21.2 Deterministic simulation tests**

Build a fake trainer loop with synthetic objective surfaces.

Scenarios:

* convex improvement region

* noisy plateaus

* delayed reward

* misleading short-term spikes

* instability-triggered blocks

## **21.3 Adapter integration tests**

Lightning:

* metric function works

* optimizer actuator wiring works

* mutable\_state actuator wiring works

* val\_epoch\_end event emitted

## **21.4 Failure tests**

* missing metric names

* actuator missing

* rollback failure

* optuna not installed

* invalid recipe bounds

## **21.5 Replay tests**

Ensure tune records replay consistently with freeze/replay modes.

# **22\. Documentation plan**

Add docs sections for:

* what `hottune` is

* when to use it

* required adapter contract

* built-in actuators

* recipe format

* observe-only mode

* safety model

* how to evolve a recipe across runs

* examples for Lightning / HF / bare torch

# **23\. Recommended v1 scope**

Keep v1 intentionally tight.

## **23.1 Included**

* metric accessor contract

* kernel actuator registry

* optimizer actuator

* mutable-state actuator

* TPE proposal

* next-val-epoch acceptance

* rollback for actuator-managed params

* recipe persistence and evolution

* observe-only mode

## **23.2 Excluded from v1**

* optimizer class swaps

* arbitrary scheduler graph mutations

* RL/meta-learning controller

* multi-objective Pareto UI

* distributed cross-worker shared tuner

* automatic architecture edits

# **24\. Minimal code changes required**

## **24.1 Lightning adapter**

Add:

* `env["metric"]`

* `env["kernel"]`

* `on_validation_epoch_end` support to emit `val_epoch_end`

The rest is already close enough because `optimizer`, `scheduler`, and `mutable_state` are already exposed.

## **24.2 Kernel**

Add:

* actuator registry

* tune artifact writers

* tune event dispatch or module event hook

* optional transaction wrapper

## **24.3 New module**

Add `hottune` controller and its storage/policy helpers.

# **25\. Suggested roadmap**

## **Phase 1**

Infrastructure only:

* metric accessor

* kernel actuator registry

* optimizer/loss actuators

* observe-only tune module

## **Phase 2**

Active bounded tuning:

* TPE proposal

* acceptance logic

* rollback

* logs and summary

## **Phase 3**

Recipe evolution:

* evolve priors

* export/import recipes

* compare runs

## **Phase 4**

Advanced:

* extra actuators

* suggest-only mode

* replay of tune recipes

* richer dashboards

# **26\. Final recommendation**

Yes, `hottune` should live **inside the single package**, as an **optional module** built around:

* adapter-provided `metric()`

* kernel-owned actuator registry

* bounded mutation search

* short-horizon evaluation

* recipe evolution across runs

