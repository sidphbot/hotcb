"""
hotcb.server.autopilot — Self-mode rule engine for autonomous training control.

Monitors metrics and proposes or auto-applies training interventions
based on configurable rules (community guidelines).

Modes: off, suggest, auto (rule-based), ai_suggest, ai_auto (LLM-driven).

Actions are executed by writing JSONL to ``hotcb.commands.jsonl``.
"""
import collections
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

log = logging.getLogger("hotcb.server.autopilot")

_VALID_MODES = {"off", "suggest", "auto", "ai_suggest", "ai_auto"}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AutopilotRule:
    """A single autopilot rule definition."""

    rule_id: str
    condition: str  # "plateau", "divergence", "overfitting", "custom"
    metric_name: str  # e.g., "val/loss", "val_loss"
    params: dict = field(default_factory=dict)
    action: dict = field(default_factory=dict)
    confidence: str = "medium"  # "high", "medium", "low"
    enabled: bool = True
    description: str = ""


@dataclass
class AutopilotAction:
    """Emitted when a rule fires."""

    action_id: str
    rule_id: str
    step: int
    wall_time: float
    condition_met: str
    proposed_action: dict
    confidence: str
    status: str  # "applied", "proposed", "rejected"


# ---------------------------------------------------------------------------
# Condition evaluators
# ---------------------------------------------------------------------------

_VALID_CONDITIONS = {"plateau", "divergence", "overfitting", "custom"}


def _eval_plateau(
    metric_history: list[float],
    params: dict,
) -> Optional[str]:
    """Return a description string if plateau detected, else None."""
    window = params.get("window", 5)
    epsilon = params.get("epsilon", 0.001)
    if len(metric_history) < window:
        return None
    recent = metric_history[-window:]
    best = min(recent)
    worst = max(recent)
    if worst - best <= epsilon:
        return (
            f"Metric plateaued: range {worst - best:.6f} <= epsilon {epsilon} "
            f"over last {window} steps"
        )
    return None


def _eval_divergence(
    metric_history: list[float],
    params: dict,
) -> Optional[str]:
    """Return description if metric diverged (increased sharply)."""
    window = params.get("window", 10)
    threshold = params.get("threshold", 2.0)
    if len(metric_history) < window:
        return None
    start_val = metric_history[-window]
    end_val = metric_history[-1]
    increase = end_val - start_val
    if increase > threshold:
        return (
            f"Metric diverged: increased by {increase:.4f} > threshold {threshold} "
            f"over last {window} steps"
        )
    return None


def _eval_overfitting(
    metrics: dict[str, float],
    params: dict,
) -> Optional[str]:
    """Check train/val loss ratio for overfitting."""
    ratio_threshold = params.get("ratio_threshold", 0.5)
    train_key = params.get("train_metric", "train_loss")
    val_key = params.get("val_metric", "val_loss")
    # Try common name variants
    train_loss = metrics.get(train_key) or metrics.get("train/loss")
    val_loss = metrics.get(val_key) or metrics.get("val/loss")
    if train_loss is None or val_loss is None:
        return None
    if val_loss == 0:
        return None
    ratio = train_loss / val_loss
    if ratio < ratio_threshold:
        return (
            f"Overfitting detected: train/val ratio {ratio:.4f} < "
            f"threshold {ratio_threshold}"
        )
    return None


def _eval_custom(
    metrics: dict[str, float],
    expression: str,
) -> Optional[str]:
    """Safely evaluate a simple metric expression."""
    # Build a restricted namespace with only metric values and builtins
    safe_ns: dict[str, Any] = {}
    for k, v in metrics.items():
        # Normalize metric names: replace / and . with _ for use as identifiers
        safe_key = k.replace("/", "_").replace(".", "_").replace("-", "_")
        safe_ns[safe_key] = v
        safe_ns[k] = v  # also allow original names via dict-style
    # Add basic math
    import math
    safe_ns["abs"] = abs
    safe_ns["min"] = min
    safe_ns["max"] = max
    safe_ns["math"] = math
    try:
        result = eval(expression, {"__builtins__": {}}, safe_ns)  # noqa: S307
        if result:
            return f"Custom condition met: {expression}"
    except NameError:
        # Metric not yet available in stream — silently skip
        pass
    except Exception as exc:
        log.debug("Custom condition eval failed: %s — %s", expression, exc)
    return None


# ---------------------------------------------------------------------------
# Autopilot Engine
# ---------------------------------------------------------------------------


class AutopilotEngine:
    """
    Rule engine that monitors metrics and proposes or auto-applies
    training interventions based on configurable rules.

    Modes
    -----
    - ``"off"``        — engine is disabled, no evaluation
    - ``"suggest"``    — evaluate rules and record proposals, never auto-apply
    - ``"auto"``       — apply high-confidence actions, propose the rest
    - ``"ai_suggest"`` — LLM proposes actions, human reviews in dashboard
    - ``"ai_auto"``    — LLM proposes and applies (with safety guards)
    """

    def __init__(self, run_dir: str, mode: str = "off", config: Any = None) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode: {mode!r}")
        self._run_dir = run_dir
        self._mode = mode
        self._config = config  # AutopilotConfig or None (uses hardcoded defaults)
        self._rules: dict[str, AutopilotRule] = {}
        self._history: list[AutopilotAction] = []
        # Per-metric history for condition evaluation
        self._metric_history: dict[str, list[float]] = collections.defaultdict(list)
        # Cooldown tracking: rule_id -> last-fired step
        self._last_fired: dict[str, int] = {}
        self._default_cooldown: int = 10
        # AI engine reference (set externally via set_ai_engine)
        self._ai_engine: Any = None

    @classmethod
    def with_default_guidelines(cls, run_dir: str, mode: str = "off", config: Any = None) -> "AutopilotEngine":
        """Create engine pre-loaded with community default guidelines."""
        engine = cls(run_dir=run_dir, mode=mode, config=config)
        from .guidelines import DEFAULT_GUIDELINES_PATH
        if os.path.exists(DEFAULT_GUIDELINES_PATH):
            engine.load_guidelines(DEFAULT_GUIDELINES_PATH)
        return engine

    # -- Mode ---------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    def set_ai_engine(self, ai_engine: Any) -> None:
        """Attach the LLM autopilot engine."""
        self._ai_engine = ai_engine

    def set_mode(self, mode: str) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode: {mode!r}")
        if mode in ("ai_suggest", "ai_auto") and self._ai_engine is None:
            raise ValueError("AI engine not configured — cannot use AI modes")
        log.info("[hotcb.autopilot] mode changed: %s -> %s", self._mode, mode)
        self._mode = mode

    def reset(self) -> None:
        """Clear all metric history, action history, cooldown state, and mode.

        Called when a new training run starts so stale data doesn't
        trigger phantom rules.  Mode is reset to ``"off"`` so autopilot
        never carries over between runs.
        """
        self._metric_history.clear()
        self._history.clear()
        self._last_fired.clear()
        self._mode = "off"
        log.info("[hotcb.autopilot] state reset (mode -> off)")

    # -- Rules --------------------------------------------------------------

    def add_rule(self, rule: AutopilotRule) -> None:
        self._rules[rule.rule_id] = rule
        log.info("[hotcb.autopilot] rule added: %s", rule.rule_id)

    def remove_rule(self, rule_id: str) -> bool:
        removed = self._rules.pop(rule_id, None)
        if removed:
            log.info("[hotcb.autopilot] rule removed: %s", rule_id)
        return removed is not None

    def update_rule(self, rule_id: str, changes: dict) -> Optional[AutopilotRule]:
        """Update an existing rule's fields. Returns updated rule or None."""
        rule = self._rules.get(rule_id)
        if rule is None:
            return None
        for key in ("condition", "metric_name", "params", "action",
                     "confidence", "enabled", "description"):
            if key in changes:
                setattr(rule, key, changes[key])
        log.info("[hotcb.autopilot] rule updated: %s", rule_id)
        return rule

    def get_rules(self) -> list[AutopilotRule]:
        return list(self._rules.values())

    # -- Guidelines ---------------------------------------------------------

    def load_guidelines(self, path: str) -> int:
        """Load rules from a YAML guidelines file. Returns count of rules loaded."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load guidelines: pip install pyyaml"
            )
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or "rules" not in data:
            raise ValueError(f"Invalid guidelines file: expected 'rules' key")
        count = 0
        for entry in data["rules"]:
            rule = AutopilotRule(
                rule_id=entry["id"],
                condition=entry["condition"],
                metric_name=entry.get("metric", ""),
                params=entry.get("params", {}),
                action=entry.get("action", {}),
                confidence=entry.get("confidence", "medium"),
                enabled=entry.get("enabled", True),
                description=entry.get("description", ""),
            )
            self.add_rule(rule)
            count += 1
        return count

    # -- Evaluation ---------------------------------------------------------

    def evaluate(
        self, step: int, metrics: dict[str, float]
    ) -> list[AutopilotAction]:
        """
        Evaluate all enabled rules against current metrics.
        Returns list of actions taken (applied or proposed).
        """
        if self._mode == "off":
            return []

        # Update metric history
        for name, value in metrics.items():
            self._metric_history[name].append(value)

        actions: list[AutopilotAction] = []

        for rule in self._rules.values():
            if not rule.enabled:
                continue
            if rule.condition not in _VALID_CONDITIONS:
                continue

            # Check cooldown
            cooldown = rule.params.get("cooldown", self._default_cooldown)
            last = self._last_fired.get(rule.rule_id, -999999)
            if step - last < cooldown:
                continue

            condition_desc = self._check_condition(rule, metrics)
            if condition_desc is None:
                continue

            # Determine status based on mode and confidence
            status = self._determine_status(rule.confidence)

            action = AutopilotAction(
                action_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                step=step,
                wall_time=time.time(),
                condition_met=condition_desc,
                proposed_action=rule.action,
                confidence=rule.confidence,
                status=status,
            )

            if status == "applied":
                self._apply_action(rule.action)

            self._history.append(action)
            self._last_fired[rule.rule_id] = step
            actions.append(action)
            log.info(
                "[hotcb.autopilot] rule %s fired at step %d: %s [%s]",
                rule.rule_id,
                step,
                condition_desc,
                status,
            )

        return actions

    def _check_condition(
        self, rule: AutopilotRule, metrics: dict[str, float]
    ) -> Optional[str]:
        """Evaluate a single rule's condition. Returns description or None."""
        if rule.condition == "plateau":
            history = self._metric_history.get(rule.metric_name, [])
            return _eval_plateau(history, rule.params)

        elif rule.condition == "divergence":
            history = self._metric_history.get(rule.metric_name, [])
            # Merge config defaults into rule params (rule params take precedence)
            params = dict(rule.params)
            if "threshold" not in params and self._config is not None:
                params["threshold"] = self._config.divergence_threshold
            return _eval_divergence(history, params)

        elif rule.condition == "overfitting":
            # Merge config defaults into rule params (rule params take precedence)
            params = dict(rule.params)
            if "ratio_threshold" not in params and self._config is not None:
                params["ratio_threshold"] = self._config.ratio_threshold
            return _eval_overfitting(metrics, params)

        elif rule.condition == "custom":
            expr = rule.params.get("expression", "")
            if not expr:
                return None
            return _eval_custom(metrics, expr)

        return None

    def _determine_status(self, confidence: str) -> str:
        """Determine action status based on mode and confidence."""
        if self._mode == "suggest":
            return "proposed"
        # mode == "auto"
        if confidence == "high":
            return "applied"
        elif confidence == "medium":
            return "applied"
        else:  # low
            return "proposed"

    def _apply_action(self, action_cmd: dict) -> None:
        """Write a command to the commands JSONL file.

        Resolves multiplier params (lr_mult, wd_mult) to absolute values
        using the latest metric history.
        """
        from ..util import append_jsonl

        cmd_path = os.path.join(self._run_dir, "hotcb.commands.jsonl")
        cmd = dict(action_cmd)  # copy
        cmd.setdefault("ts", time.time())
        cmd.setdefault("source", "autopilot")

        # Resolve multiplier params to absolute values
        params = dict(cmd.get("params", {}))
        if "lr_mult" in params:
            current_lr = self._get_latest_metric("lr")
            if current_lr and current_lr > 0:
                params["lr"] = current_lr * params.pop("lr_mult")
            else:
                params["lr"] = 1e-4 * params.pop("lr_mult")  # fallback
        if "wd_mult" in params:
            current_wd = self._get_latest_metric("weight_decay")
            if current_wd and current_wd > 0:
                params["weight_decay"] = current_wd * params.pop("wd_mult")
            else:
                params["weight_decay"] = 1e-4 * params.pop("wd_mult")  # fallback
        cmd["params"] = params

        append_jsonl(cmd_path, cmd)
        log.info("[hotcb.autopilot] applied command: %s", cmd)

    def _get_latest_metric(self, name: str) -> Optional[float]:
        """Get the most recent value for a metric."""
        history = self._metric_history.get(name, [])
        return history[-1] if history else None

    # -- Accept proposed actions --------------------------------------------

    def accept_action(self, action_id: str) -> Optional[AutopilotAction]:
        """
        Find a proposed action by ID, apply its command, and mark it as applied.
        Returns the action if found and applied, None otherwise.
        """
        for action in self._history:
            if action.action_id == action_id:
                if action.status != "proposed":
                    return None  # already applied or rejected
                self._apply_action(action.proposed_action)
                action.status = "applied"
                log.info(
                    "[hotcb.autopilot] accepted proposed action %s (rule %s, step %d)",
                    action_id,
                    action.rule_id,
                    action.step,
                )
                return action
        return None

    # -- AI mode helpers -----------------------------------------------------

    @property
    def is_ai_mode(self) -> bool:
        return self._mode in ("ai_suggest", "ai_auto")

    def evaluate_rules_for_alerts(
        self, step: int, metrics: dict[str, float]
    ) -> list[AutopilotAction]:
        """
        Run rule conditions for alert detection only (no auto-apply).
        Used in AI modes where rules act as the sensor layer.
        """
        # Update metric history
        for name, value in metrics.items():
            self._metric_history[name].append(value)

        alerts: list[AutopilotAction] = []
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            if rule.condition not in _VALID_CONDITIONS:
                continue
            cooldown = rule.params.get("cooldown", self._default_cooldown)
            last = self._last_fired.get(rule.rule_id, -999999)
            if step - last < cooldown:
                continue

            condition_desc = self._check_condition(rule, metrics)
            if condition_desc is None:
                continue

            alert = AutopilotAction(
                action_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                step=step,
                wall_time=time.time(),
                condition_met=condition_desc,
                proposed_action=rule.action,
                confidence=rule.confidence,
                status="alert",  # not proposed or applied — just an alert
            )
            alerts.append(alert)
            self._last_fired[rule.rule_id] = step
        return alerts

    async def evaluate_async(
        self, step: int, metrics: dict[str, float]
    ) -> list[AutopilotAction]:
        """
        Async evaluation for AI modes.  Runs rules for alerts,
        then checks AI cadence and invokes LLM if needed.
        """
        if self._mode == "off":
            return []

        # Non-AI modes: delegate to sync evaluate
        if not self.is_ai_mode:
            return self.evaluate(step, metrics)

        if self._ai_engine is None:
            return []

        # Run rules as alert/sensor layer (no auto-apply)
        alerts = self.evaluate_rules_for_alerts(step, metrics)

        # Check if AI should be invoked
        alert_dicts = [asdict(a) for a in alerts]
        if not self._ai_engine.should_invoke(step, alert_dicts):
            # Still record alerts in history for visibility
            for alert in alerts:
                self._history.append(alert)
            return alerts

        # Build current state from latest metrics
        current_state = {}
        for name in ("lr", "weight_decay", "grad_norm"):
            val = self._get_latest_metric(name)
            if val is not None:
                current_state[name] = val

        # Invoke AI
        decision = await self._ai_engine.invoke(
            step=step,
            metric_history=dict(self._metric_history),
            alerts=alert_dicts,
            action_history=self._ai_engine.get_history(last_n=10),
            current_state=current_state,
        )

        if decision is None:
            return alerts

        # Process AI actions
        ai_actions = []
        available_metrics = list(self._metric_history.keys())
        for act in decision.actions:
            action_name = act.get("action", "")
            params = act.get("params", {})

            # Meta-actions handled by AI engine
            if action_name == "set_key_metric":
                self._ai_engine.handle_set_key_metric(
                    params.get("metric", ""), available_metrics
                )
                continue
            elif action_name == "add_watch_metric":
                self._ai_engine.handle_watch_metric(params.get("metric", ""), add=True)
                continue
            elif action_name == "remove_watch_metric":
                self._ai_engine.handle_watch_metric(params.get("metric", ""), add=False)
                continue
            elif action_name == "request_raw_metrics":
                # Handled via next_check in ai_engine
                continue
            elif action_name == "declare_rerun":
                self._ai_engine.handle_declare_rerun(
                    params.get("verdict", ""), params.get("learnings", [])
                )
                continue
            elif action_name == "finalize_recipe":
                self._ai_engine.handle_finalize_recipe(params.get("summary", ""))
                continue
            elif action_name == "noop":
                continue

            # Translate AI action to hotcb command
            cmd = self._ai_action_to_command(action_name, params)
            if cmd is None:
                continue

            status = "applied" if self._mode == "ai_auto" else "proposed"

            ap_action = AutopilotAction(
                action_id=str(uuid.uuid4()),
                rule_id=f"ai:{action_name}",
                step=step,
                wall_time=time.time(),
                condition_met=f"AI: {decision.reasoning[:120]}",
                proposed_action=cmd,
                confidence="medium",
                status=status,
            )

            if status == "applied":
                self._apply_action(cmd)

            self._history.append(ap_action)
            ai_actions.append(ap_action)

        return alerts + ai_actions

    def _ai_action_to_command(self, action_name: str, params: dict) -> Optional[dict]:
        """Convert an AI action to a hotcb command dict."""
        if action_name == "set_lr":
            return {
                "module": "opt",
                "op": "set_params",
                "params": {"lr": params.get("lr")},
                "source": "ai_autopilot",
            }
        elif action_name == "set_lr_optimizer":
            cmd_params = {"lr": params.get("lr")}
            opt_idx = params.get("opt_idx", 0)
            if opt_idx:
                cmd_params["opt_idx"] = int(opt_idx)
            return {
                "module": "opt",
                "op": "set_params",
                "params": cmd_params,
                "source": "ai_autopilot",
            }
        elif action_name == "reduce_lr_factor":
            factor = params.get("factor", 0.5)
            current_lr = self._get_latest_metric("lr")
            lr = (current_lr or 1e-4) * factor
            return {
                "module": "opt",
                "op": "set_params",
                "params": {"lr": lr},
                "source": "ai_autopilot",
            }
        elif action_name == "set_wd":
            return {
                "module": "opt",
                "op": "set_params",
                "params": {"weight_decay": params.get("weight_decay")},
                "source": "ai_autopilot",
            }
        elif action_name == "set_loss_weight":
            term = params.get("term", "weight")
            weight = params.get("weight", 1.0)
            return {
                "module": "loss",
                "op": "set_params",
                "params": {term: weight},
                "source": "ai_autopilot",
            }
        elif action_name == "enable_callback":
            return {
                "module": "cb",
                "op": "enable",
                "id": params.get("id", ""),
                "source": "ai_autopilot",
            }
        elif action_name == "disable_callback":
            return {
                "module": "cb",
                "op": "disable",
                "id": params.get("id", ""),
                "source": "ai_autopilot",
            }

        log.warning("[hotcb.autopilot] Unknown AI action: %s", action_name)
        return None

    # -- History ------------------------------------------------------------

    @property
    def history(self) -> list[AutopilotAction]:
        return list(self._history)


# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------


def create_router(engine: Optional[AutopilotEngine] = None, ai_engine: Any = None) -> Any:
    """Build the autopilot API router. Requires FastAPI."""
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    _engine = engine
    _ai_engine = ai_engine
    router = APIRouter(prefix="/api/autopilot", tags=["autopilot"])

    class ModeRequest(BaseModel):
        mode: str

    class RuleRequest(BaseModel):
        rule_id: str
        condition: str
        metric_name: str
        params: dict = {}
        action: dict = {}
        confidence: str = "medium"
        enabled: bool = True
        description: str = ""

    class GuidelinesRequest(BaseModel):
        path: str

    @router.get("/status")
    async def get_status():
        recent = _engine.history[-10:]
        return {
            "mode": _engine.mode,
            "rules_count": len(_engine.get_rules()),
            "history_count": len(_engine.history),
            "recent_actions": [asdict(a) for a in recent],
        }

    @router.post("/mode")
    async def set_mode(body: ModeRequest):
        _engine.set_mode(body.mode)
        return {"mode": _engine.mode}

    @router.get("/mode")
    async def get_mode():
        return {"mode": _engine.mode}

    @router.get("/rules")
    async def list_rules():
        return {"rules": [asdict(r) for r in _engine.get_rules()]}

    @router.post("/rules")
    async def add_rule(body: RuleRequest):
        rule = AutopilotRule(
            rule_id=body.rule_id,
            condition=body.condition,
            metric_name=body.metric_name,
            params=body.params,
            action=body.action,
            confidence=body.confidence,
            enabled=body.enabled,
            description=body.description,
        )
        _engine.add_rule(rule)
        return {"status": "added", "rule_id": rule.rule_id}

    class RuleUpdateRequest(BaseModel):
        condition: Optional[str] = None
        metric_name: Optional[str] = None
        params: Optional[dict] = None
        action: Optional[dict] = None
        confidence: Optional[str] = None
        enabled: Optional[bool] = None
        description: Optional[str] = None

    @router.put("/rules/{rule_id}")
    async def update_rule(rule_id: str, body: RuleUpdateRequest):
        changes = {k: v for k, v in body.dict().items() if v is not None}
        updated = _engine.update_rule(rule_id, changes)
        if updated is None:
            return JSONResponse(status_code=404, content={"error": f"Rule {rule_id!r} not found"})
        return {"status": "updated", "rule": asdict(updated)}

    @router.post("/rules/{rule_id}/toggle")
    async def toggle_rule(rule_id: str):
        rules = {r.rule_id: r for r in _engine.get_rules()}
        rule = rules.get(rule_id)
        if rule is None:
            return JSONResponse(status_code=404, content={"error": f"Rule {rule_id!r} not found"})
        rule.enabled = not rule.enabled
        return {"status": "toggled", "rule_id": rule_id, "enabled": rule.enabled}

    @router.delete("/rules/{rule_id}")
    async def remove_rule(rule_id: str):
        removed = _engine.remove_rule(rule_id)
        if not removed:
            return JSONResponse(status_code=404, content={"error": f"Rule {rule_id!r} not found"})
        return {"status": "removed", "rule_id": rule_id}

    @router.post("/guidelines")
    async def load_guidelines_endpoint(body: GuidelinesRequest):
        try:
            path = body.path
            if path == "__default__":
                from .guidelines import DEFAULT_GUIDELINES_PATH
                path = DEFAULT_GUIDELINES_PATH
            count = _engine.load_guidelines(path)
            return {"status": "loaded", "rules_loaded": count}
        except Exception as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

    @router.get("/history")
    async def get_history(last_n: int = 100):
        items = _engine.history[-last_n:]
        return {"actions": [asdict(a) for a in items]}

    @router.post("/accept/{action_id}")
    async def accept_action(action_id: str):
        """Accept a proposed autopilot action and apply it."""
        result = _engine.accept_action(action_id)
        if result is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Action {action_id!r} not found or already applied"},
            )
        return {"status": "applied", "action_id": action_id}

    # -- AI autopilot endpoints ---------------------------------------------

    @router.get("/ai/status")
    async def ai_status():
        """Return AI autopilot status (config, cost, key metric, etc.)."""
        if _ai_engine is None:
            return {"enabled": False, "error": "AI engine not configured"}
        return _ai_engine.get_status()

    class AIConfigUpdate(BaseModel):
        provider: Optional[str] = None
        model: Optional[str] = None
        api_key: Optional[str] = None
        base_url: Optional[str] = None
        temperature: Optional[float] = None
        max_tokens: Optional[int] = None
        cadence: Optional[int] = None
        budget_cap: Optional[float] = None
        max_runs: Optional[int] = None

    @router.post("/ai/config")
    async def ai_config_update(body: AIConfigUpdate):
        """Update AI config at runtime."""
        if _ai_engine is None:
            return JSONResponse(status_code=400, content={"error": "AI engine not configured"})
        updates = {k: v for k, v in body.dict().items() if v is not None}
        _ai_engine.update_config(updates)
        return {"status": "updated", "config": _ai_engine.config.to_safe_dict()}

    @router.get("/ai/history")
    async def ai_history(last_n: int = 50):
        """Return recent AI decisions with reasoning."""
        if _ai_engine is None:
            return {"decisions": []}
        return {"decisions": _ai_engine.get_history(last_n)}

    class KeyMetricRequest(BaseModel):
        metric: str
        mode: Optional[str] = None  # "auto", "min", or "max"

    @router.post("/ai/key_metric")
    async def ai_set_key_metric(body: KeyMetricRequest):
        """Set key metric manually.

        Allows setting any metric name — the metric doesn't need to be
        in the stream yet (it might appear in future steps).
        Optional ``mode`` sets direction: "min", "max", or "auto" (inferred).
        """
        if _ai_engine is None:
            return JSONResponse(status_code=400, content={"error": "AI engine not configured"})
        available = list(_engine._metric_history.keys()) if _engine else []
        ok = _ai_engine.handle_set_key_metric(body.metric, available)
        if not ok:
            _ai_engine.state.key_metric = body.metric
            log.info("[hotcb.autopilot] key metric force-set to: %s", body.metric)
        # Set direction mode if provided
        if body.mode and body.mode in ("auto", "min", "max"):
            _ai_engine.state.key_metric_mode = body.mode
        _ai_engine.save_state()
        return {
            "status": "updated",
            "key_metric": body.metric,
            "mode": _ai_engine.state.key_metric_mode,
            "resolved_direction": _ai_engine.state.resolved_direction,
        }

    @router.get("/ai/state")
    async def ai_state():
        """Return full AI state (multi-run context)."""
        if _ai_engine is None:
            return {"error": "AI engine not configured"}
        return _ai_engine.state.to_dict()

    return router
