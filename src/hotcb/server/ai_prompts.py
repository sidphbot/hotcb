"""
hotcb.server.ai_prompts — Prompt assembly, trend compression, and action schema.

Builds structured context for the LLM autopilot from metrics, alerts,
and action history.  The ``TrendCompressor`` reduces raw metric streams
to slope / volatility / direction summaries so periodic updates stay
token-efficient.
"""
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

log = logging.getLogger("hotcb.server.ai_prompts")

# ---------------------------------------------------------------------------
# Action schema — constrained vocabulary the LLM must use
# ---------------------------------------------------------------------------

ACTION_SCHEMA: Dict[str, dict] = {
    "set_lr": {
        "params": {"lr": {"type": "float", "min": 1e-7, "max": 1.0}},
        "description": "Set learning rate to an absolute value",
    },
    "reduce_lr_factor": {
        "params": {"factor": {"type": "float", "min": 0.01, "max": 0.99}},
        "description": "Multiply current lr by factor (e.g. 0.5 = halve it)",
    },
    "set_lr_optimizer": {
        "params": {
            "lr": {"type": "float", "min": 1e-7, "max": 1.0},
            "opt_idx": {"type": "int", "min": 0, "description": "0-based optimizer index"},
        },
        "description": "Set learning rate for a specific optimizer (multi-optimizer setups)",
    },
    "set_wd": {
        "params": {"weight_decay": {"type": "float", "min": 0.0, "max": 1.0}},
        "description": "Set weight decay to an absolute value",
    },
    "set_loss_weight": {
        "params": {
            "term": {"type": "string"},
            "weight": {"type": "float", "min": 0.0, "max": 10.0},
        },
        "description": "Set a loss term's weight (e.g. term='weight_a', weight=0.8)",
    },
    "enable_callback": {
        "params": {"id": {"type": "string"}},
        "description": "Enable a registered callback by ID",
    },
    "disable_callback": {
        "params": {"id": {"type": "string"}},
        "description": "Disable a registered callback by ID",
    },
    "set_key_metric": {
        "params": {"metric": {"type": "string"}},
        "description": "Change the primary optimization target metric",
    },
    "add_watch_metric": {
        "params": {"metric": {"type": "string"}},
        "description": "Add a metric to the watch list for closer monitoring",
    },
    "remove_watch_metric": {
        "params": {"metric": {"type": "string"}},
        "description": "Remove a metric from the watch list",
    },
    "request_raw_metrics": {
        "params": {"metrics": {"type": "list[string]"}},
        "description": "Request raw values for specific metrics on next check-in",
    },
    "declare_rerun": {
        "params": {
            "verdict": {"type": "string"},
            "learnings": {"type": "list[string]"},
        },
        "description": "Declare this run degenerate; propose restart with learnings",
    },
    "finalize_recipe": {
        "params": {"summary": {"type": "string"}},
        "description": "Declare training satisfactory; export current recipe",
    },
    "noop": {
        "params": {},
        "description": "Do nothing — training is healthy, no intervention needed",
    },
}


# ---------------------------------------------------------------------------
# Trend compression
# ---------------------------------------------------------------------------

_TREND_LABELS = {
    "steep_down": "\u2193 steep",
    "steady_down": "\u2193 steady",
    "slow_down": "\u2193 slow",
    "flat": "\u2192 flat",
    "slow_up": "\u2191 slow",
    "rising": "\u2191 rising",
    "spike": "\u2191 spike",
}


@dataclass
class TrendSummary:
    metric: str
    trend: str  # one of _TREND_LABELS keys
    trend_label: str  # human-readable arrow label
    slope: float
    volatility: str  # "none", "low", "medium", "high"
    notable: str = ""
    last_value: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0


class TrendCompressor:
    """Compress raw metric history into trend summaries."""

    def compress(
        self, values: List[float], metric_name: str, window: int = 50
    ) -> TrendSummary:
        """Compute trend summary for a single metric over the last *window* values."""
        vals = values[-window:] if len(values) > window else list(values)
        n = len(vals)

        if n < 2:
            return TrendSummary(
                metric=metric_name,
                trend="flat",
                trend_label=_TREND_LABELS["flat"],
                slope=0.0,
                volatility="none",
                last_value=vals[-1] if vals else 0.0,
                min_value=vals[-1] if vals else 0.0,
                max_value=vals[-1] if vals else 0.0,
            )

        # Linear regression for slope
        x_mean = (n - 1) / 2.0
        y_mean = sum(vals) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0.0

        # Volatility: coefficient of variation
        std = math.sqrt(sum((v - y_mean) ** 2 for v in vals) / n)
        cv = std / abs(y_mean) if abs(y_mean) > 1e-10 else 0.0

        if cv < 0.01:
            volatility = "none"
        elif cv < 0.05:
            volatility = "low"
        elif cv < 0.15:
            volatility = "medium"
        else:
            volatility = "high"

        # Trend direction classification
        # Normalize slope relative to value magnitude
        norm_slope = slope / abs(y_mean) if abs(y_mean) > 1e-10 else slope
        if norm_slope < -0.01:
            trend = "steep_down"
        elif norm_slope < -0.002:
            trend = "steady_down"
        elif norm_slope < -0.0002:
            trend = "slow_down"
        elif norm_slope < 0.0002:
            trend = "flat"
        elif norm_slope < 0.002:
            trend = "slow_up"
        elif norm_slope < 0.01:
            trend = "rising"
        else:
            trend = "spike"

        # Notable events
        notables = []
        last_val = vals[-1]
        min_val = min(vals)
        max_val = max(vals)

        if last_val == min_val:
            notables.append("new min")
        if last_val == max_val and trend in ("slow_up", "rising", "spike"):
            notables.append("new max")

        # Check for trend reversal in second half vs first half
        if n >= 10:
            mid = n // 2
            first_half_slope = self._quick_slope(vals[:mid])
            second_half_slope = self._quick_slope(vals[mid:])
            if first_half_slope < -0.0001 and second_half_slope > 0.0001:
                notables.append("trend reversal (down\u2192up)")
            elif first_half_slope > 0.0001 and second_half_slope < -0.0001:
                notables.append("trend reversal (up\u2192down)")

        # Check for sudden spike in last few values
        if n >= 5:
            recent_3 = vals[-3:]
            prev_mean = sum(vals[-8:-3]) / min(5, max(1, len(vals[-8:-3]))) if n >= 8 else y_mean
            if prev_mean > 0 and max(recent_3) > prev_mean * 1.5:
                notables.append(f"spike at recent steps")

        return TrendSummary(
            metric=metric_name,
            trend=trend,
            trend_label=_TREND_LABELS[trend],
            slope=round(slope, 8),
            volatility=volatility,
            notable="; ".join(notables),
            last_value=round(last_val, 6),
            min_value=round(min_val, 6),
            max_value=round(max_val, 6),
        )

    def _quick_slope(self, vals: List[float]) -> float:
        n = len(vals)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(vals) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0

    def format_trend_table(self, summaries: List[TrendSummary]) -> str:
        """Format trend summaries as a markdown table for the LLM prompt."""
        lines = [
            "| Metric | Trend | Slope | Volatility | Last | Notable |",
            "|--------|-------|-------|------------|------|---------|",
        ]
        for s in summaries:
            slope_str = f"{s.slope:+.6f}/step"
            lines.append(
                f"| {s.metric} | {s.trend_label} | {slope_str} | "
                f"{s.volatility} | {s.last_value:.6g} | {s.notable} |"
            )
        return "\n".join(lines)

    def format_raw_metrics(
        self,
        metric_history: Dict[str, List[float]],
        names: List[str],
        last_n: int = 20,
    ) -> str:
        """Format raw metric values for specific metrics."""
        lines = ["## Raw Metric Values (last {} steps)".format(last_n)]
        for name in names:
            vals = metric_history.get(name, [])
            recent = vals[-last_n:]
            if not recent:
                lines.append(f"- **{name}**: no data")
            else:
                formatted = [f"{v:.6g}" for v in recent]
                lines.append(f"- **{name}**: [{', '.join(formatted)}]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the AI autopilot for hotcb, a live training control plane for PyTorch.
You observe training metrics, alerts, and trends, then decide whether to
intervene by adjusting hyperparameters, loss weights, or callbacks.

## Your Role
- You optimize training toward the **key metric** (currently: {key_metric}).
- Rule-based alerts fire on ALL metrics for health monitoring. You receive these as context.
- You make GRADUATED interventions: prefer small changes over drastic ones.
- When training is healthy, do NOTHING (use the "noop" action).
- Only intervene when there's clear evidence of a problem or opportunity.

## Key Metric
The key metric is the primary optimization target: **{key_metric}**
You can change it via the "set_key_metric" action if you believe a different
metric is more informative at this stage of training.

## Watch Metrics
Additional metrics you're monitoring closely: {watch_metrics}

## Multi-Run Context
Run {run_number} of {max_runs}.
{carried_context}

## Available Actions
{action_descriptions}

## Response Format
You MUST respond with valid JSON:
```json
{{
  "reasoning": "1-3 sentences explaining your assessment and decision",
  "actions": [
    {{"action": "<action_name>", "params": {{...}}}}
  ],
  "next_check": {{
    "mode": "at_step" | "in_n_steps" | "on_next_alert" | "periodic",
    "step": <number if at_step>,
    "n": <number if in_n_steps>,
    "interval": <number if periodic>
  }},
  "watch_metrics_raw": ["metric_name_1"]
}}
```

## Guidelines
- Prefer "noop" when metrics are trending well. Don't fix what isn't broken.
- Make small adjustments first (e.g., reduce lr by 0.5x, not 0.1x).
- After making a change, set next_check to wait enough steps to observe the effect (typically 20-50 steps).
- If you see divergence (loss spike), reduce lr aggressively.
- If you see plateau, try reducing lr by 0.5x or adjusting loss weights.
- Only declare_rerun if training is truly degenerate (loss diverged, NaN, etc.).
- You can request raw metric values for specific metrics if trend summaries aren't enough.
- Keep reasoning concise but specific — reference actual metric values.
"""


def _format_action_descriptions() -> str:
    """Format action schema as human-readable descriptions for the prompt."""
    lines = []
    for name, schema in ACTION_SCHEMA.items():
        params = schema["params"]
        param_strs = []
        for pname, pdef in params.items():
            ptype = pdef.get("type", "any")
            constraints = []
            if "min" in pdef:
                constraints.append(f"min={pdef['min']}")
            if "max" in pdef:
                constraints.append(f"max={pdef['max']}")
            constraint_str = f" ({', '.join(constraints)})" if constraints else ""
            param_strs.append(f"{pname}: {ptype}{constraint_str}")
        params_desc = ", ".join(param_strs) if param_strs else "none"
        lines.append(f"- **{name}**({params_desc}): {schema['description']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------


def build_context(
    step: int,
    metric_history: Dict[str, List[float]],
    alerts: List[dict],
    action_history: List[dict],
    current_state: Dict[str, Any],
    ai_state: dict,
    mode: str = "trend",
    watch_metrics_raw: Optional[List[str]] = None,
    capabilities: Optional[dict] = None,
) -> List[Dict[str, str]]:
    """
    Assemble the LLM message list for an autopilot invocation.

    Parameters
    ----------
    mode : str
        - ``"trend"`` — compressed trends only (periodic updates)
        - ``"full"``  — raw + trends (AI-requested check-in)
        - ``"alert"`` — full context + alert details (on-alert)
    watch_metrics_raw : list[str] | None
        Metrics for which to include raw values (from previous AI request).
    """
    compressor = TrendCompressor()

    # Build trend summaries for all metrics
    summaries = []
    for name, values in sorted(metric_history.items()):
        if values:
            summaries.append(compressor.compress(values, name))

    # System prompt
    key_metric = ai_state.get("key_metric", "val_loss")
    watch_metrics = ai_state.get("watch_metrics", [])
    run_number = ai_state.get("run_number", 1)
    max_runs = ai_state.get("max_runs", 3)
    carried_context = ai_state.get("carried_context", "No previous runs.")
    run_history = ai_state.get("run_history", [])

    if run_history:
        history_lines = ["### Previous Run History"]
        for rh in run_history[-3:]:  # last 3 runs
            history_lines.append(
                f"- **{rh.get('run_id', '?')}**: key_metric={rh.get('final_key_metric', '?')}, "
                f"verdict: {rh.get('ai_verdict', 'none')}"
            )
            for learning in rh.get("carried_learnings", []):
                history_lines.append(f"  - Learning: {learning}")
        carried_ctx_full = carried_context + "\n" + "\n".join(history_lines)
    else:
        carried_ctx_full = carried_context

    system = SYSTEM_PROMPT.format(
        key_metric=key_metric,
        watch_metrics=", ".join(watch_metrics) if watch_metrics else "none",
        run_number=run_number,
        max_runs=max_runs,
        carried_context=carried_ctx_full,
        action_descriptions=_format_action_descriptions(),
    )

    # User message with context
    user_parts = [f"## Current Step: {step}\n"]

    # Trend table (always included)
    if summaries:
        user_parts.append("## Metric Trends")
        user_parts.append(compressor.format_trend_table(summaries))
        user_parts.append("")

    # Raw values for watched/requested metrics
    raw_metrics_to_show = list(watch_metrics_raw or [])
    if mode in ("full", "alert"):
        # In full/alert mode, include raw for key metric and watch metrics
        raw_metrics_to_show = list(
            set(raw_metrics_to_show + [key_metric] + watch_metrics)
        )

    if raw_metrics_to_show:
        user_parts.append(
            compressor.format_raw_metrics(metric_history, raw_metrics_to_show)
        )
        user_parts.append("")

    # Alerts
    if alerts:
        user_parts.append("## Active Alerts")
        for alert in alerts[-10:]:
            rule_id = alert.get("rule_id", "?")
            condition = alert.get("condition_met", "")
            user_parts.append(f"- **{rule_id}**: {condition}")
        user_parts.append("")

    # Recent action history (last 5 AI actions)
    if action_history:
        user_parts.append("## Recent AI Actions (newest first)")
        for act in action_history[-5:][::-1]:
            act_step = act.get("step", "?")
            reasoning = act.get("reasoning", "")
            actions_taken = act.get("actions", [])
            actions_str = ", ".join(
                a.get("action", "?") for a in actions_taken
            )
            user_parts.append(
                f"- Step {act_step}: {actions_str} — {reasoning[:100]}"
            )
        user_parts.append("")

    # Current optimizer/loss state
    if current_state:
        user_parts.append("## Current Training State")
        for k, v in sorted(current_state.items()):
            user_parts.append(f"- **{k}**: {v}")
        user_parts.append("")

    # Training capabilities (if detected)
    if capabilities and capabilities.get("detected", False):
        user_parts.append("## Training Setup")
        cap_lines = []
        fw = capabilities.get("framework", "unknown")
        cap_lines.append(f"- **Framework**: {fw}")
        n_opts = capabilities.get("num_optimizers", 1)
        opt_names = capabilities.get("optimizer_names", [])
        if n_opts > 1:
            cap_lines.append(f"- **Optimizers**: {n_opts} — {', '.join(opt_names)}")
            cap_lines.append("  - Use `opt_idx` param (0-based) to target a specific optimizer")
        elif opt_names:
            cap_lines.append(f"- **Optimizer**: {opt_names[0]}")
        pg = capabilities.get("num_param_groups", [])
        if pg:
            cap_lines.append(f"- **Param groups per optimizer**: {pg}")
        if capabilities.get("has_scheduler"):
            sched_types = capabilities.get("scheduler_types", [])
            cap_lines.append(f"- **Scheduler**: {', '.join(sched_types) if sched_types else 'yes'}")
        ga = capabilities.get("grad_accumulation_steps", 1)
        if ga > 1:
            cap_lines.append(f"- **Grad accumulation**: {ga} steps")
        if capabilities.get("loss_state_detected"):
            ls_keys = capabilities.get("loss_state_keys", [])
            cap_lines.append(f"- **Loss terms**: {', '.join(ls_keys)}")
        clip = capabilities.get("grad_clip_value")
        if clip is not None:
            wired = capabilities.get("grad_clip_wired", False)
            cap_lines.append(f"- **Grad clip**: {clip} ({'wired' if wired else 'advisory only'})")
        user_parts.extend(cap_lines)
        user_parts.append("")

    # Available metrics list
    user_parts.append(
        f"## Available Metrics: {', '.join(sorted(metric_history.keys()))}"
    )

    user_content = "\n".join(user_parts)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Response parsing + validation
# ---------------------------------------------------------------------------


def parse_ai_response(raw: str) -> Optional[dict]:
    """Parse and validate an LLM response JSON.

    Returns a dict with keys: reasoning, actions, next_check, watch_metrics_raw.
    Returns None if parsing fails.
    """
    import json

    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("Failed to parse AI response as JSON: %s", text[:200])
        return None

    if not isinstance(data, dict):
        log.warning("AI response is not a dict")
        return None

    # Validate required fields
    reasoning = data.get("reasoning", "")
    actions = data.get("actions", [])
    next_check = data.get("next_check", {"mode": "periodic", "interval": 50})
    watch_raw = data.get("watch_metrics_raw", [])

    if not isinstance(actions, list):
        actions = []

    # Validate each action
    validated_actions = []
    for act in actions:
        if not isinstance(act, dict):
            continue
        action_name = act.get("action", "")
        if action_name not in ACTION_SCHEMA:
            log.warning("Unknown action: %s", action_name)
            continue

        params = act.get("params", {})
        schema = ACTION_SCHEMA[action_name]

        # Validate param bounds
        valid = True
        for pname, pdef in schema["params"].items():
            if pname in params:
                val = params[pname]
                if isinstance(val, (int, float)):
                    if "min" in pdef and val < pdef["min"]:
                        log.warning(
                            "Action %s param %s=%s below min %s",
                            action_name, pname, val, pdef["min"],
                        )
                        valid = False
                    if "max" in pdef and val > pdef["max"]:
                        log.warning(
                            "Action %s param %s=%s above max %s",
                            action_name, pname, val, pdef["max"],
                        )
                        valid = False

        if valid:
            validated_actions.append({"action": action_name, "params": params})

    return {
        "reasoning": str(reasoning),
        "actions": validated_actions,
        "next_check": next_check if isinstance(next_check, dict) else {},
        "watch_metrics_raw": watch_raw if isinstance(watch_raw, list) else [],
    }
