"""
hotcb.server.ai_engine — LLM-driven autopilot engine.

Replaces rule-only autopilot with an AI agent that reads metrics, alerts,
and trend summaries, then proposes hotcb commands via an OpenAI-compatible
API.  State persists in ``hotcb.ai.state.json`` for multi-run awareness.
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

log = logging.getLogger("hotcb.server.ai_engine")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AIConfig:
    """Runtime-mutable configuration for the LLM autopilot."""

    provider: str = "openai"  # "openai" (covers openai, ollama, vllm)
    model: str = "gpt-4o-mini"
    api_key: str = ""  # falls back to HOTCB_AI_KEY env
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.3
    max_tokens: int = 1024
    cadence: int = 50  # default periodic cadence (steps)
    budget_cap: float = 5.0  # USD limit
    max_runs: int = 3  # max reruns before AI stops proposing

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("HOTCB_AI_KEY", "")

    def to_safe_dict(self) -> dict:
        """Serialize with API key redacted."""
        d = asdict(self)
        if d["api_key"]:
            d["api_key"] = d["api_key"][:4] + "..." + d["api_key"][-4:]
        return d


# ---------------------------------------------------------------------------
# Persistent AI state
# ---------------------------------------------------------------------------

@dataclass
class AIState:
    """Persisted in ``hotcb.ai.state.json`` across runs."""

    key_metric: str = "val_loss"
    key_metric_mode: str = "auto"  # "auto", "min", or "max"
    watch_metrics: List[str] = field(default_factory=list)
    run_number: int = 1
    max_runs: int = 3
    run_history: List[dict] = field(default_factory=list)
    carried_context: str = ""
    next_check_step: Optional[int] = None
    cadence_override: Optional[int] = None
    # Transient per-run tracking
    watch_metrics_raw: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def resolved_direction(self) -> str:
        """Return 'min' or 'max' — resolves 'auto' from the metric name."""
        if self.key_metric_mode in ("min", "max"):
            return self.key_metric_mode
        return infer_metric_direction(self.key_metric)

    @classmethod
    def from_dict(cls, data: dict) -> "AIState":
        return cls(
            key_metric=data.get("key_metric", "val_loss"),
            key_metric_mode=data.get("key_metric_mode", "auto"),
            watch_metrics=data.get("watch_metrics", []),
            run_number=data.get("run_number", 1),
            max_runs=data.get("max_runs", 3),
            run_history=data.get("run_history", []),
            carried_context=data.get("carried_context", ""),
            next_check_step=data.get("next_check_step"),
            cadence_override=data.get("cadence_override"),
            watch_metrics_raw=data.get("watch_metrics_raw", []),
        )


def infer_metric_direction(name: str) -> str:
    """Infer whether a metric should be minimized or maximized from its name.

    Returns ``"min"`` or ``"max"``.
    """
    low = name.lower()
    # Patterns that clearly mean "lower is better"
    _MIN_PATTERNS = (
        "loss", "error", "err", "perplexity", "ppl", "mse", "mae", "rmse",
        "cer", "wer", "fid", "divergence", "regret", "cost",
    )
    # Patterns that clearly mean "higher is better"
    _MAX_PATTERNS = (
        "accuracy", "acc", "f1", "auc", "auroc", "recall", "precision",
        "score", "bleu", "rouge", "meteor", "iou", "dice", "map",
        "reward", "return", "r2", "correlation", "similarity",
        "alignment", "coherence", "fluency",
    )
    for pat in _MIN_PATTERNS:
        if pat in low:
            return "min"
    for pat in _MAX_PATTERNS:
        if pat in low:
            return "max"
    # Default: assume minimization (most common in ML)
    return "min"


# ---------------------------------------------------------------------------
# AI decision record
# ---------------------------------------------------------------------------

@dataclass
class AIDecision:
    """A single AI autopilot decision."""

    step: int
    wall_time: float
    reasoning: str
    actions: List[dict]
    next_check: dict
    watch_metrics_raw: List[str]
    status: str = "proposed"  # "proposed", "applied", "rejected"
    cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# LLM Autopilot Engine
# ---------------------------------------------------------------------------

class LLMAutopilotEngine:
    """
    Core AI decision engine.  Manages LLM calls, state persistence,
    cadence control, cost tracking, and action history.
    """

    def __init__(self, run_dir: str, config: Optional[AIConfig] = None):
        self._run_dir = run_dir
        self.config = config or AIConfig()
        self.state = AIState(max_runs=self.config.max_runs)
        self._decisions: List[AIDecision] = []
        self._total_cost: float = 0.0
        self._call_count: int = 0
        self._last_invoked_step: int = -1
        self._enabled: bool = True
        self._last_step_with_metrics: int = 0

        # Load persisted state if exists
        self.load_state()

    # -- State persistence ---------------------------------------------------

    def _state_path(self) -> str:
        return os.path.join(self._run_dir, "hotcb.ai.state.json")

    def load_state(self) -> None:
        """Load AI state from disk if available."""
        path = self._state_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.state = AIState.from_dict(data)
                log.info(
                    "[hotcb.ai] Loaded state: run %d/%d, key_metric=%s",
                    self.state.run_number,
                    self.state.max_runs,
                    self.state.key_metric,
                )
            except Exception as exc:
                log.warning("[hotcb.ai] Failed to load state: %s", exc)

    def save_state(self) -> None:
        """Persist AI state to disk."""
        path = self._state_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as exc:
            log.warning("[hotcb.ai] Failed to save state: %s", exc)

    # -- Cadence control -----------------------------------------------------

    def should_invoke(
        self, step: int, alerts: List[dict], *, force: bool = False
    ) -> bool:
        """
        Decide whether to invoke the LLM at this step.

        Returns True if:
        - An alert fired (on-alert mode)
        - Periodic cadence hit
        - AI-requested step reached
        - Forced by caller
        """
        if not self._enabled:
            return False

        if not self.config.api_key:
            return False

        # Budget exhausted
        if self._total_cost >= self.config.budget_cap:
            log.info("[hotcb.ai] Budget cap reached (%.2f >= %.2f)", self._total_cost, self.config.budget_cap)
            self._enabled = False
            return False

        if force:
            return True

        # Minimum cooldown: 10 steps between invocations
        if step - self._last_invoked_step < 10:
            return False

        # On-alert: any alert fires
        if alerts:
            return True

        # AI-requested specific step
        if self.state.next_check_step is not None and step >= self.state.next_check_step:
            return True

        # Periodic cadence
        cadence = self.state.cadence_override or self.config.cadence
        if cadence > 0 and self._last_invoked_step >= 0:
            if step - self._last_invoked_step >= cadence:
                return True

        # First invocation after enough data
        if self._last_invoked_step < 0 and step >= 20:
            return True

        return False

    def get_context_mode(self, alerts: List[dict]) -> str:
        """Determine context mode based on invocation trigger."""
        if alerts:
            return "alert"
        if (
            self.state.next_check_step is not None
            and self._last_step_with_metrics >= self.state.next_check_step
        ):
            return "full"
        return "trend"

    # -- LLM invocation ------------------------------------------------------

    async def invoke(
        self,
        step: int,
        metric_history: Dict[str, List[float]],
        alerts: List[dict],
        action_history: List[dict],
        current_state: Dict[str, Any],
    ) -> Optional[AIDecision]:
        """
        Assemble prompt, call LLM, parse response, return decision.
        """
        from .ai_prompts import build_context, parse_ai_response

        context_mode = self.get_context_mode(alerts)
        # Load training capabilities for context-aware prompts
        caps_dict = None
        try:
            from ..capabilities import TrainingCapabilities
            caps = TrainingCapabilities.load(self._run_dir)
            if caps is not None:
                caps_dict = {"detected": True, **caps.to_dict()}
        except Exception:
            pass

        messages = build_context(
            step=step,
            metric_history=metric_history,
            alerts=[asdict(a) if hasattr(a, "__dataclass_fields__") else a for a in alerts],
            action_history=[asdict(d) if hasattr(d, "__dataclass_fields__") else d for d in action_history],
            current_state=current_state,
            ai_state=self.state.to_dict(),
            mode=context_mode,
            watch_metrics_raw=self.state.watch_metrics_raw,
            capabilities=caps_dict,
        )

        self._last_invoked_step = step

        try:
            raw_response, cost = await self._call_llm(messages)
        except Exception as exc:
            log.error("[hotcb.ai] LLM call failed: %s", exc)
            return None

        self._call_count += 1
        self._total_cost += cost

        parsed = parse_ai_response(raw_response)
        if parsed is None:
            log.warning("[hotcb.ai] Failed to parse LLM response")
            return None

        decision = AIDecision(
            step=step,
            wall_time=time.time(),
            reasoning=parsed["reasoning"],
            actions=parsed["actions"],
            next_check=parsed["next_check"],
            watch_metrics_raw=parsed.get("watch_metrics_raw", []),
            cost_usd=cost,
        )

        # Process next_check
        nc = parsed["next_check"]
        nc_mode = nc.get("mode", "periodic")
        if nc_mode == "at_step":
            self.state.next_check_step = nc.get("step")
        elif nc_mode == "in_n_steps":
            self.state.next_check_step = step + nc.get("n", 50)
        elif nc_mode == "on_next_alert":
            self.state.next_check_step = None  # will fire on next alert
            self.state.cadence_override = 999999  # effectively disable periodic
        elif nc_mode == "periodic":
            interval = nc.get("interval", self.config.cadence)
            self.state.cadence_override = interval
            self.state.next_check_step = None

        # Update watch_metrics_raw for next invocation
        self.state.watch_metrics_raw = parsed.get("watch_metrics_raw", [])

        self._decisions.append(decision)
        self.save_state()

        log.info(
            "[hotcb.ai] Decision at step %d: %d actions, reasoning: %s",
            step,
            len(decision.actions),
            decision.reasoning[:80],
        )

        return decision

    async def _call_llm(self, messages: List[Dict[str, str]]) -> tuple:
        """
        Call the LLM via OpenAI-compatible API. Returns (response_text, cost_usd).
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for AI autopilot: pip install 'hotcb[ai]'"
            )

        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        # Extract response text
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No choices in LLM response")
        text = choices[0].get("message", {}).get("content", "")

        # Estimate cost from usage
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        # Rough cost estimate (gpt-4o-mini pricing as baseline)
        cost = (prompt_tokens * 0.15 + completion_tokens * 0.6) / 1_000_000

        return text, cost

    # -- Meta-action handlers ------------------------------------------------

    def handle_set_key_metric(
        self, metric: str, available_metrics: List[str]
    ) -> bool:
        """Handle set_key_metric action. Returns True if metric exists."""
        if metric in available_metrics:
            self.state.key_metric = metric
            self.save_state()
            log.info("[hotcb.ai] Key metric changed to: %s", metric)
            return True
        log.warning(
            "[hotcb.ai] Rejected key metric %s — not in available metrics", metric
        )
        return False

    def handle_watch_metric(self, metric: str, add: bool = True) -> None:
        """Add or remove a watch metric."""
        if add:
            if metric not in self.state.watch_metrics:
                self.state.watch_metrics.append(metric)
        else:
            if metric in self.state.watch_metrics:
                self.state.watch_metrics.remove(metric)
        self.save_state()

    def handle_declare_rerun(self, verdict: str, learnings: List[str]) -> Optional[dict]:
        """Handle declare_rerun. Returns rerun info or None if max runs reached."""
        if self.state.run_number >= self.state.max_runs:
            log.info("[hotcb.ai] Max runs reached, ignoring declare_rerun")
            return None

        run_entry = {
            "run_id": f"run_{self.state.run_number:03d}",
            "final_key_metric": None,  # to be filled by caller
            "ai_verdict": verdict,
            "carried_learnings": learnings,
        }
        self.state.run_history.append(run_entry)
        self.state.carried_context = (
            f"Previous run declared degenerate: {verdict}. "
            f"Learnings: {'; '.join(learnings)}"
        )
        self.save_state()

        return run_entry

    def handle_finalize_recipe(self, summary: str) -> dict:
        """Handle finalize_recipe action."""
        log.info("[hotcb.ai] Recipe finalized: %s", summary)
        return {"status": "finalized", "summary": summary}

    # -- Run lifecycle -------------------------------------------------------

    def on_run_start(self) -> None:
        """Called when a new training run starts."""
        self.load_state()
        self.state.run_number += 1
        self._decisions.clear()
        self._last_invoked_step = -1
        self._enabled = True
        self.save_state()
        log.info(
            "[hotcb.ai] Run %d started, carried context: %s",
            self.state.run_number,
            self.state.carried_context[:80] if self.state.carried_context else "none",
        )

    def on_run_end(self, final_metrics: Optional[Dict[str, float]] = None) -> None:
        """Called when training ends. Snapshot final state."""
        if final_metrics and self.state.run_history:
            last_run = self.state.run_history[-1]
            key = self.state.key_metric
            if key in final_metrics:
                last_run["final_key_metric"] = final_metrics[key]
        self.save_state()

    # -- Status / history accessors ------------------------------------------

    @property
    def decisions(self) -> List[AIDecision]:
        return list(self._decisions)

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def enabled(self) -> bool:
        return self._enabled

    def disable(self, reason: str = "") -> None:
        """Disable the AI engine (e.g., after divergence)."""
        self._enabled = False
        log.warning("[hotcb.ai] Disabled: %s", reason or "manual")

    def get_status(self) -> dict:
        """Return status summary for API."""
        return {
            "enabled": self._enabled,
            "config": self.config.to_safe_dict(),
            "call_count": self._call_count,
            "total_cost_usd": round(self._total_cost, 4),
            "budget_remaining_usd": round(
                max(0, self.config.budget_cap - self._total_cost), 4
            ),
            "key_metric": self.state.key_metric,
            "key_metric_mode": self.state.key_metric_mode,
            "key_metric_direction": self.state.resolved_direction,
            "watch_metrics": self.state.watch_metrics,
            "run_number": self.state.run_number,
            "max_runs": self.state.max_runs,
            "last_invoked_step": self._last_invoked_step,
            "next_check_step": self.state.next_check_step,
            "decisions_count": len(self._decisions),
        }

    def get_history(self, last_n: int = 50) -> List[dict]:
        """Return recent decisions as dicts."""
        items = self._decisions[-last_n:]
        return [asdict(d) for d in items]

    def update_config(self, updates: dict) -> None:
        """Update config fields at runtime."""
        for key in (
            "provider", "model", "api_key", "base_url",
            "temperature", "max_tokens", "cadence", "budget_cap", "max_runs",
        ):
            if key in updates:
                setattr(self.config, key, updates[key])
        self.state.max_runs = self.config.max_runs
        log.info("[hotcb.ai] Config updated: %s", list(updates.keys()))
