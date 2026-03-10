"""
hotcb.server.notifications — Alert and notification engine.

Monitors metric streams and fires alerts when configurable conditions are met.
Supports pluggable notification channels (WebSocket, Slack, logging).
"""
from __future__ import annotations

import asyncio
import collections
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Protocol, runtime_checkable

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

log = logging.getLogger("hotcb.server.notifications")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AlertRule:
    """Defines a condition that triggers an alert."""

    rule_id: str
    metric_name: str  # which metric to monitor
    condition: str  # "gt", "lt", "gte", "lte", "plateau", "spike"
    threshold: float  # value threshold for gt/lt/gte/lte
    window: int = 5  # steps to consider for plateau/spike detection
    cooldown_steps: int = 50  # minimum steps between repeated alerts
    channels: list[str] = field(default_factory=lambda: ["websocket"])
    enabled: bool = True


@dataclass
class Alert:
    """Emitted when an AlertRule fires."""

    alert_id: str
    rule_id: str
    metric_name: str
    condition: str
    current_value: float
    threshold: float
    step: int
    wall_time: float
    message: str


# ---------------------------------------------------------------------------
# Notification channel protocol and built-in channels
# ---------------------------------------------------------------------------


@runtime_checkable
class NotificationChannel(Protocol):
    async def send(self, alert: Alert) -> bool: ...


class WebSocketChannel:
    """Sends alerts via a broadcast callback (wired to ConnectionManager)."""

    def __init__(
        self,
        broadcast: Callable[[str, Any], Coroutine[Any, Any, None]],
    ) -> None:
        self._broadcast = broadcast

    async def send(self, alert: Alert) -> bool:
        try:
            await self._broadcast("alerts", _alert_to_dict(alert))
            return True
        except Exception:
            log.warning("WebSocketChannel failed to send alert %s", alert.alert_id)
            return False


class LogChannel:
    """Logs alerts via Python logging."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._log = logger or log

    async def send(self, alert: Alert) -> bool:
        self._log.warning(
            "[ALERT %s] %s — %s %s %s (value=%s, step=%d)",
            alert.alert_id,
            alert.metric_name,
            alert.condition,
            alert.threshold,
            alert.message,
            alert.current_value,
            alert.step,
        )
        return True


class SlackChannel:
    """Sends alerts via Slack webhook (requires slack_sdk)."""

    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url

    async def send(self, alert: Alert) -> bool:
        try:
            from slack_sdk.webhook import WebhookClient

            client = WebhookClient(self._webhook_url)
            text = f":rotating_light: *hotcb Alert* — {alert.message}"
            resp = client.send(text=text)
            return resp.status_code == 200
        except ImportError:
            log.warning("slack_sdk not installed — cannot send Slack alert")
            return False
        except Exception as exc:
            log.warning("SlackChannel error: %s", exc)
            return False


class SmtpChannel:
    """Send alerts via email (SMTP)."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        from_addr: str,
        to_addrs: list[str],
        username: str | None = None,
        password: str | None = None,
        use_tls: bool = True,
    ) -> None:
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._from_addr = from_addr
        self._to_addrs = to_addrs
        self._username = username
        self._password = password
        self._use_tls = use_tls

    async def send(self, alert: Alert) -> bool:
        """Send alert as email."""
        import smtplib
        from email.mime.text import MIMEText

        body = (
            f"hotcb Alert: {alert.message}\n\n"
            f"Rule: {alert.rule_id}\n"
            f"Metric: {alert.metric_name}\n"
            f"Condition: {alert.condition}\n"
            f"Current value: {alert.current_value}\n"
            f"Threshold: {alert.threshold}\n"
            f"Step: {alert.step}\n"
            f"Time: {alert.wall_time}\n"
        )
        msg = MIMEText(body)
        msg["Subject"] = f"[hotcb] Alert: {alert.metric_name} {alert.condition} (step {alert.step})"
        msg["From"] = self._from_addr
        msg["To"] = ", ".join(self._to_addrs)

        try:
            if self._use_tls:
                server = smtplib.SMTP(self._smtp_host, self._smtp_port)
                server.starttls()
            else:
                server = smtplib.SMTP(self._smtp_host, self._smtp_port)
            if self._username and self._password:
                server.login(self._username, self._password)
            server.sendmail(self._from_addr, self._to_addrs, msg.as_string())
            server.quit()
            return True
        except Exception as exc:
            log.warning("SmtpChannel error: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Notification engine
# ---------------------------------------------------------------------------


class NotificationEngine:
    """Evaluates metric records against alert rules and dispatches notifications."""

    def __init__(self) -> None:
        self.rules: Dict[str, AlertRule] = {}
        self.history: List[Alert] = []
        self._last_fired: Dict[str, int] = {}  # rule_id -> last step fired
        self._channels: Dict[str, NotificationChannel] = {}
        # Ring buffers for plateau / spike detection: metric_name -> deque
        self._metric_buffer: Dict[str, collections.deque] = {}
        self._max_window = 100  # cap deque size

    # -- rule management -----------------------------------------------------

    def add_rule(self, rule: AlertRule) -> None:
        self.rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> None:
        self.rules.pop(rule_id, None)
        self._last_fired.pop(rule_id, None)

    # -- channel management --------------------------------------------------

    def register_channel(self, name: str, channel: NotificationChannel) -> None:
        self._channels[name] = channel

    # -- evaluation ----------------------------------------------------------

    def evaluate(self, step: int, metrics: Dict[str, float]) -> List[Alert]:
        """Evaluate all rules against the given metrics dict.  Returns fired alerts."""
        # Update ring buffers
        for name, value in metrics.items():
            if name not in self._metric_buffer:
                self._metric_buffer[name] = collections.deque(maxlen=self._max_window)
            self._metric_buffer[name].append(value)

        fired: List[Alert] = []
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            if rule.metric_name not in metrics:
                continue

            value = metrics[rule.metric_name]

            # Cooldown check
            last = self._last_fired.get(rule.rule_id)
            if last is not None and (step - last) < rule.cooldown_steps:
                continue

            triggered = self._check_condition(rule, value)
            if not triggered:
                continue

            alert = Alert(
                alert_id=uuid.uuid4().hex[:12],
                rule_id=rule.rule_id,
                metric_name=rule.metric_name,
                condition=rule.condition,
                current_value=value,
                threshold=rule.threshold,
                step=step,
                wall_time=time.time(),
                message=self._build_message(rule, value, step),
            )
            self._last_fired[rule.rule_id] = step
            self.history.append(alert)
            fired.append(alert)

        return fired

    async def dispatch(self, alerts: List[Alert]) -> None:
        """Send fired alerts to their registered channels."""
        for alert in alerts:
            rule = self.rules.get(alert.rule_id)
            channel_names = rule.channels if rule else ["websocket"]
            for ch_name in channel_names:
                ch = self._channels.get(ch_name)
                if ch is not None:
                    try:
                        await ch.send(alert)
                    except Exception as exc:
                        log.warning("Channel %s failed: %s", ch_name, exc)

    # -- condition checkers --------------------------------------------------

    def _check_condition(self, rule: AlertRule, value: float) -> bool:
        cond = rule.condition
        if cond == "gt":
            return value > rule.threshold
        elif cond == "lt":
            return value < rule.threshold
        elif cond == "gte":
            return value >= rule.threshold
        elif cond == "lte":
            return value <= rule.threshold
        elif cond == "plateau":
            return self._check_plateau(rule, value)
        elif cond == "spike":
            return self._check_spike(rule, value)
        else:
            log.warning("Unknown condition %r in rule %s", cond, rule.rule_id)
            return False

    def _check_plateau(self, rule: AlertRule, value: float) -> bool:
        """True if metric hasn't changed by more than threshold over window steps."""
        buf = self._metric_buffer.get(rule.metric_name)
        if buf is None or len(buf) < rule.window:
            return False
        recent = list(buf)[-rule.window :]
        return (max(recent) - min(recent)) <= rule.threshold

    def _check_spike(self, rule: AlertRule, value: float) -> bool:
        """True if absolute deviation from moving average exceeds threshold."""
        buf = self._metric_buffer.get(rule.metric_name)
        if buf is None or len(buf) < 2:
            return False
        # Moving average over window (exclude current value which is last)
        recent = list(buf)[-rule.window :]
        # Exclude the latest value for the average baseline
        if len(recent) < 2:
            return False
        baseline = recent[:-1]
        avg = sum(baseline) / len(baseline)
        return abs(value - avg) > rule.threshold

    def _build_message(self, rule: AlertRule, value: float, step: int) -> str:
        if rule.condition in ("gt", "lt", "gte", "lte"):
            return (
                f"{rule.metric_name} = {value:.6g} "
                f"({rule.condition} {rule.threshold}) at step {step}"
            )
        elif rule.condition == "plateau":
            return f"{rule.metric_name} plateau detected (threshold {rule.threshold}) at step {step}"
        elif rule.condition == "spike":
            return f"{rule.metric_name} spike detected (value {value:.6g}, threshold {rule.threshold}) at step {step}"
        return f"Alert on {rule.metric_name} at step {step}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _alert_to_dict(alert: Alert) -> dict:
    return {
        "alert_id": alert.alert_id,
        "rule_id": alert.rule_id,
        "metric_name": alert.metric_name,
        "condition": alert.condition,
        "current_value": alert.current_value,
        "threshold": alert.threshold,
        "step": alert.step,
        "wall_time": alert.wall_time,
        "message": alert.message,
    }


def _rule_to_dict(rule: AlertRule) -> dict:
    return {
        "rule_id": rule.rule_id,
        "metric_name": rule.metric_name,
        "condition": rule.condition,
        "threshold": rule.threshold,
        "window": rule.window,
        "cooldown_steps": rule.cooldown_steps,
        "channels": rule.channels,
        "enabled": rule.enabled,
    }


# ---------------------------------------------------------------------------
# Pydantic models for the API
# ---------------------------------------------------------------------------


class AlertRuleCreate(BaseModel):
    rule_id: Optional[str] = None
    metric_name: str
    condition: str = Field(..., pattern=r"^(gt|lt|gte|lte|plateau|spike)$")
    threshold: float
    window: int = 5
    cooldown_steps: int = 50
    channels: List[str] = Field(default_factory=lambda: ["websocket"])
    enabled: bool = True


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


@router.get("/rules")
async def list_rules(request: Request):
    engine: NotificationEngine = request.app.state.notification_engine
    return {"rules": [_rule_to_dict(r) for r in engine.rules.values()]}


@router.post("/rules")
async def add_rule(body: AlertRuleCreate, request: Request):
    engine: NotificationEngine = request.app.state.notification_engine
    rule_id = body.rule_id or uuid.uuid4().hex[:12]
    rule = AlertRule(
        rule_id=rule_id,
        metric_name=body.metric_name,
        condition=body.condition,
        threshold=body.threshold,
        window=body.window,
        cooldown_steps=body.cooldown_steps,
        channels=body.channels,
        enabled=body.enabled,
    )
    engine.add_rule(rule)
    return {"status": "added", "rule": _rule_to_dict(rule)}


@router.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str, request: Request):
    engine: NotificationEngine = request.app.state.notification_engine
    if rule_id not in engine.rules:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=404, content={"detail": f"Rule {rule_id!r} not found"}
        )
    engine.remove_rule(rule_id)
    return {"status": "removed", "rule_id": rule_id}


@router.get("/history")
async def get_history(request: Request, last_n: int = 50):
    engine: NotificationEngine = request.app.state.notification_engine
    alerts = engine.history[-last_n:]
    return {"alerts": [_alert_to_dict(a) for a in alerts]}


@router.get("/alerts")
async def get_alerts(request: Request, last_n: int = 50):
    engine: NotificationEngine = request.app.state.notification_engine
    alerts = engine.history[-last_n:]
    return {"alerts": [_alert_to_dict(a) for a in alerts]}


@router.post("/rules/{rule_id}/test")
async def test_rule(rule_id: str, request: Request):
    engine: NotificationEngine = request.app.state.notification_engine
    rule = engine.rules.get(rule_id)
    if rule is None:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=404, content={"detail": f"Rule {rule_id!r} not found"}
        )
    # Create a fake alert
    fake_value = rule.threshold + 1.0 if rule.condition in ("gt", "gte") else rule.threshold - 1.0
    alert = Alert(
        alert_id=uuid.uuid4().hex[:12],
        rule_id=rule.rule_id,
        metric_name=rule.metric_name,
        condition=rule.condition,
        current_value=fake_value,
        threshold=rule.threshold,
        step=0,
        wall_time=time.time(),
        message=f"[TEST] {rule.metric_name} test alert for rule {rule.rule_id}",
    )
    engine.history.append(alert)
    await engine.dispatch([alert])
    return {"status": "test_fired", "alert": _alert_to_dict(alert)}


# ---------------------------------------------------------------------------
# Metrics subscriber for the tailer
# ---------------------------------------------------------------------------


def make_metrics_subscriber(
    engine: NotificationEngine,
) -> Callable[[str, list], Coroutine[Any, Any, None]]:
    """
    Returns an async callback suitable for ``tailer.subscribe("metrics", cb)``.
    Each metrics record has shape: {"step": N, "epoch": M, "wall_time": T, "metrics": {...}}
    """

    async def _on_metrics(channel: str, records: list) -> None:
        for rec in records:
            step = rec.get("step", 0)
            mets = rec.get("metrics", {})
            if mets:
                alerts = engine.evaluate(step, mets)
                if alerts:
                    await engine.dispatch(alerts)

    return _on_metrics
