from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Mutation:
    """A single proposed or applied mutation."""
    mutation_id: str
    step: int
    epoch: int
    phase_bin: str
    event: str
    actuator: str
    patch: dict
    proposal_source: str = "tpe"
    context: Dict[str, Any] = field(default_factory=dict)
    snapshot_ref: Optional[str] = None
    status: str = "proposed"  # proposed, applied, rejected, rolled_back, failed, blocked

    def to_dict(self) -> dict:
        return {
            "mutation_id": self.mutation_id,
            "step": self.step,
            "epoch": self.epoch,
            "phase_bin": self.phase_bin,
            "event": self.event,
            "actuator": self.actuator,
            "patch": self.patch,
            "proposal_source": self.proposal_source,
            "context": self.context,
            "snapshot_ref": self.snapshot_ref,
            "status": self.status,
        }


@dataclass
class Segment:
    """Evaluation window following one mutation."""
    segment_id: str
    mutation_id: str
    start_step: int
    end_step: Optional[int] = None
    horizon_type: str = "next_val_epoch_end"
    pre: Dict[str, float] = field(default_factory=dict)
    post: Dict[str, float] = field(default_factory=dict)
    delta: Dict[str, float] = field(default_factory=dict)
    stability: Dict[str, bool] = field(default_factory=lambda: {
        "nan": False, "anomaly": False, "grad_spike": False,
    })
    decision: Optional[str] = None  # accepted, rejected, rolled_back
    score_delta: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "mutation_id": self.mutation_id,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "horizon_type": self.horizon_type,
            "pre": self.pre,
            "post": self.post,
            "delta": self.delta,
            "stability": self.stability,
            "decision": self.decision,
            "score_delta": self.score_delta,
        }


@dataclass
class TuneState:
    """Mutable runtime state for the tuner."""
    mode: str = "off"  # off, observe, suggest, active, replay
    mutation_counter: int = 0
    segment_counter: int = 0
    reject_streak: int = 0
    cooldowns: Dict[str, int] = field(default_factory=dict)  # key -> decision_event_count remaining
    global_cooldown: int = 0
    active_mutation: Optional[Mutation] = None
    active_segment: Optional[Segment] = None
    active_snapshot: Optional[dict] = None
    active_snapshot_actuator: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def next_mutation_id(self) -> str:
        self.mutation_counter += 1
        return f"m_{self.mutation_counter:05d}"

    def next_segment_id(self) -> str:
        self.segment_counter += 1
        return f"s_{self.segment_counter:05d}"

    def tick_cooldowns(self) -> None:
        """Decrement cooldown counters by one decision window."""
        expired = []
        for key, remaining in self.cooldowns.items():
            self.cooldowns[key] = max(0, remaining - 1)
            if self.cooldowns[key] == 0:
                expired.append(key)
        for key in expired:
            del self.cooldowns[key]
        self.global_cooldown = max(0, self.global_cooldown - 1)

    def is_cooled_down(self, key: str) -> bool:
        if self.global_cooldown > 0:
            return False
        return self.cooldowns.get(key, 0) == 0

    def set_cooldown(self, key: str, windows: int) -> None:
        self.cooldowns[key] = windows
