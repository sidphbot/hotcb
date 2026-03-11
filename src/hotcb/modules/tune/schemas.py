from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MutationSpec:
    """Specification for a single mutation type within an actuator."""
    bounds: tuple[float, float] = (0.5, 2.0)
    prior_center: float = 1.0
    cooldown: int = 1
    risk: str = "low"
    max_step_mult: Optional[float] = None
    mode: str = "mult"  # mult, set, delta


@dataclass
class ActuatorConfig:
    """Configuration for an actuator within a tune recipe."""
    enabled: bool = True
    mutations: Dict[str, MutationSpec] = field(default_factory=dict)
    keys: Dict[str, MutationSpec] = field(default_factory=dict)


@dataclass
class ObjectiveConfig:
    primary: str = "val/loss"
    mode: str = "min"  # min or max
    backup_metrics: List[str] = field(default_factory=list)


@dataclass
class PhaseConfig:
    start_frac: float = 0.0
    end_frac: float = 1.0


@dataclass
class AcceptanceConfig:
    epsilon: float = 0.001
    horizon: str = "next_val_epoch_end"
    rollback_on_reject: bool = True


@dataclass
class SafetyConfig:
    block_on_nan: bool = True
    block_on_anomaly: bool = True
    max_global_reject_streak: int = 4


@dataclass
class SearchConfig:
    algorithm: str = "tpe"
    startup_trials: int = 8
    candidate_count: int = 24
    phase_conditioned: bool = True


@dataclass
class TuneRecipe:
    """Full tune recipe configuration."""
    version: int = 1
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    phases: Dict[str, PhaseConfig] = field(default_factory=lambda: {
        "early": PhaseConfig(0.0, 0.2),
        "mid": PhaseConfig(0.2, 0.7),
        "late": PhaseConfig(0.7, 1.0),
    })
    actuators: Dict[str, ActuatorConfig] = field(default_factory=dict)
    search: SearchConfig = field(default_factory=SearchConfig)
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    @classmethod
    def from_dict(cls, data: dict) -> TuneRecipe:
        obj_d = data.get("objective", {})
        objective = ObjectiveConfig(
            primary=obj_d.get("primary", "val/loss"),
            mode=obj_d.get("mode", "min"),
            backup_metrics=obj_d.get("backup_metrics", []),
        )

        phases = {}
        for name, pcfg in data.get("phases", {}).items():
            phases[name] = PhaseConfig(
                start_frac=pcfg.get("start_frac", 0.0),
                end_frac=pcfg.get("end_frac", 1.0),
            )
        if not phases:
            phases = {
                "early": PhaseConfig(0.0, 0.2),
                "mid": PhaseConfig(0.2, 0.7),
                "late": PhaseConfig(0.7, 1.0),
            }

        actuators: Dict[str, ActuatorConfig] = {}
        for aname, acfg in data.get("actuators", {}).items():
            mutations = {}
            for mname, mspec in acfg.get("mutations", {}).items():
                bounds = mspec.get("bounds", [0.5, 2.0])
                mutations[mname] = MutationSpec(
                    bounds=tuple(bounds),
                    prior_center=mspec.get("prior_center", 1.0),
                    cooldown=mspec.get("cooldown", 1),
                    risk=mspec.get("risk", "low"),
                    max_step_mult=mspec.get("max_step_mult"),
                    mode=mspec.get("mode", "mult"),
                )
            keys = {}
            for kname, kspec in acfg.get("keys", {}).items():
                bounds = kspec.get("bounds", [0.5, 2.0])
                keys[kname] = MutationSpec(
                    bounds=tuple(bounds),
                    prior_center=kspec.get("prior_center", 1.0),
                    cooldown=kspec.get("cooldown", 1),
                    risk=kspec.get("risk", "low"),
                    max_step_mult=kspec.get("max_step_mult"),
                    mode=kspec.get("mode", "mult"),
                )
            actuators[aname] = ActuatorConfig(
                enabled=acfg.get("enabled", True),
                mutations=mutations,
                keys=keys,
            )

        search_d = data.get("search", {})
        search = SearchConfig(
            algorithm=search_d.get("algorithm", "tpe"),
            startup_trials=search_d.get("startup_trials", 8),
            candidate_count=search_d.get("candidate_count", 24),
            phase_conditioned=search_d.get("phase_conditioned", True),
        )

        acc_d = data.get("acceptance", {})
        acceptance = AcceptanceConfig(
            epsilon=acc_d.get("epsilon", 0.001),
            horizon=acc_d.get("horizon", "next_val_epoch_end"),
            rollback_on_reject=acc_d.get("rollback_on_reject", True),
        )

        safe_d = data.get("safety", {})
        safety = SafetyConfig(
            block_on_nan=safe_d.get("block_on_nan", True),
            block_on_anomaly=safe_d.get("block_on_anomaly", True),
            max_global_reject_streak=safe_d.get("max_global_reject_streak", 4),
        )

        return cls(
            version=data.get("version", 1),
            objective=objective,
            phases=phases,
            actuators=actuators,
            search=search,
            acceptance=acceptance,
            safety=safety,
        )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "objective": {
                "primary": self.objective.primary,
                "mode": self.objective.mode,
                "backup_metrics": self.objective.backup_metrics,
            },
            "phases": {
                name: {"start_frac": p.start_frac, "end_frac": p.end_frac}
                for name, p in self.phases.items()
            },
            "actuators": {
                aname: {
                    "enabled": acfg.enabled,
                    "mutations": {
                        mname: {
                            "bounds": list(ms.bounds),
                            "prior_center": ms.prior_center,
                            "cooldown": ms.cooldown,
                            "risk": ms.risk,
                            **({"max_step_mult": ms.max_step_mult} if ms.max_step_mult else {}),
                            "mode": ms.mode,
                        }
                        for mname, ms in acfg.mutations.items()
                    },
                    "keys": {
                        kname: {
                            "bounds": list(ks.bounds),
                            "prior_center": ks.prior_center,
                            "cooldown": ks.cooldown,
                            "risk": ks.risk,
                            **({"max_step_mult": ks.max_step_mult} if ks.max_step_mult else {}),
                            "mode": ks.mode,
                        }
                        for kname, ks in acfg.keys.items()
                    },
                }
                for aname, acfg in self.actuators.items()
            },
            "search": {
                "algorithm": self.search.algorithm,
                "startup_trials": self.search.startup_trials,
                "candidate_count": self.search.candidate_count,
                "phase_conditioned": self.search.phase_conditioned,
            },
            "acceptance": {
                "epsilon": self.acceptance.epsilon,
                "horizon": self.acceptance.horizon,
                "rollback_on_reject": self.acceptance.rollback_on_reject,
            },
            "safety": {
                "block_on_nan": self.safety.block_on_nan,
                "block_on_anomaly": self.safety.block_on_anomaly,
                "max_global_reject_streak": self.safety.max_global_reject_streak,
            },
        }
