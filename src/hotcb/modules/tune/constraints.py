from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

from .schemas import TuneRecipe
from .state import TuneState


def get_phase_bin(
    step: int,
    total_steps: Optional[int],
    phases: Dict[str, Any],
) -> str:
    """Determine the current phase bin from step progress."""
    if total_steps is None or total_steps <= 0:
        return "mid"
    frac = step / total_steps
    for name, phase in phases.items():
        start = phase.start_frac if hasattr(phase, "start_frac") else phase.get("start_frac", 0.0)
        end = phase.end_frac if hasattr(phase, "end_frac") else phase.get("end_frac", 1.0)
        if start <= frac < end:
            return name
    return "late"


def check_safety_blockers(
    env: dict,
    recipe: TuneRecipe,
    metric_fn: Optional[Callable] = None,
) -> List[str]:
    """Return list of safety blocker reasons. Empty means safe to proceed."""
    blockers: List[str] = []

    if recipe.safety.block_on_nan:
        loss = env.get("loss")
        if loss is not None:
            try:
                val = float(loss)
                if math.isnan(val) or math.isinf(val):
                    blockers.append("nan_or_inf_loss")
            except (TypeError, ValueError):
                pass

    if recipe.safety.block_on_anomaly:
        anomaly_flag = env.get("anomaly_critical")
        if anomaly_flag:
            blockers.append("anomaly_critical")

    return blockers


def check_mutation_constraints(
    actuator_name: str,
    mutation_key: str,
    state: TuneState,
    recipe: TuneRecipe,
    phase_bin: str,
) -> List[str]:
    """Check cooldown and risk constraints for a mutation. Return list of block reasons."""
    blocks: List[str] = []

    if not state.is_cooled_down(f"{actuator_name}:{mutation_key}"):
        blocks.append(f"cooldown:{actuator_name}:{mutation_key}")

    if state.reject_streak >= recipe.safety.max_global_reject_streak:
        blocks.append(f"max_reject_streak:{state.reject_streak}")

    acfg = recipe.actuators.get(actuator_name)
    if acfg is None:
        blocks.append(f"actuator_not_in_recipe:{actuator_name}")
        return blocks

    if not acfg.enabled:
        blocks.append(f"actuator_disabled:{actuator_name}")
        return blocks

    # Check mutation exists in recipe
    mspec = acfg.mutations.get(mutation_key) or acfg.keys.get(mutation_key)
    if mspec is None:
        blocks.append(f"mutation_not_in_recipe:{actuator_name}:{mutation_key}")
        return blocks

    # v1: only allow low and medium risk
    if mspec.risk not in ("low", "medium"):
        blocks.append(f"risk_too_high:{mspec.risk}")

    return blocks
