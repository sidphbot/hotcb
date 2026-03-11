from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .schemas import TuneRecipe


def compute_run_stats(mutations: List[dict], segments: List[dict]) -> dict:
    """Compute summary statistics from a single run's tune logs."""
    total = len(mutations)
    applied = sum(1 for m in mutations if m.get("status") == "applied")

    seg_by_decision: Dict[str, int] = {}
    score_by_mutation_family: Dict[str, List[float]] = {}
    for seg in segments:
        d = seg.get("decision", "unknown")
        seg_by_decision[d] = seg_by_decision.get(d, 0) + 1
        mid = seg.get("mutation_id", "")
        # Find corresponding mutation
        for m in mutations:
            if m.get("mutation_id") == mid:
                family = f"{m.get('actuator', '')}:{m.get('patch', {}).get('op', '')}"
                score = seg.get("score_delta", 0.0)
                score_by_mutation_family.setdefault(family, []).append(score)
                break

    win_rates: Dict[str, float] = {}
    mean_scores: Dict[str, float] = {}
    for family, scores in score_by_mutation_family.items():
        wins = sum(1 for s in scores if s > 0)
        win_rates[family] = wins / len(scores) if scores else 0.0
        mean_scores[family] = sum(scores) / len(scores) if scores else 0.0

    accepted = seg_by_decision.get("accepted", 0)
    rejected = seg_by_decision.get("rejected", 0)
    total_decided = accepted + rejected

    return {
        "total_mutations": total,
        "applied_mutations": applied,
        "segments_by_decision": seg_by_decision,
        "win_rates": win_rates,
        "mean_scores": mean_scores,
        "accept_rate": accepted / total_decided if total_decided > 0 else 0.0,
    }


def evolve_recipe(
    base_recipe: TuneRecipe,
    run_summaries: List[dict],
    alpha: float = 0.3,
) -> TuneRecipe:
    """
    Evolve a recipe based on accumulated run summaries.

    Uses exponential moving average to shift prior centers toward winning values.
    """
    evolved = TuneRecipe.from_dict(base_recipe.to_dict())

    for summary in run_summaries:
        win_rates = summary.get("win_rates", {})
        mean_scores = summary.get("mean_scores", {})

        for family, rate in win_rates.items():
            parts = family.split(":")
            if len(parts) != 2:
                continue
            actuator_name, mutation_op = parts

            acfg = evolved.actuators.get(actuator_name)
            if acfg is None:
                continue

            mspec = acfg.mutations.get(mutation_op) or acfg.keys.get(mutation_op)
            if mspec is None:
                continue

            # Shift prior center via EMA if win rate is high
            if rate > 0.5:
                score = mean_scores.get(family, 0.0)
                if score > 0:
                    # Move prior_center slightly toward the mean winning direction
                    lo, hi = mspec.bounds
                    midpoint = (lo + hi) / 2
                    mspec.prior_center = (1 - alpha) * mspec.prior_center + alpha * midpoint

    return evolved
