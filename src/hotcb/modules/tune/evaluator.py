from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .schemas import TuneRecipe
from .state import Segment


def read_metrics(
    metric_fn: Optional[Callable],
    metric_names: list[str],
) -> Dict[str, float]:
    """Read a set of metrics via the metric accessor."""
    result: Dict[str, float] = {}
    if metric_fn is None:
        return result
    for name in metric_names:
        val = metric_fn(name)
        if val is not None:
            try:
                result[name] = float(val)
            except (TypeError, ValueError):
                pass
    return result


def score_segment(segment: Segment, recipe: TuneRecipe) -> float:
    """
    Score a completed segment.

    For min-mode objectives: gain = old - new (positive is good)
    For max-mode objectives: gain = new - old (positive is good)

    Applies instability penalty.
    """
    primary = recipe.objective.primary
    pre_val = segment.pre.get(primary)
    post_val = segment.post.get(primary)

    if pre_val is None or post_val is None:
        return 0.0

    if recipe.objective.mode == "min":
        gain = pre_val - post_val
    else:
        gain = post_val - pre_val

    # Instability penalty
    penalty = 0.0
    if segment.stability.get("nan"):
        penalty += 1.0
    if segment.stability.get("anomaly"):
        penalty += 0.5
    if segment.stability.get("grad_spike"):
        penalty += 0.2

    return gain - penalty


def evaluate_segment(
    segment: Segment,
    recipe: TuneRecipe,
    metric_fn: Optional[Callable],
    env: dict,
) -> Segment:
    """
    Complete evaluation of a segment: read post-metrics, compute deltas, score, decide.
    """
    all_metrics = [recipe.objective.primary] + recipe.objective.backup_metrics
    segment.post = read_metrics(metric_fn, all_metrics)

    # Compute deltas
    for key in set(list(segment.pre.keys()) + list(segment.post.keys())):
        pre_v = segment.pre.get(key)
        post_v = segment.post.get(key)
        if pre_v is not None and post_v is not None:
            segment.delta[key] = post_v - pre_v

    # Check stability
    loss = env.get("loss")
    if loss is not None:
        try:
            import math
            val = float(loss)
            if math.isnan(val) or math.isinf(val):
                segment.stability["nan"] = True
        except (TypeError, ValueError):
            pass

    if env.get("anomaly_critical"):
        segment.stability["anomaly"] = True

    segment.score_delta = score_segment(segment, recipe)

    # Decision
    if segment.stability.get("nan") or segment.stability.get("anomaly"):
        segment.decision = "rejected"
    elif segment.score_delta > recipe.acceptance.epsilon:
        segment.decision = "accepted"
    else:
        segment.decision = "rejected"

    return segment
