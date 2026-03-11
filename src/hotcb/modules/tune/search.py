from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from .schemas import ActuatorConfig, MutationSpec, TuneRecipe
from .state import TuneState


def _random_proposal(
    recipe: TuneRecipe,
    state: TuneState,
    phase_bin: str,
) -> Optional[Dict[str, Any]]:
    """
    Generate a random mutation proposal from the recipe search space.
    Fallback when optuna is not available.
    """
    candidates: List[Dict[str, Any]] = []

    for aname, acfg in recipe.actuators.items():
        if not acfg.enabled:
            continue

        # Optimizer mutations
        for mname, mspec in acfg.mutations.items():
            key = f"{aname}:{mname}"
            if not state.is_cooled_down(key):
                continue
            if mspec.risk not in ("low", "medium"):
                continue
            lo, hi = mspec.bounds
            value = random.uniform(lo, hi)
            # Bias toward prior center
            value = 0.7 * value + 0.3 * mspec.prior_center
            value = max(lo, min(hi, value))
            candidates.append({
                "actuator": aname,
                "mutation_key": mname,
                "patch": {"op": mname, "value": value},
            })

        # Loss key mutations
        for kname, kspec in acfg.keys.items():
            key = f"{aname}:{kname}"
            if not state.is_cooled_down(key):
                continue
            if kspec.risk not in ("low", "medium"):
                continue
            lo, hi = kspec.bounds
            if kspec.mode == "mult":
                value = random.uniform(lo, hi)
                value = 0.7 * value + 0.3 * kspec.prior_center
                value = max(lo, min(hi, value))
                candidates.append({
                    "actuator": aname,
                    "mutation_key": kname,
                    "patch": {"op": "mult", "key": kname, "value": value},
                })
            elif kspec.mode == "delta":
                value = random.uniform(lo, hi)
                candidates.append({
                    "actuator": aname,
                    "mutation_key": kname,
                    "patch": {"op": "delta", "key": kname, "value": value},
                })
            else:
                value = random.uniform(lo, hi)
                candidates.append({
                    "actuator": aname,
                    "mutation_key": kname,
                    "patch": {"op": "set", "key": kname, "value": value},
                })

    if not candidates:
        return None

    return random.choice(candidates)


def _tpe_proposal(
    recipe: TuneRecipe,
    state: TuneState,
    phase_bin: str,
    context: Dict[str, Any],
    run_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Generate a TPE-based proposal using Optuna.
    Falls back to random if optuna is not available.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return _random_proposal(recipe, state, phase_bin)

    import os

    storage = None
    if run_dir:
        db_path = os.path.join(run_dir, "hotcb.tune.study.sqlite")
        storage = f"sqlite:///{db_path}"

    try:
        study = optuna.create_study(
            study_name="hottune",
            storage=storage,
            load_if_exists=True,
            direction="maximize" if recipe.objective.mode == "max" else "minimize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=recipe.search.startup_trials,
                seed=42,
            ),
        )
    except Exception:
        # Fall back to in-memory study
        study = optuna.create_study(
            direction="maximize" if recipe.objective.mode == "max" else "minimize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=recipe.search.startup_trials,
                seed=42,
            ),
        )

    # Build search space from recipe
    eligible: List[tuple[str, str, MutationSpec, str]] = []  # (actuator, key, spec, kind)
    for aname, acfg in recipe.actuators.items():
        if not acfg.enabled:
            continue
        for mname, mspec in acfg.mutations.items():
            if mspec.risk not in ("low", "medium"):
                continue
            if not state.is_cooled_down(f"{aname}:{mname}"):
                continue
            eligible.append((aname, mname, mspec, "mutation"))
        for kname, kspec in acfg.keys.items():
            if kspec.risk not in ("low", "medium"):
                continue
            if not state.is_cooled_down(f"{aname}:{kname}"):
                continue
            eligible.append((aname, kname, kspec, "key"))

    if not eligible:
        return None

    trial = study.ask()

    # Choose which mutation to try
    choice_idx = trial.suggest_int("mutation_choice", 0, len(eligible) - 1)
    aname, mkey, mspec, kind = eligible[choice_idx]
    lo, hi = mspec.bounds
    value = trial.suggest_float(f"{aname}_{mkey}_value", lo, hi)

    # Tell the study about prior context (we'll report back with actual score later)
    # For now, prune immediately - we use the proposal
    study.tell(trial, state=optuna.trial.TrialState.PRUNED)

    if kind == "mutation":
        patch = {"op": mkey, "value": value}
    else:
        patch = {"op": mspec.mode, "key": mkey, "value": value}

    return {
        "actuator": aname,
        "mutation_key": mkey,
        "patch": patch,
    }


def propose_mutation(
    recipe: TuneRecipe,
    state: TuneState,
    phase_bin: str,
    context: Dict[str, Any],
    run_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Propose the next mutation based on recipe search config.
    Returns None if no mutation is feasible.
    """
    if recipe.search.algorithm == "tpe":
        return _tpe_proposal(recipe, state, phase_bin, context, run_dir)
    return _random_proposal(recipe, state, phase_bin)
