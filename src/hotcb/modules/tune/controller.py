from __future__ import annotations

import logging
import os
import traceback as tb_mod
from typing import Any, Callable, Dict, Optional

from ...actuators.base import BaseActuator
from ...ops import HotOp
from ..result import ModuleResult
from .constraints import check_mutation_constraints, check_safety_blockers, get_phase_bin
from .evaluator import evaluate_segment, read_metrics
from .schemas import TuneRecipe
from .search import propose_mutation
from .state import Mutation, Segment, TuneState
from .storage import (
    load_recipe_yaml,
    load_mutations_log,
    load_segments_log,
    write_mutation,
    write_segment,
    write_summary,
)
from .recipe import compute_run_stats

log = logging.getLogger("hotcb.tune")


def _load_jsonl(path: str) -> list:
    """Load JSONL file directly by path."""
    import json
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


class HotTuneController:
    """
    Online constrained hyperparameter tuner.

    Responds to control ops (enable/disable/set) and event-driven
    proposal/evaluation at safe points (val_epoch_end).
    """

    def __init__(
        self,
        auto_disable_on_error: bool = True,
        run_dir: Optional[str] = None,
        recipe: Optional[TuneRecipe] = None,
        recipe_path: Optional[str] = None,
        replay_mutations_path: Optional[str] = None,
    ) -> None:
        self.auto_disable_on_error = auto_disable_on_error
        self.run_dir = run_dir
        self.state = TuneState(mode="off")
        self.recipe = recipe or TuneRecipe()
        self._actuators: Dict[str, BaseActuator] = {}
        self._total_steps: Optional[int] = None
        self._replay_queue: list = []
        self._replay_index: int = 0

        if recipe_path and os.path.exists(recipe_path):
            try:
                data = load_recipe_yaml(recipe_path)
                self.recipe = TuneRecipe.from_dict(data)
            except Exception as e:
                log.warning("[hotcb.tune] failed to load recipe: %s", e)

        if replay_mutations_path and os.path.exists(replay_mutations_path):
            self._replay_queue = _load_jsonl(replay_mutations_path)

    def register_actuator(self, name: str, actuator: BaseActuator) -> None:
        self._actuators[name] = actuator

    def get_actuator(self, name: str) -> Optional[BaseActuator]:
        return self._actuators.get(name)

    def list_actuators(self) -> Dict[str, BaseActuator]:
        return dict(self._actuators)

    def set_total_steps(self, total: int) -> None:
        self._total_steps = total

    def apply_op(self, op: HotOp, env: dict) -> ModuleResult:
        """Handle control-plane ops for the tune module."""
        if op.op == "enable":
            mode = (op.params or {}).get("mode", "active")
            self.state.mode = mode
            replay_path = (op.params or {}).get("replay_path")
            if mode == "replay" and replay_path:
                self._replay_queue = _load_jsonl(replay_path)
                self._replay_index = 0
            return ModuleResult(decision="applied", payload={"mode": mode})

        if op.op == "disable":
            self.state.mode = "off"
            return ModuleResult(decision="applied", payload={"mode": "off"})

        if op.op == "set":
            params = op.params or {}
            self._apply_recipe_overrides(params)
            return ModuleResult(decision="applied", payload=params)

        if op.op == "status":
            return ModuleResult(
                decision="applied",
                payload={
                    "mode": self.state.mode,
                    "mutation_counter": self.state.mutation_counter,
                    "reject_streak": self.state.reject_streak,
                    "cooldowns": dict(self.state.cooldowns),
                    "actuators": list(self._actuators.keys()),
                    "active_mutation": self.state.active_mutation.mutation_id if self.state.active_mutation else None,
                },
            )

        return ModuleResult(decision="ignored", notes=f"unknown_op:{op.op}")

    def on_event(self, event: str, env: dict) -> None:
        """
        Event-driven tuning logic. Called by kernel alongside callback dispatch.
        """
        if self.state.mode == "off":
            return

        # Update total steps estimate
        total = env.get("max_steps") or env.get("total_steps")
        if total is not None:
            self._total_steps = int(total)

        metric_fn = env.get("metric")

        if event == "val_epoch_end":
            self._on_decision_point(env, metric_fn)
        elif event == "fit_start":
            self.state.tick_cooldowns()
        elif event == "run_end":
            self._on_run_end(env)

    def _on_decision_point(self, env: dict, metric_fn: Optional[Callable]) -> None:
        """Handle a decision point (val_epoch_end)."""
        step = int(env.get("step", 0))
        epoch = int(env.get("epoch", 0))

        # If we have an active segment, evaluate it
        if self.state.active_segment is not None:
            self._evaluate_active_segment(env, metric_fn)

        if self.state.mode == "off":
            return

        if self.state.mode == "observe":
            return

        if self.state.mode == "replay":
            self._replay_next_mutation(env, step, epoch)
            return

        # Tick cooldowns at each decision point
        self.state.tick_cooldowns()

        # Check safety blockers
        blockers = check_safety_blockers(env, self.recipe, metric_fn)
        if blockers:
            log.info("[hotcb.tune] blocked: %s", blockers)
            return

        # Don't propose if we already have an active mutation being evaluated
        if self.state.active_segment is not None:
            return

        phase_bin = get_phase_bin(step, self._total_steps, self.recipe.phases)

        # Build context for search
        context = self._build_context(env, metric_fn, phase_bin)

        # Propose mutation
        proposal = propose_mutation(
            self.recipe, self.state, phase_bin, context,
            run_dir=self.run_dir,
        )
        if proposal is None:
            log.debug("[hotcb.tune] no feasible mutation proposed")
            return

        actuator_name = proposal["actuator"]
        mutation_key = proposal["mutation_key"]
        patch = proposal["patch"]

        # Check constraints
        constraint_blocks = check_mutation_constraints(
            actuator_name, mutation_key, self.state, self.recipe, phase_bin,
        )
        if constraint_blocks:
            log.info("[hotcb.tune] mutation blocked by constraints: %s", constraint_blocks)
            return

        actuator = self._actuators.get(actuator_name)
        if actuator is None:
            log.warning("[hotcb.tune] actuator %s not registered", actuator_name)
            return

        # Validate
        vresult = actuator.validate(patch, env)
        if not vresult.valid:
            log.info("[hotcb.tune] validation failed: %s", vresult.errors)
            return

        # Suggest mode: log proposal without applying
        if self.state.mode == "suggest":
            mutation = Mutation(
                mutation_id=self.state.next_mutation_id(),
                step=step, epoch=epoch,
                phase_bin=phase_bin,
                event="val_epoch_end",
                actuator=actuator_name,
                patch=patch,
                context=context,
                status="suggested",
            )
            if self.run_dir:
                write_mutation(self.run_dir, mutation)
            log.info(
                "[hotcb.tune] suggested mutation %s: %s.%s = %s",
                mutation.mutation_id, actuator_name, patch.get("op"), patch.get("value"),
            )
            return

        # Snapshot
        snapshot = actuator.snapshot(env)

        # Apply
        try:
            aresult = actuator.apply(patch, env)
        except Exception as e:
            log.warning("[hotcb.tune] apply raised exception: %s", e)
            aresult = None

        if aresult is None or not aresult.success:
            log.warning("[hotcb.tune] apply failed: %s", aresult.error if aresult else "exception")
            mutation = Mutation(
                mutation_id=self.state.next_mutation_id(),
                step=step, epoch=epoch,
                phase_bin=phase_bin,
                event="val_epoch_end",
                actuator=actuator_name,
                patch=patch,
                context=context,
                status="failed",
            )
            if self.run_dir:
                write_mutation(self.run_dir, mutation)
            return

        # Create mutation record
        mutation = Mutation(
            mutation_id=self.state.next_mutation_id(),
            step=step, epoch=epoch,
            phase_bin=phase_bin,
            event="val_epoch_end",
            actuator=actuator_name,
            patch=patch,
            proposal_source=self.recipe.search.algorithm,
            context=context,
            snapshot_ref=f"snap_{self.state.mutation_counter:05d}",
            status="applied",
        )

        # Read pre-metrics
        all_metrics = [self.recipe.objective.primary] + self.recipe.objective.backup_metrics
        pre = read_metrics(metric_fn, all_metrics)

        # Create evaluation segment
        segment = Segment(
            segment_id=self.state.next_segment_id(),
            mutation_id=mutation.mutation_id,
            start_step=step,
            horizon_type=self.recipe.acceptance.horizon,
            pre=pre,
        )

        self.state.active_mutation = mutation
        self.state.active_segment = segment
        self.state.active_snapshot = snapshot
        self.state.active_snapshot_actuator = actuator_name

        # Set cooldown for this mutation
        acfg = self.recipe.actuators.get(actuator_name)
        cooldown = 1
        if acfg:
            mspec = acfg.mutations.get(mutation_key) or acfg.keys.get(mutation_key)
            if mspec:
                cooldown = mspec.cooldown
        self.state.set_cooldown(f"{actuator_name}:{mutation_key}", cooldown)

        # Write records
        if self.run_dir:
            write_mutation(self.run_dir, mutation)

        log.info(
            "[hotcb.tune] applied mutation %s: %s.%s = %s",
            mutation.mutation_id, actuator_name, patch.get("op"), patch.get("value"),
        )

    def _evaluate_active_segment(self, env: dict, metric_fn: Optional[Callable]) -> None:
        """Evaluate the active segment and accept/reject."""
        segment = self.state.active_segment
        mutation = self.state.active_mutation
        if segment is None or mutation is None:
            return

        step = int(env.get("step", 0))
        segment.end_step = step

        segment = evaluate_segment(segment, self.recipe, metric_fn, env)

        if segment.decision == "accepted":
            self.state.reject_streak = 0
            log.info(
                "[hotcb.tune] accepted %s (score_delta=%.4f)",
                mutation.mutation_id, segment.score_delta or 0,
            )
        else:
            self.state.reject_streak += 1
            # Rollback if configured
            if self.recipe.acceptance.rollback_on_reject:
                actuator = self._actuators.get(self.state.active_snapshot_actuator or "")
                if actuator and self.state.active_snapshot:
                    rresult = actuator.restore(self.state.active_snapshot, env)
                    if rresult.success:
                        segment.decision = "rolled_back"
                        mutation.status = "rolled_back"
                    else:
                        log.warning("[hotcb.tune] rollback failed: %s", rresult.error)

            # Global cooldown on reject
            self.state.global_cooldown = 1

            log.info(
                "[hotcb.tune] rejected %s (score_delta=%.4f, streak=%d)",
                mutation.mutation_id, segment.score_delta or 0, self.state.reject_streak,
            )

        # Update mutation status
        if mutation.status == "applied":
            mutation.status = segment.decision or "evaluated"

        # Write records
        if self.run_dir:
            write_mutation(self.run_dir, mutation)
            write_segment(self.run_dir, segment)

        # Record in history
        self.state.history.append({
            "mutation": mutation.to_dict(),
            "segment": segment.to_dict(),
        })

        # Clear active state
        self.state.active_mutation = None
        self.state.active_segment = None
        self.state.active_snapshot = None
        self.state.active_snapshot_actuator = None

    def _replay_next_mutation(self, env: dict, step: int, epoch: int) -> None:
        """Replay the next mutation from the replay queue."""
        if self._replay_index >= len(self._replay_queue):
            return

        record = self._replay_queue[self._replay_index]
        self._replay_index += 1

        actuator_name = record.get("actuator", "")
        patch = record.get("patch", {})
        actuator = self._actuators.get(actuator_name)

        if actuator is None:
            log.warning("[hotcb.tune] replay: actuator %s not registered", actuator_name)
            mutation = Mutation(
                mutation_id=self.state.next_mutation_id(),
                step=step, epoch=epoch,
                phase_bin=record.get("phase_bin", "unknown"),
                event="val_epoch_end",
                actuator=actuator_name,
                patch=patch,
                status="failed",
            )
            if self.run_dir:
                write_mutation(self.run_dir, mutation)
            return

        aresult = actuator.apply(patch, env)
        status = "applied" if aresult.success else "failed"
        mutation = Mutation(
            mutation_id=self.state.next_mutation_id(),
            step=step, epoch=epoch,
            phase_bin=record.get("phase_bin", "unknown"),
            event="val_epoch_end",
            actuator=actuator_name,
            patch=patch,
            proposal_source="replay",
            status=status,
        )
        if self.run_dir:
            write_mutation(self.run_dir, mutation)

        log.info(
            "[hotcb.tune] replayed mutation %s: %s [%s]",
            mutation.mutation_id, actuator_name, status,
        )

    def _on_run_end(self, env: dict) -> None:
        """Finalize at end of run."""
        # Evaluate any pending segment
        if self.state.active_segment is not None:
            metric_fn = env.get("metric")
            self._evaluate_active_segment(env, metric_fn)

        if self.run_dir:
            mutations = load_mutations_log(self.run_dir)
            segments = load_segments_log(self.run_dir)
            stats = compute_run_stats(mutations, segments)
            stats["mode"] = self.state.mode
            stats["total_steps"] = self._total_steps
            write_summary(self.run_dir, stats)

    def _build_context(self, env: dict, metric_fn: Optional[Callable], phase_bin: str) -> Dict[str, Any]:
        """Build compact context for search conditioning."""
        ctx: Dict[str, Any] = {"phase_bin": phase_bin}

        if metric_fn:
            for name in ("train/loss", "val/loss", "grad/norm"):
                val = metric_fn(name)
                if val is not None:
                    try:
                        ctx[name] = float(val)
                    except (TypeError, ValueError):
                        pass

        ctx["reject_streak"] = self.state.reject_streak
        ctx["mutation_count"] = self.state.mutation_counter

        return ctx

    def _apply_recipe_overrides(self, params: dict) -> None:
        """Apply dotted-path overrides to the recipe."""
        for key, value in params.items():
            parts = key.split(".")
            obj: Any = self.recipe
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                elif isinstance(obj, dict):
                    obj = obj.get(part, {})
                else:
                    break
            final = parts[-1]
            if hasattr(obj, final):
                setattr(obj, final, value)
            elif isinstance(obj, dict):
                obj[final] = value

    def close(self, env: Optional[dict] = None) -> None:
        """Finalize the tune controller."""
        if env:
            self._on_run_end(env)

    def status(self) -> dict:
        return {
            "mode": self.state.mode,
            "mutation_counter": self.state.mutation_counter,
            "reject_streak": self.state.reject_streak,
            "cooldowns": dict(self.state.cooldowns),
            "actuators": list(self._actuators.keys()),
            "active_mutation": self.state.active_mutation.mutation_id if self.state.active_mutation else None,
        }
