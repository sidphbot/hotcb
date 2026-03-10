from __future__ import annotations

import os
import time
import traceback as tb_mod
from typing import Dict, List, Optional, Tuple

from .freeze import FreezeState
from .ledger import append_ledger
from .ops import HotOp, command_to_hotop
from .recipe import RecipePlayer
from .util import FileCursor, read_new_jsonl, safe_mtime
from .modules import CallbackModule, HotOptController, HotLossController, HotTuneController
from .modules.result import ModuleResult
from .actuators.base import BaseActuator
from . import config as hotcb_config


class HotKernel:
    """
    Shared runtime that tails the control-plane, routes ops to modules, and writes the applied ledger.
    """

    def __init__(
        self,
        run_dir: str,
        debounce_steps: int = 1,
        poll_interval_sec: float = 0.0,
        auto_disable_on_error: bool = True,
        commands_path: Optional[str] = None,
        applied_path: Optional[str] = None,
        recipe_path: Optional[str] = None,
        freeze_path: Optional[str] = None,
        yaml_path: Optional[str] = None,
        log_path: Optional[str] = None,
        tune_recipe_path: Optional[str] = None,
        metrics_collector: Optional[object] = None,
    ) -> None:
        self.run_dir = run_dir
        self.debounce_steps = max(1, int(debounce_steps))
        self.poll_interval_sec = float(poll_interval_sec)
        self._last_poll_t: float = 0.0
        self._step_counter: int = 0
        self._seq: int = 0

        self.commands_path = commands_path or os.path.join(run_dir, "hotcb.commands.jsonl")
        self.applied_path = applied_path or os.path.join(run_dir, "hotcb.applied.jsonl")
        self.recipe_path = recipe_path or os.path.join(run_dir, "hotcb.recipe.jsonl")
        self.freeze_path = freeze_path or os.path.join(run_dir, "hotcb.freeze.json")
        self.yaml_path = yaml_path or os.path.join(run_dir, "hotcb.yaml")
        self.sources_dir = os.path.join(run_dir, "hotcb.sources")
        self.log_path = log_path or os.path.join(run_dir, "hotcb.log")
        self._metrics_collector = metrics_collector

        self._freeze_state = FreezeState.load(self.freeze_path)
        self._recipe_player = RecipePlayer(
            recipe_path=self._freeze_state.recipe_path,
            adjust_path=self._freeze_state.adjust_path,
            step_offset=self._freeze_state.step_offset,
        )
        self._yaml_mtime: float = 0.0
        self._cmd_cursor = FileCursor(path=self.commands_path, offset=0)

        self._actuators: Dict[str, BaseActuator] = {}

        self.modules: Dict[str, object] = {
            "cb": CallbackModule(auto_disable_on_error=auto_disable_on_error, log_path=self.log_path, source_capture_dir=self.sources_dir),
            "opt": HotOptController(auto_disable_on_error=auto_disable_on_error),
            "loss": HotLossController(auto_disable_on_error=auto_disable_on_error),
            "tune": HotTuneController(
                auto_disable_on_error=auto_disable_on_error,
                run_dir=run_dir,
                recipe_path=tune_recipe_path,
            ),
        }

        self._seq = self._seed_seq(self.applied_path)

    def _seed_seq(self, path: str) -> int:
        if not os.path.exists(path):
            return 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f)
            return count
        except Exception:
            return 0

    def register_actuator(self, name: str, actuator: BaseActuator) -> None:
        self._actuators[name] = actuator
        tune = self.modules.get("tune")
        if tune is not None and hasattr(tune, "register_actuator"):
            tune.register_actuator(name, actuator)

    def get_actuator(self, name: str) -> Optional[BaseActuator]:
        return self._actuators.get(name)

    def list_actuators(self) -> Dict[str, BaseActuator]:
        return dict(self._actuators)

    def _should_poll(self) -> bool:
        if self.poll_interval_sec > 0.0:
            now = time.time()
            if now - self._last_poll_t < self.poll_interval_sec:
                return False
            self._last_poll_t = now
        return (self._step_counter % self.debounce_steps) == 0

    def _reload_freeze(self) -> None:
        mtime = safe_mtime(self.freeze_path)
        if mtime == self._freeze_state._mtime:
            return
        self._freeze_state = FreezeState.load(self.freeze_path)
        self._recipe_player.reload_if_needed(
            self._freeze_state.recipe_path,
            self._freeze_state.adjust_path,
            self._freeze_state.step_offset,
        )

    def _load_yaml_ops(self) -> List[HotOp]:
        if safe_mtime(self.yaml_path) == self._yaml_mtime:
            return []
        ops = hotcb_config.load_yaml(self.yaml_path)
        self._yaml_mtime = safe_mtime(self.yaml_path)
        for op in ops:
            op.source = "yaml"
        return ops

    def _load_command_ops(self) -> List[HotOp]:
        try:
            raw_cmds, new_cursor = read_new_jsonl(self._cmd_cursor)
            self._cmd_cursor = new_cursor
        except Exception:
            return []
        return [command_to_hotop(c) for c in raw_cmds]

    def apply(self, env: Dict[str, object], events: List[str]) -> None:
        """
        Safe-point entrypoint. Poll control plane, apply ops, dispatch events.
        """
        self._step_counter += 1
        current_step = int(env.get("step", self._step_counter) or self._step_counter)
        events = events or []
        default_event = events[0] if events else "unknown"

        pending: List[Tuple[HotOp, str]] = []
        if self._should_poll():
            self._reload_freeze()
            pending.extend([(op, default_event) for op in self._load_yaml_ops()])
            pending.extend([(op, default_event) for op in self._load_command_ops()])

        # replay injection per event
        if self._freeze_state.mode in ("replay", "replay_adjusted"):
            for ev in events or [default_event]:
                pending.extend([(op, ev) for op in self._recipe_player.ops_for(current_step, ev)])

        for op, ev in pending:
            self._apply_single(op, env, event=ev, step=current_step)

        # dispatch events (callbacks module)
        cb = self.modules.get("cb")
        if cb is not None and hasattr(cb, "dispatch_events"):
            cb.dispatch_events(events, env)

        # dispatch events to tune module
        tune = self.modules.get("tune")
        if tune is not None and hasattr(tune, "on_event"):
            for ev in events:
                try:
                    tune.on_event(ev, env)
                except Exception:
                    pass  # defensive — never crash training

        # collect metrics (zero overhead when collector is None)
        if self._metrics_collector is not None:
            try:
                self._metrics_collector.collect(env)
            except Exception:
                pass  # never crash training

    def close(self, env: Optional[Dict[str, object]] = None) -> None:
        """
        Finalize the kernel at end of run. Checks strict replay policy.
        """
        # Close tune module
        tune = self.modules.get("tune")
        if tune is not None and hasattr(tune, "close"):
            try:
                tune.close(env)
            except Exception:
                pass

        if self._freeze_state.mode not in ("replay", "replay_adjusted"):
            return
        remaining = self._recipe_player.remaining_entries
        if not remaining:
            return

        step = int(env.get("step", self._step_counter) if env else self._step_counter)
        event = "run_end"

        if self._freeze_state.policy == "strict":
            for entry in remaining:
                op = HotOp(
                    module=str(entry.record.get("module", "unknown")),
                    op=str(entry.record.get("op", "unknown")),
                    id=str(entry.record.get("id")) if entry.record.get("id") is not None else None,
                    source="replay",
                )
                self._write_ledger(
                    op, event, step, decision="failed",
                    error=f"missed_step:{entry.at_step}:{entry.at_event}",
                    payload=entry.record, env=env or {},
                    notes="strict_policy_missed",
                )
            raise RuntimeError(
                f"Strict replay policy: {len(remaining)} recipe entries not applied "
                f"(run ended at step {step})"
            )

        # best_effort: log summary of missed entries
        for entry in remaining:
            op = HotOp(
                module=str(entry.record.get("module", "unknown")),
                op=str(entry.record.get("op", "unknown")),
                id=str(entry.record.get("id")) if entry.record.get("id") is not None else None,
                source="replay",
            )
            self._write_ledger(
                op, event, step, decision="failed",
                error=f"missed_step:{entry.at_step}:{entry.at_event}",
                payload=entry.record, env=env or {},
                notes="best_effort_missed",
            )

    def _apply_single(self, op: HotOp, env: dict, event: str, step: int) -> None:
        # freeze enforcement
        if op.source == "external" and self._freeze_state.mode == "prod" and op.module in {"cb", "opt", "loss", "tune"}:
            self._write_ledger(op, event, step, decision="ignored_freeze", payload=op.to_dict(), env=env)
            return
        if op.source == "external" and self._freeze_state.mode in {"replay", "replay_adjusted"} and op.module in {"cb", "opt", "loss", "tune"}:
            self._write_ledger(op, event, step, decision="ignored_replay", payload=op.to_dict(), env=env)
            return

        if op.module == "core":
            decision, error, payload = self._apply_core_op(op)
            self._write_ledger(op, event, step, decision=decision, error=error, payload=payload, env=env)
            return

        mod = self.modules.get(op.module)
        if mod is None:
            self._write_ledger(op, event, step, decision="failed", error=f"unknown_module:{op.module}", payload=op.to_dict(), env=env)
            return

        result: Optional[ModuleResult] = None
        try:
            if hasattr(mod, "apply_op"):
                result = mod.apply_op(op, env)
        except Exception as e:  # pragma: no cover - defensive
            self._write_ledger(op, event, step, decision="failed", error=str(e), payload=op.to_dict(), env=env, traceback_str=tb_mod.format_exc())
            return

        if result is None:
            self._write_ledger(op, event, step, decision="failed", error="no_result", payload=op.to_dict(), env=env)
            return

        payload = result.payload if result.payload is not None else op.to_dict()
        self._write_ledger(op, event, step, decision=result.decision, error=result.error, payload=payload, notes=result.notes, env=env, traceback_str=result.traceback)

    def _apply_core_op(self, op: HotOp) -> Tuple[str, Optional[str], Optional[dict]]:
        if op.op == "freeze":
            cfg = {
                "mode": op.mode or "off",
                "recipe_path": op.recipe_path or self._freeze_state.recipe_path,
                "adjust_path": op.adjust_path or self._freeze_state.adjust_path,
                "policy": self._freeze_state.policy,
                "step_offset": self._freeze_state.step_offset,
            }
            try:
                with open(self.freeze_path, "w", encoding="utf-8") as f:
                    import json

                    f.write(json.dumps(cfg))
                self._freeze_state = FreezeState.load(self.freeze_path)
                self._recipe_player.reload_if_needed(
                    self._freeze_state.recipe_path,
                    self._freeze_state.adjust_path,
                    self._freeze_state.step_offset,
                )
                return "applied", None, cfg
            except Exception as e:
                return "failed", str(e), cfg
        if op.op == "unfreeze":
            try:
                if os.path.exists(self.freeze_path):
                    os.remove(self.freeze_path)
                self._freeze_state = FreezeState(mode="off")
                return "applied", None, {"mode": "off"}
            except Exception as e:
                return "failed", str(e), {"mode": "off"}
        return "ignored", f"unknown_core_op:{op.op}", op.to_dict()

    def _write_ledger(
        self,
        op: HotOp,
        event: str,
        step: int,
        decision: str,
        env: Optional[dict],
        error: Optional[str] = None,
        payload: Optional[dict] = None,
        notes: Optional[str] = None,
        traceback_str: Optional[str] = None,
    ) -> None:
        self._seq += 1
        entry = {
            "seq": self._seq,
            "step": step,
            "epoch": env.get("epoch") if isinstance(env, dict) else None,
            "event": event,
            "phase": env.get("phase") if isinstance(env, dict) else None,
            "module": op.module,
            "op": op.op,
            "id": op.id,
            "source": op.source,
            "decision": decision,
            "payload": payload,
            "error": error,
            "traceback": traceback_str,
            "notes": notes,
        }
        append_ledger(self.applied_path, entry)
