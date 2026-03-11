from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from .util import append_jsonl, ensure_dir


def _cmd_path(run_dir: str) -> str:
    return os.path.join(run_dir, "hotcb.commands.jsonl")


def _cfg_path(run_dir: str) -> str:
    return os.path.join(run_dir, "hotcb.yaml")


def _applied_path(run_dir: str) -> str:
    return os.path.join(run_dir, "hotcb.applied.jsonl")


def _recipe_path(run_dir: str) -> str:
    return os.path.join(run_dir, "hotcb.recipe.jsonl")


def _freeze_path(run_dir: str) -> str:
    return os.path.join(run_dir, "hotcb.freeze.json")


def _parse_kv(pairs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for s in pairs:
        if "=" not in s:
            raise SystemExit(f"Expected key=value, got: {s}")
        k, v = s.split("=", 1)
        v = v.strip()

        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue

        if v.startswith("{") or v.startswith("["):
            try:
                out[k] = json.loads(v)
                continue
            except Exception:
                pass

        try:
            if "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
            continue
        except Exception:
            pass

        out[k] = v
    return out


def cmd_init(args: argparse.Namespace) -> None:
    run_dir = args.dir
    ensure_dir(run_dir)
    cfg = _cfg_path(run_dir)
    if not os.path.exists(cfg):
        with open(cfg, "w", encoding="utf-8") as f:
            f.write("version: 1\n")
        print(f"Wrote {cfg}")
    else:
        print(f"Exists: {cfg}")

    for p in [_cmd_path(run_dir), _applied_path(run_dir), _recipe_path(run_dir), _freeze_path(run_dir)]:
        if not os.path.exists(p):
            ensure_dir(os.path.dirname(p))
            with open(p, "w", encoding="utf-8") as f:
                pass
            print(f"Created {p}")
        else:
            print(f"Exists: {p}")


def _append_command(run_dir: str, obj: Dict[str, Any]) -> None:
    append_jsonl(_cmd_path(run_dir), obj)


def cmd_cb(args: argparse.Namespace) -> None:
    op = args.cb_command
    obj: Dict[str, Any] = {"module": "cb", "op": op, "id": args.id}
    if op == "set_params":
        obj["params"] = _parse_kv(args.kv or [])
    if op == "load":
        if not args.file and not args.path:
            raise SystemExit("Provide --file for python_file or --path for module import")
        obj["target"] = {"kind": "python_file" if args.file else "module", "path": args.file or args.path, "symbol": args.symbol}
        obj["init"] = _parse_kv(args.init or [])
        if args.enabled is not None:
            obj["enabled"] = args.enabled
    _append_command(args.dir, obj)
    print(f"queued cb {op} for {args.id}")


def cmd_opt(args: argparse.Namespace) -> None:
    op = args.opt_command
    obj: Dict[str, Any] = {"module": "opt", "op": op, "id": args.id}
    if op == "set_params":
        obj["params"] = _parse_kv(args.kv or [])
    _append_command(args.dir, obj)
    print(f"queued opt {op} for {args.id}")


def cmd_loss(args: argparse.Namespace) -> None:
    op = args.loss_command
    obj: Dict[str, Any] = {"module": "loss", "op": op, "id": args.id}
    if op == "set_params":
        obj["params"] = _parse_kv(args.kv or [])
    _append_command(args.dir, obj)
    print(f"queued loss {op} for {args.id}")


def cmd_freeze(args: argparse.Namespace) -> None:
    cfg = {
        "mode": args.mode,
        "recipe_path": args.recipe,
        "adjust_path": args.adjust,
        "policy": args.policy,
        "step_offset": args.step_offset,
    }
    ensure_dir(args.dir)
    with open(_freeze_path(args.dir), "w", encoding="utf-8") as f:
        f.write(json.dumps(cfg))
    print(f"Wrote freeze state -> {_freeze_path(args.dir)}")


_OPT_KEYS = {"lr", "weight_decay", "clip_norm", "scheduler_scale", "scheduler_drop", "group", "groups"}


def _infer_module(keys: set) -> str:
    """Auto-route kv keys to opt or loss module (spec §15.4)."""
    if keys & _OPT_KEYS:
        return "opt"
    for k in keys:
        if k.endswith("_w") or k.startswith("terms.") or k.startswith("ramps."):
            return "loss"
    raise SystemExit(
        f"Cannot auto-route keys {keys} to opt or loss. Use explicit subcommand."
    )


def cmd_status(args: argparse.Namespace) -> None:
    run_dir = args.dir
    # Freeze state
    fp = _freeze_path(run_dir)
    freeze = {}
    if os.path.exists(fp) and os.path.getsize(fp) > 0:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                freeze = json.load(f)
        except Exception:
            pass
    print(f"Freeze mode: {freeze.get('mode', 'off')}")
    if freeze.get("recipe_path"):
        print(f"  recipe: {freeze['recipe_path']}")
    if freeze.get("adjust_path"):
        print(f"  adjust: {freeze['adjust_path']}")
    if freeze.get("policy"):
        print(f"  policy: {freeze['policy']}")

    # Latest applied entries per module
    applied = _applied_path(run_dir)
    if not os.path.exists(applied):
        print("No applied ledger found.")
        return
    latest: Dict[str, dict] = {}
    try:
        with open(applied, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                mod = rec.get("module", "")
                key = f"{mod}:{rec.get('id', '')}"
                latest[key] = rec
    except Exception:
        pass
    if latest:
        print(f"\nLast applied entries ({len(latest)} handles):")
        for key, rec in latest.items():
            decision = rec.get("decision", "?")
            step = rec.get("step", "?")
            op = rec.get("op", "?")
            print(f"  {key}: {op} @ step {step} [{decision}]")


def cmd_sugar_enable(args: argparse.Namespace) -> None:
    """Syntactic sugar: `hotcb enable <id>` defaults to cb."""
    obj: Dict[str, Any] = {"module": "cb", "op": "enable", "id": args.id}
    _append_command(args.dir, obj)
    print(f"queued cb enable for {args.id}")


def cmd_sugar_disable(args: argparse.Namespace) -> None:
    """Syntactic sugar: `hotcb disable <id>` defaults to cb."""
    obj: Dict[str, Any] = {"module": "cb", "op": "disable", "id": args.id}
    _append_command(args.dir, obj)
    print(f"queued cb disable for {args.id}")


def cmd_sugar_set(args: argparse.Namespace) -> None:
    """Syntactic sugar: `hotcb set k=v` auto-routes to opt or loss."""
    params = _parse_kv(args.kv or [])
    module = _infer_module(set(params.keys()))
    obj: Dict[str, Any] = {"module": module, "op": "set_params", "id": args.id, "params": params}
    _append_command(args.dir, obj)
    print(f"queued {module} set_params for {args.id}")


def cmd_tune(args: argparse.Namespace) -> None:
    op = args.tune_command
    obj: Dict[str, Any] = {"module": "tune", "op": op}
    if op == "enable":
        mode = getattr(args, "mode", "active")
        obj["params"] = {"mode": mode}
    elif op == "set":
        obj["op"] = "set"
        obj["params"] = _parse_kv(args.kv or [])
    _append_command(args.dir, obj)
    print(f"queued tune {op}")


def cmd_tune_status(args: argparse.Namespace) -> None:
    run_dir = args.dir
    recipe_path = os.path.join(run_dir, "hotcb.tune.recipe.yaml")
    summary_path = os.path.join(run_dir, "hotcb.tune.summary.json")

    if os.path.exists(recipe_path):
        print(f"Tune recipe: {recipe_path}")
    else:
        print("No tune recipe found.")

    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            print(f"Mode: {summary.get('mode', '?')}")
            print(f"Mutations: {summary.get('total_mutations', 0)} total, {summary.get('applied_mutations', 0)} applied")
            print(f"Accept rate: {summary.get('accept_rate', 0):.1%}")
            segs = summary.get("segments_by_decision", {})
            for d, c in segs.items():
                print(f"  {d}: {c}")
        except Exception:
            print("Failed to read tune summary.")
    else:
        print("No tune summary found.")


def cmd_bench(args: argparse.Namespace) -> None:
    """Run benchmarks."""
    from .bench.tasks import BUILTIN_TASKS
    from .bench.runner import BenchmarkRunner
    from .bench.report import BenchmarkReport

    task_factory = BUILTIN_TASKS.get(args.task)
    if task_factory is None:
        raise SystemExit(f"Unknown task: {args.task}. Available: {list(BUILTIN_TASKS)}")

    max_steps = args.max_steps
    task = task_factory(max_steps=max_steps) if max_steps else task_factory()

    runner = BenchmarkRunner(output_dir=args.output_dir)
    conditions = [c.strip() for c in args.conditions.split(",")]

    for cond in conditions:
        if cond == "baseline":
            result = runner.run_baseline(task)
        elif cond == "auto_tune":
            result = runner.run_with_hotcb(task)
        elif cond == "recipe_replay":
            # Need a recipe from a prior auto_tune run
            prev = [r for r in runner.results if r.recipe_path]
            if not prev:
                print("Skipping recipe_replay: no recipe available")
                continue
            result = runner.run_recipe_replay(task, prev[-1].recipe_path)
        else:
            print(f"Unknown condition: {cond}")
            continue
        print(f"  {cond}: loss={result.final_metrics.get('loss', '?'):.6f} "
              f"steps={result.total_steps} time={result.total_time_sec:.3f}s")

    report = BenchmarkReport(runner.results)
    print()
    print(report.summary_table())
    report.to_json(os.path.join(args.output_dir, "benchmark.json"))
    report.to_csv(os.path.join(args.output_dir, "benchmark.csv"))
    print(f"\nResults saved to {args.output_dir}/")


def cmd_bench_eval(args: argparse.Namespace) -> None:
    """Run autopilot evaluation against a published benchmark."""
    from .bench.eval_autopilot import AutopilotEval

    ev = AutopilotEval(output_dir=args.output_dir)

    phases = [p.strip() for p in args.phases.split(",")]

    for phase in phases:
        if phase == "baseline":
            print(f"Running published baseline for {args.task} ...")
            result = ev.run_published_baseline(args.task)
            acc = result.final_metrics.get("val_accuracy", "?")
            print(f"  Baseline done: val_accuracy={acc}  time={result.total_time_sec:.1f}s")
        elif phase == "autopilot":
            print(f"Running autopilot challenge for {args.task} ...")
            result = ev.run_autopilot_challenge(
                args.task,
                guidelines_path=args.guidelines,
            )
            acc = result.final_metrics.get("val_accuracy", "?")
            print(f"  Autopilot done: val_accuracy={acc}  time={result.total_time_sec:.1f}s")
        else:
            print(f"Unknown phase: {phase}")

    print()
    print(ev.report())


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the dashboard server."""
    from .server.app import run_server, create_app

    multi_dirs = None
    if args.dirs:
        multi_dirs = [d.strip() for d in args.dirs.split(",") if d.strip()]

    autopilot_mode = getattr(args, "autopilot", None)
    key_metric = getattr(args, "key_metric", None)

    if autopilot_mode and autopilot_mode != "off":
        # Need to create app manually to configure autopilot before start
        import uvicorn

        app = create_app(
            args.dir,
            poll_interval=args.poll_interval,
            multi_dirs=multi_dirs,
        )

        # Configure autopilot
        ap_engine = app.state.autopilot_engine
        ai_engine = getattr(app.state, "ai_engine", None)

        if key_metric and ai_engine:
            ai_engine.state.key_metric = key_metric
            ai_engine.save_state()

        try:
            ap_engine.set_mode(autopilot_mode)
            print(f"Autopilot mode: {autopilot_mode}")
            if key_metric:
                print(f"Key metric: {key_metric}")
        except ValueError as e:
            print(f"Warning: {e}")

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        run_server(
            run_dir=args.dir,
            host=args.host,
            port=args.port,
            poll_interval=args.poll_interval,
            multi_dirs=multi_dirs,
        )


def cmd_demo(args: argparse.Namespace) -> None:
    """Launch a demo: synthetic training + live dashboard."""
    autopilot = getattr(args, "autopilot", "off")
    key_metric = getattr(args, "key_metric", None)

    if autopilot != "off":
        # Use launch() API to get autopilot wired before training starts
        from .launch import launch

        config = "multitask" if args.golden else "simple"
        handle = launch(
            config=config,
            run_dir=args.demo_dir if args.demo_dir else None,
            autopilot=autopilot,
            key_metric=key_metric or "val_loss",
            max_steps=args.max_steps,
            max_time=getattr(args, "max_time", None),
            step_delay=args.step_delay,
            host=args.host,
            port=args.port,
            serve=True,
            block=True,
        )
        return

    if args.golden:
        from .golden_demo import run_golden_demo

        run_golden_demo(
            host=args.host,
            port=args.port,
            max_steps=args.max_steps,
            step_delay=args.step_delay,
            run_dir=args.demo_dir if args.demo_dir else None,
        )
    else:
        from .demo import run_demo

        run_demo(
            host=args.host,
            port=args.port,
            max_steps=args.max_steps,
            step_delay=args.step_delay,
            run_dir=args.demo_dir if args.demo_dir else None,
        )


def cmd_launch(args: argparse.Namespace) -> None:
    """Launch training + dashboard + autopilot in one command."""
    from .launch import launch

    import sys

    train_fn = getattr(args, "train_fn", None)
    config = getattr(args, "config", "multitask")

    w = sys.stderr.write
    w("\n")
    w("  hotcb launch\n")
    w(f"  autopilot: {args.autopilot}\n")
    if args.key_metric:
        w(f"  key metric: {args.key_metric}\n")
    w(f"  dashboard: http://{args.host}:{args.port}\n")
    w("\n")

    handle = launch(
        train_fn=train_fn,
        config=config,
        config_file=getattr(args, "config_file", None),
        run_dir=args.dir if args.dir != "." else None,
        autopilot=args.autopilot,
        key_metric=args.key_metric or "val_loss",
        ai_model=getattr(args, "ai_model", "gpt-4o-mini"),
        ai_budget=getattr(args, "ai_budget", 5.0),
        ai_cadence=getattr(args, "ai_cadence", 50),
        max_steps=args.max_steps,
        max_time=getattr(args, "max_time", None),
        step_delay=args.step_delay,
        host=args.host,
        port=args.port,
        seed=getattr(args, "seed", None),
        serve=True,
        block=True,
    )


def cmd_tune_export_recipe(args: argparse.Namespace) -> None:
    run_dir = args.dir
    out = args.out or os.path.join(run_dir, "hotcb.tune.recipe.yaml")
    summary_path = os.path.join(run_dir, "hotcb.tune.summary.json")
    if not os.path.exists(summary_path):
        print(f"No tune summary at {summary_path}")
        raise SystemExit(1)
    # Just copy the summary as a starting point
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        ensure_dir(os.path.dirname(out) or ".")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Exported tune summary -> {out}")
    except Exception as e:
        print(f"Failed: {e}")
        raise SystemExit(1)


def cmd_recipe_validate(args: argparse.Namespace) -> None:
    """Validate a recipe file for schema correctness."""
    path = args.recipe or _recipe_path(args.dir)
    if not os.path.exists(path):
        print(f"Recipe file not found: {path}")
        raise SystemExit(1)
    errors: List[str] = []
    entries = 0
    required_fields = {"at", "module", "op"}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"  line {i}: invalid JSON: {e}")
                continue
            entries += 1
            missing = required_fields - set(rec.keys())
            if missing:
                errors.append(f"  line {i}: missing fields: {missing}")
            at = rec.get("at", {})
            if not isinstance(at, dict) or "step" not in at:
                errors.append(f"  line {i}: 'at' must contain 'step'")
            if rec.get("module") not in {"cb", "opt", "loss", "tune"}:
                errors.append(f"  line {i}: module must be cb/opt/loss, got '{rec.get('module')}'")
    if errors:
        print(f"Recipe {path}: {len(errors)} errors in {entries} entries:")
        for e in errors:
            print(e)
        raise SystemExit(1)
    print(f"Recipe {path}: {entries} entries, valid.")


def cmd_recipe_patch_template(args: argparse.Namespace) -> None:
    """Generate a YAML patch template from a recipe file."""
    recipe_path = args.recipe
    output_path = args.output
    if not os.path.exists(recipe_path):
        print(f"Recipe file not found: {recipe_path}")
        raise SystemExit(1)

    seen: list = []
    seen_keys: set = set()
    try:
        with open(recipe_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (rec.get("module"), rec.get("op"), rec.get("id"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    seen.append({"module": rec.get("module", ""), "op": rec.get("op", ""), "id": rec.get("id", "")})
    except FileNotFoundError:
        print(f"Recipe file not found: {recipe_path}")
        raise SystemExit(1)

    lines: List[str] = [
        f"# Generated from {recipe_path}",
        "patches:",
    ]
    for entry in seen:
        lines.append("  - match:")
        lines.append(f"      module: {entry['module']}")
        lines.append(f"      op: {entry['op']}")
        lines.append(f"      id: {entry['id']}")
        lines.append("    # replace_params: {}")
        lines.append("    # shift_step: 0")
        lines.append("    # drop: false")

    ensure_dir(os.path.dirname(output_path) or ".")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote patch template with {len(seen)} entries -> {output_path}")


def cmd_recipe_export(args: argparse.Namespace) -> None:
    applied = _applied_path(args.dir)
    out_path = args.out or _recipe_path(args.dir)
    entries: List[dict] = []
    try:
        with open(applied, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("decision") != "applied":
                    continue
                if rec.get("module") not in {"cb", "opt", "loss"}:
                    continue
                payload = rec.get("payload") or {}
                entry = {
                    "at": {"step": rec.get("step", 0), "event": rec.get("event", "train_step_end")},
                    "module": rec.get("module"),
                    "op": rec.get("op"),
                    "id": rec.get("id"),
                }
                # merge payload keys that map to op
                for k in ("params", "target", "init", "enabled"):
                    if k in payload:
                        entry[k] = payload[k]
                entries.append(entry)
    except FileNotFoundError:
        print(f"No applied ledger at {applied}")
        return

    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in entries:
            f.write(json.dumps(rec) + "\n")
    print(f"Exported recipe with {len(entries)} entries -> {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hotcb", description="hotcb live training control plane")
    p.add_argument("--dir", default=".", help="Run directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="Bootstrap run directory")
    pi.set_defaults(func=cmd_init)

    ps = sub.add_parser("status", help="Show run status")
    ps.set_defaults(func=cmd_status)

    # Syntactic sugar commands (spec §15.4)
    pen = sub.add_parser("enable", help="Enable a callback (sugar for cb enable)")
    pen.add_argument("id")
    pen.set_defaults(func=cmd_sugar_enable)

    pdis = sub.add_parser("disable", help="Disable a callback (sugar for cb disable)")
    pdis.add_argument("id")
    pdis.set_defaults(func=cmd_sugar_disable)

    pset_sugar = sub.add_parser("set", help="Set params (auto-routes to opt or loss)")
    pset_sugar.add_argument("--id", default="main")
    pset_sugar.add_argument("kv", nargs="*")
    pset_sugar.set_defaults(func=cmd_sugar_set)

    pf = sub.add_parser("freeze", help="Write freeze state file")
    pf.add_argument("--mode", choices=["off", "prod", "replay", "replay_adjusted"], required=True)
    pf.add_argument("--recipe", help="Recipe path for replay modes")
    pf.add_argument("--adjust", help="Adjustment overlay path")
    pf.add_argument("--policy", default="best_effort", choices=["best_effort", "strict"])
    pf.add_argument("--step-offset", type=int, default=0)
    pf.set_defaults(func=cmd_freeze)

    pr = sub.add_parser("recipe", help="Recipe utilities")
    sr = pr.add_subparsers(dest="recipe_cmd", required=True)
    pre = sr.add_parser("export", help="Export recipe from applied ledger")
    pre.add_argument("--out", help="Output path (default: runs/<dir>/hotcb.recipe.jsonl)")
    pre.set_defaults(func=cmd_recipe_export)

    prv = sr.add_parser("validate", help="Validate a recipe file")
    prv.add_argument("--recipe", help="Recipe path to validate")
    prv.set_defaults(func=cmd_recipe_validate)

    p_pt = sr.add_parser("patch-template", help="Generate adjust.yaml template from recipe")
    p_pt.add_argument("--recipe", default="hotcb.recipe.jsonl")
    p_pt.add_argument("--output", default="hotcb.adjust.yaml")
    p_pt.set_defaults(func=cmd_recipe_patch_template)

    pcb = sub.add_parser("cb", help="Callback module commands")
    pcb_sub = pcb.add_subparsers(dest="cb_command", required=True)
    for name in ["enable", "disable", "unload"]:
        ps = pcb_sub.add_parser(name)
        ps.add_argument("id")
        ps.set_defaults(func=cmd_cb)
    pset = pcb_sub.add_parser("set_params")
    pset.add_argument("id")
    pset.add_argument("kv", nargs="*")
    pset.set_defaults(func=cmd_cb)
    pload = pcb_sub.add_parser("load")
    pload.add_argument("id")
    pload.add_argument("--file", help="Python file path", dest="file")
    pload.add_argument("--path", help="Module path")
    pload.add_argument("--symbol", required=True)
    pload.add_argument("--enabled", action=argparse.BooleanOptionalAction, default=None)
    pload.add_argument("--init", nargs="*", default=[])
    pload.set_defaults(func=cmd_cb)

    popt = sub.add_parser("opt", help="Optimizer control")
    opt_sub = popt.add_subparsers(dest="opt_command", required=True)
    for name in ["enable", "disable"]:
        po = opt_sub.add_parser(name)
        po.add_argument("--id", default="main")
        po.set_defaults(func=cmd_opt)
    pset_opt = opt_sub.add_parser("set_params")
    pset_opt.add_argument("--id", default="main")
    pset_opt.add_argument("kv", nargs="*")
    pset_opt.set_defaults(func=cmd_opt)

    ploss = sub.add_parser("loss", help="Loss control")
    loss_sub = ploss.add_subparsers(dest="loss_command", required=True)
    for name in ["enable", "disable"]:
        pl = loss_sub.add_parser(name)
        pl.add_argument("--id", default="main")
        pl.set_defaults(func=cmd_loss)
    pset_loss = loss_sub.add_parser("set_params")
    pset_loss.add_argument("--id", default="main")
    pset_loss.add_argument("kv", nargs="*")
    pset_loss.set_defaults(func=cmd_loss)

    pbench = sub.add_parser("bench", help="Run benchmarks")
    bench_sub = pbench.add_subparsers(dest="bench_cmd")

    # `hotcb bench run` — original benchmark runner
    pbench_run = bench_sub.add_parser("run", help="Run benchmark conditions")
    pbench_run.add_argument("--task", default="synthetic_quadratic",
                            help="Task name (synthetic_quadratic, synthetic_classification, cifar10_resnet20)")
    pbench_run.add_argument("--output-dir", default="./bench_output", help="Output directory")
    pbench_run.add_argument("--conditions", default="baseline,auto_tune",
                            help="Comma-separated conditions to run")
    pbench_run.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    pbench_run.set_defaults(func=cmd_bench)

    # `hotcb bench eval` — autopilot evaluation
    pbench_eval = bench_sub.add_parser("eval", help="Run autopilot evaluation against published benchmark")
    pbench_eval.add_argument("--task", default="cifar10_resnet20",
                             help="Task name (cifar10_resnet20)")
    pbench_eval.add_argument("--output-dir", default="./eval_output", help="Output directory")
    pbench_eval.add_argument("--phases", default="baseline,autopilot",
                             help="Comma-separated phases: baseline, autopilot")
    pbench_eval.add_argument("--guidelines", default=None,
                             help="Path to YAML guidelines file for autopilot rules")
    pbench_eval.set_defaults(func=cmd_bench_eval)

    # Also allow bare `hotcb bench` to fall through to `run` for backwards compat
    pbench.add_argument("--task", default="synthetic_quadratic",
                        help="Task name (synthetic_quadratic, synthetic_classification, cifar10_resnet20)")
    pbench.add_argument("--output-dir", default="./bench_output", help="Output directory")
    pbench.add_argument("--conditions", default="baseline,auto_tune",
                        help="Comma-separated conditions to run")
    pbench.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    pbench.set_defaults(func=cmd_bench)

    pdemo = sub.add_parser("demo", help="Launch synthetic training with live dashboard")
    pdemo.add_argument("--golden", action="store_true",
                       help="Run the golden demo (multi-task with recipe-driven loss shifts and feature capture)")
    pdemo.add_argument("--host", default="0.0.0.0", help="Bind host")
    pdemo.add_argument("--port", type=int, default=8421, help="Bind port")
    pdemo.add_argument("--max-steps", type=int, default=500, help="Number of training steps")
    pdemo.add_argument("--max-time", type=float, default=None, help="Wall-clock time limit in seconds (stops training when reached)")
    pdemo.add_argument("--step-delay", type=float, default=0.15, help="Seconds between steps")
    pdemo.add_argument("--demo-dir", default=None, help="Run directory (default: temp dir)")
    pdemo.add_argument("--autopilot", choices=["off", "suggest", "auto", "ai_suggest", "ai_auto"],
                        default="off", help="Start with autopilot mode enabled")
    pdemo.add_argument("--key-metric", default=None, help="Primary optimization metric for AI autopilot")
    pdemo.set_defaults(func=cmd_demo)

    pserve = sub.add_parser("serve", help="Start the live dashboard server")
    pserve.add_argument("--host", default="0.0.0.0", help="Bind host")
    pserve.add_argument("--port", type=int, default=8421, help="Bind port")
    pserve.add_argument("--poll-interval", type=float, default=0.5, help="JSONL poll interval (seconds)")
    pserve.add_argument("--dirs", help="Comma-separated additional run dirs for multi-run comparison")
    pserve.add_argument("--autopilot", choices=["off", "suggest", "auto", "ai_suggest", "ai_auto"],
                         default="off", help="Start with autopilot mode enabled (server only, no training)")
    pserve.add_argument("--key-metric", default=None, help="Primary optimization metric for AI autopilot")
    pserve.set_defaults(func=cmd_serve)

    plaunch = sub.add_parser("launch", help="Start training + dashboard + autopilot in one command")
    plaunch.add_argument("--config", default="multitask", help="Built-in config: simple, multitask, finetune")
    plaunch.add_argument("--config-file", default=None, help="Path to hotcb.launch.json (values used as defaults, CLI flags override)")
    plaunch.add_argument("--train-fn", default=None, help="Custom training function (module.path:fn_name)")
    plaunch.add_argument("--host", default="0.0.0.0", help="Bind host")
    plaunch.add_argument("--port", type=int, default=8421, help="Bind port")
    plaunch.add_argument("--max-steps", type=int, default=800, help="Number of training steps")
    plaunch.add_argument("--max-time", type=float, default=None, help="Wall-clock time limit in seconds (stops training when reached)")
    plaunch.add_argument("--step-delay", type=float, default=0.12, help="Seconds between steps")
    plaunch.add_argument("--autopilot", choices=["off", "suggest", "auto", "ai_suggest", "ai_auto"],
                          default="off", help="Autopilot mode")
    plaunch.add_argument("--key-metric", default="val_loss", help="Primary optimization metric")
    plaunch.add_argument("--ai-model", default="gpt-4o-mini", help="LLM model for AI autopilot")
    plaunch.add_argument("--ai-budget", type=float, default=5.0, help="Max USD for AI calls")
    plaunch.add_argument("--ai-cadence", type=int, default=50, help="Steps between AI check-ins")
    plaunch.add_argument("--seed", type=int, default=None, help="Random seed")
    plaunch.set_defaults(func=cmd_launch)

    ptune = sub.add_parser("tune", help="Tune module control")
    tune_sub = ptune.add_subparsers(dest="tune_command", required=True)
    pt_enable = tune_sub.add_parser("enable", help="Enable tuning")
    pt_enable.add_argument("--mode", default="active", choices=["active", "observe", "suggest"])
    pt_enable.set_defaults(func=cmd_tune)
    pt_disable = tune_sub.add_parser("disable", help="Disable tuning")
    pt_disable.set_defaults(func=cmd_tune)
    pt_status = tune_sub.add_parser("status", help="Show tune status")
    pt_status.set_defaults(func=cmd_tune_status)
    pt_set = tune_sub.add_parser("set", help="Set tune recipe params")
    pt_set.add_argument("kv", nargs="*")
    pt_set.set_defaults(func=cmd_tune)
    pt_export = tune_sub.add_parser("export-recipe", help="Export tune recipe")
    pt_export.add_argument("--out", help="Output path")
    pt_export.set_defaults(func=cmd_tune_export_recipe)

    return p


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
