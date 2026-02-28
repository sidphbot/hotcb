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
            if rec.get("module") not in {"cb", "opt", "loss"}:
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

    return p


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
