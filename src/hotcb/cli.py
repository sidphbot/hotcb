from __future__ import annotations

import argparse
import json
import os
import shlex
from typing import Any, Dict, List, Tuple

DEFAULT_CONFIG = """version: 1
callbacks: {}
"""

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _cmd_path(run_dir: str) -> str:
    return os.path.join(run_dir, "hotcb.commands.jsonl")

def _cfg_path(run_dir: str) -> str:
    return os.path.join(run_dir, "hotcb.yaml")

def _append_cmd(run_dir: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(run_dir)
    p = _cmd_path(run_dir)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _parse_kv(pairs: List[str]) -> Dict[str, Any]:
    """
    Parse key=value pairs. Values are parsed as:
      - true/false -> bool
      - int/float
      - json objects/arrays if value starts with { or [
      - else string (supports shell quoting)
    """
    out: Dict[str, Any] = {}
    for s in pairs:
        if "=" not in s:
            raise SystemExit(f"Expected key=value, got: {s}")
        k, v = s.split("=", 1)
        v = v.strip()

        # allow users to pass quoted strings in shells; keep as-is
        # (they'll arrive without quotes usually, but keep logic simple)
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
            continue

        # json literal
        if v.startswith("{") or v.startswith("["):
            try:
                out[k] = json.loads(v)
                continue
            except Exception:
                pass

        # numeric
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
    _ensure_dir(run_dir)

    cfg = _cfg_path(run_dir)
    if not os.path.exists(cfg):
        with open(cfg, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG)
        print(f"Wrote {cfg}")
    else:
        print(f"Exists: {cfg}")

    cmdp = _cmd_path(run_dir)
    if not os.path.exists(cmdp):
        with open(cmdp, "w", encoding="utf-8") as f:
            pass
        print(f"Created {cmdp}")
    else:
        print(f"Exists: {cmdp}")

def cmd_enable(args: argparse.Namespace) -> None:
    _append_cmd(args.dir, {"op": "enable", "id": args.id})
    print(f"Enabled {args.id}")

def cmd_disable(args: argparse.Namespace) -> None:
    _append_cmd(args.dir, {"op": "disable", "id": args.id})
    print(f"Disabled {args.id}")

def cmd_set(args: argparse.Namespace) -> None:
    params = _parse_kv(args.kv)
    _append_cmd(args.dir, {"op": "set_params", "id": args.id, "params": params})
    print(f"Set params for {args.id}: {list(params.keys())}")

def cmd_load(args: argparse.Namespace) -> None:
    init = _parse_kv(args.init or [])
    target: Dict[str, Any]
    if args.file:
        target = {"kind": "python_file", "path": args.file, "symbol": args.symbol}
    else:
        # module path like mypkg.callbacks.foo
        target = {"kind": "module", "path": args.module, "symbol": args.symbol}

    payload: Dict[str, Any] = {
        "op": "load",
        "id": args.id,
        "target": target,
        "init": init,
        "enabled": bool(args.enabled),
    }
    _append_cmd(args.dir, payload)
    print(f"Loaded {args.id} from {target['kind']}:{target['path']}:{target['symbol']} (enabled={args.enabled})")

def cmd_unload(args: argparse.Namespace) -> None:
    _append_cmd(args.dir, {"op": "unload", "id": args.id})
    print(f"Unloaded {args.id}")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hotcb", description="Hot callback controller CLI")
    p.add_argument("--dir", default=".", help="Run directory containing hotcb.yaml and hotcb.commands.jsonl")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="Initialize hotcb.yaml and hotcb.commands.jsonl in --dir")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("enable", help="Enable a callback by id")
    sp.add_argument("id")
    sp.set_defaults(func=cmd_enable)

    sp = sub.add_parser("disable", help="Disable a callback by id")
    sp.add_argument("id")
    sp.set_defaults(func=cmd_disable)

    sp = sub.add_parser("set", help="Set callback params (key=value ...)")
    sp.add_argument("id")
    sp.add_argument("kv", nargs="+", help="key=value pairs")
    sp.set_defaults(func=cmd_set)

    sp = sub.add_parser("load", help="Load a callback from a python file or module")
    sp.add_argument("id")
    src = sp.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="Path to a .py file to load")
    src.add_argument("--module", help="Module path to import, e.g. mypkg.callbacks.foo")
    sp.add_argument("--symbol", required=True, help="Class name inside file/module")
    sp.add_argument("--enabled", action="store_true", help="Enable immediately after loading")
    sp.add_argument("--init", nargs="*", default=[], help="Init kwargs key=value ...")
    sp.set_defaults(func=cmd_load)

    sp = sub.add_parser("unload", help="Unload a callback instance (also disables)")
    sp.add_argument("id")
    sp.set_defaults(func=cmd_unload)

    return p

def main() -> None:
    p = build_parser()
    args = p.parse_args()
    args.func(args)