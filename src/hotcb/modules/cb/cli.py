# src/hotcb/cli.py
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

DEFAULT_CONFIG = """version: 1
callbacks: {}
"""


def _ensure_dir(p: str) -> None:
    """
    Ensure directory exists (mkdir -p behavior).

    Parameters
    ----------
    p:
        Directory path. If empty, does nothing.

    Notes
    -----
    This is used for `--dir` run directory creation and for output file dirs.
    """
    if not p:
        return
    os.makedirs(p, exist_ok=True)


def _cmd_path(run_dir: str) -> str:
    """
    Compute the canonical commands JSONL path for a run directory.

    Parameters
    ----------
    run_dir:
        Run directory. Typically a folder containing:
          - hotcb.yaml
          - hotcb.commands.jsonl
          - hotcb.log (optional)

    Returns
    -------
    str
        <run_dir>/hotcb.commands.jsonl
    """
    return os.path.join(run_dir, "hotcb.commands.jsonl")


def _cfg_path(run_dir: str) -> str:
    """
    Compute the canonical YAML config path for a run directory.

    Parameters
    ----------
    run_dir:
        Run directory path.

    Returns
    -------
    str
        <run_dir>/hotcb.yaml
    """
    return os.path.join(run_dir, "hotcb.yaml")


def _append_cmd(run_dir: str, obj: Dict[str, Any]) -> None:
    """
    Append a single JSON object as a line to the commands JSONL file.

    Parameters
    ----------
    run_dir:
        Run directory containing hotcb.commands.jsonl (created if missing).

    obj:
        JSON-serializable dict representing a controller command.
        Common forms:
          - {"op":"enable","id":"cb1"}
          - {"op":"set_params","id":"cb1","params":{"every":10}}
          - {"op":"load","id":"cb2","target":{...},"init":{...},"enabled":true}

    Notes
    -----
    - Commands are append-only to preserve an audit trail.
    - `HotController` reads only new lines via byte offset cursor.
    """
    _ensure_dir(run_dir)
    p = _cmd_path(run_dir)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _parse_kv(pairs: List[str]) -> Dict[str, Any]:
    """
    Parse CLI key=value pairs into a dict with light type inference.

    Accepted types
    --------------
    - "true"/"false" (case-insensitive) -> bool
    - integers -> int
    - floats -> float
    - JSON objects/arrays if the value begins with '{' or '[' -> parsed via json.loads
    - otherwise -> string (as provided by the shell after its own quote processing)

    Parameters
    ----------
    pairs:
        List of strings, each in the form "key=value".

        Examples:
          - ["every=10", "prefix=[m]"]
          - ["threshold=0.25", "enabled=true"]
          - ["paths=[\"loss\",\"outputs.logits\"]"]   # JSON

    Returns
    -------
    Dict[str, Any]
        Parsed mapping.

    Raises
    ------
    SystemExit
        If any item is not in key=value form.

    Example
    -------
    >>> _parse_kv(["every=10", "gpu=true", "meta={\"a\":1}"])
    {'every': 10, 'gpu': True, 'meta': {'a': 1}}
    """
    out: Dict[str, Any] = {}
    for s in pairs:
        if "=" not in s:
            raise SystemExit(f"Expected key=value, got: {s}")
        k, v = s.split("=", 1)
        v = v.strip()

        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
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
    """
    CLI subcommand: `hotcb init`

    Creates (if missing) in --dir:
      - hotcb.yaml (empty template)
      - hotcb.commands.jsonl (empty file)

    This is a convenience to bootstrap a run directory.

    Example
    -------
    $ hotcb --dir runs/exp1 init
    """
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
    """
    CLI subcommand: `hotcb enable <id>`

    Appends an enable op to commands.jsonl.

    Example
    -------
    $ hotcb --dir runs/exp1 enable feat_viz
    """
    _append_cmd(args.dir, {"op": "enable", "id": args.id})
    print(f"Enabled {args.id}")


def cmd_disable(args: argparse.Namespace) -> None:
    """
    CLI subcommand: `hotcb disable <id>`

    Appends a disable op to commands.jsonl.

    Example
    -------
    $ hotcb --dir runs/exp1 disable feat_viz
    """
    _append_cmd(args.dir, {"op": "disable", "id": args.id})
    print(f"Disabled {args.id}")


def cmd_set(args: argparse.Namespace) -> None:
    """
    CLI subcommand: `hotcb set <id> key=value ...`

    Appends a set_params op to commands.jsonl.

    Examples
    --------
    $ hotcb --dir runs/exp1 set timing every=10 window=200
    $ hotcb --dir runs/exp1 set tstats paths=loss,outputs.logits every=25
    $ hotcb --dir runs/exp1 set guard raise_on_trigger=true
    """
    params = _parse_kv(args.kv)
    _append_cmd(args.dir, {"op": "set_params", "id": args.id, "params": params})
    print(f"Set params for {args.id}: {list(params.keys())}")


def cmd_load(args: argparse.Namespace) -> None:
    """
    CLI subcommand: `hotcb load <id> ...`

    Load a callback from a Python module path OR from a Python file path.

    Required flags
    --------------
    Exactly one of:
      --file   /path/to/callback.py
      --module mypkg.callbacks.foo

    And:
      --symbol <ClassName>

    Optional flags
    --------------
    --enabled
        If provided, enable immediately after loading.

    --init key=value ...
        Constructor kwargs for the callback (applied only on first instantiation).
        If you pass `id`, it will be overwritten by controller to match <id>.

    Examples
    --------
    Load from a file:
      $ hotcb --dir runs/exp1 load my_diag --file /tmp/my_diag.py --symbol MyDiag --enabled --init msg="hello"

    Load from a module:
      $ hotcb --dir runs/exp1 load timing --module hotcb.callbacks.timing --symbol TimingCallback --enabled --init every=50 window=200
    """
    init = _parse_kv(args.init or [])
    if args.file:
        target = {"kind": "python_file", "path": args.file, "symbol": args.symbol}
    else:
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
    """
    CLI subcommand: `hotcb unload <id>`

    Unloads the callback instance from memory (optional feature).
    This also disables the callback.

    Use this when:
      - you want to free resources (files, threads),
      - you no longer need the callback and want to reset its state.

    Example
    -------
    $ hotcb --dir runs/exp1 unload feat_viz
    """
    _append_cmd(args.dir, {"op": "unload", "id": args.id})
    print(f"Unloaded {args.id}")


def build_parser() -> argparse.ArgumentParser:
    """
    Build an argparse parser for the hotcb CLI.

    Returns
    -------
    argparse.ArgumentParser
        Parser with subcommands:
          - init
          - enable
          - disable
          - set
          - load
          - unload

    Notes
    -----
    - The CLI is intentionally simple: it only appends commands to a JSONL file.
    - The training process must run `HotController` with commands_path pointing
      to this same JSONL file for the commands to be applied.
    """
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
    """
    CLI entrypoint for `hotcb`.

    This is registered in pyproject.toml:
      [project.scripts]
      hotcb = "hotcb.cli:main"

    Example
    -------
    $ hotcb --help
    $ hotcb --dir runs/exp1 init
    """
    p = build_parser()
    args = p.parse_args()
    args.func(args)