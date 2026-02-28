#!/usr/bin/env python3
"""
create_hotcb_structure.py
Creates the hotcb project skeleton.
"""

from pathlib import Path

# Root project name
ROOT = Path("hotcb")

# File structure definition
FILES = [
    "pyproject.toml",
    "README.md",
    "LICENSE",

    "src/hotcb/__init__.py",
    "src/hotcb/controller.py",
    "src/hotcb/loader.py",
    "src/hotcb/ops.py",
    "src/hotcb/protocol.py",
    "src/hotcb/util.py",
    "src/hotcb/config.py",
    "src/hotcb/cli.py",

    "src/hotcb/adapters/__init__.py",
    "src/hotcb/adapters/lightning.py",
    "src/hotcb/adapters/hf.py",

    "examples/callbacks/feat_viz.py",
    "examples/callbacks/print_metrics.py",
    "examples/lightning_train.py",
    "examples/hf_train.py",
    "examples/hotcb.yaml",
]


def main():
    for file in FILES:
        path = ROOT / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        print(f"Created {path}")


if __name__ == "__main__":
    main()