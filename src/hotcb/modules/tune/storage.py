from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .state import Mutation, Segment


def _append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def write_mutation(run_dir: str, mutation: Mutation) -> None:
    path = os.path.join(run_dir, "hotcb.tune.mutations.jsonl")
    _append_jsonl(path, mutation.to_dict())


def write_segment(run_dir: str, segment: Segment) -> None:
    path = os.path.join(run_dir, "hotcb.tune.segments.jsonl")
    _append_jsonl(path, segment.to_dict())


def write_summary(run_dir: str, summary: dict) -> None:
    path = os.path.join(run_dir, "hotcb.tune.summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


def load_recipe_yaml(path: str) -> dict:
    """Load a tune recipe from YAML file."""
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        raise ImportError("pyyaml required for tune recipe loading: pip install hotcb[tune]")


def save_recipe_yaml(path: str, data: dict) -> None:
    """Save a tune recipe to YAML file."""
    try:
        import yaml  # type: ignore
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        # Fallback to JSON
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def load_mutations_log(run_dir: str) -> List[dict]:
    path = os.path.join(run_dir, "hotcb.tune.mutations.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_segments_log(run_dir: str) -> List[dict]:
    path = os.path.join(run_dir, "hotcb.tune.segments.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
