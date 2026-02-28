# tests/test_config_yaml_parse.py
from __future__ import annotations

import pytest

from hotcb.modules.cb.config import parse_yaml_config, ConfigError


def test_parse_yaml_requires_pyyaml(tmp_path):
    p = tmp_path / "hotcb.yaml"
    p.write_text("version: 1\ncallbacks: {}\n", encoding="utf-8")

    # If PyYAML isn't installed, this should raise ConfigError
    try:
        ops = parse_yaml_config(str(p))
        assert isinstance(ops, list)
    except ConfigError:
        pytest.skip("PyYAML not installed in this environment (expected for core install).")