"""Unit tests for HotLossController (spec §19.7)."""
from __future__ import annotations

import pytest

from hotcb.modules.loss import HotLossController
from hotcb.ops import HotOp


def _make_mutable_state():
    return {"weights": {}, "terms": {}, "ramps": {}}


def _op(op="set_params", params=None, id="main"):
    return HotOp(module="loss", op=op, id=id, params=params)


class TestWeightsMutation:
    def test_distill_w_suffix(self):
        ctrl = HotLossController()
        ls = _make_mutable_state()
        result = ctrl.apply_op(_op(params={"distill_w": 0.2, "depth_w": 1.5}), {"mutable_state": ls})
        assert result.decision == "applied"
        assert ls["weights"]["distill"] == 0.2
        assert ls["weights"]["depth"] == 1.5

    def test_fallback_to_weights_bucket(self):
        ctrl = HotLossController()
        ls = _make_mutable_state()
        result = ctrl.apply_op(_op(params={"custom_metric": 0.5}), {"mutable_state": ls})
        assert result.decision == "applied"
        assert ls["weights"]["custom_metric"] == 0.5


class TestTermsToggle:
    def test_terms_dot_prefix(self):
        ctrl = HotLossController()
        ls = _make_mutable_state()
        result = ctrl.apply_op(_op(params={"terms.aux_depth": False, "terms.aux_heatmap": True}), {"mutable_state": ls})
        assert result.decision == "applied"
        assert ls["terms"]["aux_depth"] is False
        assert ls["terms"]["aux_heatmap"] is True

    def test_terms_as_dict(self):
        ctrl = HotLossController()
        ls = _make_mutable_state()
        result = ctrl.apply_op(_op(params={"terms": {"aux_depth": False, "aux_heatmap": True}}), {"mutable_state": ls})
        assert result.decision == "applied"
        assert ls["terms"]["aux_depth"] is False
        assert ls["terms"]["aux_heatmap"] is True


class TestRamps:
    def test_ramps_dot_prefix(self):
        ctrl = HotLossController()
        ls = _make_mutable_state()
        ramp_cfg = {"type": "linear", "warmup_frac": 0.2, "end": 2.0}
        result = ctrl.apply_op(_op(params={"ramps.depth": ramp_cfg}), {"mutable_state": ls})
        assert result.decision == "applied"
        assert ls["ramps"]["depth"] == ramp_cfg

    def test_ramps_as_dict(self):
        ctrl = HotLossController()
        ls = _make_mutable_state()
        ramp_cfg = {"type": "linear", "end": 2.0}
        result = ctrl.apply_op(_op(params={"ramps": {"depth": ramp_cfg}}), {"mutable_state": ls})
        assert result.decision == "applied"
        assert ls["ramps"]["depth"] == ramp_cfg


class TestMissingMutableState:
    def test_no_mutable_state_in_env(self):
        ctrl = HotLossController()
        result = ctrl.apply_op(_op(params={"distill_w": 0.2}), {})
        assert result.decision == "failed"
        assert result.error == "missing_mutable_state"

    def test_resolve_mutable_state_callable(self):
        ctrl = HotLossController()
        ls = _make_mutable_state()
        env = {"resolve_mutable_state": lambda: ls}
        result = ctrl.apply_op(_op(params={"distill_w": 0.3}), env)
        assert result.decision == "applied"
        assert ls["weights"]["distill"] == 0.3


class TestEnableDisable:
    def test_disabled_handle_skips(self):
        ctrl = HotLossController()
        ls = _make_mutable_state()
        env = {"mutable_state": ls}
        ctrl.apply_op(HotOp(module="loss", op="disable", id="main"), env)
        result = ctrl.apply_op(_op(params={"distill_w": 0.2}), env)
        assert result.decision == "skipped_noop"
        assert result.notes == "handle_disabled"
        assert ls["weights"] == {}

    def test_re_enable_then_apply(self):
        ctrl = HotLossController()
        ls = _make_mutable_state()
        env = {"mutable_state": ls}
        ctrl.apply_op(HotOp(module="loss", op="disable", id="main"), env)
        ctrl.apply_op(HotOp(module="loss", op="enable", id="main"), env)
        result = ctrl.apply_op(_op(params={"distill_w": 0.2}), env)
        assert result.decision == "applied"
        assert ls["weights"]["distill"] == 0.2


class TestAutoDisableOnError:
    def test_error_disables_handle(self):
        ctrl = HotLossController(auto_disable_on_error=True)

        class BrokenDict(dict):
            def setdefault(self, key, default=None):
                raise RuntimeError("broken")

        env = {"mutable_state": BrokenDict()}
        result = ctrl.apply_op(_op(params={"distill_w": 0.2}), env)
        assert result.decision == "failed"
        assert "broken" in result.error
        # handle should be disabled
        assert ctrl.handles["main"].enabled is False

    def test_no_auto_disable_when_off(self):
        ctrl = HotLossController(auto_disable_on_error=False)

        class BrokenDict(dict):
            def setdefault(self, key, default=None):
                raise RuntimeError("broken")

        env = {"mutable_state": BrokenDict()}
        result = ctrl.apply_op(_op(params={"distill_w": 0.2}), env)
        assert result.decision == "failed"
        assert ctrl.handles["main"].enabled is True


class TestStatus:
    def test_status_structure(self):
        ctrl = HotLossController()
        ls = _make_mutable_state()
        ctrl.apply_op(_op(params={"distill_w": 0.2}), {"mutable_state": ls})
        status = ctrl.status()
        assert "main" in status
        assert status["main"]["enabled"] is True
        assert "distill_w" in status["main"]["last_params"]
        assert status["main"]["last_error"] is None


class TestUnknownOp:
    def test_unknown_op_ignored(self):
        ctrl = HotLossController()
        result = ctrl.apply_op(HotOp(module="loss", op="reset", id="main"), {"mutable_state": {}})
        assert result.decision == "ignored"
        assert "unknown_op" in (result.notes or "")
