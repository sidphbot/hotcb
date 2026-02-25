import pytest

pytest.importorskip("transformers")

from hotcb.adapters.hf import HotHFCallback
from hotcb.util import resolve_tensorboard_writer


class DummyController:
    def apply(self, env, events):
        pass


class FakeTBWriter:
    def add_scalar(self, key, value, step):
        pass


class FakeTBLogger:
    def __init__(self, w):
        self.experiment = w


class Dummy:
    pass


def test_hf_env_allows_logger_resolution_if_user_passes_logger_in_extra():
    cb = HotHFCallback(DummyController())

    args = Dummy()
    state = Dummy()
    state.global_step = 5
    state.epoch = 0.0
    control = Dummy()

    w = FakeTBWriter()
    tb = FakeTBLogger(w)

    env = cb._env(args, state, control, phase="train", logger=tb)
    got = resolve_tensorboard_writer(env)
    assert got is w