import pytest

pl = pytest.importorskip("pytorch_lightning")

from hotcb.modules.cb.adapters.lightning import HotCallbackController
from hotcb.modules.cb.util import resolve_tensorboard_writer


class DummyController:
    def apply(self, env, events):
        pass


class FakeTBWriter:
    def __init__(self):
        self.calls = []

    def add_scalar(self, key, value, step):
        self.calls.append((key, value, step))


class FakeTBLogger:
    def __init__(self, w):
        self.experiment = w


class FakeTrainer:
    def __init__(self, logger):
        self.global_step = 3
        self.current_epoch = 1
        self.logger = logger
        self.loggers = [logger]
        self._printed = []

    def print(self, s):
        self._printed.append(s)


def test_lightning_env_resolves_tb_writer_via_trainer_logger():
    w = FakeTBWriter()
    tb = FakeTBLogger(w)
    tr = FakeTrainer(tb)

    hot = HotCallbackController(DummyController())
    env = hot._env(tr, object(), phase="train")

    got = resolve_tensorboard_writer(env)
    assert got is w