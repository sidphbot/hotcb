import pytest

from hotcb.util import (
    iter_env_loggers,
    resolve_tensorboard_writer,
    resolve_mlflow,
    resolve_comet_experiment,
    log_scalar,
)


class FakeTBWriter:
    def __init__(self):
        self.calls = []

    def add_scalar(self, key, value, step):
        self.calls.append((key, value, step))


class FakeTBLogger:
    def __init__(self, writer):
        self.experiment = writer


class FakeMLflowClient:
    def __init__(self):
        self.calls = []

    def log_metric(self, run_id, key, value, step=None):
        self.calls.append((run_id, key, value, step))


class FakeMLflowLogger:
    def __init__(self, client, run_id):
        self.experiment = client
        self.run_id = run_id


class FakeCometExperiment:
    __module__ = "comet_ml"  # important hint for resolver

    def __init__(self):
        self.calls = []

    def log_metric(self, key, value, step=None):
        self.calls.append((key, value, step))


class FakeCometLogger:
    def __init__(self, exp):
        self.experiment = exp


class FakeTrainer:
    def __init__(self, logger=None, loggers=None):
        self.logger = logger
        self.loggers = loggers or []


def test_iter_env_loggers_collects_multiple_sources_and_dedupes():
    w = FakeTBWriter()
    tb = FakeTBLogger(w)
    tr = FakeTrainer(logger=tb, loggers=[tb])

    env = {
        "logger": tb,
        "trainer": tr,
        "loggers": [tb, tb],
    }
    out = iter_env_loggers(env)
    assert len(out) == 1
    assert out[0] is tb


def test_resolve_tensorboard_writer_from_logger_experiment():
    w = FakeTBWriter()
    tb = FakeTBLogger(w)
    env = {"logger": tb}
    got = resolve_tensorboard_writer(env)
    assert got is w


def test_resolve_mlflow_from_logger_experiment_and_run_id():
    client = FakeMLflowClient()
    lg = FakeMLflowLogger(client, "run-123")
    env = {"loggers": [lg]}
    got = resolve_mlflow(env)
    assert got is not None
    exp, run_id = got
    assert exp is client
    assert run_id == "run-123"


def test_resolve_comet_experiment_uses_comet_hints():
    exp = FakeCometExperiment()
    lg = FakeCometLogger(exp)
    env = {"loggers": [lg]}
    got = resolve_comet_experiment(env)
    assert got is exp


def test_log_scalar_prefers_any_available_backend():
    w = FakeTBWriter()
    tb = FakeTBLogger(w)

    client = FakeMLflowClient()
    ml = FakeMLflowLogger(client, "run-xyz")

    exp = FakeCometExperiment()
    comet_logger = FakeCometLogger(exp)

    env = {"loggers": [tb, ml, comet_logger], "step": 7}

    ok = log_scalar(env, "loss", 1.25, step=10)
    assert ok is True

    assert w.calls == [("loss", 1.25, 10)]
    assert client.calls == [("run-xyz", "loss", 1.25, 10)]
    assert exp.calls == [("loss", 1.25, 10)]