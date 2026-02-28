import pytest

pl = pytest.importorskip("lightning")
from lightning.pytorch import loggers as pl_loggers

from hotcb.modules.cb.util import resolve_tensorboard_writer


def test_resolve_tensorboard_writer_with_real_lightning_tensorboard_logger(tmp_path):
    # This does not require tensorboard package installed; Lightning's logger object exists regardless.
    # The experiment property may lazily create a writer depending on environment.
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(tmp_path))

    env = {"logger": tb_logger}

    writer = resolve_tensorboard_writer(env)

    # We don't assert exact type because SummaryWriter backend may vary (tensorboard vs tensorboardX),
    # but it should behave like a TB writer (add_scalar).
    assert writer is not None
    assert hasattr(writer, "add_scalar")
    assert callable(writer.add_scalar)