from pathlib import Path

from jax_rl.logging import TensorBoardLogger
from jax_rl.types import LogEvent


def test_tensorboard_writer_creates_event_file(tmp_path: Path):
    run_dir = tmp_path / "tb_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = TensorBoardLogger(str(run_dir))
    writer.log_dict({"loss_total": 1.23, "done_fraction": 0.5}, step=32, event=LogEvent.TRAIN)
    writer.flush()
    writer.close()

    assert run_dir.exists()
    event_files = list(run_dir.glob("events.*"))
    assert len(event_files) > 0