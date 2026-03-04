from pathlib import Path

from purejax_ppo.logging import create_tensorboard_writer, log_scalar_metrics


def test_tensorboard_writer_creates_event_file(tmp_path: Path):
    writer = create_tensorboard_writer(str(tmp_path), "tb_test")
    log_scalar_metrics(writer, {"loss_total": 1.23, "reward_mean": 2.0}, step=32)
    writer.flush()
    writer.close()

    run_dir = tmp_path / "tb_test"
    assert run_dir.exists()
    event_files = list(run_dir.glob("events.*"))
    assert len(event_files) > 0