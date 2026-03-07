from jax_rl.cli import _has_explicit_tensorboard_run_name_override, inject_run_id
from jax_rl.configs.config import ExperimentConfig


def test_inject_run_id_mutates_checkpoint_dir_and_sets_tensorboard_run_name(monkeypatch):
    config = ExperimentConfig()
    config.system.name = "alphazero"
    config.io.checkpoint.checkpoint_dir = "checkpoints"
    config.io.name = "run-alpha"

    updated, run_id = inject_run_id(config)

    assert run_id == "alphazero_run_alpha"
    assert updated.io.checkpoint.checkpoint_dir.endswith(
        "checkpoints/alphazero/run_alpha"
    )
    assert updated.io.name == "run_alpha"


def test_inject_run_id_preserves_tensorboard_name_when_requested(monkeypatch):
    config = ExperimentConfig()
    config.system.name = "ppo"
    config.io.checkpoint.checkpoint_dir = "checkpoints"
    config.io.name = None

    updated, run_id = inject_run_id(
        config,
        preserve_tensorboard_run_name=True,
        run_name_override="ignored",
    )

    assert run_id == "ppo_ppo"
    assert updated.io.checkpoint.checkpoint_dir.endswith(
        "checkpoints/ppo/ppo"
    )
    assert updated.io.name == "ppo"


def test_inject_run_id_does_not_mutate_resume_from(monkeypatch):
    config = ExperimentConfig()
    config.system.name = "alphazero"
    config.io.checkpoint.resume_from = "checkpoints/old/run/checkpoint_42"

    updated, _ = inject_run_id(config)

    assert updated.io.checkpoint.resume_from == "checkpoints/old/run/checkpoint_42"


def test_detects_explicit_tensorboard_run_name_override_key():
    assert _has_explicit_tensorboard_run_name_override(
        ["io.logger.tensorboard_run_name=my_run"]
    )
    assert _has_explicit_tensorboard_run_name_override(
        ["ppo.io.logger.tensorboard_run_name=my_run"]
    )
    assert not _has_explicit_tensorboard_run_name_override(
        ["io.logger.tensorboard_logdir=runs_tb"]
    )


def test_inject_run_id_appends_checkpoint_name_when_set(monkeypatch):
    config = ExperimentConfig()
    config.system.name = "ppo"
    config.io.checkpoint.checkpoint_dir = "checkpoints"
    config.io.checkpoint.checkpoint_name = "best-model:v1"

    updated, _ = inject_run_id(config)

    assert updated.io.checkpoint.checkpoint_dir.endswith(
        "checkpoints/ppo/best_model_v1"
    )
