from jax_rl.cli import _has_explicit_tensorboard_run_name_override, inject_run_id
from jax_rl.configs.config import ExperimentConfig


def test_inject_run_id_mutates_checkpoint_dir_and_sets_tensorboard_run_name(monkeypatch):
    config = ExperimentConfig()
    config.system.name = "alphazero"
    config.env.env_name = "rustpool:BinPack-v0"
    config.checkpointing.checkpoint_dir = "checkpoints"
    config.logging.tensorboard_logdir = "runs_tb"
    config.logging.tensorboard_run_name = "default"

    monkeypatch.setattr("jax_rl.cli.time.strftime", lambda _: "20260306_111213")

    updated, run_id = inject_run_id(config)

    assert run_id == "alphazero_rustpool_BinPack_v0_20260306_111213"
    assert updated.checkpointing.checkpoint_dir.endswith(
        "checkpoints/alphazero/rustpool_BinPack_v0/20260306_111213"
    )
    assert updated.logging.tensorboard_run_name == run_id


def test_inject_run_id_preserves_tensorboard_name_when_requested(monkeypatch):
    config = ExperimentConfig()
    config.system.name = "ppo"
    config.env.env_name = "rlpallet:UldEnv-v2"
    config.checkpointing.checkpoint_dir = "checkpoints"
    config.logging.tensorboard_logdir = "runs_tb"
    config.logging.tensorboard_run_name = "my-custom:name"

    monkeypatch.setattr("jax_rl.cli.time.strftime", lambda _: "20260306_121314")

    updated, run_id = inject_run_id(
        config,
        preserve_tensorboard_run_name=True,
        run_name_override=config.logging.tensorboard_run_name,
    )

    assert run_id == "ppo_rlpallet_UldEnv_v2_my_custom_name"
    assert updated.checkpointing.checkpoint_dir.endswith(
        "checkpoints/ppo/rlpallet_UldEnv_v2/my_custom_name"
    )
    assert updated.logging.tensorboard_run_name == "my-custom:name"


def test_inject_run_id_does_not_mutate_resume_from(monkeypatch):
    config = ExperimentConfig()
    config.system.name = "alphazero"
    config.env.env_name = "rustpool:BinPack-v0"
    config.checkpointing.resume_from = "checkpoints/old/run/checkpoint_42"

    monkeypatch.setattr("jax_rl.cli.time.strftime", lambda _: "20260306_131415")

    updated, _ = inject_run_id(config)

    assert updated.checkpointing.resume_from == "checkpoints/old/run/checkpoint_42"


def test_detects_explicit_tensorboard_run_name_override_key():
    assert _has_explicit_tensorboard_run_name_override(
        ["logging.tensorboard_run_name=my_run"]
    )
    assert _has_explicit_tensorboard_run_name_override(
        ["ppo.logging.tensorboard_run_name=my_run"]
    )
    assert not _has_explicit_tensorboard_run_name_override(
        ["logging.tensorboard_logdir=runs_tb"]
    )


def test_inject_run_id_appends_checkpoint_name_when_set(monkeypatch):
    config = ExperimentConfig()
    config.system.name = "ppo"
    config.env.env_name = "rustpool:BinPack-v0"
    config.checkpointing.checkpoint_dir = "checkpoints"
    config.checkpointing.checkpoint_name = "best-model:v1"

    monkeypatch.setattr("jax_rl.cli.time.strftime", lambda _: "20260306_141516")

    updated, _ = inject_run_id(config)

    assert updated.checkpointing.checkpoint_dir.endswith(
        "checkpoints/ppo/rustpool_BinPack_v0/20260306_141516/best_model_v1"
    )
