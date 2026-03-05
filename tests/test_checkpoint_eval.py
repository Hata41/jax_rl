from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from jax_rl.configs.config import PPOConfig
from jax_rl.systems.ppo.eval import evaluate
from jax_rl.systems.ppo.update import make_actor_optimizer, make_critic_optimizer
from jax_rl.utils.checkpoint import Checkpointer
from jax_rl.utils.exceptions import CheckpointRestoreError
from jax_rl.networks import init_policy_value_params
from jax_rl.utils.types import TrainState


def test_checkpoint_roundtrip(tmp_path: Path):
    config = PPOConfig()
    key = jax.random.PRNGKey(0)
    params = init_policy_value_params(
        key,
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [32, 32]},
        obs_dim=4,
        action_dims=2,
    )
    actor_optimizer = make_actor_optimizer(config)
    critic_optimizer = make_critic_optimizer(config)
    state = TrainState(
        params=params,
        actor_opt_state=actor_optimizer.init(params.state),
        critic_opt_state=critic_optimizer.init(params.state),
    )

    save_key = jax.random.PRNGKey(42)
    checkpointer = Checkpointer(
        checkpoint_dir=str(tmp_path),
        max_to_keep=2,
        keep_period=None,
        save_interval_steps=1,
        metadata={"config": {"env_name": config.env_name}},
    )
    success = checkpointer.save(
        timestep=1,
        train_state=state,
        key=save_key,
        metric=1.23,
    )
    assert success

    loaded = checkpointer.restore(template_train_state=state, template_key=save_key)

    assert loaded["step"] == 1
    loaded_state = loaded["train_state"]
    state_tree_match = jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.allclose, state, loaded_state)
    )
    assert state_tree_match
    assert jnp.allclose(loaded["key"], save_key)


def test_max_to_keep(tmp_path: Path):
    config = PPOConfig()
    key = jax.random.PRNGKey(7)
    params = init_policy_value_params(
        key,
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [16, 16]},
        obs_dim=4,
        action_dims=2,
    )
    actor_optimizer = make_actor_optimizer(config)
    critic_optimizer = make_critic_optimizer(config)
    state = TrainState(
        params=params,
        actor_opt_state=actor_optimizer.init(params.state),
        critic_opt_state=critic_optimizer.init(params.state),
    )

    checkpointer = Checkpointer(
        checkpoint_dir=str(tmp_path),
        max_to_keep=2,
        keep_period=None,
        save_interval_steps=1,
        metadata={"config": {"seed": config.seed}},
    )
    for step in range(1, 5):
        assert checkpointer.save(
            timestep=step,
            train_state=state,
            key=jax.random.PRNGKey(step),
            metric=float(step),
        )

    assert checkpointer.all_steps() == (3, 4)


def test_evaluate_returns_expected_keys():
    config = PPOConfig(seed=1)
    params = init_policy_value_params(
        jax.random.PRNGKey(1),
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [16, 16]},
        obs_dim=4,
        action_dims=2,
    )
    metrics = evaluate(params, config, num_episodes=2, max_steps_per_episode=16)

    for key in ["return_mean", "return_std", "return_min", "return_max", "episodes"]:
        assert key in metrics
    assert metrics["episodes"] == 2


def test_restore_nonexistent_explicit_path_raises_checkpoint_restore_error(tmp_path: Path):
    checkpointer = Checkpointer(checkpoint_dir=str(tmp_path / "manager"))
    missing_dir = tmp_path / "does-not-exist"

    with pytest.raises(CheckpointRestoreError, match="does not exist"):
        checkpointer.restore(checkpoint_path=str(missing_dir))