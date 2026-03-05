import os
import importlib

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import jax.numpy as jnp
import optax
import pytest

from jax_rl.configs.config import PPOConfig
from jax_rl.systems.ppo.anakin.system import train
from jax_rl.utils.exceptions import ConfigDivisibilityError
from jax_rl.utils.types import TrainState

train_module = importlib.import_module("jax_rl.systems.ppo.anakin.system")

def test_train_raises_when_num_envs_not_divisible_by_device_count(monkeypatch):
    monkeypatch.setattr(train_module.jax, "local_device_count", lambda: 4)

    config = PPOConfig(num_envs=6)
    with pytest.raises(ConfigDivisibilityError, match="num_envs must be divisible"):
        train(config)


def test_pmean_synchronizes_replicas_across_devices():
    if jax.local_device_count() < 2:
        pytest.skip("Requires at least 2 local devices")

    optimizer = optax.sgd(learning_rate=1.0)
    base_actor_params = {"w": jnp.array(0.0, dtype=jnp.float32)}
    base_critic_params = {"w": jnp.array(0.0, dtype=jnp.float32)}
    base_state = TrainState(
        params={"actor": base_actor_params, "critic": base_critic_params},
        actor_opt_state=optimizer.init(base_actor_params),
        critic_opt_state=optimizer.init(base_critic_params),
    )
    num_devices = 2
    replicated_state = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (num_devices,) + x.shape),
        base_state,
    )

    def mock_ppo_update(state: TrainState, rollout_data: jax.Array):
        def loss_fn(params):
            return jnp.mean((params["w"] - rollout_data) ** 2)

        grads = jax.grad(loss_fn)(state.params["actor"])
        grads = jax.lax.pmean(grads, axis_name="device")
        updates, next_actor_opt_state = optimizer.update(
            grads,
            state.actor_opt_state,
            state.params["actor"],
        )
        next_actor_params = optax.apply_updates(state.params["actor"], updates)
        return TrainState(
            params={"actor": next_actor_params, "critic": state.params["critic"]},
            actor_opt_state=next_actor_opt_state,
            critic_opt_state=state.critic_opt_state,
        )

    pmap_update = jax.pmap(mock_ppo_update, axis_name="device")

    per_device_rollout_data = jnp.array(
        [
            [1.0, 2.0],
            [-3.0, -1.0],
        ],
        dtype=jnp.float32,
    )

    next_state = pmap_update(replicated_state, per_device_rollout_data)

    assert jnp.array_equal(next_state.params["actor"]["w"][0], next_state.params["actor"]["w"][1])
