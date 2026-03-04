import jax
import jax.numpy as jnp

from jax_rl.env import make_stoa_env
from jax_rl.rollout import _extract_done_and_truncated


class MockTimeStep:
    def __init__(self, discount, is_last):
        self.discount = jnp.asarray(discount, dtype=jnp.float32)
        self._is_last = jnp.asarray(is_last, dtype=jnp.bool_)

    def last(self):
        return self._is_last


def test_timestep_done_and_truncated_parsing():
    timestep = MockTimeStep(
        discount=jnp.array([0.0, 1.0, 0.5], dtype=jnp.float32),
        is_last=jnp.array([True, True, False], dtype=jnp.bool_),
    )

    done, truncated = _extract_done_and_truncated(timestep)

    assert jnp.array_equal(done, jnp.array([True, False, False], dtype=jnp.bool_))
    assert jnp.array_equal(truncated, jnp.array([False, True, False], dtype=jnp.bool_))


def test_stoa_autoreset_exposes_next_obs_in_extras():
    env, env_params = make_stoa_env("CartPole-v1", num_envs_per_device=1)

    key = jax.random.PRNGKey(0)
    key, reset_key, action_key = jax.random.split(key, 3)

    env_state, timestep = env.reset(reset_key, env_params)
    assert "next_obs" in timestep.extras
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda a, b: a.shape == b.shape,
            timestep.extras["next_obs"],
            timestep.observation,
        )
    )

    action = env.action_space(env_params).sample(action_key)
    action = jnp.expand_dims(action, axis=0)
    next_env_state, next_timestep = env.step(env_state, action, env_params)
    del next_env_state
    assert "next_obs" in next_timestep.extras
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda a, b: a.shape == b.shape,
            next_timestep.extras["next_obs"],
            next_timestep.observation,
        )
    )
