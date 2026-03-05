import jax
import jax.numpy as jnp

from jax_rl.systems.ppo.rollout import _extract_done_and_truncated


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
