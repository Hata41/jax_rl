import jax.numpy as jnp

from jax_rl.systems.alphazero.anakin.steps import _flatten_batch_time


def test_flatten_batch_time_keeps_feature_axes_for_rank4():
    x = jnp.zeros((8, 1, 6, 3), dtype=jnp.float32)
    y = _flatten_batch_time(x)
    assert y.shape == (8, 6, 3)


def test_flatten_batch_time_keeps_feature_axes_for_rank3():
    x = jnp.zeros((8, 1, 12), dtype=jnp.float32)
    y = _flatten_batch_time(x)
    assert y.shape == (8, 12)


def test_flatten_batch_time_reduces_batch_and_time_axes():
    x = jnp.zeros((4, 5, 7), dtype=jnp.float32)
    y = _flatten_batch_time(x)
    assert y.shape == (20, 7)
