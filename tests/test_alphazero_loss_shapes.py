import jax.numpy as jnp
import jax

from jax_rl.systems.alphazero.anakin.steps import _flatten_batch_time
from jax_rl.systems.ppo.advantages import compute_gae


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


def _legacy_value_targets(
    reward: jax.Array,
    done: jax.Array,
    search_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> jax.Array:
    discounts = (1.0 - done[:, :-1]) * float(gamma)
    deltas = reward[:, :-1] + discounts * search_value[:, 1:] - search_value[:, :-1]

    def _scan_adv(next_adv, elems):
        delta_t, discount_t = elems
        adv_t = delta_t + discount_t * float(gae_lambda) * next_adv
        return adv_t, adv_t

    init_adv = jnp.zeros_like(deltas[:, -1])
    _, reversed_adv = jax.lax.scan(
        _scan_adv,
        init_adv,
        (jnp.swapaxes(deltas, 0, 1)[::-1], jnp.swapaxes(discounts, 0, 1)[::-1]),
    )
    adv = jnp.swapaxes(reversed_adv[::-1], 0, 1)
    return adv + search_value[:, :-1]


def test_alphazero_value_targets_match_compute_gae_returns():
    reward = jnp.array(
        [[0.4, -0.2, 0.6, 0.1], [1.2, 0.5, -0.3, 0.9]],
        dtype=jnp.float32,
    )
    done = jnp.array(
        [[False, False, True, True], [False, True, False, False]],
        dtype=jnp.bool_,
    )
    search_value = jnp.array(
        [[0.3, 0.2, 0.1, -0.2], [0.7, 0.4, 0.6, 0.5]],
        dtype=jnp.float32,
    )
    gamma = 0.99
    gae_lambda = 0.95

    legacy_targets = _legacy_value_targets(
        reward=reward,
        done=done.astype(jnp.float32),
        search_value=search_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    _, gae_returns = jax.vmap(
        lambda reward_row, done_row, value_row: compute_gae(
            rewards=reward_row[:-1],
            dones=done_row[:-1],
            truncated=jnp.zeros_like(done_row[:-1]),
            values=value_row[:-1],
            last_values=value_row[-1],
            gamma=gamma,
            gae_lambda=gae_lambda,
        ),
        in_axes=(0, 0, 0),
    )(reward, done, search_value)

    assert jnp.allclose(gae_returns, legacy_targets, atol=1e-5)
