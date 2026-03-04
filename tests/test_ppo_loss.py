import jax
import jax.numpy as jnp

from jax_rl.losses import ppo_loss
from jax_rl.networks import init_policy_value_params, policy_value_apply
from jax_rl.types import FlattenBatch


def test_ppo_loss_is_finite():
    key = jax.random.PRNGKey(0)
    params = init_policy_value_params(key, obs_dim=4, action_dims=2, hidden_sizes=(32, 32))

    obs = jax.random.normal(jax.random.PRNGKey(1), (64, 4))
    dist, values = policy_value_apply(params.graphdef, params.state, obs)
    actions = dist.sample(jax.random.PRNGKey(2))
    old_log_probs = dist.log_prob(actions)

    batch = FlattenBatch(
        obs=obs,
        actions=actions,
        old_log_probs=old_log_probs,
        old_values=values,
        advantages=jnp.ones((64,), dtype=jnp.float32),
        returns=values + 0.25,
    )

    loss, metrics = ppo_loss(
        params.graphdef,
        params.state,
        batch,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )

    assert jnp.isfinite(loss)
    for value in metrics.values():
        assert jnp.isfinite(value)


def test_clip_fraction_in_range():
    key = jax.random.PRNGKey(3)
    params = init_policy_value_params(key, obs_dim=4, action_dims=2, hidden_sizes=(16, 16))

    obs = jax.random.normal(jax.random.PRNGKey(4), (32, 4))
    dist, values = policy_value_apply(params.graphdef, params.state, obs)
    actions = dist.sample(jax.random.PRNGKey(5))
    old_log_probs = jnp.zeros((32,), dtype=jnp.float32)

    batch = FlattenBatch(
        obs=obs,
        actions=actions,
        old_log_probs=old_log_probs,
        old_values=values,
        advantages=jnp.ones((32,), dtype=jnp.float32),
        returns=values,
    )

    _, metrics = ppo_loss(
        params.graphdef,
        params.state,
        batch,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )

    clip_fraction = metrics["clip_fraction"]
    assert clip_fraction >= 0.0
    assert clip_fraction <= 1.0


def test_ppo_loss_extreme_log_ratio_stays_finite():
    key = jax.random.PRNGKey(11)
    params = init_policy_value_params(key, obs_dim=4, action_dims=2, hidden_sizes=(16, 16))

    obs = jax.random.normal(jax.random.PRNGKey(12), (32, 4))
    dist, values = policy_value_apply(params.graphdef, params.state, obs)
    actions = dist.sample(jax.random.PRNGKey(13))
    old_log_probs = jnp.full((32,), -1e6, dtype=jnp.float32)

    batch = FlattenBatch(
        obs=obs,
        actions=actions,
        old_log_probs=old_log_probs,
        old_values=values,
        advantages=jnp.ones((32,), dtype=jnp.float32),
        returns=values,
    )

    loss, metrics = ppo_loss(
        params.graphdef,
        params.state,
        batch,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )

    assert jnp.isfinite(loss)
    for value in metrics.values():
        assert jnp.isfinite(value)