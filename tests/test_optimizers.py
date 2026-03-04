import jax
import jax.numpy as jnp
import optax

from jax_rl.config import PPOConfig
from jax_rl.networks import init_policy_value_params
from jax_rl.update import make_actor_optimizer, make_critic_optimizer


def _allclose_tree(tree_a, tree_b):
    leaves_a = jax.tree_util.tree_leaves(tree_a)
    leaves_b = jax.tree_util.tree_leaves(tree_b)
    return all(jnp.allclose(a, b) for a, b in zip(leaves_a, leaves_b))


def test_actor_and_critic_optimizers_are_separate():
    config = PPOConfig(actor_lr=1.0, critic_lr=0.0)
    params = init_policy_value_params(
        jax.random.PRNGKey(0),
        obs_dim=4,
        action_dims=2,
        hidden_sizes=(16, 16),
    )

    actor_optimizer = make_actor_optimizer(config)
    critic_optimizer = make_critic_optimizer(config)
    actor_opt_state = actor_optimizer.init(params.state)
    critic_opt_state = critic_optimizer.init(params.state)

    actor_grads = jax.tree_util.tree_map(jnp.ones_like, params.state)
    critic_grads = jax.tree_util.tree_map(jnp.zeros_like, params.state)

    actor_updates, actor_opt_state = actor_optimizer.update(
        actor_grads,
        actor_opt_state,
        params.state,
    )
    critic_updates, critic_opt_state = critic_optimizer.update(
        critic_grads,
        critic_opt_state,
        params.state,
    )

    del actor_opt_state, critic_opt_state

    next_state = optax.apply_updates(
        optax.apply_updates(params.state, actor_updates),
        critic_updates,
    )

    assert not _allclose_tree(next_state, params.state)


def test_actor_learning_rate_schedule_reaches_zero_at_total_opt_steps():
    config = PPOConfig(
        total_timesteps=8_192,
        num_envs=8,
        num_steps=16,
        update_epochs=4,
        minibatch_size=32,
        actor_lr=3e-4,
    )
    expected_total_steps = (
        config.num_updates
        * config.update_epochs
        * (config.rollout_batch_size // config.minibatch_size)
    )

    optimizer = make_actor_optimizer(config)
    params = init_policy_value_params(
        jax.random.PRNGKey(123),
        obs_dim=4,
        action_dims=2,
        hidden_sizes=(16, 16),
    ).state
    opt_state = optimizer.init(params)

    grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    for _ in range(expected_total_steps):
        _, opt_state = optimizer.update(grads, opt_state, params)

    current_lr = opt_state[1].hyperparams["learning_rate"]
    assert float(current_lr) == 0.0
