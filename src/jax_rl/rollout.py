import jax
import jax.numpy as jnp

from .networks import policy_value_apply
from .types import Array, RolloutBatch, Transition


def _extract_done_and_truncated(timestep):
    done = jnp.asarray(timestep.discount == 0.0, dtype=jnp.bool_)
    truncated = jnp.asarray(
        jnp.logical_and(timestep.last(), timestep.discount != 0.0),
        dtype=jnp.bool_,
    )
    return done, truncated


def _extract_bootstrap_obs(timestep):
    extras = timestep.extras
    if hasattr(extras, "get"):
        return extras.get("next_obs", timestep.observation)
    return timestep.observation


def _extract_episode_info(timestep, dones: Array):
    zero_float = jnp.zeros_like(dones, dtype=jnp.float32)
    zero_bool = jnp.zeros_like(dones, dtype=jnp.bool_)
    metrics = timestep.extras.get("episode_metrics", {})
    episode_returns = jnp.asarray(metrics.get("episode_return", zero_float), dtype=jnp.float32)
    episode_lengths = jnp.asarray(metrics.get("episode_length", zero_float), dtype=jnp.float32)
    completed = jnp.asarray(metrics.get("is_terminal_step", zero_bool), dtype=jnp.bool_)
    return episode_returns, episode_lengths, completed


def collect_rollout(
    params,
    env,
    env_params,
    env_state,
    obs: Array,
    key: Array,
    num_steps: int,
):
    def step_fn(carry, _):
        curr_obs, curr_env_state, curr_key = carry
        curr_key, action_key = jax.random.split(curr_key)

        dist, values = policy_value_apply(params.graphdef, params.state, curr_obs)
        actions = dist.sample(action_key)
        log_probs = dist.log_prob(actions)

        next_env_state, timestep = jax.vmap(env.step, in_axes=(0, 0, None))(
            curr_env_state,
            actions,
            env_params,
        )
        next_obs = timestep.observation
        rewards = timestep.reward
        dones, truncated = _extract_done_and_truncated(timestep)
        episode_returns, episode_lengths, completed = _extract_episode_info(timestep, dones)

        bootstrap_obs = _extract_bootstrap_obs(timestep)
        _, bootstrap_values = policy_value_apply(params.graphdef, params.state, bootstrap_obs)

        transition = Transition(
            obs=curr_obs,
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            dones=dones,
            truncated=truncated,
            values=values,
            bootstrap_values=bootstrap_values,
        )
        info_batch = {
            "episode_return": episode_returns,
            "episode_length": episode_lengths,
            "is_terminal_step": completed,
        }
        return (next_obs, next_env_state, curr_key), (transition, info_batch)

    (next_obs, next_env_state, next_key), (transitions, infos) = jax.lax.scan(
        step_fn,
        (obs, env_state, key),
        xs=None,
        length=num_steps,
    )
    last_values = transitions.bootstrap_values[-1]
    batch = RolloutBatch(
        obs=transitions.obs,
        actions=transitions.actions,
        log_probs=transitions.log_probs,
        rewards=transitions.rewards,
        dones=transitions.dones,
        truncated=transitions.truncated,
        values=transitions.values,
        bootstrap_values=transitions.bootstrap_values,
    )
    return batch, last_values, next_obs, next_env_state, next_key, infos