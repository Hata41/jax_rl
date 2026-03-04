import jax
import jax.numpy as jnp

from .config import PPOConfig
from .env import make_stoa_env
from .networks import policy_value_apply
from .types import PolicyValueParams


def evaluate(
    params: PolicyValueParams,
    config: PPOConfig,
    num_episodes: int = 10,
    max_steps_per_episode: int = 1_000,
):
    env, env_params = make_stoa_env(config.env_name, num_envs_per_device=1)
    key = jax.random.PRNGKey(config.seed)

    episode_returns = []
    total_steps = 0

    for _ in range(num_episodes):
        key, reset_key = jax.random.split(key)
        env_state, timestep = env.reset(reset_key, env_params)
        obs = timestep.observation

        total_reward = 0.0
        step_count = 0

        while (not bool(jnp.asarray(timestep.last()).all())) and step_count < max_steps_per_episode:
            dist, _ = policy_value_apply(params.graphdef, params.state, obs)
            action = dist.mode()

            key, step_key = jax.random.split(key)
            del step_key
            env_state, timestep = env.step(env_state, action, env_params)
            obs = timestep.observation
            total_reward += float(jnp.sum(jnp.asarray(timestep.reward, dtype=jnp.float32)))
            step_count += 1

        episode_returns.append(total_reward)
        total_steps += step_count

    returns = jnp.asarray(episode_returns, dtype=jnp.float32)
    return {
        "return_mean": float(jnp.mean(returns)),
        "return_std": float(jnp.std(returns)),
        "return_min": float(jnp.min(returns)),
        "return_max": float(jnp.max(returns)),
        "episodes": int(num_episodes),
        "steps": int(total_steps),
    }