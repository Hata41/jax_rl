from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp

from ...envs.env import make_stoa_env
from ...networks import policy_value_apply
from ...systems.alphazero.steps import extract_root_embedding
from ...utils.types import PolicyValueParams
from .steps import SPO, make_recurrent_fn, make_root_fn
from .types import CategoricalDualParams, SPOParams


def _zero_metrics() -> dict[str, float | int]:
    return {
        "return_mean": 0.0,
        "return_std": 0.0,
        "return_min": 0.0,
        "return_max": 0.0,
        "episodes": 0,
        "steps": 0,
    }


def _release_generated_ids(env: Any, generated_state_ids: jax.Array, chunk_size: int) -> None:
    if not hasattr(env, "release_batch"):
        return

    flat_ids = jnp.asarray(generated_state_ids, dtype=jnp.int32).reshape(-1)
    valid_mask = flat_ids > 0
    safe_ids = jnp.where(valid_mask, flat_ids, -1)

    remainder = safe_ids.shape[0] % chunk_size
    pad_size = (chunk_size - remainder) % chunk_size
    padded_ids = jnp.pad(safe_ids, (0, pad_size), constant_values=-1)
    release_chunks = padded_ids.reshape((-1, chunk_size))

    dummy_state = jnp.zeros((chunk_size,), dtype=jnp.int32)
    for chunk_idx in range(int(release_chunks.shape[0])):
        _ = env.release_batch(dummy_state, release_chunks[chunk_idx])


def evaluate(
    *,
    params: PolicyValueParams,
    config,
    env_name: str,
    seed: int,
    num_episodes: int = 10,
    max_steps_per_episode: int = 1_000,
    greedy: bool = True,
    env_kwargs: dict[str, Any] | None = None,
    action_selection: str = "policy",
) -> dict[str, float | int]:
    if num_episodes <= 0:
        return _zero_metrics()

    selection = str(action_selection).lower()
    if selection not in {"policy", "search"}:
        raise ValueError(
            f"Unsupported SPO action_selection '{action_selection}'. Expected 'policy' or 'search'."
        )

    env, env_params = make_stoa_env(
        env_name,
        num_envs_per_device=int(num_episodes),
        env_kwargs=dict(env_kwargs or {}),
    )
    is_rustpool = str(env_name).lower().startswith(("rustpool:", "rlpallet:"))

    spo_params = SPOParams(
        actor_online=params,
        actor_target=params,
        critic_online=params,
        critic_target=params,
        dual_params=CategoricalDualParams(
            log_temperature=jnp.asarray(getattr(config.system, "dual_init_log_temperature", -2.0), dtype=jnp.float32),
            log_alpha=jnp.asarray(getattr(config.system, "dual_init_log_alpha", -2.0), dtype=jnp.float32),
        ),
    )

    root_fn = None
    search = None
    if selection == "search":
        recurrent_fn = make_recurrent_fn(
            env=env,
            env_params=env_params,
            gamma=float(getattr(config.system, "search_gamma", 0.99)),
            is_rustpool=is_rustpool,
        )
        root_fn = make_root_fn(config)
        search = SPO(config, recurrent_fn)

    key = jax.random.PRNGKey(int(seed))
    key, reset_key = jax.random.split(key)
    env_state, timestep = env.reset(reset_key, cast(Any, env_params))
    obs = timestep.observation

    active_mask = jnp.ones((int(num_episodes),), dtype=jnp.float32)
    returns = jnp.zeros((int(num_episodes),), dtype=jnp.float32)
    steps = jnp.zeros((int(num_episodes),), dtype=jnp.float32)

    try:
        for _ in range(int(max_steps_per_episode)):
            key, action_key, search_key = jax.random.split(key, 3)

            if selection == "search":
                assert root_fn is not None and search is not None
                root_embedding = extract_root_embedding(
                    env=env,
                    env_state=env_state,
                    obs=obs,
                    is_rustpool=is_rustpool,
                )
                root = root_fn(spo_params, obs, root_embedding, action_key)
                search_output = search.search(spo_params, search_key, root)
                action = jnp.asarray(search_output.action, dtype=jnp.int32)
                if is_rustpool:
                    _release_generated_ids(
                        env,
                        generated_state_ids=jnp.asarray(search_output.generated_state_ids, dtype=jnp.int32),
                        chunk_size=int(num_episodes),
                    )
            else:
                dist, _ = policy_value_apply(params.graphdef, params.state, cast(Any, obs))
                action = jax.lax.cond(
                    jnp.asarray(greedy),
                    lambda _: dist.mode(),
                    lambda sample_key: dist.sample(sample_key),
                    action_key,
                )

            env_state, timestep = env.step(env_state, action, cast(Any, env_params))
            reward = jnp.asarray(timestep.reward, dtype=jnp.float32)
            done = jnp.asarray(timestep.last(), dtype=jnp.float32)

            returns = returns + reward * active_mask
            steps = steps + active_mask
            active_mask = active_mask * (1.0 - done)
            obs = timestep.observation

            if bool(jnp.all(active_mask == 0.0)):
                break
    finally:
        if hasattr(env, "close"):
            env.close()

    return {
        "return_mean": float(jnp.mean(returns)),
        "return_std": float(jnp.std(returns)),
        "return_min": float(jnp.min(returns)),
        "return_max": float(jnp.max(returns)),
        "episodes": int(num_episodes),
        "steps": int(jnp.sum(steps)),
    }
