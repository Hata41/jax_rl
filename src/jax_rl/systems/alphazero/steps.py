from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import mctx

from ...networks import policy_value_apply
from ...utils.exceptions import EnvironmentInterfaceError, NumericalInstabilityError


def _distribution_logits(dist: Any) -> jax.Array:
    if hasattr(dist, "logits"):
        return jnp.asarray(dist.logits, dtype=jnp.float32)
    if hasattr(dist, "logits_per_dim"):
        logits_per_dim = tuple(jnp.asarray(x, dtype=jnp.float32) for x in dist.logits_per_dim)
        return jnp.concatenate(logits_per_dim, axis=-1)
    raise TypeError("Policy distribution must expose 'logits' or 'logits_per_dim'.")


def make_root_fn() -> Callable[[Any, Any, Any, jax.Array], Any]:
    def root_fn(params, observation, state_embedding, _):
        dist, value = policy_value_apply(params.graphdef, params.state, observation)
        logits = _distribution_logits(dist)
        return mctx.RootFnOutput(
            prior_logits=logits,
            value=jnp.asarray(value, dtype=jnp.float32),
            embedding=state_embedding,
        )

    return root_fn


def make_recurrent_fn(
    *,
    env: Any,
    env_params: Any,
    gamma: float,
    is_rustpool: bool,
) -> Callable[[Any, jax.Array, jax.Array, Any], tuple[Any, Any]]:
    gamma_value = jnp.asarray(gamma, dtype=jnp.float32)
    is_vectorized_env = hasattr(env, "_vmap_step") or type(env).__name__ == "VmapWrapper"

    if is_rustpool:
        if not hasattr(env, "simulate_batch"):
            raise EnvironmentInterfaceError(
                "Rustpool recurrent path requires 'simulate_batch(state, state_ids, actions)'."
            )

        def recurrent_fn_rustpool(params, _, action, state_embedding):
            state_ids = jnp.asarray(state_embedding, dtype=jnp.int32)
            dummy_state = jnp.zeros_like(state_ids, dtype=jnp.int32)
            _, next_timestep = env.simulate_batch(dummy_state, state_ids, action)
            next_state_embedding = jnp.asarray(next_timestep.extras["state_id"], dtype=jnp.int32)

            dist, value = policy_value_apply(
                params.graphdef,
                params.state,
                next_timestep.observation,
            )
            logits = _distribution_logits(dist)
            timestep_discount = jnp.asarray(next_timestep.discount, dtype=jnp.float32)
            recurrent_output = mctx.RecurrentFnOutput(
                reward=jnp.asarray(next_timestep.reward, dtype=jnp.float32),
                discount=timestep_discount * gamma_value,
                prior_logits=logits,
                value=timestep_discount * jnp.asarray(value, dtype=jnp.float32),
            )
            return recurrent_output, next_state_embedding

        return recurrent_fn_rustpool

    def recurrent_fn_jax(params, _, action, state_embedding):
        if is_vectorized_env:
            next_state_embedding, next_timestep = env.step(state_embedding, action, env_params)
        else:
            next_state_embedding, next_timestep = jax.vmap(
                lambda state, selected_action: env.step(state, selected_action, env_params)
            )(state_embedding, action)

        dist, value = policy_value_apply(
            params.graphdef,
            params.state,
            next_timestep.observation,
        )
        logits = _distribution_logits(dist)
        timestep_discount = jnp.asarray(next_timestep.discount, dtype=jnp.float32)
        recurrent_output = mctx.RecurrentFnOutput(
            reward=jnp.asarray(next_timestep.reward, dtype=jnp.float32),
            discount=timestep_discount * gamma_value,
            prior_logits=logits,
            value=timestep_discount * jnp.asarray(value, dtype=jnp.float32),
        )
        return recurrent_output, next_state_embedding

    return recurrent_fn_jax


def parse_search_method(name: str):
    normalized = str(name).lower()
    if normalized == "muzero":
        return mctx.muzero_policy
    if normalized == "gumbel":
        return mctx.gumbel_muzero_policy
    raise ValueError(f"Unsupported AlphaZero search_method '{name}'.")


def make_search_apply_fn(*, config, recurrent_fn):
    search_method = parse_search_method(config.system.search_method)
    search_method_kwargs = dict(getattr(config.system, "search_method_kwargs", {}) or {})

    def apply_fn(params, key, root, observation):
        invalid_actions = None
        if isinstance(observation, dict) and "action_mask" in observation:
            action_mask = jnp.asarray(observation["action_mask"], dtype=jnp.bool_)
            invalid_actions = jnp.logical_not(action_mask)

        kwargs = {
            "params": params,
            "rng_key": key,
            "root": root,
            "recurrent_fn": recurrent_fn,
            "num_simulations": int(config.system.num_simulations),
            "max_depth": int(config.system.max_depth),
            "dirichlet_alpha": float(config.system.dirichlet_alpha),
            "dirichlet_fraction": float(config.system.dirichlet_fraction),
        }
        kwargs.update(search_method_kwargs)
        if invalid_actions is not None:
            kwargs["invalid_actions"] = invalid_actions
        return search_method(**kwargs)

    return apply_fn


def extract_root_embedding(*, env: Any, env_state: Any, obs: Any, is_rustpool: bool) -> Any:
    if not is_rustpool:
        return env_state

    if not hasattr(env, "snapshot"):
        raise EnvironmentInterfaceError("Rustpool root extraction requires 'snapshot(state, env_ids)'.")

    if isinstance(obs, dict) and "action_mask" in obs:
        batch_size = int(jnp.asarray(obs["action_mask"]).shape[0])
    else:
        leaves = jax.tree_util.tree_leaves(obs)
        batch_size = int(jnp.asarray(leaves[0]).shape[0]) if leaves else 1

    env_ids = jnp.arange(batch_size, dtype=jnp.int32)
    dummy_state = jnp.zeros((batch_size,), dtype=jnp.int32)
    snapshot_output = env.snapshot(dummy_state, env_ids)

    if isinstance(snapshot_output, tuple) and len(snapshot_output) == 2:
        _, snapshot_payload = snapshot_output
        if hasattr(snapshot_payload, "shape"):
            return jnp.asarray(snapshot_payload, dtype=jnp.int32)
        extras = getattr(snapshot_payload, "extras", None)
        if isinstance(extras, dict) and extras.get("state_id") is not None:
            return jnp.asarray(extras["state_id"], dtype=jnp.int32)
    return jnp.asarray(snapshot_output, dtype=jnp.int32)


def release_rustpool_embeddings(*, env: Any, state: Any, search_tree: Any) -> jax.Array:
    if not hasattr(env, "release_batch"):
        raise EnvironmentInterfaceError("Rustpool cleanup requires 'release_batch(state, state_ids)'.")

    # 1. Flatten the search tree node embeddings and visitation mask
    flat_ids = jnp.asarray(search_tree.embeddings, dtype=jnp.int32).reshape(-1)
    valid_mask = jnp.asarray(search_tree.node_visits).reshape(-1) > 0
    
    # 2. Mask out unvisited nodes with -1. 
    # Rust's HashMap::remove will safely ignore -1 (usize::MAX).
    safe_ids = jnp.where(valid_mask, flat_ids, -1)
    
    # 3. Call the wrapper natively!
    # The Stoa wrapper's internal io_callback will now correctly trace 
    # and resolve jax.lax.axis_index("device") just like it does in PPO.
    return env.release_batch(state, safe_ids)

def assert_finite_search_output(search_output: Any) -> None:
    is_finite = search_output_is_finite(search_output)
    if not bool(is_finite):
        raise NumericalInstabilityError("AlphaZero search produced non-finite values.")


def search_output_is_finite(search_output: Any) -> jax.Array:
    action_weights = jnp.asarray(search_output.action_weights)
    search_values = jnp.asarray(search_output.search_tree.node_values)
    weights_finite = jnp.all(jnp.isfinite(action_weights))
    values_finite = jnp.all(jnp.isfinite(search_values))
    return jnp.logical_and(weights_finite, values_finite)
