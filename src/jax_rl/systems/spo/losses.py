from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .types import CategoricalDualParams

_MPO_FLOAT_EPSILON = 1e-8
_MIN_LOG_TEMPERATURE = -18.0
_MIN_LOG_ALPHA = -18.0


def clip_categorical_mpo_params(params: CategoricalDualParams) -> CategoricalDualParams:
    return params._replace(
        log_temperature=jnp.maximum(_MIN_LOG_TEMPERATURE, params.log_temperature),
        log_alpha=jnp.maximum(_MIN_LOG_ALPHA, params.log_alpha),
    )


def get_temperature_from_params(params: CategoricalDualParams) -> jax.Array:
    return jax.nn.softplus(params.log_temperature) + _MPO_FLOAT_EPSILON


def compute_weights_and_temperature_loss(
    q_values: jax.Array,
    epsilon: float,
    temperature: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    tempered_q_values = jax.lax.stop_gradient(q_values) / temperature
    normalized_weights = jax.nn.softmax(tempered_q_values, axis=-1)
    log_num_particles = jnp.log(jnp.asarray(q_values.shape[-1], dtype=jnp.float32))
    log_normalizer = jax.nn.logsumexp(tempered_q_values, axis=-1) - log_num_particles
    loss_temperature = temperature * (epsilon + jnp.mean(log_normalizer))
    return normalized_weights, loss_temperature


def _sampled_log_prob(distribution: Any, sampled_actions: jax.Array) -> jax.Array:
    actions = jnp.asarray(sampled_actions, dtype=jnp.int32)
    if hasattr(distribution, "logits"):
        logits = jnp.asarray(distribution.logits, dtype=jnp.float32)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        gathered = jnp.take_along_axis(log_probs[:, None, :], actions[..., None], axis=-1)
        return gathered.squeeze(-1)

    if hasattr(distribution, "logits_per_dim"):
        logits_per_dim = tuple(jnp.asarray(x, dtype=jnp.float32) for x in distribution.logits_per_dim)
        per_dim = []
        for dim_idx, logits in enumerate(logits_per_dim):
            dim_actions = actions[..., dim_idx]
            dim_log_probs = jax.nn.log_softmax(logits, axis=-1)
            dim_gathered = jnp.take_along_axis(
                dim_log_probs[:, None, :],
                dim_actions[..., None],
                axis=-1,
            ).squeeze(-1)
            per_dim.append(dim_gathered)
        return jnp.sum(jnp.stack(per_dim, axis=-1), axis=-1)

    raise TypeError("Policy distribution must expose 'logits' or 'logits_per_dim'.")


def categorical_mpo_loss(
    dual_params: CategoricalDualParams,
    online_action_distribution: Any,
    target_action_distribution: Any,
    sampled_actions: jax.Array,
    q_values: jax.Array,
    epsilon: float,
    epsilon_policy: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    dual_params = clip_categorical_mpo_params(dual_params)

    temperature = get_temperature_from_params(dual_params).squeeze()
    alpha = jax.nn.softplus(dual_params.log_alpha).squeeze() + _MPO_FLOAT_EPSILON

    normalized_weights, loss_temperature = compute_weights_and_temperature_loss(
        q_values=q_values,
        epsilon=epsilon,
        temperature=temperature,
    )

    online_log_prob = _sampled_log_prob(online_action_distribution, sampled_actions)
    target_log_prob = _sampled_log_prob(target_action_distribution, sampled_actions)

    loss_policy = -jnp.mean(jnp.sum(jax.lax.stop_gradient(normalized_weights) * online_log_prob, axis=-1))

    kl = target_log_prob - online_log_prob
    mean_kl = jnp.mean(kl)
    loss_kl = jax.lax.stop_gradient(alpha) * mean_kl
    loss_alpha = alpha * (epsilon_policy - jax.lax.stop_gradient(mean_kl))

    loss_dual = loss_alpha + loss_temperature
    loss = loss_policy + loss_kl + loss_dual

    metrics = {
        "temperature": temperature,
        "alpha": alpha,
        "loss_temperature": loss_temperature,
        "loss_alpha": loss_alpha,
        "loss_policy": loss_policy,
        "loss_kl": loss_kl,
        "kl_mean": mean_kl,
        "entropy_online": jnp.mean(jnp.asarray(online_action_distribution.entropy(), dtype=jnp.float32)),
        "entropy_target": jnp.mean(jnp.asarray(target_action_distribution.entropy(), dtype=jnp.float32)),
        "q_min": jnp.min(q_values),
        "q_max": jnp.max(q_values),
    }

    finite_loss = jnp.isfinite(loss)
    finite_metrics = jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in metrics.values()]))
    metrics["loss_is_finite"] = jnp.logical_and(finite_loss, finite_metrics)

    return loss, metrics


def multidiscrete_mpo_loss(
    dual_params: CategoricalDualParams,
    online_action_distribution: Any,
    target_action_distribution: Any,
    sampled_actions: jax.Array,
    q_values: jax.Array,
    epsilon: float,
    epsilon_policy: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    return categorical_mpo_loss(
        dual_params=dual_params,
        online_action_distribution=online_action_distribution,
        target_action_distribution=target_action_distribution,
        sampled_actions=sampled_actions,
        q_values=q_values,
        epsilon=epsilon,
        epsilon_policy=epsilon_policy,
    )
