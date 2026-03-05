import jax.numpy as jnp
from typing import cast

from ...networks import CategoricalPolicyDist, MultiDiscretePolicyDist, policy_value_apply
from ...utils.types import Array, FlattenBatch


DistributionLike = CategoricalPolicyDist | MultiDiscretePolicyDist


def compute_actor_loss(
    dist: DistributionLike,
    batch: FlattenBatch,
    clip_epsilon: float,
    entropy_coef: float,
):
    safe_old_log_probs = jnp.where(
        jnp.isfinite(batch.old_log_probs),
        batch.old_log_probs,
        jnp.zeros_like(batch.old_log_probs),
    )
    safe_advantages = jnp.where(
        jnp.isfinite(batch.advantages),
        batch.advantages,
        jnp.zeros_like(batch.advantages),
    )
    new_log_probs = jnp.asarray(dist.log_prob(batch.actions), dtype=jnp.float32)
    safe_new_log_probs = jnp.where(
        jnp.isfinite(new_log_probs),
        new_log_probs,
        safe_old_log_probs,
    )
    log_ratio = jnp.clip(safe_new_log_probs - safe_old_log_probs, -20.0, 20.0)
    ratio = jnp.exp(log_ratio)
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    policy_loss_1 = ratio * safe_advantages
    policy_loss_2 = clipped_ratio * safe_advantages
    policy_loss = -jnp.mean(jnp.minimum(policy_loss_1, policy_loss_2))
    entropy_terms = jnp.asarray(dist.entropy(), dtype=jnp.float32)
    safe_entropy_terms = jnp.where(
        jnp.isfinite(entropy_terms),
        entropy_terms,
        jnp.zeros_like(entropy_terms),
    )
    entropy = jnp.mean(safe_entropy_terms)
    actor_loss = policy_loss - entropy_coef * entropy
    metrics = {
        "loss_policy": policy_loss,
        "entropy": entropy,
        "clip_fraction": jnp.mean(jnp.abs(ratio - 1.0) > clip_epsilon),
    }
    return actor_loss, metrics


def compute_critic_loss(
    new_values: Array,
    batch: FlattenBatch,
    clip_epsilon: float,
    value_coef: float,
):
    new_values = jnp.asarray(new_values, dtype=jnp.float32)
    safe_new_values = jnp.where(jnp.isfinite(new_values), new_values, batch.old_values)
    safe_old_values = jnp.where(
        jnp.isfinite(batch.old_values),
        batch.old_values,
        jnp.zeros_like(batch.old_values),
    )
    safe_returns = jnp.where(
        jnp.isfinite(batch.returns),
        batch.returns,
        safe_old_values,
    )
    value_pred_clipped = safe_old_values + jnp.clip(
        safe_new_values - safe_old_values,
        -clip_epsilon,
        clip_epsilon,
    )
    value_loss_unclipped = (safe_new_values - safe_returns) ** 2
    value_loss_clipped = (value_pred_clipped - safe_returns) ** 2
    value_loss = 0.5 * jnp.mean(jnp.maximum(value_loss_unclipped, value_loss_clipped))
    critic_loss = value_coef * value_loss
    metrics = {
        "loss_value": value_loss,
    }
    return critic_loss, metrics


def ppo_loss(
    graphdef,
    state,
    batch: FlattenBatch,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
):
    """Compute the clipped PPO objective with value clipping and entropy regularization.

    The policy objective uses

    ``L^CLIP = E[min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]``

    where ``r_t(θ)=π_θ(a_t|s_t)/π_{θ_old}(a_t|s_t)`` and ``ε=clip_epsilon``.

    The critic term uses clipped value regression:

    ``V^clip_t = V_old_t + clip(V_t - V_old_t, -ε, ε)``

    and ``L^VF = 0.5 * E[max((V_t - R_t)^2, (V^clip_t - R_t)^2)]``.

    The final minimized loss is

    ``L = -L^CLIP + value_coef * L^VF - entropy_coef * H[π_θ]``.
    """
    dist, new_values = policy_value_apply(graphdef, state, batch.obs)
    dist = cast(DistributionLike, dist)
    new_values = jnp.asarray(new_values, dtype=jnp.float32)
    actor_loss, actor_metrics = compute_actor_loss(
        dist=dist,
        batch=batch,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
    )
    critic_loss, critic_metrics = compute_critic_loss(
        new_values=new_values,
        batch=batch,
        clip_epsilon=clip_epsilon,
        value_coef=value_coef,
    )

    total_loss = actor_loss + critic_loss
    metrics = {
        "loss_total": total_loss,
        **actor_metrics,
        **critic_metrics,
    }
    return total_loss, metrics