import jax.numpy as jnp

from .networks import policy_value_apply
from .types import FlattenBatch


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
    new_log_probs = dist.log_prob(batch.actions)

    log_ratio = jnp.clip(new_log_probs - batch.old_log_probs, -20.0, 20.0)
    ratio = jnp.exp(log_ratio)
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    policy_loss_1 = ratio * batch.advantages
    policy_loss_2 = clipped_ratio * batch.advantages
    policy_loss = -jnp.mean(jnp.minimum(policy_loss_1, policy_loss_2))

    value_pred_clipped = batch.old_values + jnp.clip(
        new_values - batch.old_values,
        -clip_epsilon,
        clip_epsilon,
    )
    value_loss_unclipped = (new_values - batch.returns) ** 2
    value_loss_clipped = (value_pred_clipped - batch.returns) ** 2
    value_loss = 0.5 * jnp.mean(jnp.maximum(value_loss_unclipped, value_loss_clipped))

    entropy = jnp.mean(dist.entropy())

    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    total_loss = jnp.nan_to_num(total_loss, nan=0.0, posinf=1e6, neginf=-1e6)
    metrics = {
        "loss_total": total_loss,
        "loss_policy": jnp.nan_to_num(policy_loss, nan=0.0, posinf=1e6, neginf=-1e6),
        "loss_value": jnp.nan_to_num(value_loss, nan=0.0, posinf=1e6, neginf=-1e6),
        "entropy": jnp.nan_to_num(entropy, nan=0.0, posinf=1e6, neginf=-1e6),
        "clip_fraction": jnp.nan_to_num(
            jnp.mean(jnp.abs(ratio - 1.0) > clip_epsilon),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ),
    }
    return total_loss, metrics