import jax
import jax.numpy as jnp

from ...utils.types import Array


def compute_gae(
    rewards: Array,
    dones: Array,
    truncated: Array,
    values: Array,
    last_values: Array,
    gamma: float,
    gae_lambda: float,
    bootstrap_values: Array | None = None,
):
    """Compute Generalized Advantage Estimation (GAE) and returns.

    For each timestep ``t``, this computes temporal-difference residuals

    ``δ_t = r_t + γ * V_{t+1} * (1 - terminated_t) - V_t``

    and discounted advantage recursion

    ``A_t = δ_t + γ * λ * (1 - terminated_t) * A_{t+1}``.

    ``gamma`` controls discounting horizon, while ``gae_lambda`` controls
    bias-variance tradeoff between one-step TD (low λ) and Monte Carlo-like
    estimates (high λ).
    """
    dtype = values.dtype
    rewards = rewards.astype(dtype)
    values = values.astype(dtype)
    last_values = last_values.astype(dtype)
    done_mask = dones.astype(jnp.bool_)
    truncated_mask = truncated.astype(jnp.bool_)
    gamma = jnp.asarray(gamma, dtype=dtype)
    gae_lambda = jnp.asarray(gae_lambda, dtype=dtype)

    if bootstrap_values is not None:
        bootstrap_values = bootstrap_values.astype(dtype)

        def gae_scan(carry, transition):
            gae = carry
            reward, done, trunc, value, bootstrap_value = transition
            terminated = jnp.logical_and(done, jnp.logical_not(trunc)).astype(dtype)
            delta = reward + gamma * bootstrap_value * (1.0 - terminated) - value
            gae = delta + gamma * gae_lambda * (1.0 - terminated) * gae
            return gae, gae

        _, advantages_rev = jax.lax.scan(
            gae_scan,
            jnp.zeros_like(last_values),
            (
                rewards[::-1],
                done_mask[::-1],
                truncated_mask[::-1],
                values[::-1],
                bootstrap_values[::-1],
            ),
        )
    else:
        def gae_scan(carry, transition):
            gae, next_value = carry
            reward, done, trunc, value = transition
            terminated = jnp.logical_and(done, jnp.logical_not(trunc)).astype(dtype)
            delta = reward + gamma * next_value * (1.0 - terminated) - value
            gae = delta + gamma * gae_lambda * (1.0 - terminated) * gae
            return (gae, value), gae

        (_, _), advantages_rev = jax.lax.scan(
            gae_scan,
            (jnp.zeros_like(last_values), last_values),
            (rewards[::-1], done_mask[::-1], truncated_mask[::-1], values[::-1]),
        )
    advantages = advantages_rev[::-1]
    advantages = jnp.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
    returns = jnp.nan_to_num(advantages + values, nan=0.0, posinf=0.0, neginf=0.0)
    return advantages, returns