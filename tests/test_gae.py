import jax.numpy as jnp

from jax_rl.advantages import compute_gae


def test_gae_matches_manual_computation():
    rewards = jnp.array([[1.0], [1.0], [1.0]], dtype=jnp.float32)
    dones = jnp.array([[0.0], [0.0], [1.0]], dtype=jnp.float32)
    truncated = jnp.array([[0.0], [0.0], [0.0]], dtype=jnp.float32)
    values = jnp.array([[0.5], [0.5], [0.5]], dtype=jnp.float32)
    last_values = jnp.array([0.0], dtype=jnp.float32)

    gamma = 0.99
    lam = 0.95
    advantages, returns = compute_gae(rewards, dones, truncated, values, last_values, gamma, lam)

    manual = []
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(3)):
        done = float(dones[t, 0])
        value = float(values[t, 0])
        reward = float(rewards[t, 0])
        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * lam * (1.0 - done) * gae
        manual.append(gae)
        next_value = value
    manual_advantages = jnp.array(list(reversed(manual)), dtype=jnp.float32).reshape(3, 1)

    assert jnp.allclose(advantages, manual_advantages, atol=1e-6)
    assert jnp.allclose(returns, advantages + values, atol=1e-6)


def test_gae_bootstraps_on_truncation_but_not_termination():
    gamma = 0.99
    lam = 0.95

    rewards = jnp.array([[0.0], [0.0], [1.0]], dtype=jnp.float32)
    dones = jnp.array([[0.0], [0.0], [1.0]], dtype=jnp.float32)
    values = jnp.array([[0.1], [0.2], [0.3]], dtype=jnp.float32)
    last_values = jnp.array([0.7], dtype=jnp.float32)

    truncated_true = jnp.array([[0.0], [0.0], [1.0]], dtype=jnp.float32)
    advantages_trunc, _ = compute_gae(
        rewards,
        dones,
        truncated_true,
        values,
        last_values,
        gamma,
        lam,
    )
    expected_trunc_last = rewards[-1] + gamma * last_values - values[-1]
    assert jnp.allclose(advantages_trunc[-1], expected_trunc_last, atol=1e-6)

    truncated_false = jnp.array([[0.0], [0.0], [0.0]], dtype=jnp.float32)
    advantages_term, _ = compute_gae(
        rewards,
        dones,
        truncated_false,
        values,
        last_values,
        gamma,
        lam,
    )
    expected_term_last = rewards[-1] - values[-1]
    assert jnp.allclose(advantages_term[-1], expected_term_last, atol=1e-6)