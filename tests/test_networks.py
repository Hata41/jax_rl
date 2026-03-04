import jax
import jax.numpy as jnp

from jax_rl.networks import init_policy_value_params, policy_value_apply


def _path_token(entry) -> str:
    if hasattr(entry, "name"):
        return str(entry.name)
    if hasattr(entry, "key"):
        return str(entry.key)
    if hasattr(entry, "idx"):
        return str(entry.idx)
    return str(entry)


def _find_first_kernel(state, module_prefix: str):
    kernels = []

    def _collector(path, leaf):
        tokens = [_path_token(entry) for entry in path]
        if any(token.startswith(module_prefix) for token in tokens) and any(
            token == "kernel" for token in tokens
        ):
            kernels.append(leaf)
        return leaf

    jax.tree_util.tree_map_with_path(_collector, state)
    assert kernels, f"No kernel found for module prefix '{module_prefix}'"
    return kernels[0]


def test_multidiscrete_action_space_output_shapes():
    key = jax.random.PRNGKey(0)
    params = init_policy_value_params(
        key,
        obs_dim=4,
        action_dims=(2, 3),
        hidden_sizes=(32, 32),
    )

    obs = jax.random.normal(jax.random.PRNGKey(1), (8, 4))
    dist, _ = policy_value_apply(params.graphdef, params.state, obs)

    samples = dist.sample(jax.random.PRNGKey(2))
    assert samples.shape == (8, 2)

    log_probs = dist.log_prob(samples)
    assert log_probs.shape == (8,)


def test_modular_forward_pass_value_shape():
    key = jax.random.PRNGKey(3)
    params = init_policy_value_params(
        key,
        obs_dim=4,
        action_dims=2,
        hidden_sizes=(64, 64),
    )

    obs = jnp.zeros((16, 4), dtype=jnp.float32)
    _, values = policy_value_apply(params.graphdef, params.state, obs)

    assert values.shape == (16,)


def test_orthogonal_head_scales_actor_vs_critic_variance():
    key = jax.random.PRNGKey(7)
    params = init_policy_value_params(
        key,
        obs_dim=8,
        action_dims=4,
        hidden_sizes=(128, 128),
    )

    actor_w = _find_first_kernel(params.state, module_prefix="actor_head")
    critic_w = _find_first_kernel(params.state, module_prefix="critic_head")

    actor_var = jnp.var(actor_w)
    critic_var = jnp.var(critic_w)

    assert critic_var > actor_var * 100.0
