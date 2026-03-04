import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from jax_rl.networks import (
    BinPackPolicyValueModel,
    TransformerBlock,
    _flatten_binpack_logits,
    init_policy_value_params,
    policy_value_apply,
)


class _CustomTargetModule(nnx.Module):
    last_hidden_dim: int | None = None

    def __init__(self, obs_dim: int, action_dims: int, hidden_dim: int, rngs: nnx.Rngs):
        type(self).last_hidden_dim = int(hidden_dim)
        self.proj = nnx.Linear(obs_dim, hidden_dim, rngs=rngs)
        self.actor = nnx.Linear(hidden_dim, action_dims, rngs=rngs)
        self.critic = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, obs: jax.Array):
        h = jnp.tanh(self.proj(obs))
        return self.actor(h), self.critic(h).squeeze(-1)


_CUSTOM_TARGET_PATH = f"{__name__}._CustomTargetModule"


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
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [32, 32]},
        obs_dim=4,
        action_dims=(2, 3),
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
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [64, 64]},
        obs_dim=4,
        action_dims=2,
    )

    obs = jnp.zeros((16, 4), dtype=jnp.float32)
    _, values = policy_value_apply(params.graphdef, params.state, obs)

    assert values.shape == (16,)


def test_orthogonal_head_scales_actor_vs_critic_variance():
    key = jax.random.PRNGKey(7)
    params = init_policy_value_params(
        key,
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [128, 128]},
        obs_dim=8,
        action_dims=4,
    )

    actor_w = _find_first_kernel(params.state, module_prefix="actor_head")
    critic_w = _find_first_kernel(params.state, module_prefix="critic_head")

    actor_var = jnp.var(actor_w)
    critic_var = jnp.var(critic_w)

    assert critic_var > actor_var * 100.0


def _dummy_binpack_obs(batch_size: int = 2, max_ems: int = 40, max_items: int = 20, rotations: int = 6):
    max_actions = max_ems * max_items * rotations
    return {
        "ems_pos": jnp.ones((batch_size, max_ems, 6), dtype=jnp.float32),
        "item_dims": jnp.ones((batch_size, max_items, 3), dtype=jnp.float32),
        "ems_mask": jnp.ones((batch_size, max_ems), dtype=jnp.bool_),
        "item_mask": jnp.ones((batch_size, max_items), dtype=jnp.bool_),
        "action_mask": jnp.ones((batch_size, max_actions), dtype=jnp.bool_),
    }


def test_binpack_transformer_forward_shapes():
    model = BinPackPolicyValueModel(
        hidden_dim=32,
        action_dim=40 * 20 * 6,
        num_heads=2,
        num_layers=1,
        ems_feature_dim=6,
        item_feature_dim=3,
        rngs=nnx.Rngs(jax.random.PRNGKey(123)),
    )
    obs = _dummy_binpack_obs(batch_size=2)
    logits, values = model(obs)

    assert logits.shape == (2, 40 * 20 * 6)
    assert values.shape == (2,)


def test_binpack_transformer_masking_and_pooling_edge_cases():
    model = BinPackPolicyValueModel(
        hidden_dim=32,
        action_dim=40 * 20 * 6,
        num_heads=2,
        num_layers=1,
        ems_feature_dim=6,
        item_feature_dim=3,
        rngs=nnx.Rngs(jax.random.PRNGKey(321)),
    )
    obs = _dummy_binpack_obs(batch_size=2)
    obs["action_mask"] = obs["action_mask"].at[:, 1::7].set(False)
    obs["ems_mask"] = jnp.zeros_like(obs["ems_mask"], dtype=jnp.bool_)

    logits, values = model(obs)

    masked_logits = logits[:, 1::7]
    assert jnp.all(masked_logits == jnp.asarray(-1e9, dtype=logits.dtype))
    assert jnp.all(jnp.isfinite(values))


def test_transformer_block_masked_tokens_do_not_leak_context():
    block = TransformerBlock(dim=8, num_heads=2, rngs=nnx.Rngs(jax.random.PRNGKey(2026)))

    q = jnp.asarray([[[0.5, -0.5, 0.3, 0.1, -0.2, 0.7, -0.1, 0.9]]], dtype=jnp.float32)
    k = jnp.asarray(
        [[[0.1, 0.2, -0.1, 0.3, 0.2, -0.4, 0.6, -0.5], [0.7, -0.2, 0.4, 0.9, -0.3, 0.1, -0.8, 0.2]]],
        dtype=jnp.float32,
    )
    v_clean = jnp.asarray(
        [[[0.2, -0.1, 0.4, -0.3, 0.1, 0.0, 0.5, -0.2], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
        dtype=jnp.float32,
    )
    v_padded_extreme = v_clean.at[:, 1, :].set(
        jnp.asarray([1200.0, -900.0, 800.0, -700.0, 600.0, -500.0, 400.0, -300.0], dtype=jnp.float32)
    )

    mask = jnp.asarray([[[[True, False]]]], dtype=jnp.bool_)
    out_clean = block(q, k, v_clean, mask=mask)
    out_padded = block(q, k, v_padded_extreme, mask=mask)

    assert jnp.allclose(out_clean, out_padded, rtol=0.0, atol=1e-5)


def test_binpack_actor_index_alignment_item_ems_rotation_order():
    batch_size = 1
    num_ems = 2
    num_items = 2
    num_rotations = 6

    score_grid = jnp.zeros((batch_size, num_ems, num_items), dtype=jnp.float32)
    score_grid = score_grid.at[0, 0, 1].set(10.0)
    action_mask = jnp.ones((batch_size, num_items * num_ems * num_rotations), dtype=jnp.bool_)

    logits = _flatten_binpack_logits(score_grid, action_mask)
    argmax_idx = int(jnp.argmax(logits[0]))

    expected_idx = (1 * num_ems * num_rotations) + (0 * num_rotations) + 0
    assert argmax_idx == expected_idx


def test_binpack_actor_rotational_consistency_before_masking():
    batch_size = 1
    num_ems = 3
    num_items = 2
    num_rotations = 6

    score_grid = jnp.asarray(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
        dtype=jnp.float32,
    )
    action_mask = jnp.ones((batch_size, num_items * num_ems * num_rotations), dtype=jnp.bool_)

    logits = _flatten_binpack_logits(score_grid, action_mask)
    logits_4d = logits.reshape((batch_size, num_items, num_ems, num_rotations))

    first_rot = logits_4d[..., :1]
    tiled_first = jnp.tile(first_rot, (1, 1, 1, num_rotations))
    assert jnp.allclose(logits_4d, tiled_first)


def test_network_target_instantiation_custom_module_state_structure():
    params = init_policy_value_params(
        jax.random.PRNGKey(99),
        network_config={
            "_target_": _CUSTOM_TARGET_PATH,
            "hidden_dim": 48,
        },
        obs_dim=4,
        action_dims=3,
    )

    proj_kernel = _find_first_kernel(params.state, module_prefix="proj")
    assert proj_kernel.shape == (4, 48)


def test_network_parameter_injection_hidden_dim_reaches_constructor():
    _CustomTargetModule.last_hidden_dim = None
    init_policy_value_params(
        jax.random.PRNGKey(100),
        network_config={
            "_target_": _CUSTOM_TARGET_PATH,
            "hidden_dim": 128,
        },
        obs_dim=6,
        action_dims=2,
    )

    assert _CustomTargetModule.last_hidden_dim == 128


def test_missing_network_target_raises_value_error():
    with pytest.raises(ValueError, match="_target_"):
        init_policy_value_params(
            jax.random.PRNGKey(101),
            network_config={"hidden_sizes": [16, 16]},
            obs_dim=4,
            action_dims=2,
        )
