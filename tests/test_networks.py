import jax
import jax.numpy as jnp
from flax import nnx
import pytest

from jax_rl.networks import (
    BinPackPolicyValueModel,
    CategoricalPolicyDist,
    ModularPolicyValueModel,
    PolicyValueModel,
    RustpalletInputAdapterV2,
    _flatten_binpack_logits,
    flatten_observation_features,
    init_policy_value_params,
    policy_value_apply,
)
from jax_rl.utils.exceptions import NetworkTargetResolutionError


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


def test_policy_value_shapes():
    key = jax.random.PRNGKey(0)
    params = init_policy_value_params(
        key,
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [64, 64]},
        obs_dim=4,
        action_dims=2,
    )

    obs = jnp.zeros((16, 4), dtype=jnp.float32)
    dist, values = policy_value_apply(params.graphdef, params.state, obs)

    assert dist.sample(jax.random.PRNGKey(1)).shape == (16,)
    assert values.shape == (16,)


def test_invalid_network_target_raises_error():
    with pytest.raises(NetworkTargetResolutionError):
        init_policy_value_params(
            key=jax.random.PRNGKey(11),
            network_config={"_target_": "invalid.path.Model"},
            obs_dim=4,
            action_dims=2,
        )


def test_shared_vs_separate_torso_features():
    obs = jnp.ones((8, 4), dtype=jnp.float32)

    shared_model = PolicyValueModel(
        obs_dim=4,
        action_dims=2,
        hidden_sizes=(32, 32),
        rngs=nnx.Rngs(jax.random.PRNGKey(101)),
        shared_torso=True,
    )
    separate_model = PolicyValueModel(
        obs_dim=4,
        action_dims=2,
        hidden_sizes=(32, 32),
        rngs=nnx.Rngs(jax.random.PRNGKey(202)),
        shared_torso=False,
    )

    shared_obs_features, _ = flatten_observation_features(obs)
    shared_actor_features = shared_model.shared_torso(shared_obs_features)
    shared_critic_features = shared_model.shared_torso(shared_obs_features)
    assert jnp.allclose(shared_actor_features, shared_critic_features)

    separate_obs_features, _ = flatten_observation_features(obs)
    separate_actor_features = separate_model.actor_torso(separate_obs_features)
    separate_critic_features = separate_model.critic_torso(separate_obs_features)
    assert not jnp.allclose(separate_actor_features, separate_critic_features)

    separate_params = init_policy_value_params(
        key=jax.random.PRNGKey(303),
        network_config={
            "_target_": "jax_rl.networks.PolicyValueModel",
            "hidden_sizes": [32, 32],
            "shared_torso": False,
        },
        obs_dim=4,
        action_dims=2,
    )
    shared_params = init_policy_value_params(
        key=jax.random.PRNGKey(404),
        network_config={
            "_target_": "jax_rl.networks.PolicyValueModel",
            "hidden_sizes": [32, 32],
            "shared_torso": True,
        },
        obs_dim=4,
        action_dims=2,
    )

    separate_paths = [
        "/".join(_path_token(entry) for entry in path)
        for path, _ in jax.tree_util.tree_leaves_with_path(separate_params.state)
    ]
    shared_paths = [
        "/".join(_path_token(entry) for entry in path)
        for path, _ in jax.tree_util.tree_leaves_with_path(shared_params.state)
    ]
    assert any("actor_torso" in path for path in separate_paths)
    assert any("critic_torso" in path for path in separate_paths)
    assert any("shared_torso" in path for path in shared_paths)


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


def _dummy_v2_obs(batch_size: int = 2, max_ems: int = 10, max_items: int = 5):
    rotations = 6
    return {
        "uld_dims": jnp.full((batch_size, 3), 10.0, dtype=jnp.float32),
        "max_weight": jnp.full((batch_size,), 100.0, dtype=jnp.float32),
        "item_dims": jnp.full((batch_size, max_items, 3), 2.0, dtype=jnp.float32),
        "item_pos": jnp.full((batch_size, max_items, 3), 1.0, dtype=jnp.float32),
        "item_weights": jnp.full((batch_size, max_items), 4.0, dtype=jnp.float32),
        "ems_dims": jnp.full((batch_size, max_ems, 3), 8.0, dtype=jnp.float32),
        "ems_pos": jnp.full((batch_size, max_ems, 3), 1.5, dtype=jnp.float32),
        "group_counts": jnp.full((batch_size, max_items), 2.0, dtype=jnp.float32),
        "item_mask": jnp.ones((batch_size, max_items), dtype=jnp.bool_),
        "ems_mask": jnp.ones((batch_size, max_ems), dtype=jnp.bool_),
        "action_mask": jnp.ones((batch_size, max_items * max_ems * rotations), dtype=jnp.bool_),
    }


def test_adapter_v2_feature_generation():
    d_model = 32
    adapter = RustpalletInputAdapterV2(d_model=d_model, rngs=nnx.Rngs(jax.random.PRNGKey(2001)))
    obs = _dummy_v2_obs(batch_size=2, max_ems=10, max_items=5)

    ems_embeddings, item_embeddings, ems_mask, expanded_item_mask = adapter(obs)

    assert ems_embeddings.shape == (2, 10, d_model)
    assert item_embeddings.shape == (2, 5 * 6, d_model)
    assert ems_mask.shape == (2, 10)
    assert expanded_item_mask.shape == (2, 5 * 6)


def test_adapter_v2_fallback_logic():
    d_model = 16
    adapter = RustpalletInputAdapterV2(d_model=d_model, rngs=nnx.Rngs(jax.random.PRNGKey(2002)))
    obs = _dummy_v2_obs(batch_size=2, max_ems=10, max_items=5)
    obs.pop("group_counts")

    ems_embeddings, item_embeddings, ems_mask, expanded_item_mask = adapter(obs)

    assert ems_embeddings.shape == (2, 10, d_model)
    assert item_embeddings.shape == (2, 5 * 6, d_model)
    assert ems_mask.shape == (2, 10)
    assert expanded_item_mask.shape == (2, 5 * 6)


def test_sinusoidal_encoding_variance():
    adapter = RustpalletInputAdapterV2(d_model=32, rngs=nnx.Rngs(jax.random.PRNGKey(2003)))
    enc_1 = adapter._get_sinusoidal_encoding(jnp.array([[1.0]], dtype=jnp.float32))
    enc_2 = adapter._get_sinusoidal_encoding(jnp.array([[2.0]], dtype=jnp.float32))

    assert not jnp.allclose(enc_1, enc_2)


def test_modular_model_forward_pass():
    class DummyInputAdapter(nnx.Module):
        def __call__(self, obs: dict):
            return (
                jnp.asarray(obs["ems_embeddings"], dtype=jnp.float32),
                jnp.asarray(obs["item_embeddings"], dtype=jnp.float32),
                jnp.asarray(obs["ems_mask"], dtype=jnp.bool_),
                jnp.asarray(obs["item_mask"], dtype=jnp.bool_),
            )

    class DummyTorso(nnx.Module):
        def __call__(self, ems_embeddings, item_embeddings, ems_mask, item_mask):
            del ems_mask, item_mask
            return ems_embeddings, item_embeddings

    class DummyActorHead(nnx.Module):
        def __call__(self, ems_embeddings, item_embeddings, action_mask):
            del ems_embeddings, item_embeddings
            return jnp.zeros(action_mask.shape, dtype=jnp.float32)

    class DummyCriticHead(nnx.Module):
        def __call__(self, ems_embeddings, item_embeddings, ems_mask, item_mask):
            del item_embeddings, ems_mask, item_mask
            return jnp.zeros((ems_embeddings.shape[0],), dtype=jnp.float32)

    model = ModularPolicyValueModel(
        input_adapter=DummyInputAdapter(),
        shared_torso=DummyTorso(),
        actor_head=DummyActorHead(),
        critic_head=DummyCriticHead(),
    )
    graphdef, state = nnx.split(model)

    obs = {
        "ems_embeddings": jnp.ones((2, 10, 8), dtype=jnp.float32),
        "item_embeddings": jnp.ones((2, 30, 8), dtype=jnp.float32),
        "ems_mask": jnp.ones((2, 10), dtype=jnp.bool_),
        "item_mask": jnp.ones((2, 30), dtype=jnp.bool_),
        "action_mask": jnp.ones((2, 300), dtype=jnp.bool_),
    }
    dist, values = policy_value_apply(graphdef, state, obs)

    assert isinstance(dist, CategoricalPolicyDist)
    assert values.shape == (2,)


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
