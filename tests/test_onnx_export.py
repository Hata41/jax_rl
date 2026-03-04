import numpy as np
import jax

from jax_rl.export import export_model_to_onnx
from jax_rl.networks import init_policy_value_params, policy_value_apply


def _run_onnx_inference(model_path, obs):
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    input_names = [inp.name for inp in session.get_inputs()]
    if isinstance(obs, dict):
        feed = {
            name: np.asarray(obs[key])
            for name, key in zip(input_names, sorted(obs.keys()))
        }
    else:
        feed = {input_names[0]: obs}
    return session.run(None, feed)


def test_discrete_action_space_export_and_equivalence(tmp_path):
    key = jax.random.PRNGKey(0)
    params = init_policy_value_params(
        key,
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [32, 32]},
        obs_dim=4,
        action_dims=2,
    )

    rng = np.random.default_rng(123)
    obs = rng.standard_normal((1, 4)).astype(np.float32)

    dist, values = policy_value_apply(params.graphdef, params.state, obs)
    expected_logits = np.asarray(dist.logits)
    expected_values = np.asarray(values)

    model_path = tmp_path / "policy_value_discrete.onnx"
    export_model_to_onnx(params, obs_shape=(4,), filepath=str(model_path))

    ort_logits, ort_values = _run_onnx_inference(model_path, obs)

    np.testing.assert_allclose(ort_logits, expected_logits, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(ort_values, expected_values, rtol=0.0, atol=1e-5)


def test_multidiscrete_action_space_export_and_equivalence(tmp_path):
    key = jax.random.PRNGKey(7)
    params = init_policy_value_params(
        key,
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [32, 32]},
        obs_dim=4,
        action_dims=(2, 3),
    )

    rng = np.random.default_rng(999)
    obs = rng.standard_normal((1, 4)).astype(np.float32)

    dist, values = policy_value_apply(params.graphdef, params.state, obs)
    expected_logits = np.concatenate([np.asarray(x) for x in dist.logits_per_dim], axis=-1)
    expected_values = np.asarray(values)

    model_path = tmp_path / "policy_value_multidiscrete.onnx"
    export_model_to_onnx(params, obs_shape=(4,), filepath=str(model_path))

    ort_logits, ort_values = _run_onnx_inference(model_path, obs)

    assert ort_logits.shape == (1, 5)
    np.testing.assert_allclose(ort_logits, expected_logits, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(ort_values, expected_values, rtol=0.0, atol=1e-5)


def test_binpack_transformer_export_and_equivalence(tmp_path):
    key = jax.random.PRNGKey(17)
    params = init_policy_value_params(
        key,
        network_config={
            "_target_": "jax_rl.networks.BinPackPolicyValueModel",
            "hidden_dim": 32,
            "num_heads": 2,
            "num_layers": 1,
        },
        obs_dim=1,
        action_dims=40 * 20 * 6,
        ems_feature_dim=6,
        item_feature_dim=3,
    )

    obs = {
        "ems_pos": np.ones((1, 40, 6), dtype=np.float32),
        "item_dims": np.ones((1, 20, 3), dtype=np.float32),
        "ems_mask": np.ones((1, 40), dtype=np.bool_),
        "item_mask": np.ones((1, 20), dtype=np.bool_),
        "action_mask": np.ones((1, 40 * 20 * 6), dtype=np.bool_),
    }
    obs["action_mask"][:, 5::11] = False

    dist, values = policy_value_apply(params.graphdef, params.state, obs)
    expected_logits = np.asarray(dist.logits)
    expected_values = np.asarray(values)

    model_path = tmp_path / "policy_value_binpack.onnx"
    export_model_to_onnx(
        params,
        obs_shape={
            "ems_pos": (40, 6),
            "item_dims": (20, 3),
            "ems_mask": (40,),
            "item_mask": (20,),
            "action_mask": (40 * 20 * 6,),
        },
        filepath=str(model_path),
    )

    ort_logits, ort_values = _run_onnx_inference(model_path, obs)

    np.testing.assert_allclose(ort_logits, expected_logits, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(ort_values, expected_values, rtol=0.0, atol=1e-5)
    assert np.all(ort_logits[:, 5::11] == -1e9)
