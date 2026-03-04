import numpy as np
import jax

from purejax_ppo.export import export_model_to_onnx
from purejax_ppo.networks import init_policy_value_params, policy_value_apply


def _run_onnx_inference(model_path, obs):
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: obs})


def test_discrete_action_space_export_and_equivalence(tmp_path):
    key = jax.random.PRNGKey(0)
    params = init_policy_value_params(
        key,
        obs_dim=4,
        action_dims=2,
        hidden_sizes=(32, 32),
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
        obs_dim=4,
        action_dims=(2, 3),
        hidden_sizes=(32, 32),
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
