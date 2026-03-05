from collections.abc import Mapping
from typing import Any, cast
import warnings

import jax
import jax.numpy as jnp
import jax2onnx

from ..networks import CategoricalPolicyDist, policy_value_apply

jax.config.update("jax_enable_x64", False)
warnings.filterwarnings(
    "ignore",
    message=r".*float64.*truncat.*",
    category=UserWarning,
)


def export_model_to_onnx(params, obs_shape: tuple | Mapping[str, tuple], filepath: str) -> None:
    onnx_forward_fn: Any
    if isinstance(obs_shape, Mapping):
        obs_keys = tuple(sorted(obs_shape.keys()))
        dummy_inputs = [
            jnp.zeros(
                (1, *tuple(obs_shape[key])),
                dtype=(jnp.bool_ if "mask" in key else jnp.float32),
            )
            for key in obs_keys
        ]

        def _onnx_forward_fn_mapping(*obs_leaves):
            obs = {key: value for key, value in zip(obs_keys, obs_leaves)}
            dist, values = policy_value_apply(params.graphdef, params.state, obs)

            if isinstance(dist, CategoricalPolicyDist):
                logits = dist.logits
            else:
                logits = jnp.concatenate(list(dist.logits_per_dim), axis=-1)

            return logits, values

        onnx_forward_fn = _onnx_forward_fn_mapping

    else:
        dummy_input = jnp.zeros((1, *obs_shape), dtype=jnp.float32)
        dummy_inputs = [dummy_input]

        def _onnx_forward_fn_dense(obs):
            dist, values = policy_value_apply(params.graphdef, params.state, obs)

            if isinstance(dist, CategoricalPolicyDist):
                logits = dist.logits
            else:
                logits = jnp.concatenate(list(dist.logits_per_dim), axis=-1)

            return logits, values

        onnx_forward_fn = _onnx_forward_fn_dense

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Explicitly requested dtype float64 requested.*truncated to dtype float32.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"When `eqx\.nn\.BatchNorm\(\.\.\., mode=\.\.\.\)` is unspecified it defaults to 'ema'.*",
            category=UserWarning,
        )
        onnx_model = jax2onnx.to_onnx(onnx_forward_fn, cast(Any, dummy_inputs))

    with open(filepath, "wb") as f:
        f.write(onnx_model.SerializeToString())
