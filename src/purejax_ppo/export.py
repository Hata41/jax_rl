import warnings

import jax
import jax.numpy as jnp
import jax2onnx

from .networks import policy_value_apply

jax.config.update("jax_enable_x64", False)
warnings.filterwarnings(
    "ignore",
    message=r".*float64.*truncat.*",
    category=UserWarning,
)


def export_model_to_onnx(params, obs_shape: tuple, filepath: str) -> None:
    def onnx_forward_fn(obs):
        dist, values = policy_value_apply(params.graphdef, params.state, obs)

        if hasattr(dist, "logits"):
            logits = dist.logits
        else:
            logits = jnp.concatenate(list(dist.logits_per_dim), axis=-1)

        return logits, values

    dummy_input = jnp.zeros((1, *obs_shape), dtype=jnp.float32)
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
        onnx_model = jax2onnx.to_onnx(onnx_forward_fn, [dummy_input])

    with open(filepath, "wb") as f:
        f.write(onnx_model.SerializeToString())
