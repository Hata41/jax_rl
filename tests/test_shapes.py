import jax
import jax.numpy as jnp

from jax_rl.networks import init_policy_value_params, policy_value_apply


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