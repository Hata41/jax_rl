import jax
import jax.numpy as jnp
import pytest
from stoa import AutoResetWrapper
from stoa.core_wrappers.vmap import VmapWrapper
from stoa.env_types import StepType, TimeStep

from jax_rl.config import PPOConfig
from jax_rl.env import make_stoa_env
from jax_rl.networks import init_policy_value_params, policy_value_apply
from jax_rl.rollout import collect_rollout
from jax_rl.train import train


def _wrapper_chain_types(env):
    chain = []
    current = env
    while True:
        chain.append(type(current))
        if not hasattr(current, "_env"):
            break
        current = current._env
    return chain


def test_make_stoa_env_routing():
    pytest.importorskip("rustpool")
    pytest.importorskip("jaxpallet")

    rust_env, rust_params = make_stoa_env("rustpool:BinPack-v0", num_envs_per_device=4)
    del rust_params
    rust_chain = _wrapper_chain_types(rust_env)
    assert VmapWrapper not in rust_chain
    assert AutoResetWrapper not in rust_chain

    jp_env, jp_params = make_stoa_env("jaxpallet:PMC-PLD", num_envs_per_device=4)
    del jp_params
    assert isinstance(jp_env, VmapWrapper)


class _MockBatchedEnv:
    def step(self, state, actions, env_params):
        del env_params
        assert actions.shape == (4,)
        next_obs = state + 1.0
        timestep = TimeStep(
            step_type=jnp.full((4,), StepType.MID, dtype=jnp.int8),
            reward=jnp.ones((4,), dtype=jnp.float32),
            discount=jnp.ones((4,), dtype=jnp.float32),
            observation=next_obs,
            extras={},
        )
        return next_obs, timestep


def test_batched_rollout_no_vmap():
    params = init_policy_value_params(
        key=jax.random.PRNGKey(0),
        obs_dim=3,
        action_dims=2,
        hidden_sizes=(16,),
    )
    env = _MockBatchedEnv()
    env_state = jnp.zeros((4, 3), dtype=jnp.float32)
    obs = jnp.zeros((4, 3), dtype=jnp.float32)

    batch, *_ = collect_rollout(
        params=params,
        env=env,
        env_params=None,
        env_state=env_state,
        obs=obs,
        key=jax.random.PRNGKey(1),
        num_steps=2,
    )

    assert batch.obs.shape == (2, 4, 3)
    assert batch.actions.shape == (2, 4)


def test_policy_action_masking():
    params = init_policy_value_params(
        key=jax.random.PRNGKey(2),
        obs_dim=3,
        action_dims=4,
        hidden_sizes=(16,),
    )

    obs = {
        "feature_a": jnp.ones((2, 2), dtype=jnp.float32),
        "feature_b": jnp.ones((2, 1), dtype=jnp.float32),
        "action_mask": jnp.array(
            [[True, False, True, False], [False, True, False, True]],
            dtype=jnp.bool_,
        ),
    }

    dist, _ = policy_value_apply(params.graphdef, params.state, obs)
    logits = dist.logits

    assert jnp.all(logits[0, jnp.array([1, 3])] <= -1e8)
    assert jnp.all(logits[1, jnp.array([0, 2])] <= -1e8)


def test_train_pipeline_dry_run():
    pytest.importorskip("jaxpallet")

    config = PPOConfig(
        env_name="jaxpallet:PMC-PLD",
        total_timesteps=4,
        num_envs=2,
        num_steps=2,
        update_epochs=1,
        minibatch_size=2,
        hidden_size=16,
        hidden_layers=1,
        eval_episodes=0,
        log_every=1,
        save_interval_steps=0,
        tensorboard_logdir=None,
    )

    result = train(config)
    assert isinstance(result, dict)
    assert "metrics" in result
