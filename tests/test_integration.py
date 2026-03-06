import jax
import jax.numpy as jnp
import importlib
import pytest
from stoa import AutoResetWrapper
from stoa.core_wrappers.vmap import VmapWrapper
from stoa.env_types import StepType, TimeStep

from jax_rl.configs.config import (
    ArchConfig,
    CheckpointConfig,
    EnvConfig,
    ExperimentConfig,
    LoggingConfig,
    SystemConfig,
)
from jax_rl.envs.env import BatchedRecordEpisodeMetrics, RustpoolObsWrapper, make_stoa_env
from jax_rl.networks import init_policy_value_params, policy_value_apply
from jax_rl.systems.ppo.anakin.system import train
from jax_rl.systems.ppo.rollout import collect_rollout


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


def test_rustpool_obs_keys_are_canonicalized():
    pytest.importorskip("rustpool")

    env, env_params = make_stoa_env("rustpool:BinPack-v0", num_envs_per_device=1)
    _, timestep = env.reset(jax.random.PRNGKey(0), env_params)
    obs = timestep.observation

    assert isinstance(obs, dict)
    assert "action_mask" in obs
    assert "ems_pos" in obs
    assert "item_dims" in obs
    assert "item_mask" in obs

    assert "next_obs" in timestep.extras
    next_obs = timestep.extras["next_obs"]
    assert "ems_pos" in next_obs
    assert "item_dims" in next_obs
    assert "item_mask" in next_obs


def test_rlpallet_env_routing_and_wrapping():
    pytest.importorskip("rlpallet")
    pytest.importorskip("rustpool")

    env, env_params = make_stoa_env(
        "rlpallet:UldEnv-v2",
        num_envs_per_device=2,
        env_kwargs={"max_items": 10},
    )
    assert env_params is None
    assert isinstance(env, BatchedRecordEpisodeMetrics)
    assert isinstance(env._env, RustpoolObsWrapper)

    _state, _timestep = env.reset(jax.random.PRNGKey(0))


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
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [16]},
        obs_dim=3,
        action_dims=2,
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
    assert jnp.allclose(batch.obs[1], batch.obs[0] + 1.0)


def test_policy_action_masking():
    params = init_policy_value_params(
        key=jax.random.PRNGKey(2),
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [16]},
        obs_dim=3,
        action_dims=4,
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

    config = ExperimentConfig(
        env=EnvConfig(env_name="jaxpallet:PMC-PLD"),
        arch=ArchConfig(
            total_timesteps=4,
            num_envs=2,
            num_steps=2,
        ),
        system=SystemConfig(
            update_epochs=1,
            minibatch_size=2,
        ),
        logging=LoggingConfig(log_every=1, tensorboard_logdir=None),
        checkpointing=CheckpointConfig(save_interval_steps=0),
        network={
            "_target_": "jax_rl.networks.BinPackPolicyValueModel",
            "hidden_dim": 16,
            "num_heads": 2,
            "num_layers": 1,
        },
        evaluations={},
    )

    result = train(config)
    assert isinstance(result, dict)
    assert "metrics" in result


def test_train_pipeline_evaluators_closed(monkeypatch):
    train_module = importlib.import_module("jax_rl.systems.ppo.anakin.system")
    counters = {"init": 0, "run": 0, "close": 0}

    class _FakeEvaluator:
        def __init__(self, env_name, num_episodes, max_steps_per_episode, greedy, env_kwargs=None):
            del env_name, max_steps_per_episode, greedy, env_kwargs
            self.num_episodes = int(num_episodes)
            counters["init"] += 1

        def run(self, replicated_params, seed):
            del replicated_params, seed
            counters["run"] += 1
            return {
                "return_mean": 1.0,
                "return_std": 0.0,
                "return_min": 1.0,
                "return_max": 1.0,
                "episodes": self.num_episodes,
                "steps": 1,
            }

        def close(self):
            counters["close"] += 1

    monkeypatch.setattr(train_module, "Evaluator", _FakeEvaluator)

    num_envs = jax.local_device_count()
    config = ExperimentConfig(
        env=EnvConfig(env_name="CartPole-v1"),
        arch=ArchConfig(
            total_timesteps=num_envs * 8,
            num_envs=num_envs,
            num_steps=8,
        ),
        system=SystemConfig(
            update_epochs=1,
            minibatch_size=num_envs * 8,
        ),
        logging=LoggingConfig(tensorboard_logdir=None),
        checkpointing=CheckpointConfig(save_interval_steps=0),
        evaluations={
            "eval_1": {
                "env_name": "CartPole-v1",
                "eval_every": 1,
                "num_episodes": num_envs,
            },
            "eval_2": {
                "env_name": "CartPole-v1",
                "eval_every": 1,
                "num_episodes": num_envs,
            },
        },
    )

    train(config)

    assert counters["init"] == 2
    assert counters["close"] == 2
    assert counters["run"] >= 2
