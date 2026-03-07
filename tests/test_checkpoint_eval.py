from dataclasses import replace
import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from jax_rl.configs.config import ArchConfig, CheckpointConfig, EnvConfig, ExperimentConfig, IOConfig, LoggingConfig, SystemConfig
from jax_rl.systems.ppo.eval import Evaluator, evaluate
from jax_rl.systems.ppo.anakin.system import train
from jax_rl.systems.ppo.anakin.factory import _init_train_state, _setup_environment
from jax_rl.systems.ppo.update import make_actor_optimizer, make_critic_optimizer
from jax_rl.utils.checkpoint import Checkpointer, resolve_resume_from
from jax_rl.utils.exceptions import CheckpointRestoreError, ConfigDivisibilityError
from jax_rl.networks import init_policy_value_params
from jax_rl.utils.types import PolicyValueParams, TrainState


def test_checkpoint_roundtrip(tmp_path: Path):
    config = ExperimentConfig()
    key = jax.random.PRNGKey(0)
    params = init_policy_value_params(
        key,
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [32, 32]},
        obs_dim=4,
        action_dims=2,
    )
    actor_optimizer = make_actor_optimizer(config)
    critic_optimizer = make_critic_optimizer(config)
    state = TrainState(
        params=params,
        actor_opt_state=actor_optimizer.init(params.state),
        critic_opt_state=critic_optimizer.init(params.state),
    )

    save_key = jax.random.PRNGKey(42)
    checkpointer = Checkpointer(
        checkpoint_dir=str(tmp_path),
        max_to_keep=2,
        keep_period=None,
        save_interval_steps=1,
        metadata={"config": {"env_name": config.env.env_name}},
    )
    success = checkpointer.save(
        timestep=1,
        train_state=state,
        key=save_key,
        metric=1.23,
    )
    assert success

    loaded = checkpointer.restore(template_train_state=state, template_key=save_key)

    assert loaded["step"] == 1
    loaded_state = loaded["train_state"]
    state_tree_match = jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.allclose, state, loaded_state)
    )
    assert state_tree_match
    assert jnp.allclose(loaded["key"], save_key)


def test_max_to_keep(tmp_path: Path):
    config = ExperimentConfig()
    key = jax.random.PRNGKey(7)
    params = init_policy_value_params(
        key,
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [16, 16]},
        obs_dim=4,
        action_dims=2,
    )
    actor_optimizer = make_actor_optimizer(config)
    critic_optimizer = make_critic_optimizer(config)
    state = TrainState(
        params=params,
        actor_opt_state=actor_optimizer.init(params.state),
        critic_opt_state=critic_optimizer.init(params.state),
    )

    checkpointer = Checkpointer(
        checkpoint_dir=str(tmp_path),
        max_to_keep=2,
        keep_period=None,
        save_interval_steps=1,
        metadata={"config": {"seed": config.env.seed}},
    )
    for step in range(1, 5):
        assert checkpointer.save(
            timestep=step,
            train_state=state,
            key=jax.random.PRNGKey(step),
            metric=float(step),
        )

    assert checkpointer.all_steps() == (3, 4)


def test_evaluate_compiled_scan():
    config = ExperimentConfig(env=EnvConfig(seed=1))
    params = init_policy_value_params(
        jax.random.PRNGKey(1),
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [16, 16]},
        obs_dim=4,
        action_dims=2,
    )
    metrics = evaluate(
        params,
        env_name="CartPole-v1",
        seed=config.env.seed,
        num_episodes=2,
        max_steps_per_episode=16,
        greedy=True,
    )

    for key in ["return_mean", "return_std", "return_min", "return_max", "episodes"]:
        assert key in metrics
    assert metrics["episodes"] == 2


def test_evaluator_initialization_divisibility(monkeypatch):
    monkeypatch.setattr(jax, "local_device_count", lambda: 4)

    with pytest.raises(ConfigDivisibilityError):
        Evaluator(
            env_name="CartPole-v1",
            num_episodes=6,
            max_steps_per_episode=16,
            greedy=True,
        )


def test_evaluator_run_and_close():
    num_devices = jax.local_device_count()
    params = init_policy_value_params(
        jax.random.PRNGKey(3),
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [16, 16]},
        obs_dim=4,
        action_dims=2,
    )
    replicated_params = PolicyValueParams(
        graphdef=params.graphdef,
        state=jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (num_devices,) + x.shape),
            params.state,
        ),
    )

    evaluator = Evaluator(
        env_name="CartPole-v1",
        num_episodes=num_devices,
        max_steps_per_episode=16,
        greedy=True,
    )
    metrics = evaluator.run(replicated_params=replicated_params, seed=7)

    assert "return_mean" in metrics
    assert "episodes" in metrics
    assert metrics["episodes"] == num_devices
    evaluator.close()


def test_restore_nonexistent_explicit_path_raises_checkpoint_restore_error(tmp_path: Path):
    checkpointer = Checkpointer(checkpoint_dir=str(tmp_path / "manager"))
    missing_dir = tmp_path / "does-not-exist"

    with pytest.raises(CheckpointRestoreError, match="does not exist"):
        checkpointer.restore(checkpoint_path=str(missing_dir))


def test_resolve_resume_from_supports_algo_prefixed_shorthand(tmp_path: Path):
    env_token = "rlpallet_UldEnv_v2"
    ppo_leaf = tmp_path / "ppo" / env_token / "20260306_191108" / "save_ppo" / "42"
    spo_leaf = tmp_path / "spo" / env_token / "20260306_193210" / "spo_after" / "1"
    ppo_leaf.mkdir(parents=True)
    spo_leaf.mkdir(parents=True)

    current_checkpoint_dir = tmp_path / "spo" / env_token / "20260306_200000" / "spo_after"

    resolved_default = resolve_resume_from(
        checkpoint_dir=str(current_checkpoint_dir),
        env_name="rlpallet:UldEnv-v2",
        resume_from="spo_after",
        source_algo="spo",
    )
    assert resolved_default == str(spo_leaf.parent)

    resolved_prefixed_ppo = resolve_resume_from(
        checkpoint_dir=str(current_checkpoint_dir),
        env_name="rlpallet:UldEnv-v2",
        resume_from="ppo/save_ppo",
        source_algo="spo",
    )
    assert resolved_prefixed_ppo == str(ppo_leaf.parent)


def test_resolve_resume_from_returns_leaf_when_no_steps_exist(tmp_path: Path):
    env_token = "rlpallet_UldEnv_v2"
    ppo_leaf = tmp_path / "ppo" / env_token / "20260306_191108" / "save_ppo"
    (ppo_leaf / "metadata").mkdir(parents=True)

    current_checkpoint_dir = tmp_path / "spo" / env_token / "20260306_200000" / "spo_after"

    resolved_prefixed_ppo = resolve_resume_from(
        checkpoint_dir=str(current_checkpoint_dir),
        env_name="rlpallet:UldEnv-v2",
        resume_from="ppo/save_ppo",
        source_algo="spo",
    )
    assert resolved_prefixed_ppo == str(ppo_leaf)


def test_resolve_resume_from_supports_flat_algo_layout(tmp_path: Path):
    ppo_leaf = tmp_path / "ppo" / "save_ppo_flat"
    (ppo_leaf / "1").mkdir(parents=True)

    current_checkpoint_dir = tmp_path / "spo" / "spo_after"

    resolved_prefixed_ppo = resolve_resume_from(
        checkpoint_dir=str(current_checkpoint_dir),
        env_name="rlpallet:UldEnv-v2",
        resume_from="ppo/save_ppo_flat",
        source_algo="spo",
    )
    assert resolved_prefixed_ppo == str(ppo_leaf)


@pytest.mark.integration
def test_train_resume_from_checkpoint(tmp_path: Path):
    num_devices = jax.local_device_count()
    rollout_batch_size = num_devices * 8
    checkpoint_root = tmp_path / "resume_ckpts"

    config = ExperimentConfig(
        env=EnvConfig(env_name="CartPole-v1", seed=0),
        arch=ArchConfig(
            total_timesteps=2 * rollout_batch_size,
            num_envs=num_devices,
            num_steps=8,
        ),
        system=SystemConfig(
            update_epochs=1,
            minibatch_size=rollout_batch_size,
        ),
        io=IOConfig(
            logger=LoggingConfig(log_every=1, tensorboard_logdir=None),
            checkpoint=CheckpointConfig(
                checkpoint_dir=str(checkpoint_root),
                save_interval_steps=1,
                max_to_keep=3,
            ),
        ),
        evaluations={},
    )

    first_result = train(config)
    checkpoint_path = first_result["checkpoint_path"]

    assert checkpoint_path is not None
    assert Path(checkpoint_path).exists()

    resumed_checkpointing = replace(config.io.checkpoint, resume_from=str(checkpoint_path))
    resumed_io = replace(config.io, checkpoint=resumed_checkpointing)
    resumed_arch = replace(config.arch, total_timesteps=3 * rollout_batch_size)
    resumed_config = replace(
        config,
        io=resumed_io,
        arch=resumed_arch,
    )

    resumed_result = train(resumed_config)

    assert resumed_result["start_update"] == 2


def test_transfer_weights_only_resets_optimizers(tmp_path: Path):
    num_devices = jax.local_device_count()
    rollout_batch_size = num_devices * 8
    checkpoint_root = tmp_path / "transfer_ckpts"

    base_config = ExperimentConfig(
        env=EnvConfig(env_name="CartPole-v1", seed=0),
        arch=ArchConfig(
            total_timesteps=2 * rollout_batch_size,
            num_envs=num_devices,
            num_steps=8,
        ),
        system=SystemConfig(
            update_epochs=1,
            minibatch_size=rollout_batch_size,
        ),
        io=IOConfig(
            logger=LoggingConfig(log_every=1, tensorboard_logdir=None),
            checkpoint=CheckpointConfig(
                checkpoint_dir=str(checkpoint_root),
                save_interval_steps=1,
                max_to_keep=3,
            ),
        ),
        evaluations={},
    )

    actor_optimizer = make_actor_optimizer(base_config)
    critic_optimizer = make_critic_optimizer(base_config)

    params = init_policy_value_params(
        jax.random.PRNGKey(5),
        network_config=base_config.network,
        obs_dim=4,
        action_dims=2,
    )
    actor_opt_state = actor_optimizer.init(params.state)
    critic_opt_state = critic_optimizer.init(params.state)

    zero_grads = jax.tree_util.tree_map(jnp.zeros_like, params.state)
    _, advanced_actor_opt_state = actor_optimizer.update(
        zero_grads,
        actor_opt_state,
        params=params.state,
    )
    _, advanced_critic_opt_state = critic_optimizer.update(
        zero_grads,
        critic_opt_state,
        params=params.state,
    )

    mutated_state = TrainState(
        params=params,
        actor_opt_state=advanced_actor_opt_state,
        critic_opt_state=advanced_critic_opt_state,
    )
    checkpoint_key = jax.random.PRNGKey(42)
    checkpointer = Checkpointer(
        checkpoint_dir=str(checkpoint_root),
        max_to_keep=3,
        keep_period=None,
        save_interval_steps=1,
        metadata={"config": {"transfer_weights_only": True}},
    )
    assert checkpointer.save(
        timestep=5,
        train_state=mutated_state,
        key=checkpoint_key,
        metric=0.0,
    )
    resume_path = checkpointer.checkpoint_path_for_step(5)

    transfer_config = replace(
        base_config,
        io=replace(
            base_config.io,
            checkpoint=replace(
                base_config.io.checkpoint,
                resume_from=resume_path,
                transfer_weights_only=True,
            ),
        ),
    )

    _, _, obs_space, obs_dim, action_dims, n_devices, _ = _setup_environment(transfer_config)
    restored_train_state_repl, restored_actor_opt, restored_critic_opt, _, start_update, _ = _init_train_state(
        config=transfer_config,
        obs_space=obs_space,
        obs_dim=obs_dim,
        action_dims=action_dims,
        num_devices=n_devices,
    )
    restored_train_state = jax.tree_util.tree_map(lambda x: x[0], restored_train_state_repl)

    expected_fresh_actor_state = restored_actor_opt.init(restored_train_state.params.state)
    expected_fresh_critic_state = restored_critic_opt.init(restored_train_state.params.state)

    assert start_update == 0
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.allclose, restored_train_state.params, mutated_state.params)
    )
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            jnp.allclose,
            restored_train_state.actor_opt_state,
            expected_fresh_actor_state,
        )
    )
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            jnp.allclose,
            restored_train_state.critic_opt_state,
            expected_fresh_critic_state,
        )
    )
    assert not jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            jnp.allclose,
            restored_train_state.actor_opt_state,
            advanced_actor_opt_state,
        )
    )
    assert not jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            jnp.allclose,
            restored_train_state.critic_opt_state,
            advanced_critic_opt_state,
        )
    )


@pytest.mark.integration
def test_train_resume_with_transfer_weights_only(tmp_path: Path):
    num_devices = jax.local_device_count()
    rollout_batch_size = num_devices * 8
    checkpoint_root = tmp_path / "resume_transfer_ckpts"

    config = ExperimentConfig(
        env=EnvConfig(env_name="CartPole-v1", seed=0),
        arch=ArchConfig(
            total_timesteps=2 * rollout_batch_size,
            num_envs=num_devices,
            num_steps=8,
        ),
        system=SystemConfig(
            update_epochs=1,
            minibatch_size=rollout_batch_size,
        ),
        io=IOConfig(
            logger=LoggingConfig(log_every=1, tensorboard_logdir=None),
            checkpoint=CheckpointConfig(
                checkpoint_dir=str(checkpoint_root),
                save_interval_steps=1,
                max_to_keep=3,
            ),
        ),
        evaluations={},
    )

    first_result = train(config)
    checkpoint_path = first_result["checkpoint_path"]
    assert checkpoint_path is not None

    resumed_checkpointing = replace(
        config.io.checkpoint,
        resume_from=str(checkpoint_path),
        transfer_weights_only=True,
    )
    resumed_io = replace(config.io, checkpoint=resumed_checkpointing)
    resumed_arch = replace(config.arch, total_timesteps=2 * rollout_batch_size)
    resumed_config = replace(
        config,
        io=resumed_io,
        arch=resumed_arch,
    )

    resumed_result = train(resumed_config)

    assert resumed_result["start_update"] == 0


@pytest.mark.integration
def test_evaluator_greedy_behavior(monkeypatch):
    eval_module = importlib.import_module("jax_rl.systems.ppo.eval")

    class _TinyTimeStep:
        def __init__(self, observation, reward):
            self.observation = observation
            self.reward = reward

        def last(self):
            return jnp.zeros((self.reward.shape[0],), dtype=jnp.bool_)

    class _TinyEvalEnv:
        def __init__(self, num_envs_per_device: int):
            self.num_envs_per_device = int(num_envs_per_device)

        def reset(self, key, env_params):
            del key, env_params
            obs = jnp.zeros((self.num_envs_per_device, 4), dtype=jnp.float32)
            timestep = _TinyTimeStep(
                observation=obs,
                reward=jnp.zeros((self.num_envs_per_device,), dtype=jnp.float32),
            )
            return obs, timestep

        def step(self, state, action, env_params):
            del env_params
            next_obs = state + 1.0
            reward = jnp.asarray(action, dtype=jnp.float32)
            timestep = _TinyTimeStep(observation=next_obs, reward=reward)
            return next_obs, timestep

        def close(self):
            return None

    class _DeterministicPolicyDist:
        def __init__(self, batch_size: int):
            self.batch_size = int(batch_size)

        def mode(self):
            return jnp.zeros((self.batch_size,), dtype=jnp.int32)

        def sample(self, key):
            del key
            return jnp.ones((self.batch_size,), dtype=jnp.int32)

    def _fake_make_stoa_env(env_name, num_envs_per_device, env_kwargs=None):
        del env_name, env_kwargs
        return _TinyEvalEnv(num_envs_per_device), None

    def _fake_policy_value_apply(graphdef, state, obs):
        del graphdef, state
        batch_size = int(obs.shape[0])
        return _DeterministicPolicyDist(batch_size), jnp.zeros((batch_size,), dtype=jnp.float32)

    monkeypatch.setattr(eval_module, "make_stoa_env", _fake_make_stoa_env)
    monkeypatch.setattr(eval_module, "policy_value_apply", _fake_policy_value_apply)

    num_devices = jax.local_device_count()
    params = init_policy_value_params(
        jax.random.PRNGKey(21),
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [16, 16]},
        obs_dim=4,
        action_dims=2,
    )
    replicated_params = PolicyValueParams(
        graphdef=params.graphdef,
        state=jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (num_devices,) + x.shape),
            params.state,
        ),
    )

    greedy_eval = Evaluator(
        env_name="CartPole-v1",
        num_episodes=num_devices,
        max_steps_per_episode=5,
        greedy=True,
    )
    greedy_metrics = greedy_eval.run(replicated_params=replicated_params, seed=0)
    greedy_eval.close()

    stochastic_eval = Evaluator(
        env_name="CartPole-v1",
        num_episodes=num_devices,
        max_steps_per_episode=5,
        greedy=False,
    )
    stochastic_metrics = stochastic_eval.run(replicated_params=replicated_params, seed=0)
    stochastic_eval.close()

    assert greedy_metrics["return_mean"] < stochastic_metrics["return_mean"]

    assert greedy_metrics["episodes"] == num_devices
    assert stochastic_metrics["episodes"] == num_devices