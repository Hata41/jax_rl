from dataclasses import asdict
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from ....configs.config import ExperimentConfig
from ....envs.env import make_stoa_env
from ....networks import init_policy_value_params
from ....utils.checkpoint import Checkpointer, resolve_resume_from
from ....utils.exceptions import ConfigDivisibilityError
from ....utils.jax_utils import normalize_restored_train_state_and_key, replicate_tree
from ....utils.shapes import space_feature_dim, space_flat_dim
from ....utils.types import RunnerState, SystemComponents, TrainState
from ..update import make_actor_optimizer, make_critic_optimizer
def _infer_action_dims(action_space) -> int | tuple[int, ...]:
    if isinstance(action_space, tuple) and len(action_space) == 1:
        return int(action_space[0])
    if isinstance(action_space, list) and len(action_space) == 1:
        return int(action_space[0])
    if hasattr(action_space, "num_values"):
        action_space_obj = cast(Any, action_space)
        num_values = np.asarray(action_space_obj.num_values)
        if num_values.ndim == 0:
            return int(num_values)
        return tuple(int(v) for v in num_values.tolist())
    raise NotImplementedError(
        "Action space not supported. Only Discrete and MultiDiscrete are allowed."
    )


def _setup_environment(config: ExperimentConfig):
    if config.rollout_batch_size % config.system.minibatch_size != 0:
        raise ConfigDivisibilityError(
            "minibatch_size must divide num_envs * num_steps, "
            f"got {config.system.minibatch_size} and {config.rollout_batch_size}."
        )
    if config.num_updates < 1:
        raise ValueError("total_timesteps is too small for one PPO update.")

    num_devices = config.local_device_count
    if config.arch.num_envs % num_devices != 0:
        raise ConfigDivisibilityError(
            "num_envs must be divisible by local device count, "
            f"got num_envs={config.arch.num_envs} and num_devices={num_devices}."
        )
    if config.system.minibatch_size % num_devices != 0:
        raise ConfigDivisibilityError(
            "minibatch_size must be divisible by local device count, "
            f"got minibatch_size={config.system.minibatch_size} and num_devices={num_devices}."
        )
    num_envs_per_device = config.arch.num_envs // num_devices

    env, env_params = make_stoa_env(
        config.env.env_name,
        num_envs_per_device=num_envs_per_device,
        env_kwargs=config.env.env_kwargs,
    )
    obs_space = env.observation_space(cast(Any, env_params))
    obs_dim = space_flat_dim(obs_space)
    action_dims = _infer_action_dims(env.action_space(cast(Any, env_params)))

    return env, env_params, obs_space, obs_dim, action_dims, num_devices, num_envs_per_device


def _init_train_state(
    config: ExperimentConfig,
    obs_space,
    obs_dim: int,
    action_dims: int | tuple[int, ...],
    num_devices: int,
):
    key = jax.random.PRNGKey(config.env.seed)
    key, init_net_key = jax.random.split(key, 2)

    actor_optimizer = make_actor_optimizer(config)
    critic_optimizer = make_critic_optimizer(config)
    start_update = 0

    initial_params = init_policy_value_params(
        init_net_key,
        network_config=config.network,
        obs_dim=obs_dim,
        action_dims=action_dims,
        ems_feature_dim=space_feature_dim(obs_space, "ems_pos", default=6),
        item_feature_dim=space_feature_dim(obs_space, "item_dims", default=3),
    )
    initial_train_state = TrainState(
        params=initial_params,
        actor_opt_state=actor_optimizer.init(initial_params.state),
        critic_opt_state=critic_optimizer.init(initial_params.state),
    )

    checkpointer = Checkpointer(
        checkpoint_dir=config.io.checkpoint.checkpoint_dir,
        max_to_keep=config.io.checkpoint.max_to_keep,
        keep_period=config.io.checkpoint.keep_period,
        save_interval_steps=config.io.checkpoint.save_interval_steps,
        metadata={"config": asdict(config)},
    )

    if config.io.checkpoint.resume_from:
        resume_path = resolve_resume_from(
            checkpoint_dir=config.io.checkpoint.checkpoint_dir,
            env_name=config.env.env_name,
            resume_from=config.io.checkpoint.resume_from,
            source_algo="ppo",
        )
        payload = checkpointer.restore(
            checkpoint_path=resume_path,
            template_train_state=initial_train_state,
            template_key=key,
        )
        train_state, restored_key = normalize_restored_train_state_and_key(
            payload["train_state"],
            payload["key"],
        )
        start_update = int(payload["step"])
        if restored_key is not None:
            key = restored_key
        if config.io.checkpoint.transfer_weights_only:
            start_update = 0
            train_state = TrainState(
                params=train_state.params,
                actor_opt_state=actor_optimizer.init(train_state.params.state),
                critic_opt_state=critic_optimizer.init(train_state.params.state),
            )
    else:
        train_state = initial_train_state

    if start_update > config.num_updates:
        raise ValueError(
            "Checkpoint update index is larger than target num_updates, "
            f"got checkpoint={start_update}, target={config.num_updates}."
        )

    train_state = replicate_tree(train_state)
    return train_state, actor_optimizer, critic_optimizer, checkpointer, start_update, key


def _init_runner_state(per_device_train_state, runner_key, env, env_params):
    next_key, reset_key = jax.random.split(runner_key)
    env_state, timestep = env.reset(reset_key, env_params)
    return RunnerState(
        train_state=per_device_train_state,
        env_state=env_state,
        obs=timestep.observation,
        key=next_key,
    )


def build_system(config: ExperimentConfig) -> SystemComponents:
    (
        env,
        env_params,
        obs_space,
        obs_dim,
        action_dims,
        num_devices,
        num_envs_per_device,
    ) = _setup_environment(config)

    (
        train_state,
        actor_optimizer,
        critic_optimizer,
        checkpointer,
        start_update,
        key,
    ) = _init_train_state(
        config=config,
        obs_space=obs_space,
        obs_dim=obs_dim,
        action_dims=action_dims,
        num_devices=num_devices,
    )

    key, runner_seed_key = jax.random.split(key)
    keys = jax.random.split(runner_seed_key, num_devices)
    pmap_init_runner_state = jax.pmap(
        lambda per_device_train_state, runner_key: _init_runner_state(
            per_device_train_state,
            runner_key,
            env,
            env_params,
        ),
        axis_name="device",
    )
    runner_state = pmap_init_runner_state(train_state, keys)

    return SystemComponents(
        runner_state=runner_state,
        env=env,
        env_params=env_params,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        checkpointer=checkpointer,
        start_update=start_update,
        num_devices=num_devices,
        num_envs_per_device=num_envs_per_device,
    )
