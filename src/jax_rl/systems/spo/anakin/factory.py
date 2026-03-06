from __future__ import annotations

from dataclasses import asdict
import importlib
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import optax
from optax import GradientTransformation
from stoa.environment import Environment

from ....configs.config import ExperimentConfig
from ....envs.env import make_stoa_env
from ....networks import init_policy_value_params
from ....systems.ppo.update import make_actor_optimizer, make_critic_optimizer
from ....utils.checkpoint import Checkpointer
from ....utils.exceptions import ConfigDivisibilityError
from ....utils.jax_utils import replicate_tree
from ....utils.shapes import space_feature_dim, space_flat_dim
from ..types import CategoricalDualParams, SPOOptStates, SPOParams, SPOTrainState, SPOTransition


class SPOComponents(NamedTuple):
    env: Environment
    env_params: Any
    actor_optimizer: GradientTransformation
    critic_optimizer: GradientTransformation
    dual_optimizer: GradientTransformation
    checkpointer: Checkpointer
    runner_state: Any
    num_devices: int
    num_envs_per_device: int
    is_rustpool: bool
    buffer_add_fn: Any
    buffer_sample_fn: Any


def _take_first_env_sample(obs):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x)[0], obs)


def _make_dummy_transition(obs, num_particles: int) -> SPOTransition:
    obs_sample = _take_first_env_sample(obs)
    return SPOTransition(
        done=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        action=jnp.asarray(0, dtype=jnp.int32),
        sampled_actions=jnp.zeros((num_particles,), dtype=jnp.int32),
        sampled_actions_weights=jnp.zeros((num_particles,), dtype=jnp.float32),
        reward=jnp.asarray(0.0, dtype=jnp.float32),
        search_value=jnp.asarray(0.0, dtype=jnp.float32),
        obs=obs_sample,
        bootstrap_obs=obs_sample,
        sampled_advantages=jnp.zeros((num_particles,), dtype=jnp.float32),
        info={
            "episode_return": jnp.asarray(0.0, dtype=jnp.float32),
            "episode_length": jnp.asarray(0.0, dtype=jnp.float32),
            "is_terminal_step": jnp.asarray(False, dtype=jnp.bool_),
            "search_finite": jnp.asarray(True, dtype=jnp.bool_),
            "released_state_ids": jnp.asarray(0.0, dtype=jnp.float32),
            "invalid_action_rate": jnp.asarray(0.0, dtype=jnp.float32),
        },
    )


def build_system(config: ExperimentConfig, runner_state_cls: type):
    if str(config.system.name).lower() != "spo":
        raise ValueError("SPO build_system requires config.system.name='spo'.")

    num_devices = max(config.local_device_count, 1)
    if config.arch.num_envs % num_devices != 0:
        raise ConfigDivisibilityError(
            "num_envs must be divisible by local device count, "
            f"got num_envs={config.arch.num_envs} and num_devices={num_devices}."
        )
    if config.system.total_buffer_size % num_devices != 0:
        raise ConfigDivisibilityError(
            "total_buffer_size must be divisible by local device count for pmap compatibility."
        )
    if config.system.total_batch_size % num_devices != 0:
        raise ConfigDivisibilityError(
            "total_batch_size must be divisible by local device count for pmap compatibility."
        )

    num_envs_per_device = config.arch.num_envs // num_devices
    env, env_params = make_stoa_env(
        config.env.env_name,
        num_envs_per_device=num_envs_per_device,
        env_kwargs=config.env.env_kwargs,
    )

    obs_space = env.observation_space(cast(Any, env_params))
    obs_dim = space_flat_dim(obs_space)
    action_space = env.action_space(cast(Any, env_params))
    if isinstance(action_space, tuple) and len(action_space) == 1:
        action_dims: int | tuple[int, ...] = int(action_space[0])
    elif isinstance(action_space, list) and len(action_space) == 1:
        action_dims = int(action_space[0])
    elif hasattr(action_space, "num_values"):
        num_values = jnp.asarray(cast(Any, action_space).num_values)
        if num_values.ndim == 0:
            action_dims = int(num_values)
        else:
            action_dims = tuple(int(v) for v in num_values.tolist())
    else:
        raise NotImplementedError("Action space not supported. Only Discrete and MultiDiscrete are allowed.")

    key = jax.random.PRNGKey(config.env.seed)
    key, actor_key, critic_key = jax.random.split(key, 3)

    actor_online = init_policy_value_params(
        actor_key,
        network_config=config.network,
        obs_dim=obs_dim,
        action_dims=action_dims,
        ems_feature_dim=space_feature_dim(obs_space, "ems_pos", default=6),
        item_feature_dim=space_feature_dim(obs_space, "item_dims", default=3),
    )
    critic_online = init_policy_value_params(
        critic_key,
        network_config=config.network,
        obs_dim=obs_dim,
        action_dims=action_dims,
        ems_feature_dim=space_feature_dim(obs_space, "ems_pos", default=6),
        item_feature_dim=space_feature_dim(obs_space, "item_dims", default=3),
    )

    params = SPOParams(
        actor_online=actor_online,
        actor_target=actor_online,
        critic_online=critic_online,
        critic_target=critic_online,
        dual_params=CategoricalDualParams(
            log_temperature=jnp.asarray(config.system.dual_init_log_temperature, dtype=jnp.float32),
            log_alpha=jnp.asarray(config.system.dual_init_log_alpha, dtype=jnp.float32),
        ),
    )

    actor_optimizer = make_actor_optimizer(config)
    critic_optimizer = make_critic_optimizer(config)
    dual_optimizer = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(config.system.dual_lr),
    )

    opt_states = SPOOptStates(
        actor_opt_state=actor_optimizer.init(params.actor_online.state),
        critic_opt_state=critic_optimizer.init(params.critic_online.state),
        dual_opt_state=dual_optimizer.init(params.dual_params),
    )
    train_state = SPOTrainState(params=params, opt_states=opt_states)

    checkpointer = Checkpointer(
        checkpoint_dir=config.checkpointing.checkpoint_dir,
        max_to_keep=config.checkpointing.max_to_keep,
        keep_period=config.checkpointing.keep_period,
        save_interval_steps=config.checkpointing.save_interval_steps,
        metadata={"config": asdict(config)},
    )

    fbx = importlib.import_module("flashbax")
    key, buffer_reset_key = jax.random.split(key)
    _, buffer_init_timestep = env.reset(buffer_reset_key, cast(Any, env_params))
    dummy_transition = _make_dummy_transition(
        obs=buffer_init_timestep.observation,
        num_particles=int(config.system.num_particles),
    )

    local_buffer_size = config.system.total_buffer_size // num_devices
    local_batch_size = max(config.system.total_batch_size // num_devices, 1)
    buffer = fbx.make_trajectory_buffer(
        max_size=int(local_buffer_size),
        min_length_time_axis=int(config.system.sample_sequence_length),
        sample_batch_size=int(local_batch_size),
        sample_sequence_length=int(config.system.sample_sequence_length),
        period=int(config.system.period),
        add_batch_size=int(num_envs_per_device),
    )
    buffer_state = buffer.init(dummy_transition)

    replicated_train_state = replicate_tree(train_state)
    replicated_buffer_state = replicate_tree(buffer_state)

    key, runner_seed_key = jax.random.split(key)
    keys = jax.random.split(runner_seed_key, num_devices)

    def _init_runner_state(per_device_train_state, per_device_buffer_state, runner_key):
        next_key, reset_key = jax.random.split(runner_key)
        env_state, timestep = env.reset(reset_key, cast(Any, env_params))
        return runner_state_cls(
            train_state=per_device_train_state,
            buffer_state=per_device_buffer_state,
            env_state=env_state,
            obs=timestep.observation,
            key=next_key,
        )

    pmap_init_runner_state = jax.pmap(_init_runner_state, axis_name="device")
    runner_state = pmap_init_runner_state(replicated_train_state, replicated_buffer_state, keys)

    is_rustpool = str(config.env.env_name).lower().startswith(("rustpool:", "rlpallet:"))

    return SPOComponents(
        env=env,
        env_params=env_params,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        dual_optimizer=dual_optimizer,
        checkpointer=checkpointer,
        runner_state=runner_state,
        num_devices=num_devices,
        num_envs_per_device=num_envs_per_device,
        is_rustpool=is_rustpool,
        buffer_add_fn=buffer.add,
        buffer_sample_fn=buffer.sample,
    )
