from __future__ import annotations

from dataclasses import asdict
import importlib
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp

from ....configs.config import ExperimentConfig
from ....envs.env import make_stoa_env
from ....networks import init_policy_value_params
from ....systems.ppo.update import make_actor_optimizer, make_critic_optimizer
from ....utils.checkpoint import Checkpointer
from ....utils.exceptions import ConfigDivisibilityError, SearchTreeCapacityError
from ....utils.shapes import space_feature_dim, space_flat_dim
from ....utils.types import TrainState
from ..search_types import ExItTransition


class AlphaZeroComponents(NamedTuple):
    env: Any
    env_params: Any
    actor_optimizer: Any
    critic_optimizer: Any
    checkpointer: Checkpointer
    runner_state: Any
    num_devices: int
    num_envs_per_device: int
    is_rustpool: bool
    num_actions: int
    buffer_add_fn: Any
    buffer_sample_fn: Any


def _replicate_tree(tree, num_devices: int):
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (num_devices,) + x.shape),
        tree,
    )


def _infer_action_dims(action_space) -> int | tuple[int, ...]:
    if isinstance(action_space, tuple) and len(action_space) == 1:
        return int(action_space[0])
    if isinstance(action_space, list) and len(action_space) == 1:
        return int(action_space[0])
    if hasattr(action_space, "num_values"):
        num_values = jnp.asarray(cast(Any, action_space).num_values)
        if num_values.ndim == 0:
            return int(num_values)
        return tuple(int(v) for v in num_values.tolist())
    raise NotImplementedError("Action space not supported. Only Discrete and MultiDiscrete are allowed.")


def _estimate_tree_memory_bytes(config: ExperimentConfig) -> int:
    nodes_per_env = int(config.system.num_simulations) * int(config.system.max_depth)
    return int(config.system.num_envs) * max(nodes_per_env, 1) * 16


def _take_first_env_sample(obs):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x)[0], obs)


def _make_dummy_transition(obs, num_actions: int) -> ExItTransition:
    obs_sample = _take_first_env_sample(obs)
    return ExItTransition(
        done=jnp.asarray(False, dtype=jnp.bool_),
        action=jnp.asarray(0, dtype=jnp.int32),
        reward=jnp.asarray(0.0, dtype=jnp.float32),
        search_value=jnp.asarray(0.0, dtype=jnp.float32),
        search_policy=jnp.zeros((num_actions,), dtype=jnp.float32),
        obs=obs_sample,
        info={
            "episode_return": jnp.asarray(0.0, dtype=jnp.float32),
            "episode_length": jnp.asarray(0.0, dtype=jnp.float32),
            "is_terminal_step": jnp.asarray(False, dtype=jnp.bool_),
            "search_finite": jnp.asarray(True, dtype=jnp.bool_),
        },
    )


def build_system(config: ExperimentConfig, runner_state_cls: type):
    if str(config.system.name).lower() != "alphazero":
        raise ValueError("AlphaZero build_system requires config.system.name='alphazero'.")

    estimated_bytes = _estimate_tree_memory_bytes(config)
    budget_bytes = int(config.system.tree_memory_budget_mb) * 1024 * 1024
    if estimated_bytes > budget_bytes:
        raise SearchTreeCapacityError(
            "Configured MCTS tree budget exceeds memory boundary: "
            f"estimated={estimated_bytes} budget={budget_bytes}."
        )

    num_devices = max(config.local_device_count, 1)
    if config.system.num_envs % num_devices != 0:
        raise ConfigDivisibilityError(
            "num_envs must be divisible by local device count, "
            f"got num_envs={config.system.num_envs} and num_devices={num_devices}."
        )
    if config.system.total_buffer_size % num_devices != 0:
        raise ConfigDivisibilityError(
            "total_buffer_size must be divisible by local device count for pmap compatibility."
        )
    if config.system.total_batch_size % num_devices != 0:
        raise ConfigDivisibilityError(
            "total_batch_size must be divisible by local device count for pmap compatibility."
        )

    num_envs_per_device = config.system.num_envs // num_devices
    env, env_params = make_stoa_env(
        config.env.env_name,
        num_envs_per_device=num_envs_per_device,
        env_kwargs=config.env.env_kwargs,
    )

    obs_space = env.observation_space(cast(Any, env_params))
    obs_dim = space_flat_dim(obs_space)
    action_dims = _infer_action_dims(env.action_space(cast(Any, env_params)))
    num_actions = int(action_dims) if isinstance(action_dims, int) else int(sum(action_dims))

    key = jax.random.PRNGKey(config.env.seed)
    key, init_net_key = jax.random.split(key, 2)

    params = init_policy_value_params(
        init_net_key,
        network_config=config.network,
        obs_dim=obs_dim,
        action_dims=action_dims,
        ems_feature_dim=space_feature_dim(obs_space, "ems_pos", default=6),
        item_feature_dim=space_feature_dim(obs_space, "item_dims", default=3),
    )
    actor_optimizer = make_actor_optimizer(config)
    critic_optimizer = make_critic_optimizer(config)
    train_state = TrainState(
        params=params,
        actor_opt_state=actor_optimizer.init(params.state),
        critic_opt_state=critic_optimizer.init(params.state),
    )

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
        num_actions=num_actions,
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

    replicated_train_state = _replicate_tree(train_state, num_devices)
    replicated_buffer_state = _replicate_tree(buffer_state, num_devices)

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

    return AlphaZeroComponents(
        env=env,
        env_params=env_params,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        checkpointer=checkpointer,
        runner_state=runner_state,
        num_devices=num_devices,
        num_envs_per_device=num_envs_per_device,
        is_rustpool=is_rustpool,
        num_actions=num_actions,
        buffer_add_fn=buffer.add,
        buffer_sample_fn=buffer.sample,
    )
