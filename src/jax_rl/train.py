import jax
import jax.numpy as jnp
import numpy as np
import time
from dataclasses import asdict
from pathlib import Path

from .checkpoint import Checkpointer
from .config import PPOConfig
from .env import make_stoa_env
from .eval import evaluate
from .logging import create_tensorboard_writer, log_scalar_metrics
from .networks import init_policy_value_params
from .rollout import collect_rollout
from .types import RunnerState, TrainState
from .update import make_actor_optimizer, make_critic_optimizer, ppo_update


def _hidden_sizes(config: PPOConfig):
    return tuple([config.hidden_size] * config.hidden_layers)


def _unreplicate(tree):
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def _reshape_env_batch(tree, num_devices: int, num_envs_per_device: int):
    return jax.tree_util.tree_map(
        lambda x: x.reshape((num_devices, num_envs_per_device) + x.shape[1:]),
        tree,
    )


def _replicate_tree(tree, num_devices: int):
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (num_devices,) + x.shape),
        tree,
    )


def _safe_steps_per_second(work_units: float, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0.0:
        return float("nan")
    return float(work_units / elapsed_seconds)


def _prefix_metrics(prefix: str, metrics: dict) -> dict[str, float]:
    return {f"{prefix}/{key}": float(np.asarray(value)) for key, value in metrics.items()}


def _extract_completed_episode_metrics(rollout_infos: dict) -> dict[str, float]:
    returns = np.asarray(
        rollout_infos.get("returned_episode_returns", rollout_infos.get("episode_return")),
        dtype=np.float32,
    )
    lengths = np.asarray(
        rollout_infos.get("returned_episode_lengths", rollout_infos.get("episode_length")),
        dtype=np.float32,
    )
    completed = np.asarray(
        rollout_infos.get("returned_episode", rollout_infos.get("is_terminal_step")),
        dtype=bool,
    )
    completed_returns = returns[completed]
    if completed_returns.size == 0:
        return {}
    completed_lengths = lengths[completed]
    return {
        "act/episode_return": float(completed_returns.mean()),
        "act/episode_length": float(completed_lengths.mean()),
    }


def _extract_learning_rate(actor_opt_state) -> float:
    stack = [actor_opt_state]
    while stack:
        current = stack.pop()
        if hasattr(current, "hyperparams"):
            hyperparams = getattr(current, "hyperparams")
            if isinstance(hyperparams, dict) and "learning_rate" in hyperparams:
                return float(np.asarray(hyperparams["learning_rate"]))
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (tuple, list)):
            stack.extend(current)
    return float("nan")


def train(config: PPOConfig):
    if config.rollout_batch_size % config.minibatch_size != 0:
        raise ValueError(
            "minibatch_size must divide num_envs * num_steps, "
            f"got {config.minibatch_size} and {config.rollout_batch_size}."
        )
    if config.num_updates < 1:
        raise ValueError("total_timesteps is too small for one PPO update.")

    num_devices = config.local_device_count
    if config.num_envs % num_devices != 0:
        raise ValueError(
            "num_envs must be divisible by local device count, "
            f"got num_envs={config.num_envs} and num_devices={num_devices}."
        )
    if config.minibatch_size % num_devices != 0:
        raise ValueError(
            "minibatch_size must be divisible by local device count, "
            f"got minibatch_size={config.minibatch_size} and num_devices={num_devices}."
        )
    num_envs_per_device = config.num_envs // num_devices

    env, env_params = make_stoa_env(config.env_name)
    obs_shape = env.observation_space(env_params).shape
    obs_dim = int(np.prod(obs_shape))

    action_space = env.action_space(env_params)
    if hasattr(action_space, "num_values"):
        num_values = np.asarray(action_space.num_values)
        if num_values.ndim == 0:
            action_dims = int(num_values)
        else:
            action_dims = tuple(int(v) for v in num_values.tolist())
    else:
        raise NotImplementedError(
            "Action space not supported. Only Discrete and MultiDiscrete are allowed."
        )

    key = jax.random.PRNGKey(config.seed)
    key, init_net_key = jax.random.split(key, 2)

    actor_optimizer = make_actor_optimizer(config)
    critic_optimizer = make_critic_optimizer(config)
    start_update = 0

    initial_params = init_policy_value_params(
        init_net_key,
        obs_dim=obs_dim,
        action_dims=action_dims,
        hidden_sizes=_hidden_sizes(config),
    )
    initial_train_state = TrainState(
        params=initial_params,
        actor_opt_state=actor_optimizer.init(initial_params.state),
        critic_opt_state=critic_optimizer.init(initial_params.state),
    )

    checkpointer = Checkpointer(
        checkpoint_dir=config.checkpoint_dir,
        max_to_keep=config.max_to_keep,
        keep_period=config.keep_period,
        save_interval_steps=config.save_interval_steps,
        metadata={"config": asdict(config)},
    )

    if config.resume_from:
        payload = checkpointer.restore(
            checkpoint_path=config.resume_from,
            template_train_state=initial_train_state,
            template_key=key,
        )
        train_state = payload["train_state"]
        start_update = int(payload["step"])
        if payload["key"] is not None:
            key = payload["key"]
    else:
        train_state = initial_train_state

    if start_update > config.num_updates:
        raise ValueError(
            "Checkpoint update index is larger than target num_updates, "
            f"got checkpoint={start_update}, target={config.num_updates}."
        )

    train_state = _replicate_tree(train_state, num_devices)

    key, reset_key = jax.random.split(key)

    reset_keys = jax.random.split(reset_key, config.num_envs)
    env_state, timesteps = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
    obs = timesteps.observation
    obs = obs.reshape((num_devices, num_envs_per_device) + obs.shape[1:])
    env_state = _reshape_env_batch(env_state, num_devices, num_envs_per_device)
    keys = jax.random.split(key, num_devices)
    runner_state = RunnerState(train_state=train_state, env_state=env_state, obs=obs, key=keys)

    def rollout_step(state: RunnerState):
        batch, last_values, next_obs, next_env_state, next_key, rollout_infos = collect_rollout(
            params=state.train_state.params,
            env=env,
            env_params=env_params,
            env_state=state.env_state,
            obs=state.obs,
            key=state.key,
            num_steps=config.num_steps,
        )
        rollout_metrics = {
            "reward_mean": jnp.mean(batch.rewards),
            "done_fraction": jnp.mean(batch.dones.astype(jnp.float32)),
        }
        next_state = RunnerState(
            train_state=state.train_state,
            env_state=next_env_state,
            obs=next_obs,
            key=next_key,
        )
        return next_state, (batch, last_values, rollout_infos, rollout_metrics)

    def update_step(state: RunnerState, batch, last_values):
        next_train_state, ppo_metrics, next_key = ppo_update(
            train_state=state.train_state,
            rollout_batch=batch,
            last_values=last_values,
            key=state.key,
            config=config,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
        )

        next_state = RunnerState(
            train_state=next_train_state,
            env_state=state.env_state,
            obs=state.obs,
            key=next_key,
        )
        return next_state, ppo_metrics

    pmap_rollout = jax.pmap(rollout_step, axis_name="device")
    pmap_update = jax.pmap(update_step, axis_name="device")
    latest_metrics = None
    latest_checkpoint = None
    remaining_updates = config.num_updates - start_update
    num_minibatches = config.rollout_batch_size // config.minibatch_size
    writer = create_tensorboard_writer(config.tensorboard_logdir, config.tensorboard_run_name)
    tensorboard_run_dir = (
        str(Path(config.tensorboard_logdir) / config.tensorboard_run_name)
        if config.tensorboard_logdir
        else None
    )

    if remaining_updates < 1:
        if writer is not None:
            writer.flush()
            writer.close()
        return {
            "num_updates": config.num_updates,
            "start_update": start_update,
            "ran_updates": 0,
            "metrics": {},
            "checkpoint_path": None,
            "tensorboard_run_dir": tensorboard_run_dir,
            "params": _unreplicate(runner_state.train_state.params),
        }

    for local_update_idx in range(remaining_updates):
        global_update_idx = start_update + local_update_idx

        rollout_start = time.time()
        runner_state, rollout_outputs = pmap_rollout(runner_state)
        jax.block_until_ready(runner_state.obs)
        rollout_elapsed = time.time() - rollout_start
        rollout_batch, last_values, rollout_infos, rollout_metrics = rollout_outputs
        rollout_metrics = _unreplicate(rollout_metrics)
        rollout_infos = _unreplicate(rollout_infos)

        act_metrics = _prefix_metrics("act", rollout_metrics)
        act_metrics["act/steps_per_second"] = _safe_steps_per_second(
            num_devices * num_envs_per_device * config.num_steps,
            rollout_elapsed,
        )
        act_metrics.update(_extract_completed_episode_metrics(rollout_infos))

        update_start = time.time()
        runner_state, train_metrics = pmap_update(runner_state, rollout_batch, last_values)
        jax.block_until_ready(runner_state.obs)
        update_elapsed = time.time() - update_start
        train_metrics = _unreplicate(train_metrics)

        train_metrics = _prefix_metrics("train", train_metrics)
        train_metrics["train/steps_per_second"] = _safe_steps_per_second(
            config.update_epochs * num_minibatches,
            update_elapsed,
        )
        actor_opt_state = _unreplicate(runner_state.train_state.actor_opt_state)
        train_metrics["train/learning_rate"] = _extract_learning_rate(actor_opt_state)

        log_step = (global_update_idx + 1) * config.rollout_batch_size
        misc_metrics = {"misc/timestep": float(log_step)}

        eval_metrics = {}
        if (
            config.eval_episodes > 0
            and config.eval_every > 0
            and global_update_idx % config.eval_every == 0
        ):
            eval_start = time.time()
            eval_results = evaluate(
                _unreplicate(runner_state.train_state.params),
                config,
                num_episodes=config.eval_episodes,
            )
            eval_elapsed = time.time() - eval_start
            eval_metrics = _prefix_metrics("eval", eval_results)
            eval_metrics["eval/steps_per_second"] = _safe_steps_per_second(
                float(eval_results.get("steps", 0)),
                eval_elapsed,
            )

        merged_metrics = {
            **act_metrics,
            **train_metrics,
            **misc_metrics,
            **eval_metrics,
        }
        latest_metrics = merged_metrics
        log_scalar_metrics(
            writer,
            merged_metrics,
            log_step,
        )
        if (
            global_update_idx % config.log_every == 0
            or local_update_idx == remaining_updates - 1
        ):
            metrics_str = " ".join(
                f"{k}={float(v):.4f}" for k, v in sorted(merged_metrics.items())
            )
            print(f"update={global_update_idx + 1}/{config.num_updates} {metrics_str}")
        if config.save_interval_steps > 0 and (
            (global_update_idx + 1) % config.save_interval_steps == 0
            or local_update_idx == remaining_updates - 1
        ):
            train_state_to_save = _unreplicate(runner_state.train_state)
            key_to_save = _unreplicate(runner_state.key)
            eval_metric = float(eval_metrics.get("eval/return_mean", float("-inf")))
            checkpointer.save(
                timestep=global_update_idx + 1,
                train_state=train_state_to_save,
                key=key_to_save,
                metric=eval_metric,
            )
            latest_checkpoint = checkpointer.checkpoint_path_for_step(global_update_idx + 1)

    if writer is not None:
        writer.flush()
        writer.close()

    return {
        "num_updates": config.num_updates,
        "start_update": start_update,
        "ran_updates": remaining_updates,
        "metrics": latest_metrics if latest_metrics else {},
        "checkpoint_path": latest_checkpoint,
        "tensorboard_run_dir": tensorboard_run_dir,
        "params": _unreplicate(runner_state.train_state.params),
    }