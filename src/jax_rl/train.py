import jax
import jax.numpy as jnp
import numpy as np
import time
from dataclasses import asdict

from .checkpoint import Checkpointer
from .config import PPOConfig
from .env import make_stoa_env
from .eval import evaluate
from .logging import extract_completed_episode_metrics, jaxRL_Logger
from .networks import flatten_observation_features, init_policy_value_params
from .rollout import collect_rollout
from .types import LogEvent, RunnerState, TrainState
from .update import make_actor_optimizer, make_critic_optimizer, ppo_update


def _hidden_sizes(config: PPOConfig):
    return tuple([config.hidden_size] * config.hidden_layers)


def _unreplicate(tree):
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def _replicate_tree(tree, num_devices: int):
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (num_devices,) + x.shape),
        tree,
    )


def _safe_steps_per_second(work_units: float, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0.0:
        return float("nan")
    return float(work_units / elapsed_seconds)


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


def _space_flat_dim(space) -> int:
    shape = getattr(space, "shape", None)
    if shape is not None:
        return int(np.prod(shape))

    spaces = getattr(space, "spaces", None)
    if isinstance(spaces, dict):
        return int(
            sum(
                _space_flat_dim(subspace)
                for key, subspace in spaces.items()
                if key != "action_mask"
            )
        )
    if isinstance(space, dict):
        return int(
            sum(
                _space_flat_dim(subspace)
                for key, subspace in space.items()
                if key != "action_mask"
            )
        )

    if isinstance(space, tuple) and len(space) >= 1:
        shape = space[0]
        if isinstance(shape, (tuple, list)):
            return int(np.prod(shape))
    if isinstance(space, list):
        return int(np.prod(space))

    if hasattr(space, "generate_value"):
        sample_obs = space.generate_value()
        flat_obs, _ = flatten_observation_features(sample_obs, batch_ndim=0)
        return int(np.prod(flat_obs.shape))

    raise ValueError(f"Unsupported observation space for flat dim inference: {type(space)}")


def _space_feature_dim(obs_space, key: str, default: int) -> int:
    spaces = getattr(obs_space, "spaces", None)
    if isinstance(spaces, dict) and key in spaces:
        leaf = spaces[key]
        shape = getattr(leaf, "shape", None)
        if shape is not None:
            return int(shape[-1])

    if isinstance(obs_space, dict) and key in obs_space:
        leaf = obs_space[key]
        if isinstance(leaf, tuple) and len(leaf) >= 1 and isinstance(leaf[0], (tuple, list)):
            return int(leaf[0][-1])
    return int(default)


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

    env, env_params = make_stoa_env(config.env_name, num_envs_per_device=num_envs_per_device)
    obs_space = env.observation_space(env_params)
    obs_dim = _space_flat_dim(obs_space)

    action_space = env.action_space(env_params)
    if isinstance(action_space, tuple) and len(action_space) == 1:
        action_dims = int(action_space[0])
    elif isinstance(action_space, list) and len(action_space) == 1:
        action_dims = int(action_space[0])
    elif hasattr(action_space, "num_values"):
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
        model_kind=(
            "binpack"
            if (config.env_name.startswith("rustpool:") or config.env_name.startswith("jaxpallet:"))
            else None
        ),
        ems_feature_dim=_space_feature_dim(obs_space, "ems_pos", default=6),
        item_feature_dim=_space_feature_dim(obs_space, "item_dims", default=3),
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

    def init_runner_state(per_device_train_state, runner_key):
        next_key, reset_key = jax.random.split(runner_key)
        env_state, timestep = env.reset(reset_key, env_params)
        return RunnerState(
            train_state=per_device_train_state,
            env_state=env_state,
            obs=timestep.observation,
            key=next_key,
        )

    key, runner_seed_key = jax.random.split(key)
    keys = jax.random.split(runner_seed_key, num_devices)
    pmap_init_runner_state = jax.pmap(init_runner_state, axis_name="device")
    runner_state = pmap_init_runner_state(train_state, keys)

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
    logger = jaxRL_Logger.from_config(config)
    logger.log_config(config)
    tensorboard_run_dir = (
        str(config.tensorboard_logdir + "/" + config.tensorboard_run_name)
        if config.tensorboard_logdir
        else None
    )

    if remaining_updates < 1:
        logger.flush()
        logger.close()
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

        act_metrics = dict(rollout_metrics)
        act_metrics["steps_per_second"] = _safe_steps_per_second(
            num_devices * num_envs_per_device * config.num_steps,
            rollout_elapsed,
        )
        act_metrics.update(extract_completed_episode_metrics(rollout_infos))

        update_start = time.time()
        runner_state, train_metrics = pmap_update(runner_state, rollout_batch, last_values)
        jax.block_until_ready(runner_state.obs)
        update_elapsed = time.time() - update_start
        train_metrics = _unreplicate(train_metrics)

        train_metrics = dict(train_metrics)
        train_metrics["steps_per_second"] = _safe_steps_per_second(
            config.update_epochs * num_minibatches,
            update_elapsed,
        )
        actor_opt_state = _unreplicate(runner_state.train_state.actor_opt_state)
        train_metrics["learning_rate"] = _extract_learning_rate(actor_opt_state)

        log_step = (global_update_idx + 1) * config.rollout_batch_size
        misc_metrics = {"timestep": float(log_step)}

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
            eval_metrics = dict(eval_results)
            eval_metrics["steps_per_second"] = _safe_steps_per_second(
                float(eval_results.get("steps", 0)),
                eval_elapsed,
            )

        logger.log(act_metrics, log_step, LogEvent.ACT)
        logger.log(train_metrics, log_step, LogEvent.TRAIN)
        logger.log(misc_metrics, log_step, LogEvent.ABSOLUTE)
        if eval_metrics:
            logger.log(eval_metrics, log_step, LogEvent.EVAL)

        merged_metrics = {}
        merged_metrics.update(logger.materialize(act_metrics, LogEvent.ACT))
        merged_metrics.update(logger.materialize(train_metrics, LogEvent.TRAIN))
        merged_metrics.update(logger.materialize(misc_metrics, LogEvent.ABSOLUTE))
        if eval_metrics:
            merged_metrics.update(logger.materialize(eval_metrics, LogEvent.EVAL))
        latest_metrics = merged_metrics
        if config.save_interval_steps > 0 and (
            (global_update_idx + 1) % config.save_interval_steps == 0
            or local_update_idx == remaining_updates - 1
        ):
            train_state_to_save = _unreplicate(runner_state.train_state)
            key_to_save = _unreplicate(runner_state.key)
            eval_metric = float(eval_metrics.get("return_mean", float("-inf")))
            checkpointer.save(
                timestep=global_update_idx + 1,
                train_state=train_state_to_save,
                key=key_to_save,
                metric=eval_metric,
            )
            latest_checkpoint = checkpointer.checkpoint_path_for_step(global_update_idx + 1)

    logger.flush()
    logger.close()

    return {
        "num_updates": config.num_updates,
        "start_update": start_update,
        "ran_updates": remaining_updates,
        "metrics": latest_metrics if latest_metrics else {},
        "checkpoint_path": latest_checkpoint,
        "tensorboard_run_dir": tensorboard_run_dir,
        "params": _unreplicate(runner_state.train_state.params),
    }