import time
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np

from ....configs.config import PPOConfig
from ....envs.env import make_stoa_env
from ....networks import init_policy_value_params
from ....utils.checkpoint import Checkpointer
from ....utils.exceptions import ConfigDivisibilityError
from ....utils.logging import extract_completed_episode_metrics, jaxRL_Logger
from ....utils.shapes import space_feature_dim, space_flat_dim
from ....utils.types import LogEvent, RunnerState, TrainState
from ..eval import Evaluator
from ..update import make_actor_optimizer, make_critic_optimizer
from .steps import make_ppo_steps


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


def _prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def build_system(config: PPOConfig):
    if config.rollout_batch_size % config.minibatch_size != 0:
        raise ConfigDivisibilityError(
            "minibatch_size must divide num_envs * num_steps, "
            f"got {config.minibatch_size} and {config.rollout_batch_size}."
        )
    if config.num_updates < 1:
        raise ValueError("total_timesteps is too small for one PPO update.")

    num_devices = config.local_device_count
    if config.num_envs % num_devices != 0:
        raise ConfigDivisibilityError(
            "num_envs must be divisible by local device count, "
            f"got num_envs={config.num_envs} and num_devices={num_devices}."
        )
    if config.minibatch_size % num_devices != 0:
        raise ConfigDivisibilityError(
            "minibatch_size must be divisible by local device count, "
            f"got minibatch_size={config.minibatch_size} and num_devices={num_devices}."
        )
    num_envs_per_device = config.num_envs // num_devices

    env, env_params = make_stoa_env(config.env_name, num_envs_per_device=num_envs_per_device)
    obs_space = env.observation_space(env_params)
    obs_dim = space_flat_dim(obs_space)

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

    return {
        "runner_state": runner_state,
        "env": env,
        "env_params": env_params,
        "actor_optimizer": actor_optimizer,
        "critic_optimizer": critic_optimizer,
        "checkpointer": checkpointer,
        "start_update": start_update,
        "num_devices": num_devices,
        "num_envs_per_device": num_envs_per_device,
    }


def train(config: PPOConfig):
    system = build_system(config)

    runner_state = system["runner_state"]
    env = system["env"]
    env_params = system["env_params"]
    actor_optimizer = system["actor_optimizer"]
    critic_optimizer = system["critic_optimizer"]
    checkpointer = system["checkpointer"]
    start_update = system["start_update"]
    num_devices = system["num_devices"]
    num_envs_per_device = system["num_envs_per_device"]

    pmap_rollout, pmap_update = make_ppo_steps(
        config,
        env,
        env_params,
        actor_optimizer,
        critic_optimizer,
    )

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

    evaluators: dict[str, Evaluator] = {}
    eval_every_by_name: dict[str, int] = {}
    for eval_name, eval_cfg in (config.evaluations or {}).items():
        eval_cfg = dict(eval_cfg)
        num_episodes = int(eval_cfg.get("num_episodes", 10))
        if num_episodes <= 0:
            continue
        evaluators[eval_name] = Evaluator(
            env_name=str(eval_cfg.get("env_name", config.env_name)),
            num_episodes=num_episodes,
            max_steps_per_episode=int(eval_cfg.get("max_steps_per_episode", 1_000)),
            greedy=bool(eval_cfg.get("greedy", True)),
        )
        eval_every_by_name[eval_name] = int(eval_cfg.get("eval_every", 10))

    try:
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
            for eval_name, evaluator in evaluators.items():
                eval_every = eval_every_by_name.get(eval_name, 10)
                if eval_every <= 0 or global_update_idx % eval_every != 0:
                    continue

                eval_start = time.time()
                eval_results = evaluator.run(
                    replicated_params=runner_state.train_state.params,
                    seed=int(config.seed + global_update_idx),
                )
                eval_elapsed = time.time() - eval_start

                prefixed = _prefixed_metrics(eval_name, dict(eval_results))
                prefixed[f"{eval_name}/steps_per_second"] = _safe_steps_per_second(
                    float(eval_results.get("steps", 0)),
                    eval_elapsed,
                )
                eval_metrics.update(prefixed)

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
                eval_return_values = [
                    float(value)
                    for key, value in eval_metrics.items()
                    if key.endswith("/return_mean")
                ]
                eval_metric = max(eval_return_values) if eval_return_values else float("-inf")
                checkpointer.save(
                    timestep=global_update_idx + 1,
                    train_state=train_state_to_save,
                    key=key_to_save,
                    metric=eval_metric,
                )
                latest_checkpoint = checkpointer.checkpoint_path_for_step(global_update_idx + 1)
    finally:
        for evaluator in evaluators.values():
            evaluator.close()
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
