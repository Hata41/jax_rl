import time

import jax

from ....configs.config import ExperimentConfig
from ....utils.jax_utils import unreplicate_tree
from ....utils.logging import extract_completed_episode_metrics, extract_learning_rate, jaxRL_Logger
from ....utils.runtime import PhaseTimer
from ....utils.types import LogEvent
from ..eval import EvaluationManager, Evaluator
from .factory import build_system
from .steps import make_ppo_steps


def train(config: ExperimentConfig):
    system = build_system(config)

    runner_state = system.runner_state
    env = system.env
    env_params = system.env_params
    actor_optimizer = system.actor_optimizer
    critic_optimizer = system.critic_optimizer
    checkpointer = system.checkpointer
    start_update = system.start_update
    num_devices = system.num_devices
    num_envs_per_device = system.num_envs_per_device

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
    num_minibatches = config.rollout_batch_size // config.system.minibatch_size

    logger = jaxRL_Logger.from_config(config)
    logger.log_config(config)
    tensorboard_run_dir = (
        str(config.logging.tensorboard_logdir + "/" + config.logging.tensorboard_run_name)
        if config.logging.tensorboard_logdir
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
            "params": unreplicate_tree(runner_state.train_state.params),
        }

    evaluation_manager = EvaluationManager(
        evaluations=config.evaluations,
        default_env_name=config.env.env_name,
        default_env_kwargs=config.env.env_kwargs,
        evaluator_cls=Evaluator,
        now_fn=time.time,
    )

    try:
        for local_update_idx in range(remaining_updates):
            global_update_idx = start_update + local_update_idx

            timer = PhaseTimer(now_fn=time.time)
            with timer.phase("act"):
                runner_state, rollout_outputs = pmap_rollout(runner_state)
                jax.block_until_ready(runner_state.obs)
            rollout_batch, last_values, rollout_infos, rollout_metrics = rollout_outputs
            rollout_metrics = unreplicate_tree(rollout_metrics)
            rollout_infos = unreplicate_tree(rollout_infos)

            act_metrics = dict(rollout_metrics)
            act_metrics["steps_per_second"] = timer.steps_per_second(
                "act",
                num_devices * num_envs_per_device * config.arch.num_steps,
            )
            act_metrics.update(extract_completed_episode_metrics(rollout_infos))

            with timer.phase("train"):
                runner_state, train_metrics = pmap_update(runner_state, rollout_batch, last_values)
                jax.block_until_ready(runner_state.obs)
            train_metrics = unreplicate_tree(train_metrics)

            train_metrics = dict(train_metrics)
            train_metrics["steps_per_second"] = timer.steps_per_second(
                "train",
                config.system.update_epochs * num_minibatches,
            )
            actor_opt_state = unreplicate_tree(runner_state.train_state.actor_opt_state)
            train_metrics["learning_rate"] = extract_learning_rate(actor_opt_state)

            log_step = (global_update_idx + 1) * config.rollout_batch_size
            misc_metrics = {"timestep": float(log_step)}

            eval_metrics = evaluation_manager.run_if_needed(
                update_idx=global_update_idx,
                params=runner_state.train_state.params,
                seed=int(config.env.seed + global_update_idx),
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

            if config.checkpointing.save_interval_steps > 0 and (
                (global_update_idx + 1) % config.checkpointing.save_interval_steps == 0
                or local_update_idx == remaining_updates - 1
            ):
                train_state_to_save = unreplicate_tree(runner_state.train_state)
                key_to_save = unreplicate_tree(runner_state.key)
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
        evaluation_manager.close()
        logger.flush()
        logger.close()

    return {
        "num_updates": config.num_updates,
        "start_update": start_update,
        "ran_updates": remaining_updates,
        "metrics": latest_metrics if latest_metrics else {},
        "checkpoint_path": latest_checkpoint,
        "tensorboard_run_dir": tensorboard_run_dir,
        "params": unreplicate_tree(runner_state.train_state.params),
    }
