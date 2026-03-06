from __future__ import annotations

import time

import jax.numpy as jnp

from ....configs.config import ExperimentConfig
from ....configs.evaluations import resolve_eval_env
from ....utils.exceptions import NumericalInstabilityError
from ....utils.jax_utils import unreplicate_tree
from ....utils.logging import extract_learning_rate, jaxRL_Logger
from ....utils.runtime import PhaseTimer
from ....utils.types import LogEvent
from ..eval import evaluate as evaluate_spo
from .factory import build_system
from .steps import SPORunnerState, make_spo_steps


def _run_warmup_rollouts(config: ExperimentConfig, pmap_rollout, runner_state):
    warmup_steps = max(int(getattr(config.system, "warmup_steps", 0)), 0)
    if warmup_steps == 0:
        return runner_state

    rollout_len = max(int(config.arch.num_steps), 1)
    warmup_rollouts = (warmup_steps + rollout_len - 1) // rollout_len

    for _ in range(warmup_rollouts):
        runner_state, (_, rollout_metrics) = pmap_rollout(runner_state)
        rollout_metrics = dict(unreplicate_tree(rollout_metrics))
        if not bool(jnp.all(rollout_metrics.get("search_finite", jnp.asarray(True, dtype=jnp.bool_)))):
            raise NumericalInstabilityError("SPO warmup search produced non-finite values.")

    return runner_state


def _run_evaluations_if_needed(
    *,
    config: ExperimentConfig,
    update_idx: int,
    params,
    seed: int,
    now_fn,
) -> dict[str, float]:
    eval_metrics: dict[str, float] = {}

    for eval_name, eval_cfg in (config.evaluations or {}).items():
        cfg = dict(eval_cfg)
        num_episodes = int(cfg.get("num_episodes", 10))
        if num_episodes <= 0:
            continue
        env_name, env_kwargs = resolve_eval_env(
            cfg,
            default_env_name=config.env.env_name,
            default_env_kwargs=config.env.env_kwargs,
        )

        eval_every = int(cfg.get("eval_every", 10))
        if eval_every <= 0 or update_idx % eval_every != 0:
            continue

        phase_name = f"eval:{eval_name}"
        timer = PhaseTimer(now_fn=now_fn)
        with timer.phase(phase_name):
            eval_results = evaluate_spo(
                params=params,
                config=config,
                env_name=env_name,
                seed=int(seed),
                num_episodes=num_episodes,
                max_steps_per_episode=int(cfg.get("max_steps_per_episode", 1_000)),
                greedy=bool(cfg.get("greedy", True)),
                env_kwargs=env_kwargs,
                action_selection=str(cfg.get("action_selection", "policy")),
            )

        prefixed = {
            f"{eval_name}/{key}": float(value) if hasattr(value, "__float__") else value
            for key, value in dict(eval_results).items()
        }
        prefixed[f"{eval_name}/steps_per_second"] = timer.steps_per_second(
            phase_name,
            float(eval_results.get("steps", 0)),
        )
        eval_metrics.update(prefixed)

    return eval_metrics


def train(config: ExperimentConfig):
    system = build_system(config, SPORunnerState)

    pmap_rollout, pmap_update = make_spo_steps(
        config=config,
        env=system.env,
        env_params=system.env_params,
        actor_optimizer=system.actor_optimizer,
        critic_optimizer=system.critic_optimizer,
        dual_optimizer=system.dual_optimizer,
        is_rustpool=system.is_rustpool,
        num_envs_per_device=system.num_envs_per_device,
        buffer_add_fn=system.buffer_add_fn,
        buffer_sample_fn=system.buffer_sample_fn,
    )

    runner_state = _run_warmup_rollouts(config, pmap_rollout, system.runner_state)
    latest_metrics: dict[str, float] = {}
    latest_checkpoint = None

    logger = jaxRL_Logger.from_config(config)
    logger.log_config(config)
    tensorboard_run_dir = (
        str(config.logging.tensorboard_logdir + "/" + config.logging.tensorboard_run_name)
        if config.logging.tensorboard_logdir
        else None
    )

    try:
        for update_idx in range(config.num_updates):
            timer = PhaseTimer(now_fn=time.time)
            with timer.phase("act"):
                runner_state, rollout_outputs = pmap_rollout(runner_state)
            _, rollout_metrics = rollout_outputs
            rollout_metrics = dict(unreplicate_tree(rollout_metrics))
            if not bool(jnp.all(rollout_metrics.get("search_finite", jnp.asarray(True, dtype=jnp.bool_)))):
                raise NumericalInstabilityError("SPO search produced non-finite values.")

            with timer.phase("train"):
                runner_state, train_metrics = pmap_update(runner_state, rollout_outputs)

            act_metrics = dict(rollout_metrics)
            act_metrics["steps_per_second"] = timer.steps_per_second(
                "act",
                config.arch.num_envs * config.arch.num_steps,
            )

            train_metrics = dict(unreplicate_tree(train_metrics))
            loss_is_finite = train_metrics.get("loss_is_finite", jnp.asarray(True, dtype=jnp.bool_))
            search_finite = train_metrics.get("search_finite", jnp.asarray(True, dtype=jnp.bool_))
            if not bool(jnp.all(loss_is_finite)):
                raise NumericalInstabilityError("SPO update produced non-finite MPO loss outputs.")
            if not bool(jnp.all(search_finite)):
                raise NumericalInstabilityError("SPO search produced non-finite values.")

            train_metrics["steps_per_second"] = timer.steps_per_second(
                "train",
                max(config.system.learner_updates_per_cycle, 1),
            )
            actor_opt_state = unreplicate_tree(runner_state.train_state.opt_states.actor_opt_state)
            train_metrics["learning_rate"] = extract_learning_rate(actor_opt_state)

            log_step = (update_idx + 1) * config.rollout_batch_size
            misc_metrics = {"timestep": float(log_step)}

            eval_metrics = _run_evaluations_if_needed(
                config=config,
                update_idx=update_idx,
                params=unreplicate_tree(runner_state.train_state.params.actor_online),
                seed=int(config.env.seed + update_idx),
                now_fn=time.time,
            )

            logger.log(act_metrics, log_step, LogEvent.ACT)
            logger.log(train_metrics, log_step, LogEvent.TRAIN)
            logger.log(misc_metrics, log_step, LogEvent.ABSOLUTE)
            if eval_metrics:
                logger.log(eval_metrics, log_step, LogEvent.EVAL)

            latest_metrics = {}
            latest_metrics.update(logger.materialize(act_metrics, LogEvent.ACT))
            latest_metrics.update(logger.materialize(train_metrics, LogEvent.TRAIN))
            latest_metrics.update(logger.materialize(misc_metrics, LogEvent.ABSOLUTE))
            if eval_metrics:
                latest_metrics.update(logger.materialize(eval_metrics, LogEvent.EVAL))

            if config.checkpointing.save_interval_steps > 0 and (
                (update_idx + 1) % config.checkpointing.save_interval_steps == 0
                or update_idx == config.num_updates - 1
            ):
                train_state_to_save = unreplicate_tree(runner_state.train_state)
                key_to_save = unreplicate_tree(runner_state.key)
                eval_return_values = [
                    float(value)
                    for key, value in eval_metrics.items()
                    if key.endswith("/return_mean")
                ]
                checkpoint_metric = (
                    max(eval_return_values)
                    if eval_return_values
                    else float(-latest_metrics.get("train/loss_actor_dual", 0.0))
                )
                system.checkpointer.save(
                    timestep=update_idx + 1,
                    train_state=train_state_to_save,
                    key=key_to_save,
                    metric=checkpoint_metric,
                )
                latest_checkpoint = system.checkpointer.checkpoint_path_for_step(update_idx + 1)
    finally:
        logger.flush()
        logger.close()

    return {
        "num_updates": config.num_updates,
        "start_update": 0,
        "ran_updates": config.num_updates,
        "metrics": latest_metrics,
        "checkpoint_path": latest_checkpoint,
        "tensorboard_run_dir": tensorboard_run_dir,
        "params": unreplicate_tree(runner_state.train_state.params.actor_online),
    }
