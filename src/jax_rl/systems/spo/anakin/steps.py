from __future__ import annotations

from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import optax
from optax import GradientTransformation
from stoa.environment import Environment

from ....configs.config import ExperimentConfig
from ....networks import policy_value_apply
from ....systems.alphazero.steps import extract_root_embedding
from ....systems.ppo.advantages import compute_gae
from ....utils.exceptions import EnvironmentInterfaceError
from ..losses import categorical_mpo_loss, multidiscrete_mpo_loss
from ..steps import SPO, make_recurrent_fn, make_root_fn
from ..types import SPOOptStates, SPOOutput, SPOTrainState, SPOTransition


class SPORunnerState(NamedTuple):
    train_state: SPOTrainState
    buffer_state: Any
    env_state: Any
    obs: Any
    key: jax.Array


def _flatten_batch_time(x: jax.Array) -> jax.Array:
    x = jnp.asarray(x)
    if x.ndim < 2:
        return x
    return jnp.reshape(x, (-1,) + x.shape[2:])


def _tree_flatten_batch_time(tree: Any) -> Any:
    return jax.tree_util.tree_map(_flatten_batch_time, tree)


def _actor_dual_loss(train_state: SPOTrainState, sequence: Any, config: ExperimentConfig):
    obs = _tree_flatten_batch_time(sequence.obs)
    sampled_actions = _flatten_batch_time(sequence.sampled_actions)
    sampled_advantages = _flatten_batch_time(sequence.sampled_advantages)

    online_dist, _ = policy_value_apply(
        train_state.params.actor_online.graphdef,
        train_state.params.actor_online.state,
        obs,
    )
    target_dist, _ = policy_value_apply(
        train_state.params.actor_target.graphdef,
        train_state.params.actor_target.state,
        obs,
    )

    if hasattr(online_dist, "logits_per_dim"):
        return multidiscrete_mpo_loss(
            dual_params=train_state.params.dual_params,
            online_action_distribution=online_dist,
            target_action_distribution=target_dist,
            sampled_actions=sampled_actions,
            q_values=sampled_advantages,
            epsilon=float(config.system.mpo_epsilon),
            epsilon_policy=float(config.system.mpo_epsilon_policy),
        )

    return categorical_mpo_loss(
        dual_params=train_state.params.dual_params,
        online_action_distribution=online_dist,
        target_action_distribution=target_dist,
        sampled_actions=sampled_actions,
        q_values=sampled_advantages,
        epsilon=float(config.system.mpo_epsilon),
        epsilon_policy=float(config.system.mpo_epsilon_policy),
    )


def _critic_loss(train_state: SPOTrainState, sequence: Any, config: ExperimentConfig):
    rewards = jnp.asarray(sequence.reward, dtype=jnp.float32)
    dones = jnp.asarray(sequence.done, dtype=jnp.bool_)
    truncated = jnp.asarray(sequence.truncated, dtype=jnp.bool_)

    obs = _tree_flatten_batch_time(sequence.obs)
    bootstrap_obs = _tree_flatten_batch_time(sequence.bootstrap_obs)

    _, critic_target_values_flat = policy_value_apply(
        train_state.params.critic_target.graphdef,
        train_state.params.critic_target.state,
        obs,
    )
    _, critic_target_bootstrap_flat = policy_value_apply(
        train_state.params.critic_target.graphdef,
        train_state.params.critic_target.state,
        bootstrap_obs,
    )

    critic_target_values = jnp.reshape(jnp.asarray(critic_target_values_flat, dtype=jnp.float32), rewards.shape)
    critic_target_bootstrap = jnp.reshape(
        jnp.asarray(critic_target_bootstrap_flat, dtype=jnp.float32),
        rewards.shape,
    )

    _, returns = jax.vmap(
        lambda reward_row, done_row, trunc_row, value_row, bootstrap_row: compute_gae(
            rewards=reward_row,
            dones=done_row,
            truncated=trunc_row,
            values=value_row,
            last_values=jnp.asarray(0.0, dtype=jnp.float32),
            gamma=float(config.system.gamma),
            gae_lambda=float(config.system.gae_lambda),
            bootstrap_values=bootstrap_row,
        ),
        in_axes=(0, 0, 0, 0, 0),
    )(
        rewards,
        dones,
        truncated,
        critic_target_values,
        critic_target_bootstrap,
    )

    target_value = jnp.reshape(jnp.asarray(returns, dtype=jnp.float32), (-1,))

    _, value_pred = policy_value_apply(
        train_state.params.critic_online.graphdef,
        train_state.params.critic_online.state,
        obs,
    )
    value_pred = jnp.reshape(jnp.asarray(value_pred, dtype=jnp.float32), (-1,))
    loss = jnp.mean((value_pred - target_value) ** 2)
    metrics = {
        "loss_critic": loss,
        "value_mean": jnp.mean(value_pred),
        "target_value_mean": jnp.mean(target_value),
    }
    return loss, metrics


def make_spo_steps(
    config: ExperimentConfig,
    env: Environment,
    env_params: Any,
    actor_optimizer: GradientTransformation,
    critic_optimizer: GradientTransformation,
    dual_optimizer: GradientTransformation,
    is_rustpool: bool,
    num_envs_per_device: int,
    buffer_add_fn: Any,
    buffer_sample_fn: Any,
):
    recurrent_fn = make_recurrent_fn(
        env=env,
        env_params=env_params,
        gamma=float(config.system.search_gamma),
        is_rustpool=is_rustpool,
    )
    root_fn = make_root_fn(config)
    search = SPO(config, recurrent_fn)

    def rollout_step(state: SPORunnerState):
        def _scan_env(carry, _):
            train_state, buffer_state, env_state, obs, key = carry
            key, root_key, search_key = jax.random.split(key, 3)

            root_embedding = extract_root_embedding(
                env=env,
                env_state=env_state,
                obs=obs,
                is_rustpool=is_rustpool,
            )
            root = root_fn(train_state.params, obs, root_embedding, root_key)
            search_output: SPOOutput = search.search(train_state.params, search_key, root)
            finite_search = jnp.logical_and(
                jnp.all(jnp.isfinite(jnp.asarray(search_output.sampled_action_weights, dtype=jnp.float32))),
                jnp.all(jnp.isfinite(jnp.asarray(search_output.value, dtype=jnp.float32))),
            )

            action = jnp.asarray(search_output.action, dtype=jnp.int32)
            invalid_action_rate = jnp.asarray(0.0, dtype=jnp.float32)
            if (
                isinstance(obs, dict)
                and "action_mask" in obs
                and action.ndim == 1
                and jnp.asarray(obs["action_mask"]).ndim == 2
            ):
                action_mask = jnp.asarray(obs["action_mask"], dtype=jnp.bool_)
                selected_valid = jnp.take_along_axis(
                    action_mask,
                    action[:, None],
                    axis=-1,
                ).squeeze(-1)
                invalid_action_rate = jnp.mean(jnp.logical_not(selected_valid).astype(jnp.float32))

            next_env_state, timestep = env.step(env_state, action, cast(Any, env_params))

            released_count = jnp.asarray(0.0, dtype=jnp.float32)
            if is_rustpool:
                if not hasattr(env, "release_batch"):
                    raise EnvironmentInterfaceError(
                        "Rustpool rollout cleanup requires 'release_batch(state, state_ids)'."
                    )
                generated = jnp.asarray(search_output.generated_state_ids, dtype=jnp.int32).reshape(-1)
                valid_mask = generated > 0
                safe_ids = jnp.where(valid_mask, generated, -1)

                chunk_size = int(num_envs_per_device)
                remainder = safe_ids.shape[0] % chunk_size
                pad_size = (chunk_size - remainder) % chunk_size
                padded_ids = jnp.pad(safe_ids, (0, pad_size), constant_values=-1)
                release_chunks = padded_ids.reshape((-1, chunk_size))
                dummy_state = jnp.zeros((chunk_size,), dtype=jnp.int32)
                for chunk_idx in range(int(release_chunks.shape[0])):
                    _ = cast(Any, env).release_batch(dummy_state, release_chunks[chunk_idx])
                released_count = jnp.asarray(jnp.sum(valid_mask), dtype=jnp.float32)

            done = jnp.asarray(timestep.last(), dtype=jnp.bool_).reshape(-1)
            extras = timestep.extras if isinstance(timestep.extras, dict) else {}
            info = extras.get("episode_metrics", {}) if isinstance(extras, dict) else {}
            info_dict: dict[str, Any] = cast(dict[str, Any], info if isinstance(info, dict) else {})
            truncated = jnp.asarray(extras.get("truncated", jnp.zeros_like(done)), dtype=jnp.bool_)

            transition = SPOTransition(
                done=done,
                truncated=truncated,
                action=action,
                sampled_actions=jnp.asarray(search_output.sampled_actions, dtype=jnp.int32),
                sampled_actions_weights=jnp.asarray(search_output.sampled_action_weights, dtype=jnp.float32),
                reward=jnp.asarray(timestep.reward, dtype=jnp.float32),
                search_value=jnp.asarray(search_output.value, dtype=jnp.float32),
                obs=obs,
                bootstrap_obs=timestep.observation,
                sampled_advantages=jnp.asarray(search_output.sampled_advantages, dtype=jnp.float32),
                info={
                    "episode_return": jnp.asarray(
                        info_dict.get("episode_return", jnp.zeros((num_envs_per_device,), dtype=jnp.float32)),
                        dtype=jnp.float32,
                    ),
                    "episode_length": jnp.asarray(
                        info_dict.get("episode_length", jnp.zeros((num_envs_per_device,), dtype=jnp.float32)),
                        dtype=jnp.float32,
                    ),
                    "is_terminal_step": jnp.asarray(
                        info_dict.get("is_terminal_step", jnp.zeros((num_envs_per_device,), dtype=jnp.bool_)),
                        dtype=jnp.bool_,
                    ),
                    "search_finite": jnp.full((num_envs_per_device,), finite_search, dtype=jnp.bool_),
                    "released_state_ids": jnp.full((num_envs_per_device,), released_count, dtype=jnp.float32),
                    "invalid_action_rate": jnp.full(
                        (num_envs_per_device,),
                        invalid_action_rate,
                        dtype=jnp.float32,
                    ),
                },
            )
            next_obs = timestep.observation
            return (train_state, buffer_state, next_env_state, next_obs, key), transition

        (train_state, buffer_state, next_env_state, next_obs, next_key), transitions = jax.lax.scan(
            _scan_env,
            (state.train_state, state.buffer_state, state.env_state, state.obs, state.key),
            xs=None,
            length=config.arch.num_steps,
        )
        traj = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), transitions)
        next_state = SPORunnerState(
            train_state=train_state,
            buffer_state=buffer_state,
            env_state=next_env_state,
            obs=next_obs,
            key=next_key,
        )
        rollout_metrics = {
            "search_finite": jnp.all(jnp.asarray(transitions.info["search_finite"], dtype=jnp.bool_)),
            "released_state_ids": jnp.sum(jnp.asarray(transitions.info["released_state_ids"], dtype=jnp.float32)),
            "invalid_action_rate": jnp.mean(
                jnp.asarray(transitions.info["invalid_action_rate"], dtype=jnp.float32)
            ),
        }
        return next_state, (traj, rollout_metrics)

    def update_step(state: SPORunnerState, rollout_outputs):
        traj, rollout_metrics = rollout_outputs
        buffer_state = buffer_add_fn(state.buffer_state, traj)

        def _update_once(carry, _):
            current_train_state, current_buffer_state, current_key = carry
            current_key, sample_key = jax.random.split(current_key)
            sampled = buffer_sample_fn(current_buffer_state, sample_key).experience

            def _actor_dual_loss_for_grad(actor_state, dual_params):
                replaced = SPOTrainState(
                    params=current_train_state.params._replace(
                        actor_online=current_train_state.params.actor_online._replace(state=actor_state),
                        dual_params=dual_params,
                    ),
                    opt_states=current_train_state.opt_states,
                )
                return _actor_dual_loss(replaced, sampled, config)

            def _critic_loss_for_grad(critic_state):
                replaced = SPOTrainState(
                    params=current_train_state.params._replace(
                        critic_online=current_train_state.params.critic_online._replace(state=critic_state),
                    ),
                    opt_states=current_train_state.opt_states,
                )
                return _critic_loss(replaced, sampled, config)

            (actor_dual_loss_value, actor_metrics), (actor_grads, dual_grads) = jax.value_and_grad(
                _actor_dual_loss_for_grad,
                has_aux=True,
                argnums=(0, 1),
            )(
                current_train_state.params.actor_online.state,
                current_train_state.params.dual_params,
            )
            (critic_loss_value, critic_metrics), critic_grads = jax.value_and_grad(
                _critic_loss_for_grad,
                has_aux=True,
            )(
                current_train_state.params.critic_online.state,
            )

            actor_grads = jax.lax.pmean(actor_grads, axis_name="device")
            critic_grads = jax.lax.pmean(critic_grads, axis_name="device")
            dual_grads = jax.lax.pmean(dual_grads, axis_name="device")

            actor_updates, next_actor_opt_state = actor_optimizer.update(
                actor_grads,
                current_train_state.opt_states.actor_opt_state,
                current_train_state.params.actor_online.state,
            )
            critic_updates, next_critic_opt_state = critic_optimizer.update(
                critic_grads,
                current_train_state.opt_states.critic_opt_state,
                current_train_state.params.critic_online.state,
            )
            dual_updates, next_dual_opt_state = dual_optimizer.update(
                dual_grads,
                current_train_state.opt_states.dual_opt_state,
                current_train_state.params.dual_params,
            )

            next_actor_online_state = optax.apply_updates(
                current_train_state.params.actor_online.state,
                actor_updates,
            )
            next_critic_online_state = optax.apply_updates(
                current_train_state.params.critic_online.state,
                critic_updates,
            )
            next_dual_params = optax.apply_updates(current_train_state.params.dual_params, dual_updates)

            tau = float(config.system.target_tau)
            next_actor_target_state = optax.incremental_update(
                next_actor_online_state,
                current_train_state.params.actor_target.state,
                step_size=tau,
            )
            next_critic_target_state = optax.incremental_update(
                next_critic_online_state,
                current_train_state.params.critic_target.state,
                step_size=tau,
            )

            next_train_state = SPOTrainState(
                params=current_train_state.params._replace(
                    actor_online=current_train_state.params.actor_online._replace(state=next_actor_online_state),
                    actor_target=current_train_state.params.actor_target._replace(state=next_actor_target_state),
                    critic_online=current_train_state.params.critic_online._replace(state=next_critic_online_state),
                    critic_target=current_train_state.params.critic_target._replace(state=next_critic_target_state),
                    dual_params=next_dual_params,
                ),
                opt_states=SPOOptStates(
                    actor_opt_state=next_actor_opt_state,
                    critic_opt_state=next_critic_opt_state,
                    dual_opt_state=next_dual_opt_state,
                ),
            )

            metrics = {
                "loss_actor_dual": actor_dual_loss_value,
                "loss_critic": critic_loss_value,
                **actor_metrics,
                **critic_metrics,
            }
            metrics = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name="device"), metrics)
            return (next_train_state, current_buffer_state, current_key), metrics

        (next_train_state, next_buffer_state, next_key), metrics = jax.lax.scan(
            _update_once,
            (state.train_state, buffer_state, state.key),
            xs=None,
            length=max(int(config.system.learner_updates_per_cycle), 1),
        )
        train_metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        train_metrics = dict(train_metrics)
        train_metrics["search_finite"] = jnp.asarray(rollout_metrics["search_finite"], dtype=jnp.bool_)
        train_metrics["released_state_ids"] = jnp.asarray(
            rollout_metrics["released_state_ids"], dtype=jnp.float32
        )
        train_metrics["invalid_action_rate"] = jnp.asarray(
            rollout_metrics["invalid_action_rate"], dtype=jnp.float32
        )

        next_state = SPORunnerState(
            train_state=next_train_state,
            buffer_state=next_buffer_state,
            env_state=state.env_state,
            obs=state.obs,
            key=next_key,
        )
        return next_state, train_metrics

    pmap_rollout = jax.pmap(rollout_step, axis_name="device")
    pmap_update = jax.pmap(update_step, axis_name="device")
    return pmap_rollout, pmap_update
