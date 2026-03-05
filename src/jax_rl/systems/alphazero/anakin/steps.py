from __future__ import annotations

from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import optax

from ....configs.config import ExperimentConfig
from ....networks import policy_value_apply
from ....systems.ppo.update import _zero_out_except_module
from ....utils.exceptions import EnvironmentInterfaceError, NumericalInstabilityError
from ....utils.types import TrainState
from ..search_types import ExItTransition
from ..steps import (
    extract_root_embedding,
    make_recurrent_fn,
    make_root_fn,
    make_search_apply_fn,
    release_rustpool_embeddings,
    search_output_is_finite,
)


class AlphaZeroRunnerState(NamedTuple):
    train_state: TrainState
    buffer_state: Any
    env_state: Any
    obs: Any
    key: jax.Array


def _distribution_logits(dist: Any) -> jax.Array:
    if hasattr(dist, "logits"):
        return jnp.asarray(dist.logits, dtype=jnp.float32)
    if hasattr(dist, "logits_per_dim"):
        return jnp.concatenate(
            tuple(jnp.asarray(x, dtype=jnp.float32) for x in dist.logits_per_dim),
            axis=-1,
        )
    raise TypeError("Policy distribution must expose 'logits' or 'logits_per_dim'.")


def _compute_value_targets(
    reward: jax.Array,
    done: jax.Array,
    search_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> jax.Array:
    reward = jnp.asarray(reward, dtype=jnp.float32)
    done = jnp.asarray(done, dtype=jnp.float32)
    search_value = jnp.asarray(search_value, dtype=jnp.float32)

    seq_len = reward.shape[1]
    if seq_len <= 1:
        return search_value

    discounts = (1.0 - done[:, :-1]) * float(gamma)
    deltas = reward[:, :-1] + discounts * search_value[:, 1:] - search_value[:, :-1]

    def _scan_adv(next_adv, elems):
        delta_t, discount_t = elems
        adv_t = delta_t + discount_t * float(gae_lambda) * next_adv
        return adv_t, adv_t

    init_adv = jnp.zeros_like(deltas[:, -1])
    _, reversed_adv = jax.lax.scan(
        _scan_adv,
        init_adv,
        (jnp.swapaxes(deltas, 0, 1)[::-1], jnp.swapaxes(discounts, 0, 1)[::-1]),
    )
    adv = jnp.swapaxes(reversed_adv[::-1], 0, 1)
    return adv + search_value[:, :-1]


def _flatten_batch_time(x: jax.Array) -> jax.Array:
    x = jnp.asarray(x)
    if x.ndim < 2:
        return x
    return jnp.reshape(x, (-1,) + x.shape[2:])


def _loss_and_grads(train_state: TrainState, sequence: Any, config: ExperimentConfig):
    reward = jnp.asarray(sequence.reward, dtype=jnp.float32)
    done = jnp.asarray(sequence.done, dtype=jnp.float32)
    search_value = jnp.asarray(sequence.search_value, dtype=jnp.float32)
    search_policy = jnp.asarray(sequence.search_policy, dtype=jnp.float32)

    if reward.shape[1] <= 1:
        obs = jax.tree_util.tree_map(_flatten_batch_time, sequence.obs)
        target_policy = _flatten_batch_time(search_policy)
        target_value = jnp.reshape(_flatten_batch_time(search_value), (-1,))
    else:
        obs = jax.tree_util.tree_map(lambda x: _flatten_batch_time(jnp.asarray(x)[:, :-1]), sequence.obs)
        target_policy = _flatten_batch_time(search_policy[:, :-1])
        target_value = jnp.reshape(
            _compute_value_targets(
                reward=reward,
                done=done,
                search_value=search_value,
                gamma=float(config.system.gamma),
                gae_lambda=float(config.system.gae_lambda),
            ),
            (-1,),
        )

    def loss_fn(param_state):
        dist, predicted_value = policy_value_apply(train_state.params.graphdef, param_state, obs)
        logits = _distribution_logits(dist)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        policy_loss = -jnp.mean(jnp.sum(target_policy * log_probs, axis=-1))

        predicted_value = jnp.reshape(jnp.asarray(predicted_value, dtype=jnp.float32), (-1,))
        value_loss = jnp.mean((predicted_value - target_value) ** 2)

        entropy = jnp.mean(jnp.asarray(dist.entropy(), dtype=jnp.float32))
        actor_loss = policy_loss - float(config.system.entropy_coef) * entropy
        critic_loss = float(config.system.value_coef) * value_loss
        total_loss = actor_loss + critic_loss
        metrics = {
            "loss_total": total_loss,
            "loss_policy": policy_loss,
            "loss_value": value_loss,
            "entropy": entropy,
            "loss_is_finite": jnp.isfinite(total_loss),
        }
        return total_loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params.state)
    del loss
    return grads, metrics


def make_alphazero_steps(
    config: ExperimentConfig,
    env: Any,
    env_params: Any,
    actor_optimizer: Any,
    critic_optimizer: Any,
    is_rustpool: bool,
    num_envs_per_device: int,
    buffer_add_fn: Any,
    buffer_sample_fn: Any,
):
    recurrent_fn = make_recurrent_fn(
        env=env,
        env_params=env_params,
        gamma=float(config.system.gamma),
        is_rustpool=is_rustpool,
    )
    root_fn = make_root_fn()
    search_apply_fn = make_search_apply_fn(config=config, recurrent_fn=recurrent_fn)

    def rollout_step(state: AlphaZeroRunnerState):
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
            search_output = search_apply_fn(train_state.params, search_key, root, obs)
            search_finite = search_output_is_finite(search_output)

            action = jnp.asarray(search_output.action, dtype=jnp.int32)
            next_env_state, timestep = env.step(env_state, action, cast(Any, env_params))

            if is_rustpool:
                if not hasattr(env, "release_batch"):
                    raise EnvironmentInterfaceError(
                        "Rustpool rollout cleanup requires 'release_batch(state, state_ids)'."
                    )
                dummy_state = jnp.zeros((num_envs_per_device,), dtype=jnp.int32)
                _ = release_rustpool_embeddings(
                    env=env,
                    state=dummy_state,
                    search_tree=search_output.search_tree,
                )

            done = jnp.asarray(timestep.last(), dtype=jnp.bool_).reshape(-1)
            extras = timestep.extras if isinstance(timestep.extras, dict) else {}
            info = extras.get("episode_metrics", {}) if isinstance(extras, dict) else {}
            info_dict: dict[str, Any] = cast(dict[str, Any], info if isinstance(info, dict) else {})
            search_value = jnp.asarray(search_output.search_tree.node_values[:, 0], dtype=jnp.float32)

            transition = ExItTransition(
                done=done,
                action=action,
                reward=jnp.asarray(timestep.reward, dtype=jnp.float32),
                search_value=search_value,
                search_policy=jnp.asarray(search_output.action_weights, dtype=jnp.float32),
                obs=obs,
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
                    "search_finite": jnp.full(
                        (num_envs_per_device,),
                        search_finite,
                        dtype=jnp.bool_,
                    ),
                },
            )

            next_obs = timestep.observation
            return (train_state, buffer_state, next_env_state, next_obs, key), transition

        (train_state, buffer_state, next_env_state, next_obs, next_key), transitions = jax.lax.scan(
            _scan_env,
            (state.train_state, state.buffer_state, state.env_state, state.obs, state.key),
            xs=None,
            length=config.system.num_steps,
        )
        traj = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), transitions)
        next_state = AlphaZeroRunnerState(
            train_state=train_state,
            buffer_state=buffer_state,
            env_state=next_env_state,
            obs=next_obs,
            key=next_key,
        )
        return next_state, traj

    def update_step(state: AlphaZeroRunnerState, traj):
        buffer_state = buffer_add_fn(state.buffer_state, traj)

        def _update_once(carry, _):
            current_train_state, current_buffer_state, current_key = carry
            current_key, sample_key = jax.random.split(current_key)
            sampled = buffer_sample_fn(current_buffer_state, sample_key).experience
            grads, metrics = _loss_and_grads(current_train_state, sampled, config)

            grads = jax.lax.pmean(grads, axis_name="device")
            actor_grads = _zero_out_except_module(
                grads,
                module_prefixes=("actor_", "shared_", "input_adapter"),
            )
            critic_grads = _zero_out_except_module(
                grads,
                module_prefixes=("critic_",),
            )

            actor_updates, next_actor_opt_state = actor_optimizer.update(
                actor_grads,
                current_train_state.actor_opt_state,
                current_train_state.params.state,
            )
            critic_updates, next_critic_opt_state = critic_optimizer.update(
                critic_grads,
                current_train_state.critic_opt_state,
                current_train_state.params.state,
            )

            merged_updates = jax.tree_util.tree_map(lambda a, c: a + c, actor_updates, critic_updates)
            next_param_state = optax.apply_updates(current_train_state.params.state, merged_updates)
            next_train_state = TrainState(
                params=current_train_state.params._replace(state=next_param_state),
                actor_opt_state=next_actor_opt_state,
                critic_opt_state=next_critic_opt_state,
            )
            metrics = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name="device"), metrics)
            return (next_train_state, current_buffer_state, current_key), metrics

        (next_train_state, next_buffer_state, next_key), metrics = jax.lax.scan(
            _update_once,
            (state.train_state, buffer_state, state.key),
            xs=None,
            length=max(int(config.system.learner_updates_per_cycle), 1),
        )
        train_metrics = jax.tree_util.tree_map(jnp.mean, metrics)

        next_state = AlphaZeroRunnerState(
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
