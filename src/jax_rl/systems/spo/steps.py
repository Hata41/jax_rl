from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from ...networks import policy_value_apply
from ...utils.exceptions import EnvironmentInterfaceError
from .types import Particles, SPOOutput, SPOParams, SPORecurrentFnOutput, SPORootFnOutput


def _distribution_logits(dist: Any) -> jax.Array:
    if hasattr(dist, "logits"):
        return jnp.asarray(dist.logits, dtype=jnp.float32)
    if hasattr(dist, "logits_per_dim"):
        logits_per_dim = tuple(jnp.asarray(x, dtype=jnp.float32) for x in dist.logits_per_dim)
        return jnp.concatenate(logits_per_dim, axis=-1)
    raise TypeError("Policy distribution must expose 'logits' or 'logits_per_dim'.")


def _broadcast_tree(struct: Any, batch_size: int, num_particles: int) -> Any:
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(
            jnp.expand_dims(jnp.asarray(x), axis=1),
            (batch_size, num_particles) + jnp.asarray(x).shape[1:],
        ),
        struct,
    )


def _reshape_timestep_tree(next_timestep: Any, batch_size: int, num_particles: int) -> Any:
    leaves = jax.tree_util.tree_leaves(next_timestep)
    if not leaves:
        return next_timestep
    leading_dim = int(jnp.asarray(leaves[0]).shape[0])
    if leading_dim != batch_size * num_particles:
        raise ValueError(
            "Rustpool simulate_batch did not return the expected flat [B*P] shape. "
            f"Got leading dim {leading_dim} for B={batch_size}, P={num_particles}."
        )
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x).reshape((batch_size, num_particles) + jnp.asarray(x).shape[1:]),
        next_timestep,
    )


def _flatten_particle_obs(obs: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x).reshape((-1,) + jnp.asarray(x).shape[2:]),
        obs,
    )


def _reshape_flat_batch(x: jax.Array, batch_size: int, num_particles: int) -> jax.Array:
    x = jnp.asarray(x)
    return x.reshape((batch_size, num_particles) + x.shape[1:])


def _apply_safe_action_mask(logits: jax.Array, action_mask: jax.Array) -> jax.Array:
    mask = jnp.asarray(action_mask, dtype=jnp.bool_)
    if mask.ndim == 1:
        mask = mask[jnp.newaxis, :]
    has_any_valid = jnp.any(mask, axis=-1, keepdims=True)
    safe_mask = jnp.where(has_any_valid, mask, jnp.ones_like(mask, dtype=jnp.bool_))
    return jnp.where(safe_mask, logits, jnp.asarray(-1e9, dtype=logits.dtype))


def _sanitize_terminal_rustpool_inputs(
    *,
    state_embedding: jax.Array,
    sampled_actions: jax.Array,
    terminal_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    alive_mask = jnp.logical_not(jnp.asarray(terminal_mask, dtype=jnp.bool_))
    flat_alive = alive_mask.reshape(-1)
    has_alive = jnp.any(flat_alive)

    flat_states = jnp.asarray(state_embedding, dtype=jnp.int32).reshape(-1)
    flat_actions = jnp.asarray(sampled_actions, dtype=jnp.int32).reshape(-1)

    ref_idx = jnp.argmax(flat_alive.astype(jnp.int32))
    ref_state_id = flat_states[ref_idx]
    ref_action = flat_actions[ref_idx]

    safe_state_embedding = jnp.where(alive_mask, state_embedding, ref_state_id)
    safe_sampled_actions = jnp.where(alive_mask, sampled_actions, ref_action)
    return safe_state_embedding, safe_sampled_actions, has_alive


def make_root_fn(config) -> Callable[[SPOParams, Any, Any, jax.Array], SPORootFnOutput]:
    num_particles = int(config.system.num_particles)
    dirichlet_alpha = float(config.system.dirichlet_alpha)
    dirichlet_fraction = float(config.system.dirichlet_fraction)

    def root_fn(params: SPOParams, observation: Any, state_embedding: Any, key: jax.Array) -> SPORootFnOutput:
        actor_dist, _ = policy_value_apply(
            params.actor_target.graphdef,
            params.actor_target.state,
            observation,
        )
        _, critic_value = policy_value_apply(
            params.critic_target.graphdef,
            params.critic_target.state,
            observation,
        )
        logits = _distribution_logits(actor_dist)
        batch_size = int(logits.shape[0])

        key_noise, key_actions = jax.random.split(key)
        prior_probs = jax.nn.softmax(logits, axis=-1)
        noise = jax.random.dirichlet(
            key_noise,
            alpha=jnp.full((prior_probs.shape[-1],), dirichlet_alpha, dtype=jnp.float32),
            shape=(batch_size,),
        )
        mixed_probs = (1.0 - dirichlet_fraction) * prior_probs + dirichlet_fraction * noise

        if isinstance(observation, dict) and "action_mask" in observation:
            action_mask = jnp.asarray(observation["action_mask"], dtype=jnp.bool_)
            has_any_valid = jnp.any(action_mask, axis=-1, keepdims=True)
            safe_mask = jnp.where(has_any_valid, action_mask, jnp.ones_like(action_mask, dtype=jnp.bool_))
            mixed_probs = jnp.where(safe_mask, mixed_probs, 0.0)
            mixed_probs = mixed_probs / jnp.clip(jnp.sum(mixed_probs, axis=-1, keepdims=True), a_min=1e-8)
            mixed_logits = jnp.where(
                safe_mask,
                jnp.log(jnp.clip(mixed_probs, a_min=1e-8)),
                jnp.asarray(-1e9, dtype=jnp.float32),
            )
        else:
            mixed_logits = jnp.log(jnp.clip(mixed_probs, a_min=1e-8))

        action_keys = jax.random.split(key_actions, batch_size)
        particle_actions = jax.vmap(
            lambda sample_key, sample_logits: jax.random.categorical(
                sample_key,
                sample_logits,
                shape=(num_particles,),
            ),
            in_axes=(0, 0),
        )(action_keys, mixed_logits)

        return SPORootFnOutput(
            particle_logits=jnp.broadcast_to(
                jnp.expand_dims(mixed_logits, axis=1),
                (batch_size, num_particles, mixed_logits.shape[-1]),
            ),
            particle_actions=jnp.asarray(particle_actions, dtype=jnp.int32),
            particle_state_embedding=_broadcast_tree(state_embedding, batch_size, num_particles),
            particle_values=jnp.broadcast_to(
                jnp.expand_dims(jnp.asarray(critic_value, dtype=jnp.float32), axis=1),
                (batch_size, num_particles),
            ),
        )

    return root_fn


def make_recurrent_fn(
    *,
    env: Any,
    env_params: Any,
    gamma: float,
    is_rustpool: bool,
) -> Callable[[SPOParams, jax.Array, jax.Array, Any], tuple[SPORecurrentFnOutput, Any]]:
    gamma_value = jnp.asarray(gamma, dtype=jnp.float32)

    if is_rustpool:
        if not hasattr(env, "snapshot") or not hasattr(env, "simulate_batch") or not hasattr(env, "release_batch"):
            raise EnvironmentInterfaceError(
                "Rustpool SPO recurrent path requires 'snapshot', 'simulate_batch', and 'release_batch'."
            )

        def recurrent_fn_rustpool(
            params: SPOParams,
            key: jax.Array,
            particle_actions: jax.Array,
            state_embedding: jax.Array,
        ) -> tuple[SPORecurrentFnOutput, jax.Array]:
            actions = jnp.asarray(particle_actions, dtype=jnp.int32)
            state_ids = jnp.asarray(state_embedding, dtype=jnp.int32)
            batch_size = actions.shape[0]
            num_particles = actions.shape[1]

            callback_capacity = None
            state_ids_struct = getattr(env, "_state_ids_struct", None)
            if hasattr(state_ids_struct, "shape") and len(state_ids_struct.shape) > 0:
                callback_capacity = int(state_ids_struct.shape[0])
            elif hasattr(env, "num_envs_per_device"):
                callback_capacity = int(getattr(env, "num_envs_per_device"))

            simulate_particlewise = callback_capacity == int(batch_size)

            if simulate_particlewise:
                per_particle_timesteps = []
                for particle_idx in range(num_particles):
                    particle_actions = actions[:, particle_idx]
                    particle_state_ids = state_ids[:, particle_idx]
                    particle_dummy_state = jnp.zeros_like(particle_state_ids, dtype=jnp.int32)
                    _, particle_timestep = env.simulate_batch(
                        particle_dummy_state,
                        particle_state_ids,
                        particle_actions,
                    )
                    per_particle_timesteps.append(particle_timestep)

                next_timestep = jax.tree_util.tree_map(
                    lambda *xs: jnp.stack(tuple(jnp.asarray(x) for x in xs), axis=1),
                    *per_particle_timesteps,
                )
            else:
                flat_actions = actions.reshape((-1,) + actions.shape[2:])
                flat_state_ids = state_ids.reshape(-1)
                dummy_state = jnp.zeros_like(flat_state_ids, dtype=jnp.int32)
                _, next_timestep_flat = env.simulate_batch(dummy_state, flat_state_ids, flat_actions)
                next_timestep = _reshape_timestep_tree(next_timestep_flat, batch_size, num_particles)

            next_state_ids = jnp.asarray(next_timestep.extras["state_id"], dtype=jnp.int32)
            invalid_state_ids = next_state_ids <= 0
            safe_next_state_ids = jnp.where(invalid_state_ids, state_ids, next_state_ids)

            flat_obs = _flatten_particle_obs(next_timestep.observation)

            actor_dist, _ = policy_value_apply(
                params.actor_target.graphdef,
                params.actor_target.state,
                flat_obs,
            )
            _, critic_value = policy_value_apply(
                params.critic_target.graphdef,
                params.critic_target.state,
                flat_obs,
            )
            flat_prior_logits = jnp.asarray(_distribution_logits(actor_dist), dtype=jnp.float32)
            if isinstance(flat_obs, dict) and "action_mask" in flat_obs:
                flat_prior_logits = _apply_safe_action_mask(
                    flat_prior_logits,
                    jnp.asarray(flat_obs["action_mask"], dtype=jnp.bool_),
                )

            flat_action_keys = jax.random.split(key, int(flat_prior_logits.shape[0]))
            flat_next_action = jax.vmap(
                lambda sample_key, sample_logits: jax.random.categorical(sample_key, sample_logits),
                in_axes=(0, 0),
            )(flat_action_keys, flat_prior_logits)

            prior_logits = _reshape_flat_batch(
                flat_prior_logits,
                batch_size,
                num_particles,
            )
            timestep_discount = jnp.asarray(next_timestep.discount, dtype=jnp.float32)
            timestep_discount = jnp.where(
                invalid_state_ids,
                jnp.zeros_like(timestep_discount),
                timestep_discount,
            )
            next_action = _reshape_flat_batch(
                jnp.asarray(flat_next_action, dtype=jnp.int32),
                batch_size,
                num_particles,
            )
            critic_value = _reshape_flat_batch(
                jnp.asarray(critic_value, dtype=jnp.float32),
                batch_size,
                num_particles,
            )

            recurrent_output = SPORecurrentFnOutput(
                reward=jnp.asarray(next_timestep.reward, dtype=jnp.float32),
                discount=timestep_discount * gamma_value,
                prior_logits=prior_logits,
                value=timestep_discount * jnp.asarray(critic_value, dtype=jnp.float32),
                next_sampled_action=next_action,
            )
            return recurrent_output, safe_next_state_ids

        return recurrent_fn_rustpool

    def recurrent_fn_jax(
        params: SPOParams,
        key: jax.Array,
        particle_actions: jax.Array,
        state_embedding: Any,
    ) -> tuple[SPORecurrentFnOutput, Any]:
        next_state_embedding, next_timestep = jax.vmap(jax.vmap(
            lambda state, action: env.step(state, action, env_params)
        ))(state_embedding, particle_actions)

        batch_size = int(particle_actions.shape[0])
        num_particles = int(particle_actions.shape[1])
        flat_obs = _flatten_particle_obs(next_timestep.observation)

        actor_dist, _ = policy_value_apply(
            params.actor_target.graphdef,
            params.actor_target.state,
            flat_obs,
        )
        _, critic_value = policy_value_apply(
            params.critic_target.graphdef,
            params.critic_target.state,
            flat_obs,
        )
        prior_logits = _reshape_flat_batch(_distribution_logits(actor_dist), batch_size, num_particles)
        timestep_discount = jnp.asarray(next_timestep.discount, dtype=jnp.float32)
        critic_value = _reshape_flat_batch(jnp.asarray(critic_value, dtype=jnp.float32), batch_size, num_particles)
        sampled_action = _reshape_flat_batch(
            jnp.asarray(actor_dist.sample(key), dtype=jnp.int32),
            batch_size,
            num_particles,
        )
        recurrent_output = SPORecurrentFnOutput(
            reward=jnp.asarray(next_timestep.reward, dtype=jnp.float32),
            discount=timestep_discount * gamma_value,
            prior_logits=prior_logits,
            value=timestep_discount * critic_value,
            next_sampled_action=sampled_action,
        )
        return recurrent_output, next_state_embedding

    return recurrent_fn_jax


class SPO:
    def __init__(self, config, recurrent_fn: Callable[[SPOParams, jax.Array, jax.Array, Any], tuple[SPORecurrentFnOutput, Any]]):
        self.config = config
        self.recurrent_fn = recurrent_fn

    def search(self, params: SPOParams, rng_key: jax.Array, root: SPORootFnOutput) -> SPOOutput:
        particles, rollout_metrics, generated_state_ids = self.rollout(params, rng_key, root)

        weights_logits = self.get_resample_logits(
            particles.resample_td_weights,
            log_temperature=params.dual_params.log_temperature,
        )
        sampled_action_weights = jax.nn.softmax(weights_logits, axis=-1)

        best_particle_idx = jnp.argmax(sampled_action_weights, axis=-1)
        action = jnp.take_along_axis(
            particles.root_actions,
            best_particle_idx[..., None],
            axis=1,
        ).squeeze(axis=1)
        value = jnp.sum(sampled_action_weights * particles.value, axis=-1)

        return SPOOutput(
            action=jnp.asarray(action, dtype=jnp.int32),
            sampled_action_weights=jnp.asarray(sampled_action_weights, dtype=jnp.float32),
            sampled_actions=jnp.asarray(particles.root_actions, dtype=jnp.int32),
            value=jnp.asarray(value, dtype=jnp.float32),
            sampled_advantages=jnp.asarray(particles.gae, dtype=jnp.float32),
            generated_state_ids=jnp.asarray(generated_state_ids, dtype=jnp.int32),
            rollout_metrics=rollout_metrics,
        )

    def rollout(
        self,
        params: SPOParams,
        rng_key: jax.Array,
        root: SPORootFnOutput,
    ) -> tuple[Particles, dict[str, jax.Array], jax.Array]:
        keys = jax.random.split(rng_key, int(self.config.system.search_depth))
        initial_particles = self.init_particles(root)
        carry = (initial_particles, initial_particles.root_actions)
        (final_particles, _), scan_metrics = jax.lax.scan(
            lambda c, x: self.one_step_rollout(c, x, params),
            carry,
            xs=(jnp.arange(int(self.config.system.search_depth), dtype=jnp.int32), keys),
        )

        rollout_metrics = {
            "ess": jnp.mean(scan_metrics["ess"], axis=0),
            "entropy": jnp.mean(scan_metrics["entropy"], axis=0),
            "mean_td_weights": jnp.mean(scan_metrics["mean_td_weights"], axis=0),
            "particles_alive": jnp.mean(scan_metrics["particles_alive"], axis=0),
            "resample": jnp.mean(scan_metrics["resample"], axis=0),
        }
        generated_state_ids = jnp.swapaxes(scan_metrics["generated_state_ids"], 0, 1)
        return final_particles, rollout_metrics, generated_state_ids

    def one_step_rollout(
        self,
        particles_and_actions: tuple[Particles, jax.Array],
        depth_count_and_key: tuple[jax.Array, jax.Array],
        params: SPOParams,
    ) -> tuple[tuple[Particles, jax.Array], dict[str, jax.Array]]:
        particles, sampled_actions = particles_and_actions
        current_depth, key = depth_count_and_key
        key_resampling, recurrent_step_key, next_action_key = jax.random.split(key, 3)

        use_rustpool_terminal_guard = isinstance(particles.state_embedding, jax.Array)
        if use_rustpool_terminal_guard:
            safe_embedding, safe_actions, has_alive = _sanitize_terminal_rustpool_inputs(
                state_embedding=jnp.asarray(particles.state_embedding, dtype=jnp.int32),
                sampled_actions=jnp.asarray(sampled_actions, dtype=jnp.int32),
                terminal_mask=particles.terminal,
            )

            def _run_recurrent(_):
                return self.recurrent_fn(
                    params,
                    recurrent_step_key,
                    safe_actions,
                    safe_embedding,
                )

            def _skip_recurrent(_):
                batch_size, num_particles = particles.terminal.shape
                recurrent_output = SPORecurrentFnOutput(
                    reward=jnp.zeros((batch_size, num_particles), dtype=jnp.float32),
                    discount=jnp.zeros((batch_size, num_particles), dtype=jnp.float32),
                    prior_logits=jnp.asarray(particles.prior_logits, dtype=jnp.float32),
                    value=jnp.asarray(particles.value, dtype=jnp.float32),
                    next_sampled_action=jnp.zeros((batch_size, num_particles), dtype=jnp.int32),
                )
                return recurrent_output, jnp.asarray(particles.state_embedding, dtype=jnp.int32)

            recurrent_output, next_state_embedding = jax.lax.cond(
                has_alive,
                _run_recurrent,
                _skip_recurrent,
                operand=None,
            )

            alive_mask = jnp.logical_not(jnp.asarray(particles.terminal, dtype=jnp.bool_))
            recurrent_output = SPORecurrentFnOutput(
                reward=jnp.where(alive_mask, recurrent_output.reward, jnp.zeros_like(recurrent_output.reward)),
                discount=jnp.where(alive_mask, recurrent_output.discount, jnp.zeros_like(recurrent_output.discount)),
                prior_logits=jnp.where(
                    alive_mask[..., None],
                    recurrent_output.prior_logits,
                    jnp.asarray(particles.prior_logits, dtype=jnp.float32),
                ),
                value=jnp.where(alive_mask, recurrent_output.value, jnp.asarray(particles.value, dtype=jnp.float32)),
                next_sampled_action=jnp.where(
                    alive_mask,
                    recurrent_output.next_sampled_action,
                    jnp.asarray(sampled_actions, dtype=jnp.int32),
                ),
            )
            next_state_embedding = jnp.where(
                alive_mask,
                jnp.asarray(next_state_embedding, dtype=jnp.int32),
                jnp.asarray(particles.state_embedding, dtype=jnp.int32),
            )
        else:
            recurrent_output, next_state_embedding = self.recurrent_fn(
                params,
                recurrent_step_key,
                sampled_actions,
                particles.state_embedding,
            )

        td_error = recurrent_output.reward + recurrent_output.value - particles.value
        terminal_mask = 1.0 - particles.terminal.astype(jnp.float32)
        updated_td_weights = td_error * terminal_mask + particles.resample_td_weights

        ess, entropy = self.calculate_ess_and_entropy(
            updated_td_weights,
            log_temperature=params.dual_params.log_temperature,
        )

        root_action = jnp.where(
            current_depth == 0,
            sampled_actions,
            particles.root_actions,
        )

        updated_particles = self.update_particles(
            embedding=next_state_embedding,
            updated_td_weights=updated_td_weights,
            root_action=root_action,
            recurrent_output=recurrent_output,
            particles=particles,
        )

        resample_logits = self.get_resample_logits(
            updated_td_weights,
            log_temperature=params.dual_params.log_temperature,
        )
        resampling_mode = str(self.config.system.spo_resampling_mode).lower()

        if resampling_mode == "period":
            should_resample = ((current_depth + 1) % int(self.config.system.spo_resampling_period)) == 0
            updated_particles = jax.lax.cond(
                should_resample,
                lambda _: self.resample(updated_particles, key_resampling, resample_logits),
                lambda _: updated_particles,
                operand=None,
            )
            resample_metric = jnp.full((updated_td_weights.shape[0],), should_resample, dtype=jnp.float32)
        else:
            condition = ess < (
                float(self.config.system.spo_ess_threshold) * float(self.config.system.num_particles)
            )
            resampled_particles = self.resample(updated_particles, key_resampling, resample_logits)

            def _select(new_field, old_field):
                cond = condition.reshape((condition.shape[0],) + (1,) * (old_field.ndim - 1))
                return jnp.where(cond, new_field, old_field)

            updated_particles = jax.tree_util.tree_map(_select, resampled_particles, updated_particles)
            resample_metric = condition.astype(jnp.float32)

        step_metrics = {
            "ess": ess,
            "entropy": entropy,
            "mean_td_weights": jnp.mean(updated_td_weights, axis=-1),
            "particles_alive": jnp.mean(terminal_mask, axis=-1),
            "resample": resample_metric,
            "generated_state_ids": jnp.asarray(next_state_embedding, dtype=jnp.int32),
        }

        next_logits = jnp.asarray(updated_particles.prior_logits, dtype=jnp.float32)
        flat_next_logits = next_logits.reshape((-1, next_logits.shape[-1]))
        flat_next_keys = jax.random.split(next_action_key, int(flat_next_logits.shape[0]))
        flat_next_actions = jax.vmap(
            lambda sample_key, sample_logits: jax.random.categorical(sample_key, sample_logits),
            in_axes=(0, 0),
        )(flat_next_keys, flat_next_logits)
        next_sampled_actions = flat_next_actions.reshape(next_logits.shape[:2]).astype(jnp.int32)

        return (updated_particles, next_sampled_actions), step_metrics

    def init_particles(self, root: SPORootFnOutput) -> Particles:
        batch_size = root.particle_values.shape[0]
        num_particles = int(self.config.system.num_particles)
        return Particles(
            state_embedding=root.particle_state_embedding,
            root_actions=root.particle_actions,
            resample_td_weights=jnp.zeros((batch_size, num_particles), dtype=jnp.float32),
            prior_logits=root.particle_logits,
            value=root.particle_values,
            terminal=jnp.zeros((batch_size, num_particles), dtype=jnp.bool_),
            depth=jnp.zeros((batch_size, num_particles), dtype=jnp.int32),
            gae=jnp.zeros((batch_size, num_particles), dtype=jnp.float32),
        )

    def update_particles(
        self,
        embedding: Any,
        updated_td_weights: jax.Array,
        root_action: jax.Array,
        recurrent_output: SPORecurrentFnOutput,
        particles: Particles,
    ) -> Particles:
        return Particles(
            state_embedding=embedding,
            root_actions=root_action,
            resample_td_weights=updated_td_weights,
            prior_logits=recurrent_output.prior_logits,
            value=recurrent_output.value,
            terminal=jnp.logical_or(particles.terminal, recurrent_output.discount <= 0.0),
            depth=particles.depth + 1,
            gae=self.calculate_gae(
                current_gae=particles.gae,
                value=particles.value,
                next_value=recurrent_output.value,
                reward=recurrent_output.reward,
                discount=recurrent_output.discount,
                depth=particles.depth,
                gamma=float(self.config.system.search_gamma),
                lambda_=float(self.config.system.search_gae_lambda),
            ),
        )

    def calculate_gae(
        self,
        current_gae: jax.Array,
        value: jax.Array,
        next_value: jax.Array,
        reward: jax.Array,
        discount: jax.Array,
        depth: jax.Array,
        gamma: float,
        lambda_: float,
    ) -> jax.Array:
        delta = reward + next_value - value
        return delta * (gamma * lambda_ * discount) ** depth + current_gae

    def get_resample_logits(
        self,
        td_weights: jax.Array,
        log_temperature: jax.Array,
    ) -> jax.Array:
        temperature = jax.nn.softplus(log_temperature).squeeze() + 1e-8
        return td_weights / temperature

    def resample(
        self,
        particles: Particles,
        key: jax.Array,
        resample_logits: jax.Array,
    ) -> Particles:
        num_particles = int(self.config.system.num_particles)
        batch_dim_keys = jax.random.split(key, resample_logits.shape[0])
        selection = jax.vmap(
            lambda subkey, logits: jax.random.categorical(
                subkey,
                logits,
                shape=(num_particles,),
            ),
            in_axes=(0, 0),
        )(batch_dim_keys, resample_logits)

        def _gather(per_batch_particles: Any, per_batch_selection: jax.Array):
            return jax.tree_util.tree_map(lambda x: x[per_batch_selection], per_batch_particles)

        particles_resampled = jax.vmap(_gather, in_axes=(0, 0))(particles, selection)
        return particles_resampled._replace(
            resample_td_weights=jnp.zeros_like(particles.resample_td_weights),
            gae=particles.gae,
        )

    def calculate_ess_and_entropy(self, td_weights: jax.Array, log_temperature: jax.Array) -> tuple[jax.Array, jax.Array]:
        logits = self.get_resample_logits(td_weights, log_temperature=log_temperature)
        weights = jax.nn.softmax(logits, axis=-1)
        ess = 1.0 / jnp.sum(jnp.square(weights), axis=-1)
        entropy = -jnp.sum(weights * jnp.log(jnp.clip(weights, a_min=1e-8)), axis=-1)
        return ess, entropy
