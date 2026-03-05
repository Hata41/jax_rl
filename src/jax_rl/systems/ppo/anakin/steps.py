import jax
import jax.numpy as jnp

from ....utils.types import RunnerState
from ..rollout import collect_rollout
from ..update import ppo_update


def make_ppo_steps(config, env, env_params, actor_optimizer, critic_optimizer):
    """Build and pmap PPO rollout/update steps for the device axis."""

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
    return pmap_rollout, pmap_update
