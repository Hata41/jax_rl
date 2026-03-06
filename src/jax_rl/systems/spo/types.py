from __future__ import annotations

from typing import Any, NamedTuple

import chex
import jax

from ...utils.types import PolicyValueParams


Array = jax.Array
ArrayTree = Any


class CategoricalDualParams(NamedTuple):
    log_temperature: Array
    log_alpha: Array


class SPOParams(NamedTuple):
    actor_online: PolicyValueParams
    actor_target: PolicyValueParams
    critic_online: PolicyValueParams
    critic_target: PolicyValueParams
    dual_params: CategoricalDualParams


class SPOOptStates(NamedTuple):
    actor_opt_state: Any
    critic_opt_state: Any
    dual_opt_state: Any


class SPORootFnOutput(NamedTuple):
    particle_logits: Array
    particle_actions: Array
    particle_state_embedding: ArrayTree
    particle_values: Array


class SPORecurrentFnOutput(NamedTuple):
    reward: Array
    discount: Array
    prior_logits: Array
    value: Array
    next_sampled_action: Array


class Particles(NamedTuple):
    state_embedding: ArrayTree
    root_actions: Array
    resample_td_weights: Array
    prior_logits: Array
    value: Array
    terminal: Array
    depth: Array
    gae: Array


class SPOOutput(NamedTuple):
    action: Array
    sampled_action_weights: Array
    sampled_actions: Array
    value: Array
    sampled_advantages: Array
    generated_state_ids: chex.Array
    rollout_metrics: dict[str, Array]


class SPOTransition(NamedTuple):
    done: Array
    truncated: Array
    action: Array
    sampled_actions: Array
    sampled_actions_weights: Array
    reward: Array
    search_value: Array
    obs: ArrayTree
    bootstrap_obs: ArrayTree
    sampled_advantages: Array
    info: dict[str, Array]


class SPOTrainState(NamedTuple):
    params: SPOParams
    opt_states: SPOOptStates
