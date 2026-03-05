from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

import chex
import jax
from flax import nnx

Array = jax.Array
ArrayTree = chex.ArrayTree
PRNGKey = chex.PRNGKey
Params = Any


class LogEvent(Enum):
    ACT = "act"
    TRAIN = "train"
    EVAL = "eval"
    ABSOLUTE = "absolute"
    MISC = "misc"


class PolicyValueParams(NamedTuple):
    graphdef: nnx.GraphDef
    state: nnx.State


class TrainState(NamedTuple):
    params: PolicyValueParams
    actor_opt_state: Any
    critic_opt_state: Any


class Transition(NamedTuple):
    obs: Array
    actions: Array
    log_probs: Array
    rewards: Array
    dones: Array
    truncated: Array
    values: Array
    bootstrap_values: Array


class RolloutBatch(NamedTuple):
    obs: Array
    actions: Array
    log_probs: Array
    rewards: Array
    dones: Array
    truncated: Array
    values: Array
    bootstrap_values: Array


class FlattenBatch(NamedTuple):
    obs: Array
    actions: Array
    old_log_probs: Array
    old_values: Array
    advantages: Array
    returns: Array


class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: Any
    obs: Array
    key: Array


@dataclass
class SystemComponents:
    runner_state: RunnerState
    env: Any
    env_params: Any
    actor_optimizer: Any
    critic_optimizer: Any
    checkpointer: Any
    start_update: int
    num_devices: int
    num_envs_per_device: int