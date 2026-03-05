from typing import Any, Callable, NamedTuple

import jax


Array = jax.Array
ArrayTree = Any


class ExItTransition(NamedTuple):
    done: Array
    action: Array
    reward: Array
    search_value: Array
    search_policy: Array
    obs: ArrayTree
    info: dict[str, Array]


class AlphaZeroTrainState(NamedTuple):
    params: Any
    actor_opt_state: Any
    critic_opt_state: Any
    buffer_state: Any


SearchApply = Callable[[Any, Array, Any], Any]
RootFnApply = Callable[[Any, ArrayTree, ArrayTree, Array], Any]
