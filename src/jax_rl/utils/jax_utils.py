from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax.jax_utils import replicate, unreplicate


def replicate_tree(tree):
    return replicate(tree)


def unreplicate_tree(tree):
    return unreplicate(tree)


def normalize_restored_train_state_and_key(train_state, key):
    normalized_train_state = jax.device_get(train_state)
    if key is None:
        return normalized_train_state, None

    normalized_key = jax.device_get(key)
    key_array = jnp.asarray(normalized_key)
    if key_array.ndim == 2:
        normalized_key = key_array[0]
    return normalized_train_state, normalized_key
