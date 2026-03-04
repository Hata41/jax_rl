from typing import Any, NamedTuple, Sequence

import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from .types import Array, PolicyValueParams


def _flatten_leaf_with_batch_ndim(leaf: Any, batch_ndim: int) -> Array:
    arr = jnp.asarray(leaf, dtype=jnp.float32)
    if batch_ndim <= 0:
        return arr.reshape((-1,))
    leading_shape = arr.shape[:batch_ndim]
    return arr.reshape(leading_shape + (-1,))


def flatten_observation_features(obs: Any, batch_ndim: int | None = None) -> tuple[Array, Array | None]:
    if isinstance(obs, dict):
        action_mask = obs.get("action_mask")
        obs_without_mask = {key: value for key, value in obs.items() if key != "action_mask"}
        leaves, _ = jax.tree_util.tree_flatten(obs_without_mask)
        if action_mask is not None:
            inferred_batch_ndim = max(jnp.asarray(action_mask).ndim - 1, 0)
        elif leaves:
            inferred_batch_ndim = 1 if jnp.asarray(leaves[0]).ndim > 1 else 0
        else:
            inferred_batch_ndim = 0
        batch_ndim = inferred_batch_ndim if batch_ndim is None else batch_ndim

        if leaves:
            flat_leaves = [_flatten_leaf_with_batch_ndim(leaf, batch_ndim) for leaf in leaves]
            features = jnp.concatenate(flat_leaves, axis=-1)
        elif action_mask is not None:
            features = _flatten_leaf_with_batch_ndim(action_mask, batch_ndim)
        else:
            raise ValueError("Observation dict is empty and cannot be flattened.")

        normalized_action_mask = None
        if action_mask is not None:
            normalized_action_mask = jnp.asarray(action_mask, dtype=jnp.bool_)
        return features, normalized_action_mask

    obs_arr = jnp.asarray(obs, dtype=jnp.float32)
    if batch_ndim is None:
        batch_ndim = 1 if obs_arr.ndim > 1 else 0
    if batch_ndim <= 0:
        return obs_arr.reshape((-1,)), None
    leading_shape = obs_arr.shape[:batch_ndim]
    return obs_arr.reshape(leading_shape + (-1,)), None


def _linear(
    in_dim: int,
    out_dim: int,
    scale: float,
    rngs: nnx.Rngs,
) -> nnx.Linear:
    return nnx.Linear(
        in_features=in_dim,
        out_features=out_dim,
        use_bias=True,
        kernel_init=jax.nn.initializers.orthogonal(scale),
        bias_init=jax.nn.initializers.zeros,
        rngs=rngs,
    )


class Torso(nnx.Module):
    def __init__(self, in_dim: int, hidden_sizes: Sequence[int], rngs: nnx.Rngs):
        sizes = (in_dim, *hidden_sizes)
        self.layers = nnx.List(
            [
            _linear(
                in_dim=d_in,
                out_dim=d_out,
                scale=jnp.sqrt(2.0),
                rngs=rngs,
            )
            for d_in, d_out in zip(sizes[:-1], sizes[1:])
            ]
        )

    def __call__(self, x: Array) -> Array:
        h = x
        for layer in self.layers:
            h = jnp.tanh(layer(h))
        return h


class CategoricalPolicyDist(NamedTuple):
    logits: Array

    def sample(self, key: Array) -> Array:
        return distrax.Categorical(logits=self.logits).sample(seed=key)

    def log_prob(self, actions: Array) -> Array:
        actions = jnp.asarray(actions, dtype=jnp.int32)
        return distrax.Categorical(logits=self.logits).log_prob(actions)

    def entropy(self) -> Array:
        return distrax.Categorical(logits=self.logits).entropy()

    def mode(self) -> Array:
        return jnp.argmax(self.logits, axis=-1)


class MultiDiscretePolicyDist(NamedTuple):
    logits_per_dim: tuple[Array, ...]

    def sample(self, key: Array) -> Array:
        keys = jax.random.split(key, len(self.logits_per_dim))
        samples = [
            distrax.Categorical(logits=logits).sample(seed=sample_key)
            for sample_key, logits in zip(keys, self.logits_per_dim)
        ]
        return jnp.stack(samples, axis=-1)

    def log_prob(self, actions: Array) -> Array:
        actions = jnp.asarray(actions, dtype=jnp.int32)
        per_dim_log_probs = [
            distrax.Categorical(logits=logits).log_prob(actions[..., dim_idx])
            for dim_idx, logits in enumerate(self.logits_per_dim)
        ]
        return jnp.sum(jnp.stack(per_dim_log_probs, axis=-1), axis=-1)

    def entropy(self) -> Array:
        per_dim_entropy = [
            distrax.Categorical(logits=logits).entropy() for logits in self.logits_per_dim
        ]
        return jnp.sum(jnp.stack(per_dim_entropy, axis=-1), axis=-1)

    def mode(self) -> Array:
        per_dim_mode = [jnp.argmax(logits, axis=-1) for logits in self.logits_per_dim]
        return jnp.stack(per_dim_mode, axis=-1)


def _torso_out_dim(obs_dim: int, hidden_sizes: Sequence[int]) -> int:
    return obs_dim if len(hidden_sizes) == 0 else int(hidden_sizes[-1])


class ActorHead(nnx.Module):
    def __init__(self, in_dim: int, action_dims: int | Sequence[int], rngs: nnx.Rngs):
        self._is_discrete = isinstance(action_dims, int)
        branch_dims = (int(action_dims),) if self._is_discrete else tuple(int(v) for v in action_dims)
        self.branches = nnx.List(
            [
                _linear(
                    in_dim=in_dim,
                    out_dim=branch_dim,
                    scale=0.01,
                    rngs=rngs,
                )
                for branch_dim in branch_dims
            ]
        )

    def __call__(self, features: Array):
        if self._is_discrete:
            return self.branches[0](features)
        return tuple(branch(features) for branch in self.branches)


class PolicyValueModel(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dims: int | Sequence[int],
        hidden_sizes: Sequence[int],
        rngs: nnx.Rngs,
    ):
        torso_out_dim = _torso_out_dim(obs_dim, hidden_sizes)
        self.actor_torso = Torso(obs_dim, hidden_sizes, rngs=rngs)
        self.critic_torso = Torso(obs_dim, hidden_sizes, rngs=rngs)
        self.actor_head = ActorHead(torso_out_dim, action_dims, rngs=rngs)
        self.critic_head = _linear(
            in_dim=torso_out_dim,
            out_dim=1,
            scale=1.0,
            rngs=rngs,
        )

    def __call__(self, obs: dict | Array):
        obs_features, action_mask = flatten_observation_features(obs)

        actor_features = self.actor_torso(obs_features)
        logits = self.actor_head(actor_features)
        if action_mask is not None and not isinstance(logits, tuple):
            valid_mask = action_mask
            if valid_mask.ndim == 1:
                valid_mask = valid_mask[jnp.newaxis, :]
            has_any_valid = jnp.any(valid_mask, axis=-1, keepdims=True)
            safe_mask = jnp.where(has_any_valid, valid_mask, jnp.ones_like(valid_mask, dtype=jnp.bool_))
            logits = jnp.where(safe_mask, logits, jnp.asarray(-1e9, dtype=logits.dtype))

        critic_features = self.critic_torso(obs_features)
        values = self.critic_head(critic_features).squeeze(-1)
        return logits, values


def _distribution_from_logits(logits):
    if isinstance(logits, tuple):
        return MultiDiscretePolicyDist(logits_per_dim=logits)
    return CategoricalPolicyDist(logits=logits)


def init_policy_value_params(
    key: Array,
    obs_dim: int,
    action_dims: int | Sequence[int],
    hidden_sizes: Sequence[int],
) -> PolicyValueParams:
    model = PolicyValueModel(
        obs_dim=obs_dim,
        action_dims=action_dims,
        hidden_sizes=hidden_sizes,
        rngs=nnx.Rngs(key),
    )
    graphdef, state = nnx.split(model)
    return PolicyValueParams(graphdef=graphdef, state=state)


def policy_value_apply(graphdef: nnx.GraphDef, state: nnx.State, obs: dict | Array):
    model = nnx.merge(graphdef, state)
    logits, values = model(obs)
    return _distribution_from_logits(logits), values