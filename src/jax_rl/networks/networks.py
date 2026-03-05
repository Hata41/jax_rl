import inspect
from typing import Mapping, NamedTuple, Sequence, TypeAlias

import distrax
import hydra
import jax
import jax.numpy as jnp
from flax import nnx

from ..utils.exceptions import NetworkTargetResolutionError
from ..utils.types import Array, PolicyValueParams


ObservationInput = Mapping[str, Array] | Array


def _flatten_leaf_with_batch_ndim(leaf: object, batch_ndim: int) -> Array:
    arr = jnp.asarray(leaf, dtype=jnp.float32)
    if batch_ndim <= 0:
        return arr.reshape((-1,))
    leading_shape = arr.shape[:batch_ndim]
    return arr.reshape(leading_shape + (-1,))


def flatten_observation_features(
    obs: ObservationInput,
    batch_ndim: int | None = None,
) -> tuple[Array, Array | None]:
    """Flatten observation leaves into feature vectors while preserving batch axes.

    For mapping observations, all leaves except ``action_mask`` are flattened and
    concatenated on the last dimension. If ``batch_ndim=1`` and leaves have shape
    ``[B, ...]``, the output has shape ``[B, F]`` where ``F`` is the sum of per-leaf
    flattened sizes. For non-mapping observations, the tensor is flattened similarly.
    """
    if isinstance(obs, Mapping):
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
                scale=2.0**0.5,
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
        return jnp.asarray(distrax.Categorical(logits=self.logits).log_prob(actions))

    def entropy(self) -> Array:
        return jnp.asarray(distrax.Categorical(logits=self.logits).entropy())

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
        if isinstance(action_dims, int):
            self._is_discrete = True
            branch_dims = (int(action_dims),)
        else:
            self._is_discrete = False
            branch_dims = tuple(int(v) for v in action_dims)
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


class TransformerBlock(nnx.Module):
    def __init__(self, dim: int, num_heads: int, rngs: nnx.Rngs):
        self.norm_attn = nnx.LayerNorm(num_features=dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            decode=False,
            rngs=rngs,
        )
        self.norm_mlp = nnx.LayerNorm(num_features=dim, rngs=rngs)
        self.mlp_fc1 = _linear(dim, dim * 4, scale=2.0**0.5, rngs=rngs)
        self.mlp_fc2 = _linear(dim * 4, dim, scale=1.0, rngs=rngs)

    def __call__(self, q: Array, k: Array, v: Array, mask: Array | None = None) -> Array:
        q_norm = self.norm_attn(q)
        k_norm = self.norm_attn(k)
        v_norm = self.norm_attn(v)
        attn_mask = None if mask is None else jnp.asarray(mask, dtype=jnp.bool_)
        x = q + self.attn(q_norm, k_norm, v_norm, mask=attn_mask)

        h = self.norm_mlp(x)
        h = self.mlp_fc1(h)
        h = jax.nn.relu(h)
        h = self.mlp_fc2(h)
        return x + h


def _pair_mask(query_mask: Array, key_mask: Array) -> Array:
    q = jnp.expand_dims(jnp.expand_dims(jnp.asarray(query_mask, dtype=jnp.bool_), axis=1), axis=-1)
    k = jnp.expand_dims(jnp.expand_dims(jnp.asarray(key_mask, dtype=jnp.bool_), axis=1), axis=2)
    return jnp.logical_and(q, k)


def _masked_mean(x: Array, mask: Array) -> Array:
    mask_f = jnp.asarray(mask, dtype=jnp.float32)
    mask_exp = jnp.expand_dims(mask_f, axis=-1)
    numerator = jnp.sum(x * mask_exp, axis=1)
    denominator = jnp.sum(mask_f, axis=1, keepdims=True) + jnp.asarray(1e-6, dtype=jnp.float32)
    return numerator / denominator


def _flatten_binpack_logits(score_grid: Array, action_mask: Array) -> Array:
    """Flatten item/EMS scores into action logits with rotation expansion.

    Args:
        score_grid: Pairwise score tensor of shape ``[B, E, I]`` where ``E`` is EMS
            count and ``I`` is item count.
        action_mask: Boolean mask with shape ``[B, A]`` where
            ``A = I * E * R`` for ``R`` rotations.

    Returns:
        Flattened logits of shape ``[B, I * E * R]`` by transposing to
        ``[B, I, E]``, adding a singleton rotation axis, tiling over rotations,
        then reshaping.
    """
    score_grid = jnp.transpose(score_grid, (0, 2, 1))
    score_grid = jnp.expand_dims(score_grid, axis=-1)

    num_items = int(score_grid.shape[1])
    num_ems = int(score_grid.shape[2])
    num_actions = int(action_mask.shape[-1])
    denom = max(num_items * num_ems, 1)
    num_rotations = max(num_actions // denom, 1)

    tiled = jnp.tile(score_grid, (1, 1, 1, num_rotations))
    return tiled.reshape((tiled.shape[0], -1))


class BinPackTorso(nnx.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        rngs: nnx.Rngs,
    ):
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.ems_self_blocks = nnx.List(
            [TransformerBlock(dim=hidden_dim, num_heads=num_heads, rngs=rngs) for _ in range(num_layers)]
        )
        self.item_self_blocks = nnx.List(
            [TransformerBlock(dim=hidden_dim, num_heads=num_heads, rngs=rngs) for _ in range(num_layers)]
        )
        self.ems_cross_blocks = nnx.List(
            [TransformerBlock(dim=hidden_dim, num_heads=num_heads, rngs=rngs) for _ in range(num_layers)]
        )
        self.item_cross_blocks = nnx.List(
            [TransformerBlock(dim=hidden_dim, num_heads=num_heads, rngs=rngs) for _ in range(num_layers)]
        )

    def __call__(
        self,
        ems_embeddings: Array,
        item_embeddings: Array,
        ems_mask: Array,
        item_mask: Array,
    ) -> tuple[Array, Array]:
        ems_self_mask = _pair_mask(ems_mask, ems_mask)
        items_self_mask = _pair_mask(item_mask, item_mask)
        ems_cross_mask = _pair_mask(ems_mask, item_mask)
        items_cross_mask = _pair_mask(item_mask, ems_mask)

        for layer_idx in range(self.num_layers):
            ems_embeddings = self.ems_self_blocks[layer_idx](
                ems_embeddings,
                ems_embeddings,
                ems_embeddings,
                ems_self_mask,
            )
            item_embeddings = self.item_self_blocks[layer_idx](
                item_embeddings,
                item_embeddings,
                item_embeddings,
                items_self_mask,
            )
            ems_embeddings = self.ems_cross_blocks[layer_idx](
                ems_embeddings,
                item_embeddings,
                item_embeddings,
                ems_cross_mask,
            )
            item_embeddings = self.item_cross_blocks[layer_idx](
                item_embeddings,
                ems_embeddings,
                ems_embeddings,
                items_cross_mask,
            )

        return ems_embeddings, item_embeddings


class RustpalletInputAdapterV1(nnx.Module):
    def __init__(
        self,
        hidden_dim: int,
        ems_feature_dim: int,
        item_feature_dim: int,
        rngs: nnx.Rngs,
    ):
        self.ems_embed = _linear(int(ems_feature_dim), int(hidden_dim), scale=2.0**0.5, rngs=rngs)
        self.item_embed = _linear(int(item_feature_dim), int(hidden_dim), scale=2.0**0.5, rngs=rngs)

    def __call__(self, obs: dict) -> tuple[Array, Array, Array, Array]:
        ems_pos = jnp.asarray(obs["ems_pos"], dtype=jnp.float32)
        item_dims = jnp.asarray(obs["item_dims"], dtype=jnp.float32)
        ems_mask = jnp.asarray(obs["ems_mask"], dtype=jnp.bool_)
        item_mask = jnp.asarray(obs["item_mask"], dtype=jnp.bool_)

        ems_embeddings = self.ems_embed(ems_pos)
        item_embeddings = self.item_embed(item_dims)
        return ems_embeddings, item_embeddings, ems_mask, item_mask


class RustpalletInputAdapterV2(nnx.Module):
    def __init__(self, d_model: int, rngs: nnx.Rngs):
        self.d_model = int(d_model)
        self.num_rotations = 6
        self.ems_proj = nnx.Linear(6, self.d_model, rngs=rngs)
        self.item_proj = nnx.Linear(9 + self.d_model, self.d_model, rngs=rngs)

    def _get_sinusoidal_encoding(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float32)
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2, dtype=jnp.float32)
            * (-(jnp.log(10000.0) / max(float(self.d_model), 1.0)))
        )
        phase = x * div_term
        encoding = jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)
        current_dim = int(encoding.shape[-1])
        if current_dim > self.d_model:
            return encoding[..., : self.d_model]
        if current_dim < self.d_model:
            pad_width = [(0, 0)] * encoding.ndim
            pad_width[-1] = (0, self.d_model - current_dim)
            return jnp.pad(encoding, pad_width, mode="constant")
        return encoding

    def __call__(self, obs: dict) -> tuple[Array, Array, Array, Array]:
        eps = jnp.asarray(1e-6, dtype=jnp.float32)

        uld_dims = jnp.asarray(obs["uld_dims"], dtype=jnp.float32)
        max_weight = jnp.asarray(obs["max_weight"], dtype=jnp.float32)

        ems_dims = jnp.asarray(obs["ems_dims"], dtype=jnp.float32)
        ems_pos = jnp.asarray(obs["ems_pos"], dtype=jnp.float32)
        item_dims = jnp.asarray(obs["item_dims"], dtype=jnp.float32)
        item_pos = jnp.asarray(obs["item_pos"], dtype=jnp.float32)
        item_weights = jnp.asarray(obs["item_weights"], dtype=jnp.float32)

        ems_mask = jnp.asarray(obs["ems_mask"], dtype=jnp.bool_)
        item_mask = jnp.asarray(obs["item_mask"], dtype=jnp.bool_)

        safe_uld_dims = jnp.maximum(uld_dims, eps)
        safe_max_weight = jnp.maximum(max_weight, eps)

        norm_ems_dims = ems_dims / safe_uld_dims[:, None, :]
        norm_ems_pos = ems_pos / safe_uld_dims[:, None, :]
        ems_features = jnp.concatenate([norm_ems_dims, norm_ems_pos], axis=-1)

        norm_item_dims = item_dims / safe_uld_dims[:, None, :]
        norm_item_pos = item_pos / safe_uld_dims[:, None, :]
        norm_item_weights = item_weights / safe_max_weight[:, None]

        group_counts = jnp.asarray(obs.get("group_counts", jnp.ones_like(item_weights)), dtype=jnp.float32)
        group_counts_exp = jnp.expand_dims(group_counts, axis=-1)
        log_counts = jnp.log1p(group_counts_exp)
        sin_enc = self._get_sinusoidal_encoding(group_counts_exp)
        is_last = (group_counts_exp == 1).astype(jnp.float32)

        permutations = [
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ]
        permuted_item_dims = jnp.stack(
            [norm_item_dims[..., jnp.asarray(order, dtype=jnp.int32)] for order in permutations],
            axis=2,
        )

        non_dim_item_features = jnp.concatenate(
            [
                norm_item_pos,
                jnp.expand_dims(norm_item_weights, axis=-1),
                log_counts,
                sin_enc,
                is_last,
            ],
            axis=-1,
        )
        repeated_non_dim = jnp.repeat(
            jnp.expand_dims(non_dim_item_features, axis=2),
            repeats=self.num_rotations,
            axis=2,
        )
        item_features = jnp.concatenate([permuted_item_dims, repeated_non_dim], axis=-1)

        batch_size = int(item_features.shape[0])
        num_items = int(item_features.shape[1])
        flat_item_features = item_features.reshape((batch_size, num_items * self.num_rotations, -1))

        expanded_item_mask = jnp.repeat(item_mask[..., None], self.num_rotations, axis=-1).reshape(
            (batch_size, num_items * self.num_rotations)
        )

        ems_embeddings = self.ems_proj(ems_features)
        item_embeddings = self.item_proj(flat_item_features)
        return ems_embeddings, item_embeddings, ems_mask, expanded_item_mask


class BinPackActorHead(nnx.Module):
    def __init__(self, hidden_dim: int, num_rotations: int = 6, rngs: nnx.Rngs | None = None):
        if rngs is None:
            raise ValueError("rngs is required to initialize BinPackActorHead.")
        self.num_rotations = int(num_rotations)
        self.actor_proj = _linear(int(hidden_dim), int(hidden_dim), scale=0.01, rngs=rngs)

    def __call__(self, ems_embeddings: Array, item_embeddings: Array, action_mask: Array) -> Array:
        items_projected = self.actor_proj(item_embeddings)
        score_grid = jnp.matmul(ems_embeddings, jnp.transpose(items_projected, (0, 2, 1)))

        action_mask = jnp.asarray(action_mask, dtype=jnp.bool_)
        num_actions = int(action_mask.shape[-1])
        num_ems = int(score_grid.shape[1])
        token_count = int(score_grid.shape[2])

        if (
            token_count % self.num_rotations == 0
            and token_count * num_ems == num_actions
        ):
            num_items = token_count // self.num_rotations
            score_grid_4d = score_grid.reshape(
                (score_grid.shape[0], num_ems, num_items, self.num_rotations)
            )
            logits = jnp.transpose(score_grid_4d, (0, 2, 1, 3)).reshape((score_grid.shape[0], -1))
        else:
            logits = _flatten_binpack_logits(score_grid, action_mask)

        return jnp.where(action_mask, logits, jnp.asarray(-1e9, dtype=logits.dtype))


class BinPackCriticHead(nnx.Module):
    def __init__(self, hidden_dim: int, rngs: nnx.Rngs):
        self.critic_head = _linear(int(hidden_dim) * 2, 1, scale=1.0, rngs=rngs)

    def __call__(
        self,
        ems_embeddings: Array,
        item_embeddings: Array,
        ems_mask: Array,
        item_mask: Array,
    ) -> Array:
        pooled_ems = _masked_mean(ems_embeddings, ems_mask)
        pooled_items = _masked_mean(item_embeddings, item_mask)
        critic_features = jnp.concatenate([pooled_ems, pooled_items], axis=-1)
        return self.critic_head(critic_features).squeeze(-1)


class ModularPolicyValueModel(nnx.Module):
    def __init__(
        self,
        input_adapter: nnx.Module,
        shared_torso: nnx.Module,
        actor_head: nnx.Module,
        critic_head: nnx.Module,
    ):
        self.input_adapter = input_adapter
        self.shared_torso = shared_torso
        self.actor_head = actor_head
        self.critic_head = critic_head

    def __call__(self, obs: dict):
        ems_embeddings, item_embeddings, ems_mask, item_mask = self.input_adapter(obs)
        ems_embeddings, item_embeddings = self.shared_torso(
            ems_embeddings,
            item_embeddings,
            ems_mask,
            item_mask,
        )

        action_mask = jnp.asarray(obs["action_mask"], dtype=jnp.bool_)
        logits = self.actor_head(ems_embeddings, item_embeddings, action_mask)
        values = self.critic_head(ems_embeddings, item_embeddings, ems_mask, item_mask)
        return logits, values


class BinPackPolicyValueModel(nnx.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        num_heads: int,
        num_layers: int,
        ems_feature_dim: int,
        item_feature_dim: int,
        rngs: nnx.Rngs,
    ):
        self.action_dim = int(action_dim)
        self.input_adapter = RustpalletInputAdapterV1(
            hidden_dim=hidden_dim,
            ems_feature_dim=ems_feature_dim,
            item_feature_dim=item_feature_dim,
            rngs=rngs,
        )
        self.shared_torso = BinPackTorso(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            rngs=rngs,
        )
        self.actor_head = BinPackActorHead(hidden_dim=hidden_dim, num_rotations=6, rngs=rngs)
        self.critic_head = BinPackCriticHead(hidden_dim=hidden_dim, rngs=rngs)

    def __call__(self, obs: dict):
        ems_embeddings, item_embeddings, ems_mask, item_mask = self.input_adapter(obs)
        ems_embeddings, item_embeddings = self.shared_torso(
            ems_embeddings,
            item_embeddings,
            ems_mask,
            item_mask,
        )

        action_mask = jnp.asarray(obs["action_mask"], dtype=jnp.bool_)
        logits = self.actor_head(ems_embeddings, item_embeddings, action_mask)
        values = self.critic_head(ems_embeddings, item_embeddings, ems_mask, item_mask)
        return logits, values


class PolicyValueModel(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dims: int | Sequence[int],
        hidden_sizes: Sequence[int],
        rngs: nnx.Rngs,
        shared_torso: bool = False,
    ):
        torso_out_dim = _torso_out_dim(obs_dim, hidden_sizes)
        self.use_shared_torso = bool(shared_torso)
        if self.use_shared_torso:
            self.shared_torso = Torso(obs_dim, hidden_sizes, rngs=rngs)
        else:
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

        if self.use_shared_torso:
            shared_features = self.shared_torso(obs_features)
            actor_features = shared_features
            critic_features = shared_features
        else:
            actor_features = self.actor_torso(obs_features)
            critic_features = self.critic_torso(obs_features)

        logits = self.actor_head(actor_features)
        if action_mask is not None and not isinstance(logits, tuple):
            valid_mask = action_mask
            if valid_mask.ndim == 1:
                valid_mask = valid_mask[jnp.newaxis, :]
            has_any_valid = jnp.any(valid_mask, axis=-1, keepdims=True)
            safe_mask = jnp.where(has_any_valid, valid_mask, jnp.ones_like(valid_mask, dtype=jnp.bool_))
            logits = jnp.where(safe_mask, logits, jnp.asarray(-1e9, dtype=logits.dtype))

        values = self.critic_head(critic_features).squeeze(-1)
        return logits, values


def _distribution_from_logits(logits):
    if isinstance(logits, tuple):
        return MultiDiscretePolicyDist(logits_per_dim=logits)
    return CategoricalPolicyDist(logits=logits)


DistributionLike: TypeAlias = CategoricalPolicyDist | MultiDiscretePolicyDist


def _instantiate_target_tree(
    config: object,
    shared_rngs: nnx.Rngs,
    runtime_values: Mapping[str, object] | None = None,
) -> object:
    if isinstance(config, list):
        return [_instantiate_target_tree(value, shared_rngs, None) for value in config]

    if not isinstance(config, Mapping):
        return config

    if "_target_" not in config:
        return {
            key: _instantiate_target_tree(value, shared_rngs, None)
            for key, value in config.items()
            if key != "_delete_"
        }

    target = config.get("_target_")
    if not isinstance(target, str) or not target.strip():
        raise NetworkTargetResolutionError(
            "Invalid network sub-config: missing or empty '_target_' key."
        )

    try:
        target_cls = hydra.utils.get_class(target)
    except Exception as exc:
        raise NetworkTargetResolutionError(
            f"Invalid nested '_target_' value '{target}': could not resolve import path."
        ) from exc

    constructor_sig = inspect.signature(target_cls.__init__)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in constructor_sig.parameters.values()
    )

    kwargs = {
        key: _instantiate_target_tree(value, shared_rngs, None)
        for key, value in config.items()
        if not key.startswith("_") and (accepts_kwargs or key in constructor_sig.parameters)
    }
    if "rngs" not in kwargs and (accepts_kwargs or "rngs" in constructor_sig.parameters):
        kwargs["rngs"] = shared_rngs

    if runtime_values is not None:
        runtime_kwargs = {
            name: value
            for name, value in runtime_values.items()
            if value is not None and name not in kwargs
            and (accepts_kwargs or name in constructor_sig.parameters)
        }
        kwargs.update(runtime_kwargs)

    instantiate_config = {key: value for key, value in config.items() if key.startswith("_")}
    instantiate_config = {key: value for key, value in instantiate_config.items() if key != "_delete_"}
    if instantiate_config.get("_partial_"):
        raise NetworkTargetResolutionError(
            "Network configs with '_partial_=True' are not supported for model instantiation."
        )
    try:
        return target_cls(**kwargs)
    except Exception as exc:
        raise NetworkTargetResolutionError(
            "Failed to instantiate network target from config tree. "
            f"target='{target}'."
        ) from exc


def init_policy_value_params(
    key: Array,
    network_config: Mapping[str, object],
    obs_dim: int,
    action_dims: int | Sequence[int],
    ems_feature_dim: int = 6,
    item_feature_dim: int = 3,
) -> PolicyValueParams:
    if not isinstance(network_config, Mapping):
        raise NetworkTargetResolutionError(
            "Invalid 'network' config: expected a mapping with a '_target_' field, "
            f"got {type(network_config).__name__}."
        )

    target = network_config.get("_target_")
    if not isinstance(target, str) or not target.strip():
        raise NetworkTargetResolutionError(
            "Invalid 'network' config: missing or empty '_target_' key. "
            "Define the model class path in training.network._target_."
        )

    try:
        model_cls = hydra.utils.get_class(target)
    except Exception as exc:
        raise NetworkTargetResolutionError(
            f"Invalid 'network._target_' value '{target}': could not resolve import path."
        ) from exc

    shared_rngs = nnx.Rngs(key)
    runtime_values = {
        "obs_dim": int(obs_dim),
        "action_dims": action_dims,
        "action_dim": int(action_dims) if isinstance(action_dims, int) else None,
        "ems_feature_dim": int(ems_feature_dim),
        "item_feature_dim": int(item_feature_dim),
        "rngs": shared_rngs,
    }
    try:
        model = _instantiate_target_tree(
            config={key: value for key, value in network_config.items() if key != "_delete_"},
            shared_rngs=shared_rngs,
            runtime_values=runtime_values,
        )
    except Exception as exc:
        raise NetworkTargetResolutionError(
            "Failed to instantiate network from config. "
            f"target='{target}'."
        ) from exc

    if not isinstance(model, nnx.Module):
        raise NetworkTargetResolutionError(
            f"Configured network target '{target}' must instantiate to a flax.nnx.Module, "
            f"got {type(model).__name__}."
        )

    graphdef, state = nnx.split(model)
    return PolicyValueParams(graphdef=graphdef, state=state)


def policy_value_apply(
    graphdef: nnx.GraphDef,
    state: nnx.State,
    obs: dict | Array,
) -> tuple[DistributionLike, Array]:
    model = nnx.merge(graphdef, state)
    logits, values = model(obs)
    return _distribution_from_logits(logits), jnp.asarray(values)