import jax
import jax.numpy as jnp
import optax

from .advantages import compute_gae
from .losses import ppo_loss
from ...utils.types import FlattenBatch, RolloutBatch, TrainState


def _total_opt_steps(config) -> int:
    num_minibatches = config.rollout_batch_size // config.system.minibatch_size
    return config.num_updates * config.system.update_epochs * num_minibatches


def _path_token(entry) -> str:
    if hasattr(entry, "name"):
        return str(entry.name)
    if hasattr(entry, "key"):
        return str(entry.key)
    if hasattr(entry, "idx"):
        return str(entry.idx)
    return str(entry)


def _matches_module_prefix(path, module_prefixes: tuple[str, ...]) -> bool:
    return any(
        _path_token(entry).startswith(module_prefix)
        for entry in path
        for module_prefix in module_prefixes
    )


def _zero_out_except_module(tree, module_prefixes: tuple[str, ...]):
    known_prefixes = ("actor_", "critic_", "shared_", "input_adapter")

    def _validate_prefixes(path, leaf):
        del leaf
        if not _matches_module_prefix(path, known_prefixes):
            path_tokens = "/".join(_path_token(entry) for entry in path)
            raise ValueError(
                "Unrecognized parameter prefix encountered during gradient filtering: "
                f"{path_tokens}. Expected one of {known_prefixes}."
            )
        return None

    jax.tree_util.tree_map_with_path(_validate_prefixes, tree)

    return jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaf
        if _matches_module_prefix(path, module_prefixes)
        else jnp.zeros_like(leaf),
        tree,
    )


def _base_optimizer(learning_rate: float, config):
    total_opt_steps = _total_opt_steps(config)
    transition_steps = max(total_opt_steps - 1, 1)

    schedule_name = str(config.system.lr_schedule).lower()
    if schedule_name == "linear":
        schedule = optax.linear_schedule(
            init_value=learning_rate,
            end_value=0.0,
            transition_steps=transition_steps,
        )
    elif schedule_name == "cosine":
        schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=transition_steps,
        )
    elif schedule_name == "constant":
        schedule = learning_rate
    else:
        raise ValueError(
            "Unsupported lr_schedule "
            f"'{config.system.lr_schedule}'. Expected one of: linear, constant, cosine."
        )

    optimizer_name = str(config.system.optimizer).lower()
    if optimizer_name in {"adam", "adamw", "sgd", "rmsprop"}:
        optimizer_ctor = getattr(optax, optimizer_name)
        base_opt = optax.inject_hyperparams(optimizer_ctor)(learning_rate=schedule)
    elif optimizer_name in {"schedule_free_adamw", "schedule_free_sgd"}:
        schedule_free_ctor = getattr(optax.contrib, optimizer_name, None)
        if schedule_free_ctor is None:
            raise ValueError(
                f"Optimizer '{optimizer_name}' is not available in optax.contrib for this environment."
            )
        base_opt = optax.inject_hyperparams(schedule_free_ctor)(learning_rate=learning_rate)
    else:
        raise ValueError(
            "Unsupported optimizer "
            f"'{config.system.optimizer}'. Expected one of: "
            "adam, adamw, sgd, rmsprop, schedule_free_adamw, schedule_free_sgd."
        )

    return optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        base_opt,
    )


def make_actor_optimizer(config):
    return _base_optimizer(config.system.actor_lr, config)


def make_critic_optimizer(config):
    return _base_optimizer(config.system.critic_lr, config)


def _flatten_batch(
    rollout_batch: RolloutBatch,
    advantages: jax.Array,
    returns: jax.Array,
) -> FlattenBatch:
    action_shape = rollout_batch.actions.shape[2:]
    return FlattenBatch(
        obs=jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]),
            rollout_batch.obs,
        ),
        actions=rollout_batch.actions.reshape((-1,) + action_shape),
        old_log_probs=rollout_batch.log_probs.reshape((-1,)),
        old_values=rollout_batch.values.reshape((-1,)),
        advantages=advantages.reshape((-1,)),
        returns=returns.reshape((-1,)),
    )


def ppo_update(
    train_state: TrainState,
    rollout_batch: RolloutBatch,
    last_values,
    key,
    config,
    actor_optimizer,
    critic_optimizer,
):
    advantages, returns = compute_gae(
        rewards=rollout_batch.rewards,
        dones=rollout_batch.dones,
        truncated=rollout_batch.truncated,
        values=rollout_batch.values,
        last_values=last_values,
        gamma=config.system.gamma,
        gae_lambda=config.system.gae_lambda,
        bootstrap_values=rollout_batch.bootstrap_values,
    )
    advantages = jnp.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
    returns = jnp.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    adv_mean = jnp.mean(advantages)
    adv_std = jnp.maximum(jnp.std(advantages), jnp.asarray(1e-8, dtype=advantages.dtype))
    advantages = (advantages - adv_mean) / adv_std
    dataset = _flatten_batch(rollout_batch, advantages, returns)

    num_devices = max(jax.local_device_count(), 1)
    if config.system.minibatch_size % num_devices != 0:
        raise ValueError(
            "minibatch_size must be divisible by local device count, "
            f"got minibatch_size={config.system.minibatch_size} and num_devices={num_devices}."
        )
    local_minibatch_size = config.system.minibatch_size // num_devices
    batch_size = rollout_batch.rewards.shape[0] * rollout_batch.rewards.shape[1]
    num_minibatches = batch_size // local_minibatch_size

    def epoch_step(carry, _):
        state, curr_key = carry
        curr_key, perm_key = jax.random.split(curr_key)
        permutation = jax.random.permutation(perm_key, batch_size)
        shuffled = jax.tree_util.tree_map(lambda x: x[permutation], dataset)
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((num_minibatches, local_minibatch_size) + x.shape[1:]),
            shuffled,
        )

        def minibatch_step(curr_state, minibatch):
            loss_fn = lambda state: ppo_loss(
                curr_state.params.graphdef,
                state,
                minibatch,
                config.system.clip_epsilon,
                config.system.value_coef,
                config.system.entropy_coef,
            )
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                curr_state.params.state
            )
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
                curr_state.actor_opt_state,
                curr_state.params.state,
            )
            critic_updates, next_critic_opt_state = critic_optimizer.update(
                critic_grads,
                curr_state.critic_opt_state,
                curr_state.params.state,
            )

            merged_updates = jax.tree_util.tree_map(
                lambda actor_u, critic_u: actor_u + critic_u,
                actor_updates,
                critic_updates,
            )
            next_params_state = optax.apply_updates(curr_state.params.state, merged_updates)
            next_state = TrainState(
                params=curr_state.params._replace(state=next_params_state),
                actor_opt_state=next_actor_opt_state,
                critic_opt_state=next_critic_opt_state,
            )
            metrics = dict(metrics)
            metrics["loss_total"] = loss
            metrics = jax.tree_util.tree_map(
                lambda x: jax.lax.pmean(x, axis_name="device"),
                metrics,
            )
            return next_state, metrics

        next_state, minibatch_metrics = jax.lax.scan(minibatch_step, state, minibatches)
        epoch_metrics = jax.tree_util.tree_map(jnp.mean, minibatch_metrics)
        return (next_state, curr_key), epoch_metrics

    (next_train_state, next_key), epoch_metrics = jax.lax.scan(
        epoch_step,
        (train_state, key),
        xs=None,
        length=config.system.update_epochs,
    )
    metrics = jax.tree_util.tree_map(jnp.mean, epoch_metrics)
    return next_train_state, metrics, next_key