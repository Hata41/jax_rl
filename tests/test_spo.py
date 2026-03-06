import jax
import jax.numpy as jnp
import pytest
from stoa.env_types import StepType, TimeStep
import re

from jax_rl.configs.config import (
    ArchConfig,
    CheckpointConfig,
    EnvConfig,
    ExperimentConfig,
    LoggingConfig,
    SystemConfig,
)
from jax_rl.envs.env import make_stoa_env
from jax_rl.networks import init_policy_value_params
from jax_rl.systems.alphazero.steps import extract_root_embedding
from jax_rl.systems.spo.losses import categorical_mpo_loss
from jax_rl.systems.spo.anakin.steps import _critic_loss
from jax_rl.systems.spo.steps import SPO, make_recurrent_fn, make_root_fn
from jax_rl.systems.spo.types import CategoricalDualParams, Particles, SPOParams, SPORecurrentFnOutput
from jax_rl.systems.spo.anakin.system import train
from jax_rl.utils.shapes import space_feature_dim, space_flat_dim


class _MockRustpoolEnv:
    def __init__(self):
        self.flat_shape = None
        self.released_ids = []
        self.num_envs_per_device = 2

    def snapshot(self, state, env_ids):
        del state
        return env_ids

    def simulate_batch(self, state, state_ids, actions):
        del state
        self.flat_shape = tuple(state_ids.shape)
        assert state_ids.ndim == 1
        assert actions.ndim == 1
        n = state_ids.shape[0]
        timestep = TimeStep(
            step_type=jnp.full((n,), StepType.MID, dtype=jnp.int8),
            reward=jnp.ones((n,), dtype=jnp.float32),
            discount=jnp.ones((n,), dtype=jnp.float32),
            observation={
                "ems_pos": jnp.ones((n, 2), dtype=jnp.float32),
                "item_dims": jnp.ones((n, 3), dtype=jnp.float32),
                "item_mask": jnp.ones((n, 3), dtype=jnp.bool_),
                "action_mask": jnp.ones((n, 5), dtype=jnp.bool_),
            },
            extras={"state_id": state_ids + 1},
        )
        return state_ids, timestep

    def release_batch(self, state, state_ids):
        del state
        self.released_ids.append(jnp.asarray(state_ids, dtype=jnp.int32))
        return state_ids


class _DummyDist:
    def __init__(self, logits):
        self.logits = logits

    def log_prob(self, actions):
        logp = jax.nn.log_softmax(self.logits, axis=-1)
        return jnp.take_along_axis(logp, actions[..., None], axis=-1).squeeze(-1)

    def entropy(self):
        p = jax.nn.softmax(self.logits, axis=-1)
        return -jnp.sum(p * jnp.log(jnp.clip(p, a_min=1e-8)), axis=-1)


class _LogitsOnlyDist:
    def __init__(self, logits):
        self.logits = logits



def _make_spo_params(obs_dim=8, action_dim=5):
    actor = init_policy_value_params(
        key=jax.random.PRNGKey(0),
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [8]},
        obs_dim=obs_dim,
        action_dims=action_dim,
    )
    critic = init_policy_value_params(
        key=jax.random.PRNGKey(1),
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [8]},
        obs_dim=obs_dim,
        action_dims=action_dim,
    )
    return SPOParams(
        actor_online=actor,
        actor_target=actor,
        critic_online=critic,
        critic_target=critic,
        dual_params=CategoricalDualParams(
            log_temperature=jnp.asarray(-2.0, dtype=jnp.float32),
            log_alpha=jnp.asarray(-2.0, dtype=jnp.float32),
        ),
    )


def _make_config(num_particles=8, search_depth=4):
    return ExperimentConfig(
        env=EnvConfig(env_name="rustpool:BinPack-v0"),
        arch=ArchConfig(total_timesteps=16, num_envs=4, num_steps=1),
        system=SystemConfig(
            name="spo",
            num_particles=num_particles,
            search_depth=search_depth,
            warmup_steps=0,
            learner_updates_per_cycle=1,
            total_buffer_size=64,
            total_batch_size=16,
            sample_sequence_length=1,
            period=1,
        ),
        checkpointing=CheckpointConfig(save_interval_steps=0),
        logging=LoggingConfig(tensorboard_logdir=None),
        evaluations={},
        network={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [8]},
    )


def _infer_action_dims(action_space) -> int | tuple[int, ...]:
    if isinstance(action_space, tuple) and len(action_space) == 1:
        return int(action_space[0])
    if isinstance(action_space, list) and len(action_space) == 1:
        return int(action_space[0])
    if hasattr(action_space, "num_values"):
        num_values = jnp.asarray(action_space.num_values)
        if num_values.ndim == 0:
            return int(num_values)
        return tuple(int(v) for v in num_values.tolist())
    raise NotImplementedError("Action space not supported. Only Discrete and MultiDiscrete are allowed.")


def _make_uld_spo_setup(num_envs_per_device: int = 2):
    env_kwargs = {
        "max_items": 10,
        "max_ems": 40,
        "max_episode_steps": 30,
        "uld_preset": "PMC-PLD",
        "generator_type": "twophase_physics",
        "target_groups": 10,
        "max_mult": 5,
        "prob_split_one_item": 0.3,
        "split_num_same_items": 6,
        "min_dimension": 100,
        "min_density": 1e-9,
        "max_density": 1e-2,
        "dataset_path": "data/eval_data.csv",
        "reward_type": "dense",
        "action_mode": "Oriented",
        "item_representation": "Group",
        "parallel_strategy": "None",
    }

    env, env_params = make_stoa_env(
        "rlpallet:UldEnv-v2",
        num_envs_per_device=num_envs_per_device,
        env_kwargs=env_kwargs,
    )

    obs_space = env.observation_space(env_params)
    action_dims = _infer_action_dims(env.action_space(env_params))
    params = init_policy_value_params(
        key=jax.random.PRNGKey(123),
        network_config={
            "_target_": "jax_rl.networks.ModularPolicyValueModel",
            "input_adapter": {
                "_target_": "jax_rl.networks.RustpalletInputAdapterV2",
                "d_model": 16,
            },
            "shared_torso": {
                "_target_": "jax_rl.networks.BinPackTorso",
                "hidden_dim": 16,
                "num_heads": 1,
                "num_layers": 1,
            },
            "actor_head": {
                "_target_": "jax_rl.networks.BinPackActorHead",
                "hidden_dim": 16,
                "num_rotations": 6,
            },
            "critic_head": {
                "_target_": "jax_rl.networks.BinPackCriticHead",
                "hidden_dim": 16,
            },
        },
        obs_dim=space_flat_dim(obs_space),
        action_dims=action_dims,
        ems_feature_dim=space_feature_dim(obs_space, "ems_pos", default=6),
        item_feature_dim=space_feature_dim(obs_space, "item_dims", default=3),
    )

    spo_params = SPOParams(
        actor_online=params,
        actor_target=params,
        critic_online=params,
        critic_target=params,
        dual_params=CategoricalDualParams(
            log_temperature=jnp.asarray(-2.0, dtype=jnp.float32),
            log_alpha=jnp.asarray(-2.0, dtype=jnp.float32),
        ),
    )

    config = ExperimentConfig(
        env=EnvConfig(env_name="rlpallet:UldEnv-v2", env_kwargs=env_kwargs),
        arch=ArchConfig(total_timesteps=16, num_envs=num_envs_per_device, num_steps=1),
        system=SystemConfig(name="spo", num_particles=32, search_depth=2),
        logging=LoggingConfig(tensorboard_logdir=None),
        checkpointing=CheckpointConfig(save_interval_steps=0),
        evaluations={},
    )

    return env, env_params, spo_params, config


def _release_generated_ids(env, generated_ids: jax.Array, chunk_size: int):
    flat_ids = jnp.asarray(generated_ids, dtype=jnp.int32).reshape(-1)
    valid_mask = flat_ids > 0
    safe_ids = jnp.where(valid_mask, flat_ids, -1)
    remainder = safe_ids.shape[0] % chunk_size
    pad_size = (chunk_size - remainder) % chunk_size
    padded_ids = jnp.pad(safe_ids, (0, pad_size), constant_values=-1)
    chunks = padded_ids.reshape((-1, chunk_size))
    dummy_state = jnp.zeros((chunk_size,), dtype=jnp.int32)
    for idx in range(int(chunks.shape[0])):
        _ = env.release_batch(dummy_state, chunks[idx])


def _run_minimal_spo_uld_search_steps(num_steps: int = 4):
    env, env_params, spo_params, config = _make_uld_spo_setup(num_envs_per_device=2)
    config = config.__class__(
        env=config.env,
        arch=ArchConfig(total_timesteps=16, num_envs=2, num_steps=1),
        system=SystemConfig(
            name="spo",
            num_particles=2,
            search_depth=2,
            warmup_steps=0,
            total_buffer_size=16,
            total_batch_size=4,
            sample_sequence_length=1,
            period=1,
            learner_updates_per_cycle=1,
        ),
        checkpointing=CheckpointConfig(save_interval_steps=0),
        logging=LoggingConfig(tensorboard_logdir=None),
        evaluations={},
        network=config.network,
    )
    root_fn = make_root_fn(config)
    recurrent_fn = make_recurrent_fn(
        env=env,
        env_params=env_params,
        gamma=float(config.system.search_gamma),
        is_rustpool=True,
    )
    search = SPO(config, recurrent_fn)

    key = jax.random.PRNGKey(321)
    env_state, timestep = env.reset(key, env_params)
    obs = timestep.observation
    action_mask_history = []
    action_history = []

    for step_idx in range(num_steps):
        root_embedding = extract_root_embedding(
            env=env,
            env_state=env_state,
            obs=obs,
            is_rustpool=True,
        )
        root = root_fn(spo_params, obs, root_embedding, jax.random.PRNGKey(5000 + step_idx))
        search_output = search.search(spo_params, jax.random.PRNGKey(6000 + step_idx), root)

        action = jnp.asarray(search_output.action, dtype=jnp.int32)
        action_history.append(action)
        action_mask_history.append(jnp.asarray(obs["action_mask"], dtype=jnp.bool_))

        if hasattr(env, "release_batch"):
            _release_generated_ids(
                env,
                generated_ids=jnp.asarray(search_output.generated_state_ids, dtype=jnp.int32),
                chunk_size=int(obs["action_mask"].shape[0]),
            )

        env_state, timestep = env.step(env_state, action, env_params)
        obs = timestep.observation

    return action_history, action_mask_history


def test_spo_recurrent_fn_shape_handling():
    env = _MockRustpoolEnv()
    params = _make_spo_params()
    recurrent_fn = make_recurrent_fn(
        env=env,
        env_params=None,
        gamma=0.99,
        is_rustpool=True,
    )

    batch_size, num_particles = 4, 8
    action = jnp.zeros((batch_size, num_particles), dtype=jnp.int32)
    state_embedding = jnp.arange(batch_size * num_particles, dtype=jnp.int32).reshape(batch_size, num_particles)

    recurrent_output, next_embedding = recurrent_fn(params, jax.random.PRNGKey(0), action, state_embedding)

    assert env.flat_shape == (32,)
    assert next_embedding.shape == (4, 8)
    assert recurrent_output.reward.shape == (4, 8)


def test_spo_recurrent_fn_respects_action_mask_for_sampled_actions(monkeypatch):
    env = _MockRustpoolEnv()
    params = _make_spo_params()

    def _fake_policy_value_apply(graphdef, state, obs):
        del graphdef, state
        if isinstance(obs, dict):
            batch = int(jnp.asarray(obs["action_mask"]).shape[0])
            action_dim = int(jnp.asarray(obs["action_mask"]).shape[-1])
        else:
            batch = int(jnp.asarray(obs).shape[0])
            action_dim = 5
        logits = jnp.broadcast_to(
            jnp.linspace(5.0, 1.0, action_dim, dtype=jnp.float32)[None, :],
            (batch, action_dim),
        )
        values = jnp.zeros((batch,), dtype=jnp.float32)
        return _LogitsOnlyDist(logits), values

    monkeypatch.setattr("jax_rl.systems.spo.steps.policy_value_apply", _fake_policy_value_apply)

    def _simulate_with_mask(state, state_ids, actions):
        del state, actions
        n = state_ids.shape[0]
        action_mask = jnp.zeros((n, 5), dtype=jnp.bool_).at[:, 3].set(True)
        timestep = TimeStep(
            step_type=jnp.full((n,), StepType.MID, dtype=jnp.int8),
            reward=jnp.zeros((n,), dtype=jnp.float32),
            discount=jnp.ones((n,), dtype=jnp.float32),
            observation={
                "ems_pos": jnp.ones((n, 2), dtype=jnp.float32),
                "item_dims": jnp.ones((n, 3), dtype=jnp.float32),
                "item_mask": jnp.ones((n, 3), dtype=jnp.bool_),
                "action_mask": action_mask,
            },
            extras={"state_id": state_ids + 1},
        )
        return state_ids, timestep

    monkeypatch.setattr(env, "simulate_batch", _simulate_with_mask)

    recurrent_fn = make_recurrent_fn(
        env=env,
        env_params=None,
        gamma=0.99,
        is_rustpool=True,
    )

    batch_size, num_particles = 4, 8
    actions = jnp.zeros((batch_size, num_particles), dtype=jnp.int32)
    state_embedding = jnp.arange(batch_size * num_particles, dtype=jnp.int32).reshape(batch_size, num_particles)

    recurrent_output, _ = recurrent_fn(params, jax.random.PRNGKey(0), actions, state_embedding)
    assert jnp.all(jnp.asarray(recurrent_output.next_sampled_action, dtype=jnp.int32) == 3)


def test_spo_recurrent_fn_marks_non_positive_state_ids_as_terminal(monkeypatch):
    env = _MockRustpoolEnv()
    params = _make_spo_params()

    def _fake_policy_value_apply(graphdef, state, obs):
        del graphdef, state
        if isinstance(obs, dict):
            batch = int(jnp.asarray(obs["action_mask"]).shape[0])
            action_dim = int(jnp.asarray(obs["action_mask"]).shape[-1])
        else:
            batch = int(jnp.asarray(obs).shape[0])
            action_dim = 5
        logits = jnp.zeros((batch, action_dim), dtype=jnp.float32)
        values = jnp.ones((batch,), dtype=jnp.float32)
        return _LogitsOnlyDist(logits), values

    monkeypatch.setattr("jax_rl.systems.spo.steps.policy_value_apply", _fake_policy_value_apply)

    def _simulate_with_invalid_state_ids(state, state_ids, actions):
        del state, actions
        n = state_ids.shape[0]
        timestep = TimeStep(
            step_type=jnp.full((n,), StepType.MID, dtype=jnp.int8),
            reward=jnp.ones((n,), dtype=jnp.float32),
            discount=jnp.ones((n,), dtype=jnp.float32),
            observation={
                "ems_pos": jnp.ones((n, 2), dtype=jnp.float32),
                "item_dims": jnp.ones((n, 3), dtype=jnp.float32),
                "item_mask": jnp.ones((n, 3), dtype=jnp.bool_),
                "action_mask": jnp.ones((n, 5), dtype=jnp.bool_),
            },
            extras={"state_id": jnp.zeros((n,), dtype=jnp.int32)},
        )
        return state_ids, timestep

    monkeypatch.setattr(env, "simulate_batch", _simulate_with_invalid_state_ids)

    recurrent_fn = make_recurrent_fn(
        env=env,
        env_params=None,
        gamma=0.99,
        is_rustpool=True,
    )

    batch_size, num_particles = 2, 3
    actions = jnp.zeros((batch_size, num_particles), dtype=jnp.int32)
    state_embedding = jnp.arange(batch_size * num_particles, dtype=jnp.int32).reshape(batch_size, num_particles)

    recurrent_output, next_state_ids = recurrent_fn(
        params,
        jax.random.PRNGKey(1),
        actions,
        state_embedding,
    )

    assert jnp.all(jnp.asarray(recurrent_output.discount, dtype=jnp.float32) == 0.0)
    assert jnp.array_equal(jnp.asarray(next_state_ids, dtype=jnp.int32), jnp.asarray(state_embedding, dtype=jnp.int32))


def test_spo_one_step_rollout_samples_actions_from_post_resample_logits(monkeypatch):
    config = _make_config(num_particles=2, search_depth=1)
    config = config.__class__(
        env=config.env,
        arch=config.arch,
        system=SystemConfig(
            name="spo",
            num_particles=2,
            search_depth=1,
            spo_resampling_mode="period",
            spo_resampling_period=1,
            warmup_steps=0,
            total_buffer_size=64,
            total_batch_size=16,
            sample_sequence_length=1,
            period=1,
            learner_updates_per_cycle=1,
        ),
        checkpointing=config.checkpointing,
        logging=config.logging,
        evaluations=config.evaluations,
        network=config.network,
    )

    def _fake_recurrent_fn(params, key, particle_actions, state_embedding):
        del params, key, particle_actions
        batch_size, num_particles = state_embedding.shape
        recurrent_output = SPORecurrentFnOutput(
            reward=jnp.zeros((batch_size, num_particles), dtype=jnp.float32),
            discount=jnp.ones((batch_size, num_particles), dtype=jnp.float32),
            prior_logits=jnp.zeros((batch_size, num_particles, 3), dtype=jnp.float32),
            value=jnp.zeros((batch_size, num_particles), dtype=jnp.float32),
            next_sampled_action=jnp.zeros((batch_size, num_particles), dtype=jnp.int32),
        )
        return recurrent_output, state_embedding

    search = SPO(config, _fake_recurrent_fn)

    def _fake_resample(self, particles, key, resample_logits):
        del key, resample_logits
        post_logits = jnp.full_like(particles.prior_logits, jnp.asarray(-1e9, dtype=jnp.float32))
        post_logits = post_logits.at[..., 2].set(0.0)
        return particles._replace(prior_logits=post_logits)

    monkeypatch.setattr(SPO, "resample", _fake_resample)

    particles = Particles(
        state_embedding=jnp.array([[10, 11]], dtype=jnp.int32),
        root_actions=jnp.array([[0, 1]], dtype=jnp.int32),
        resample_td_weights=jnp.zeros((1, 2), dtype=jnp.float32),
        prior_logits=jnp.zeros((1, 2, 3), dtype=jnp.float32),
        value=jnp.zeros((1, 2), dtype=jnp.float32),
        terminal=jnp.zeros((1, 2), dtype=jnp.bool_),
        depth=jnp.zeros((1, 2), dtype=jnp.int32),
        gae=jnp.zeros((1, 2), dtype=jnp.float32),
    )
    sampled_actions = jnp.zeros((1, 2), dtype=jnp.int32)

    (_, carried_actions), _ = search.one_step_rollout(
        (particles, sampled_actions),
        (jnp.asarray(0, dtype=jnp.int32), jax.random.PRNGKey(0)),
        _make_spo_params(),
    )

    assert jnp.all(jnp.asarray(carried_actions, dtype=jnp.int32) == 2)


def test_spo_recurrent_fn_ignores_env_capacity_hint():
    env = _MockRustpoolEnv()
    env.num_envs_per_device = 1
    params = _make_spo_params()
    recurrent_fn = make_recurrent_fn(
        env=env,
        env_params=None,
        gamma=0.99,
        is_rustpool=True,
    )

    batch_size, num_particles = 4, 8
    action = jnp.zeros((batch_size, num_particles), dtype=jnp.int32)
    state_embedding = jnp.arange(batch_size * num_particles, dtype=jnp.int32).reshape(batch_size, num_particles)

    recurrent_output, next_embedding = recurrent_fn(params, jax.random.PRNGKey(42), action, state_embedding)

    assert env.flat_shape == (32,)
    assert next_embedding.shape == (4, 8)
    assert recurrent_output.reward.shape == (4, 8)


def test_spo_memory_leak_prevention():
    env = _MockRustpoolEnv()
    config = _make_config(num_particles=2, search_depth=3)
    params = _make_spo_params()

    obs = {
        "ems_pos": jnp.ones((4, 2), dtype=jnp.float32),
        "item_dims": jnp.ones((4, 3), dtype=jnp.float32),
        "item_mask": jnp.ones((4, 3), dtype=jnp.bool_),
        "action_mask": jnp.ones((4, 5), dtype=jnp.bool_),
    }
    root_embedding = jnp.arange(4, dtype=jnp.int32)

    recurrent_fn = make_recurrent_fn(env=env, env_params=None, gamma=0.99, is_rustpool=True)
    root_fn = make_root_fn(config)
    search = SPO(config, recurrent_fn)

    root = root_fn(params, obs, root_embedding, jax.random.PRNGKey(1))
    search_output = search.search(params, jax.random.PRNGKey(2), root)

    generated = jnp.asarray(search_output.generated_state_ids, dtype=jnp.int32).reshape(-1)
    safe_ids = generated[generated >= 0]
    _ = env.release_batch(jnp.zeros_like(safe_ids, dtype=jnp.int32), safe_ids)

    assert env.released_ids
    assert int(env.released_ids[-1].shape[0]) == 4 * 2 * 3


def test_spo_mpo_loss_finite():
    dual_params = CategoricalDualParams(
        log_temperature=jnp.asarray(-2.0, dtype=jnp.float32),
        log_alpha=jnp.asarray(-2.0, dtype=jnp.float32),
    )

    logits_online = jnp.array([[1.2, 0.2, -0.4], [0.3, 0.9, -0.2]], dtype=jnp.float32)
    logits_target = jnp.array([[1.0, 0.1, -0.3], [0.2, 1.0, -0.1]], dtype=jnp.float32)
    sampled_actions = jnp.array([[0, 1, 2], [1, 0, 1]], dtype=jnp.int32)
    q_values = jnp.array([[0.1, 0.5, -0.3], [0.4, 0.2, -0.1]], dtype=jnp.float32)

    loss, metrics = categorical_mpo_loss(
        dual_params=dual_params,
        online_action_distribution=_DummyDist(logits_online),
        target_action_distribution=_DummyDist(logits_target),
        sampled_actions=sampled_actions,
        q_values=q_values,
        epsilon=0.1,
        epsilon_policy=0.05,
    )

    assert jnp.isfinite(loss)
    assert all(jnp.all(jnp.isfinite(value)) for value in metrics.values())


def test_spo_critic_loss_uses_gae_not_search_value():
    config = _make_config(num_particles=2, search_depth=2)
    train_params = _make_spo_params()
    from jax_rl.systems.spo.types import SPOOptStates, SPOTrainState

    train_state = SPOTrainState(
        params=train_params,
        opt_states=SPOOptStates(actor_opt_state=None, critic_opt_state=None, dual_opt_state=None),
    )

    batch_size, time_steps = 2, 2
    obs = jnp.ones((batch_size, time_steps, 8), dtype=jnp.float32)
    bootstrap_obs = jnp.ones((batch_size, time_steps, 8), dtype=jnp.float32) * 1.1
    rewards = jnp.array([[1.0, 0.5], [0.2, 0.0]], dtype=jnp.float32)
    dones = jnp.array([[False, True], [False, False]], dtype=jnp.bool_)
    truncated = jnp.zeros((batch_size, time_steps), dtype=jnp.bool_)

    sequence_base = type("Seq", (), {
        "obs": obs,
        "bootstrap_obs": bootstrap_obs,
        "reward": rewards,
        "done": dones,
        "truncated": truncated,
        "search_value": jnp.zeros((batch_size, time_steps), dtype=jnp.float32),
    })
    loss_a, _ = _critic_loss(train_state, sequence_base, config)

    sequence_changed = type("Seq", (), {
        "obs": obs,
        "bootstrap_obs": bootstrap_obs,
        "reward": rewards,
        "done": dones,
        "truncated": truncated,
        "search_value": jnp.ones((batch_size, time_steps), dtype=jnp.float32) * 1e6,
    })
    loss_b, _ = _critic_loss(train_state, sequence_changed, config)

    assert jnp.allclose(loss_a, loss_b)


@pytest.mark.integration
def test_spo_integration_dry_run():
    pytest.importorskip("rustpool")

    config = ExperimentConfig(
        env=EnvConfig(env_name="rustpool:BinPack-v0", env_kwargs={"max_items": 5}),
        arch=ArchConfig(total_timesteps=4, num_envs=2, num_steps=2),
        system=SystemConfig(
            name="spo",
            update_epochs=1,
            minibatch_size=2,
            warmup_steps=0,
            num_particles=2,
            search_depth=2,
            total_buffer_size=16,
            total_batch_size=4,
            sample_sequence_length=2,
            period=1,
            learner_updates_per_cycle=1,
        ),
        logging=LoggingConfig(log_every=1, tensorboard_logdir=None),
        checkpointing=CheckpointConfig(save_interval_steps=0),
        network={
            "_target_": "jax_rl.networks.BinPackPolicyValueModel",
            "hidden_dim": 16,
            "num_heads": 2,
            "num_layers": 1,
        },
        evaluations={},
    )

    result = train(config)
    assert isinstance(result, dict)
    assert "metrics" in result


@pytest.mark.integration
def test_spo_root_fn_uld_invalid_action_probability_zero():
    pytest.importorskip("rustpool")
    pytest.importorskip("rlpallet")

    env, env_params, spo_params, config = _make_uld_spo_setup(num_envs_per_device=2)
    root_fn = make_root_fn(config)

    key = jax.random.PRNGKey(77)
    _, timestep = env.reset(key, env_params)
    observation = timestep.observation
    action_mask = jnp.asarray(observation["action_mask"], dtype=jnp.bool_)
    root_embedding = jnp.arange(action_mask.shape[0], dtype=jnp.int32)

    root = root_fn(spo_params, observation, root_embedding, jax.random.PRNGKey(78))
    probs = jax.nn.softmax(jnp.asarray(root.particle_logits, dtype=jnp.float32), axis=-1)
    invalid_probs = probs * jnp.logical_not(action_mask)[:, None, :]

    assert jnp.all(invalid_probs == 0.0)


@pytest.mark.integration
def test_spo_root_fn_uld_invalid_action_rate_low():
    pytest.importorskip("rustpool")
    pytest.importorskip("rlpallet")

    env, env_params, spo_params, config = _make_uld_spo_setup(num_envs_per_device=2)
    root_fn = make_root_fn(config)

    key = jax.random.PRNGKey(79)
    _, timestep = env.reset(key, env_params)
    observation = timestep.observation
    action_mask = jnp.asarray(observation["action_mask"], dtype=jnp.bool_)
    root_embedding = jnp.arange(action_mask.shape[0], dtype=jnp.int32)

    invalid_count = 0
    total_count = 0
    for idx in range(32):
        root = root_fn(spo_params, observation, root_embedding, jax.random.PRNGKey(1000 + idx))
        sampled = jnp.asarray(root.particle_actions, dtype=jnp.int32)
        sampled_valid = jnp.take_along_axis(action_mask[:, None, :], sampled[..., None], axis=-1).squeeze(-1)
        invalid_count += int(jnp.sum(jnp.logical_not(sampled_valid)))
        total_count += int(sampled_valid.size)

    invalid_rate = invalid_count / max(total_count, 1)
    assert invalid_rate < 1e-6


@pytest.mark.integration
def test_spo_minimal_uld_search_no_error_prints(capfd):
    pytest.importorskip("rustpool")
    pytest.importorskip("rlpallet")

    _run_minimal_spo_uld_search_steps(num_steps=4)
    captured = capfd.readouterr()
    text = f"{captured.out}\n{captured.err}"

    error_pattern = re.compile(
        r"(Placement Failed|placement error|Invalid EMS|Critical: Empty or invalid group)",
        re.IGNORECASE,
    )
    assert error_pattern.search(text) is None, text


@pytest.mark.integration
def test_spo_minimal_uld_search_actions_follow_mask():
    pytest.importorskip("rustpool")
    pytest.importorskip("rlpallet")

    action_history, action_mask_history = _run_minimal_spo_uld_search_steps(num_steps=4)

    for actions, action_mask in zip(action_history, action_mask_history):
        selected_valid = jnp.take_along_axis(
            action_mask,
            actions[:, None],
            axis=-1,
        ).squeeze(-1)
        assert jnp.all(selected_valid)
