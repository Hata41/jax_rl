import jax
import jax.numpy as jnp
from stoa.env_types import StepType, TimeStep
from typing import cast, Any

from jax_rl.envs.env import RustpoolObsWrapper
from jax_rl.networks import init_policy_value_params
from jax_rl.systems.alphazero import steps as az_steps


class _MockRustpoolInner:
    def simulate_batch(self, state, state_ids, actions):
        del state_ids, actions
        timestep = TimeStep(
            step_type=jnp.full((2,), StepType.MID, dtype=jnp.int8),
            reward=jnp.ones((2,), dtype=jnp.float32),
            discount=jnp.ones((2,), dtype=jnp.float32),
            observation={
                "ems": jnp.ones((2, 3), dtype=jnp.float32),
                "items": jnp.ones((2, 4), dtype=jnp.float32),
                "items_mask": jnp.ones((2, 4), dtype=jnp.bool_),
            },
            extras={
                "action_mask": jnp.ones((2, 5), dtype=jnp.bool_),
                "next_obs": {
                    "ems": jnp.ones((2, 3), dtype=jnp.float32),
                    "items": jnp.ones((2, 4), dtype=jnp.float32),
                    "items_mask": jnp.ones((2, 4), dtype=jnp.bool_),
                },
            },
        )
        return state, timestep


class _MockRustpoolRecurrentEnv:
    def __init__(self):
        self.simulate_called = 0

    def simulate_batch(self, state, state_ids, actions):
        self.simulate_called += 1
        timestep = TimeStep(
            step_type=jnp.full((2,), StepType.MID, dtype=jnp.int8),
            reward=jnp.ones((2,), dtype=jnp.float32),
            discount=jnp.ones((2,), dtype=jnp.float32),
            observation=jnp.ones((2, 3), dtype=jnp.float32),
            extras={"state_id": jnp.array([7, 9], dtype=jnp.int32)},
        )
        return state, timestep


class _MockJaxEnv:
    def __init__(self):
        self.step_called = 0

    def step(self, state, action, env_params=None):
        del env_params
        self.step_called += 1
        next_state = state + action.astype(state.dtype)
        timestep = TimeStep(
            step_type=StepType.MID,
            reward=jnp.asarray(1.0, dtype=jnp.float32),
            discount=jnp.asarray(1.0, dtype=jnp.float32),
            observation=jnp.ones((3,), dtype=jnp.float32),
            extras={},
        )
        return next_state, timestep


class _MockVectorizedJaxEnv:
    _vmap_step = True

    def __init__(self):
        self.step_called = 0

    def step(self, state, action, env_params=None):
        del env_params
        self.step_called += 1
        next_state = state + action[:, None].astype(state.dtype)
        batch = state.shape[0]
        timestep = TimeStep(
            step_type=jnp.full((batch,), StepType.MID, dtype=jnp.int8),
            reward=jnp.ones((batch,), dtype=jnp.float32),
            discount=jnp.ones((batch,), dtype=jnp.float32),
            observation=jnp.ones((batch, 3), dtype=jnp.float32),
            extras={},
        )
        return next_state, timestep


class _MockReleaseEnv:
    def __init__(self):
        self.calls = []

    def release_batch(self, state, state_ids):
        self.calls.append((state, state_ids))
        return state


def _make_params():
    return init_policy_value_params(
        key=jax.random.PRNGKey(0),
        network_config={"_target_": "jax_rl.networks.PolicyValueModel", "hidden_sizes": [8]},
        obs_dim=3,
        action_dims=4,
    )


def test_rustpool_obs_wrapper_simulate_batch():
    wrapper = RustpoolObsWrapper(cast(Any, _MockRustpoolInner()))
    _, timestep = wrapper.simulate_batch(
        state=jnp.zeros((2,), dtype=jnp.int32),
        state_ids=jnp.array([1, 2], dtype=jnp.int32),
        actions=jnp.array([0, 1], dtype=jnp.int32),
    )

    assert "ems_pos" in timestep.observation
    assert "item_dims" in timestep.observation
    assert "item_mask" in timestep.observation
    assert "next_obs" in timestep.extras
    assert "ems_pos" in timestep.extras["next_obs"]


def test_recurrent_fn_routing(monkeypatch):
    params = _make_params()

    rust_env = _MockRustpoolRecurrentEnv()
    rust_recurrent = az_steps.make_recurrent_fn(
        env=rust_env,
        env_params=None,
        gamma=0.99,
        is_rustpool=True,
    )
    rust_recurrent(
        params,
        jax.random.PRNGKey(1),
        jnp.array([0, 1], dtype=jnp.int32),
        jnp.array([10, 11], dtype=jnp.int32),
    )
    assert rust_env.simulate_called == 1

    vmap_called = {"flag": False}
    real_vmap = az_steps.jax.vmap

    def _wrapped_vmap(*args, **kwargs):
        vmap_called["flag"] = True
        return real_vmap(*args, **kwargs)

    monkeypatch.setattr(az_steps.jax, "vmap", _wrapped_vmap)

    jax_env = _MockJaxEnv()
    jax_recurrent = az_steps.make_recurrent_fn(
        env=jax_env,
        env_params=None,
        gamma=0.99,
        is_rustpool=False,
    )
    jax_recurrent(
        params,
        jax.random.PRNGKey(2),
        jnp.array([0, 1], dtype=jnp.int32),
        jnp.zeros((2, 3), dtype=jnp.float32),
    )
    assert vmap_called["flag"] is True


def test_recurrent_fn_vectorized_env_skips_outer_vmap(monkeypatch):
    params = _make_params()
    vectorized_env = _MockVectorizedJaxEnv()

    vmap_called = {"flag": False}
    real_vmap = az_steps.jax.vmap

    def _wrapped_vmap(*args, **kwargs):
        vmap_called["flag"] = True
        return real_vmap(*args, **kwargs)

    monkeypatch.setattr(az_steps.jax, "vmap", _wrapped_vmap)

    recurrent = az_steps.make_recurrent_fn(
        env=vectorized_env,
        env_params=None,
        gamma=0.99,
        is_rustpool=False,
    )
    recurrent(
        params,
        jax.random.PRNGKey(3),
        jnp.array([0, 1], dtype=jnp.int32),
        jnp.zeros((2, 3), dtype=jnp.float32),
    )

    assert vectorized_env.step_called == 1
    assert vmap_called["flag"] is False


def test_alphazero_memory_cleanup_called():
    env = _MockReleaseEnv()

    class _SearchTree:
        embeddings = jnp.array([[11, 12, 12], [0, 13, 0]], dtype=jnp.int32)
        node_visits = jnp.array([[1, 1, 0], [0, 2, 0]], dtype=jnp.int32)

    token = az_steps.release_rustpool_embeddings(
        env=env,
        state=jnp.array([0, 0], dtype=jnp.int32),
        search_tree=_SearchTree,
    )
    jax.block_until_ready(token)

    assert len(env.calls) == 1
    _, released_ids = env.calls[0]
    assert released_ids.dtype == jnp.int32
    assert released_ids.tolist() == [11, 12, 13]
