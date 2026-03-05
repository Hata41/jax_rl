import jax.numpy as jnp
import pytest

from jax_rl.configs.config import ExperimentConfig
from jax_rl.utils.exceptions import NumericalInstabilityError
from jax_rl.systems.alphazero.anakin import factory as az_factory
from jax_rl.systems.alphazero.anakin import system as az_system


class _RunnerState:
    def __init__(self):
        self.obs = jnp.zeros((1,), dtype=jnp.float32)


class _Traj:
    def __init__(self, finite: bool):
        self.info = {"search_finite": jnp.asarray([finite], dtype=jnp.bool_)}


def test_run_warmup_rollouts_uses_ceiling_cycles():
    config = ExperimentConfig()
    config.system.warmup_steps = 9
    config.system.num_steps = 4

    calls = {"count": 0}

    def _pmap_rollout(runner_state):
        calls["count"] += 1
        return runner_state, _Traj(True)

    runner_state = _RunnerState()
    az_system._run_warmup_rollouts(config, _pmap_rollout, runner_state)

    assert calls["count"] == 3


def test_run_warmup_rollouts_raises_on_non_finite_search():
    config = ExperimentConfig()
    config.system.warmup_steps = 4
    config.system.num_steps = 4

    def _pmap_rollout(runner_state):
        return runner_state, _Traj(False)

    with pytest.raises(NumericalInstabilityError):
        az_system._run_warmup_rollouts(config, _pmap_rollout, _RunnerState())


def test_dummy_transition_schema_is_unbatched():
    obs = {
        "x": jnp.ones((8, 3), dtype=jnp.float32),
        "mask": jnp.ones((8, 5), dtype=jnp.bool_),
    }
    transition = az_factory._make_dummy_transition(obs=obs, num_actions=7)

    assert transition.done.shape == ()
    assert transition.action.shape == ()
    assert transition.reward.shape == ()
    assert transition.search_value.shape == ()
    assert transition.search_policy.shape == (7,)
    assert transition.obs["x"].shape == (3,)
    assert transition.obs["mask"].shape == (5,)
