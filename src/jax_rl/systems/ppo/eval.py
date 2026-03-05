import time
from typing import Any, cast

import jax
import jax.numpy as jnp

from ...envs.env import make_stoa_env
from ...networks import policy_value_apply
from ...utils.exceptions import ConfigDivisibilityError
from ...utils.runtime import PhaseTimer
from ...utils.types import PolicyValueParams


def _zero_metrics() -> dict[str, float | int]:
    return {
        "return_mean": 0.0,
        "return_std": 0.0,
        "return_min": 0.0,
        "return_max": 0.0,
        "episodes": 0,
        "steps": 0,
    }


def _replicate_tree(tree, num_devices: int):
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(jnp.asarray(x), (num_devices,) + jnp.asarray(x).shape),
        tree,
    )


def _is_replicated_state(state, num_devices: int) -> bool:
    leaves = jax.tree_util.tree_leaves(state)
    if not leaves:
        return False
    return all(
        hasattr(leaf, "shape") and len(leaf.shape) > 0 and int(leaf.shape[0]) == int(num_devices)
        for leaf in leaves
    )


def _prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


class Evaluator:
    def __init__(
        self,
        env_name: str,
        num_episodes: int,
        max_steps_per_episode: int,
        greedy: bool,
        env_kwargs: dict[str, Any] | None = None,
    ):
        self.env_name = str(env_name)
        self.num_episodes = int(num_episodes)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.greedy = bool(greedy)
        self.env_kwargs = dict(env_kwargs or {})
        self.num_devices = int(jax.local_device_count())
        self._closed = False

        if self.num_episodes <= 0:
            self.disabled = True
            self.num_envs_per_device = 0
            self.env = None
            self.env_params = None
            self._pmap_eval = None
            return

        self.disabled = False
        if self.num_episodes % self.num_devices != 0:
            raise ConfigDivisibilityError(
                "num_episodes must be divisible by local device count, "
                f"got num_episodes={self.num_episodes} and num_devices={self.num_devices}."
            )

        self.num_envs_per_device = self.num_episodes // self.num_devices
        self.env, self.env_params = make_stoa_env(
            self.env_name,
            num_envs_per_device=self.num_envs_per_device,
            env_kwargs=self.env_kwargs,
        )
        env = self.env
        assert env is not None
        env_params = self.env_params

        def _device_eval(params: PolicyValueParams, device_key):
            key, reset_key = jax.random.split(device_key)
            env_state, timestep = env.reset(reset_key, cast(Any, env_params))
            init_active_mask = jnp.ones((self.num_envs_per_device,), dtype=jnp.float32)

            def _env_step(carry, _):
                curr_env_state, curr_obs, curr_key, active_mask = carry
                curr_key, action_key, next_key = jax.random.split(curr_key, 3)

                dist, _ = policy_value_apply(params.graphdef, params.state, curr_obs)
                action = jax.lax.cond(
                    jnp.asarray(self.greedy),
                    lambda _: dist.mode(),
                    lambda sample_key: dist.sample(sample_key),
                    action_key,
                )

                next_env_state, next_timestep = env.step(curr_env_state, action, cast(Any, env_params))
                reward = jnp.asarray(next_timestep.reward, dtype=jnp.float32)
                done = jnp.asarray(next_timestep.last(), dtype=jnp.float32)

                masked_reward = reward * active_mask
                next_active_mask = active_mask * (1.0 - done)

                return (
                    next_env_state,
                    next_timestep.observation,
                    next_key,
                    next_active_mask,
                ), {
                    "reward": masked_reward,
                    "active_mask": active_mask,
                }

            _, outputs = jax.lax.scan(
                _env_step,
                (env_state, timestep.observation, key, init_active_mask),
                xs=None,
                length=self.max_steps_per_episode,
            )
            returns = jnp.sum(outputs["reward"], axis=0)
            steps = jnp.sum(outputs["active_mask"], axis=0)
            return returns, steps

        params_in_axes = cast(Any, PolicyValueParams._make((None, 0)))
        self._pmap_eval = jax.pmap(
            _device_eval,
            axis_name="device",
            in_axes=(params_in_axes, 0),
        )

    def run(self, replicated_params: PolicyValueParams, seed: int) -> dict[str, float | int]:
        if self.disabled:
            return _zero_metrics()
        assert self._pmap_eval is not None

        device_keys = jax.random.split(jax.random.PRNGKey(int(seed)), self.num_devices)
        returns, steps = self._pmap_eval(replicated_params, device_keys)

        flat_returns = jnp.reshape(returns, (self.num_episodes,))
        total_steps = jnp.sum(steps)

        return {
            "return_mean": float(jnp.mean(flat_returns)),
            "return_std": float(jnp.std(flat_returns)),
            "return_min": float(jnp.min(flat_returns)),
            "return_max": float(jnp.max(flat_returns)),
            "episodes": int(self.num_episodes),
            "steps": int(total_steps),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.env is not None and hasattr(self.env, "close"):
            self.env.close()


class EvaluationManager:
    def __init__(
        self,
        evaluations: dict[str, dict[str, Any]] | None,
        default_env_name: str,
        default_env_kwargs: dict[str, Any] | None,
        evaluator_cls=Evaluator,
        now_fn=None,
    ):
        self._evaluators: dict[str, Evaluator] = {}
        self._eval_every_by_name: dict[str, int] = {}
        self._evaluator_cls = evaluator_cls
        self._now = now_fn or time.time

        for eval_name, eval_cfg in (evaluations or {}).items():
            cfg = dict(eval_cfg)
            num_episodes = int(cfg.get("num_episodes", 10))
            if num_episodes <= 0:
                continue
            self._evaluators[eval_name] = self._evaluator_cls(
                env_name=str(cfg.get("env_name", default_env_name)),
                num_episodes=num_episodes,
                max_steps_per_episode=int(cfg.get("max_steps_per_episode", 1_000)),
                greedy=bool(cfg.get("greedy", True)),
                env_kwargs=dict(cfg.get("env_kwargs", default_env_kwargs or {})),
            )
            self._eval_every_by_name[eval_name] = int(cfg.get("eval_every", 10))

    def run_if_needed(
        self,
        update_idx: int,
        params,
        seed: int,
    ) -> dict[str, float]:
        eval_metrics: dict[str, float] = {}

        for eval_name, evaluator in self._evaluators.items():
            eval_every = self._eval_every_by_name.get(eval_name, 10)
            if eval_every <= 0 or update_idx % eval_every != 0:
                continue

            timer = PhaseTimer(now_fn=self._now)
            phase_name = f"eval:{eval_name}"
            with timer.phase(phase_name):
                eval_results = evaluator.run(
                    replicated_params=params,
                    seed=int(seed),
                )

            prefixed = _prefixed_metrics(eval_name, dict(eval_results))
            prefixed[f"{eval_name}/steps_per_second"] = timer.steps_per_second(
                phase_name,
                float(eval_results.get("steps", 0)),
            )
            eval_metrics.update(prefixed)

        return eval_metrics

    def close(self) -> None:
        for evaluator in self._evaluators.values():
            evaluator.close()


def evaluate(
    params: PolicyValueParams,
    env_name: str,
    seed: int,
    num_episodes: int = 10,
    max_steps_per_episode: int = 1_000,
    greedy: bool = True,
    env_kwargs: dict[str, Any] | None = None,
):
    evaluator = Evaluator(
        env_name=env_name,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        greedy=greedy,
        env_kwargs=env_kwargs,
    )
    try:
        if evaluator.disabled:
            return _zero_metrics()

        if _is_replicated_state(params.state, evaluator.num_devices):
            replicated_params = params
        else:
            replicated_params = PolicyValueParams(
                graphdef=params.graphdef,
                state=_replicate_tree(params.state, evaluator.num_devices),
            )
        return evaluator.run(replicated_params=replicated_params, seed=seed)
    finally:
        evaluator.close()