import time
from dataclasses import replace
from typing import Any, cast

import jax
import jax.numpy as jnp

from ...configs.config import ExperimentConfig
from ...configs.evaluations import resolve_eval_env
from ...envs.env import make_stoa_env
from ...networks import policy_value_apply
from ...utils.exceptions import ConfigDivisibilityError, NumericalInstabilityError
from ...utils.jax_utils import replicate_tree
from ...utils.runtime import PhaseTimer
from ...utils.types import PolicyValueParams
from .steps import (
    extract_root_embedding,
    make_recurrent_fn,
    make_root_fn,
    parse_search_method,
    release_rustpool_embeddings,
    search_output_is_finite,
)


def _zero_metrics() -> dict[str, float | int]:
    return {
        "return_mean": 0.0,
        "return_std": 0.0,
        "return_min": 0.0,
        "return_max": 0.0,
        "episodes": 0,
        "steps": 0,
    }
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


def _distribution_logits(dist: Any) -> jax.Array:
    if hasattr(dist, "logits"):
        return jnp.asarray(dist.logits, dtype=jnp.float32)
    if hasattr(dist, "logits_per_dim"):
        return jnp.concatenate(
            tuple(jnp.asarray(x, dtype=jnp.float32) for x in dist.logits_per_dim),
            axis=-1,
        )
    raise TypeError("Policy distribution must expose 'logits' or 'logits_per_dim'.")


class Evaluator:
    def __init__(
        self,
        config: ExperimentConfig,
        env_name: str,
        num_episodes: int,
        max_steps_per_episode: int,
        greedy: bool,
        env_kwargs: dict[str, Any] | None = None,
        action_selection: str = "policy",
    ):
        self.config = config
        self.env_name = str(env_name)
        self.num_episodes = int(num_episodes)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.greedy = bool(greedy)
        self.env_kwargs = dict(env_kwargs or {})
        self.action_selection = str(action_selection).lower()
        self.num_devices = int(jax.local_device_count())
        self._closed = False

        if self.action_selection not in {"policy", "search"}:
            raise ValueError(
                f"Unsupported evaluation action_selection '{action_selection}'. "
                "Expected 'policy' or 'search'."
            )

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

        self.is_rustpool = self.env_name.lower().startswith(("rustpool:", "rlpallet:"))
        recurrent_fn: Any = None

        if self.action_selection == "search":
            recurrent_fn = make_recurrent_fn(
                env=env,
                env_params=env_params,
                gamma=float(config.system.gamma),
                is_rustpool=self.is_rustpool,
            )
            self._root_fn = make_root_fn()
            self._search_method = parse_search_method(config.system.search_method)
            self._dirichlet_alpha = float(config.system.dirichlet_alpha)
            self._dirichlet_fraction = 0.0
        else:
            self._root_fn = None
            self._search_method = None
            self._dirichlet_alpha = 0.0
            self._dirichlet_fraction = 0.0

        def _device_eval(params: PolicyValueParams, device_key):
            key, reset_key = jax.random.split(device_key)
            env_state, timestep = env.reset(reset_key, cast(Any, env_params))
            init_active_mask = jnp.ones((self.num_envs_per_device,), dtype=jnp.float32)
            init_returns = jnp.zeros((self.num_envs_per_device,), dtype=jnp.float32)
            init_steps = jnp.zeros((self.num_envs_per_device,), dtype=jnp.float32)
            init_all_finite = jnp.asarray(True, dtype=jnp.bool_)
            init_step_count = jnp.asarray(0, dtype=jnp.int32)

            def _cond_fn(carry):
                _, _, _, active_mask, _, _, _, step_count = carry
                has_active = jnp.any(active_mask > 0.0)
                within_limit = step_count < jnp.asarray(self.max_steps_per_episode, dtype=jnp.int32)
                return jnp.logical_and(has_active, within_limit)

            def _body_fn(carry):
                curr_env_state, curr_obs, curr_key, active_mask, returns, steps, all_finite, step_count = carry
                curr_key, action_key, root_key, search_key, next_key = jax.random.split(curr_key, 5)
                search_output = None

                if self.action_selection == "search":
                    assert self._root_fn is not None
                    assert self._search_method is not None
                    assert recurrent_fn is not None
                    root_embedding = extract_root_embedding(
                        env=env,
                        env_state=curr_env_state,
                        obs=curr_obs,
                        is_rustpool=self.is_rustpool,
                    )
                    root = self._root_fn(params, curr_obs, root_embedding, root_key)

                    invalid_actions = None
                    if isinstance(curr_obs, dict) and "action_mask" in curr_obs:
                        invalid_actions = jnp.logical_not(
                            jnp.asarray(curr_obs["action_mask"], dtype=jnp.bool_)
                        )

                    search_output = self._search_method(
                        params=params,
                        rng_key=search_key,
                        root=root,
                        recurrent_fn=recurrent_fn,
                        num_simulations=int(config.system.num_simulations),
                        max_depth=int(config.system.max_depth),
                        invalid_actions=invalid_actions,
                        dirichlet_alpha=self._dirichlet_alpha,
                        dirichlet_fraction=self._dirichlet_fraction,
                    )
                    search_finite = search_output_is_finite(search_output)
                    action = jnp.argmax(search_output.action_weights, axis=-1).astype(jnp.int32)
                else:
                    dist, values = policy_value_apply(params.graphdef, params.state, curr_obs)
                    logits = _distribution_logits(dist)
                    policy_finite = jnp.logical_and(
                        jnp.all(jnp.isfinite(logits)),
                        jnp.all(jnp.isfinite(jnp.asarray(values, dtype=jnp.float32))),
                    )
                    search_finite = policy_finite
                    action = jax.lax.cond(
                        jnp.asarray(self.greedy),
                        lambda _: dist.mode(),
                        lambda sample_key: dist.sample(sample_key),
                        action_key,
                    )

                next_env_state, next_timestep = env.step(curr_env_state, action, cast(Any, env_params))
                reward = jnp.asarray(next_timestep.reward, dtype=jnp.float32)
                done = jnp.asarray(next_timestep.last(), dtype=jnp.float32)

                if self.action_selection == "search" and self.is_rustpool and search_output is not None:
                    dummy_state = jnp.zeros((self.num_envs_per_device,), dtype=jnp.int32)
                    _ = release_rustpool_embeddings(
                        env=env,
                        state=dummy_state,
                        search_tree=search_output.search_tree,
                    )

                masked_reward = reward * active_mask
                next_active_mask = active_mask * (1.0 - done)

                return (
                    next_env_state,
                    next_timestep.observation,
                    next_key,
                    next_active_mask,
                    returns + masked_reward,
                    steps + active_mask,
                    jnp.logical_and(all_finite, jnp.asarray(search_finite, dtype=jnp.bool_)),
                    step_count + jnp.asarray(1, dtype=jnp.int32),
                )

            (
                _,
                _,
                _,
                _,
                returns,
                steps,
                is_finite,
                _,
            ) = jax.lax.while_loop(
                _cond_fn,
                _body_fn,
                (
                    env_state,
                    timestep.observation,
                    key,
                    init_active_mask,
                    init_returns,
                    init_steps,
                    init_all_finite,
                    init_step_count,
                ),
            )
            return returns, steps, is_finite

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
        returns, steps, finite_flags = self._pmap_eval(replicated_params, device_keys)

        if not bool(jnp.all(finite_flags)):
            raise NumericalInstabilityError(
                f"AlphaZero evaluation produced non-finite outputs in mode '{self.action_selection}'."
            )

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
        config: ExperimentConfig,
        evaluations: dict[str, dict[str, Any]] | None,
        default_env_name: str,
        default_env_kwargs: dict[str, Any] | None,
        evaluator_cls: Any = Evaluator,
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
            env_name, env_kwargs = resolve_eval_env(
                cfg,
                default_env_name=default_env_name,
                default_env_kwargs=default_env_kwargs,
            )
            self._evaluators[eval_name] = self._evaluator_cls(
                config=config,
                env_name=env_name,
                num_episodes=num_episodes,
                max_steps_per_episode=int(cfg.get("max_steps_per_episode", 1_000)),
                greedy=bool(cfg.get("greedy", True)),
                env_kwargs=env_kwargs,
                action_selection=str(cfg.get("action_selection", "policy")),
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
    *,
    params: PolicyValueParams,
    config: ExperimentConfig,
    env_name: str,
    seed: int,
    num_episodes: int = 10,
    max_steps_per_episode: int = 1_000,
    greedy: bool = True,
    env_kwargs: dict[str, Any] | None = None,
    action_selection: str = "policy",
):
    evaluator = Evaluator(
        config=config,
        env_name=env_name,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        greedy=greedy,
        env_kwargs=env_kwargs,
        action_selection=action_selection,
    )
    try:
        if evaluator.disabled:
            return _zero_metrics()

        if _is_replicated_state(params.state, evaluator.num_devices):
            replicated_params = params
        else:
            replicated_params = PolicyValueParams(
                graphdef=params.graphdef,
                state=replicate_tree(params.state),
            )
        return evaluator.run(replicated_params=replicated_params, seed=seed)
    finally:
        evaluator.close()
