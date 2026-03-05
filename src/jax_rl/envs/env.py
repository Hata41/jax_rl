import jax.numpy as jnp
from typing import Any, Callable, cast

from gymnax import make as make_gymnax_env
from stoa import AddRNGKey, AutoResetWrapper, RecordEpisodeMetrics
from stoa.core_wrappers.episode_metrics import RecordEpisodeMetricsState
from stoa.core_wrappers.vmap import VmapWrapper
from stoa.core_wrappers.wrapper import Wrapper
from stoa.env_adapters.gymnax import GymnaxToStoa

from ..utils.exceptions import EnvironmentNotFoundError


EnvFactory = Callable[[str, int, dict[str, Any]], tuple[Any, Any]]
_ENV_REGISTRY: dict[str, EnvFactory] = {}


def register_env(prefix: str) -> Callable[[EnvFactory], EnvFactory]:
    """Register a backend environment factory under a prefix."""
    normalized = prefix.strip().lower()
    if not normalized:
        raise ValueError("Environment prefix cannot be empty.")

    def decorator(factory: EnvFactory) -> EnvFactory:
        _ENV_REGISTRY[normalized] = factory
        return factory

    return decorator


class BatchedRecordEpisodeMetrics(RecordEpisodeMetrics):
    def reset(self, rng_key, env_params=None):
        base_env_state, timestep = self._env.reset(rng_key, env_params)
        reward = jnp.asarray(timestep.reward)
        float_zeros = jnp.zeros(reward.shape, dtype=jnp.float32)
        int_zeros = jnp.zeros(reward.shape, dtype=jnp.int32)
        bool_zeros = jnp.zeros(reward.shape, dtype=jnp.bool_)

        state = RecordEpisodeMetricsState(
            base_env_state=base_env_state,
            running_count_episode_return=float_zeros,
            running_count_episode_length=int_zeros,
            episode_return=float_zeros,
            episode_length=int_zeros,
        )
        episode_metrics = {
            "episode_return": float_zeros,
            "episode_length": int_zeros,
            "is_terminal_step": bool_zeros,
        }

        new_extras = {**timestep.extras, "episode_metrics": episode_metrics}
        timestep_obj = cast(Any, timestep)
        timestep = timestep_obj.replace(extras=new_extras)
        return state, timestep

class RustpoolObsWrapper(Wrapper):
    @staticmethod
    def _normalize_observation(observation, action_mask):
        if not isinstance(observation, dict):
            observation = {"obs": observation}

        canonical = dict(observation)
        if "ems_pos" not in canonical and "ems" in canonical:
            canonical["ems_pos"] = canonical["ems"]
        if "item_dims" not in canonical and "items" in canonical:
            canonical["item_dims"] = canonical["items"]
        if "item_mask" not in canonical and "items_mask" in canonical:
            canonical["item_mask"] = canonical["items_mask"]

        if action_mask is not None:
            canonical["action_mask"] = action_mask
        return canonical

    def _normalize_timestep(self, timestep):
        extras = timestep.extras or {}
        action_mask = extras.get("action_mask") if hasattr(extras, "get") else None
        new_observation = self._normalize_observation(timestep.observation, action_mask)

        new_extras = extras
        if hasattr(extras, "get"):
            next_obs = extras.get("next_obs")
            if next_obs is not None:
                normalized_next_obs = self._normalize_observation(next_obs, action_mask)
                new_extras = {**extras, "next_obs": normalized_next_obs}

        timestep_obj = cast(Any, timestep)
        return timestep_obj.replace(observation=new_observation, extras=new_extras)

    def reset(self, rng_key, env_params=None):
        state, timestep = self._env.reset(rng_key, env_params)
        return state, self._normalize_timestep(timestep)

    def step(self, state, action, env_params=None):
        state, timestep = self._env.step(state, action, env_params)
        return state, self._normalize_timestep(timestep)

    def simulate_batch(self, state, state_ids, actions):
        state, timestep = self._env.simulate_batch(state, state_ids, actions)
        return state, self._normalize_timestep(timestep)

@register_env("rustpool")
def _make_rustpool_env(env_name: str, num_envs_per_device: int, env_kwargs: dict[str, Any]):
    task_id = env_name.split(":", 1)[1]
    from rustpool.envpool_api.stoa_wrapper import StoaRustpoolWrapper

    env = StoaRustpoolWrapper(
        task_id=task_id,
        num_envs_per_device=num_envs_per_device,
        **env_kwargs,
    )
    env = RustpoolObsWrapper(env)
    env = BatchedRecordEpisodeMetrics(env)
    return env, None


@register_env("rlpallet")
def _make_rlpallet_env(env_name: str, num_envs_per_device: int, env_kwargs: dict[str, Any]):
    task_id = env_name.split(":", 1)[1]

    import jax
    import numpy as np
    import rlpallet
    from rustpool.envpool_api.stoa_wrapper import StoaRustpoolWrapper

    class StoaRlpalletWrapper(StoaRustpoolWrapper):
        def __init__(self, task_id: str, num_envs_per_device: int, **kwargs: Any):
            self.task_id = task_id
            self.num_envs_per_device = int(num_envs_per_device)
            self.n_devices = len(jax.devices())

            self._step_type_first = self._step_type_value("FIRST", 0)
            self._step_type_mid = self._step_type_value("MID", 1)
            self._step_type_last = self._step_type_value("LAST", 2)

            base_seed = kwargs.pop("seed", 0)

            self.sharded_pools = [
                rlpallet.make(
                    task_id,
                    num_envs=self.num_envs_per_device,
                    batch_size=self.num_envs_per_device,
                    seed=base_seed + (device_index * 10000),
                    **kwargs,
                )
                for device_index in range(self.n_devices)
            ]

            self._env_ids_by_device: list[np.ndarray | None] = [None] * self.n_devices

            self.dummy_state_struct = jax.ShapeDtypeStruct(
                (self.num_envs_per_device,), jnp.int32
            )
            self._state_ids_struct = jax.ShapeDtypeStruct(
                (self.num_envs_per_device,), jnp.int32
            )

            self._build_structs_from_probe()

    env = StoaRlpalletWrapper(
        task_id=task_id,
        num_envs_per_device=num_envs_per_device,
        **env_kwargs,
    )
    env = RustpoolObsWrapper(env)
    env = BatchedRecordEpisodeMetrics(env)
    return env, None


@register_env("jaxpallet")
def _make_jaxpallet_env(env_name: str, num_envs_per_device: int, env_kwargs: dict[str, Any]):
    preset = env_name.split(":", 1)[1]
    from jaxpallet.stoa_adapter import JaxPalletToStoa

    env = JaxPalletToStoa(preset=preset, **env_kwargs)
    env = AddRNGKey(env)
    env = RecordEpisodeMetrics(env)
    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = VmapWrapper(env, num_envs=num_envs_per_device)
    return env, None


def make_stoa_env(
    env_name: str,
    num_envs_per_device: int,
    env_kwargs: dict[str, Any] | None = None,
):
    kwargs_payload = dict(env_kwargs or {})
    prefix, has_prefix, _ = env_name.partition(":")
    factory = _ENV_REGISTRY.get(prefix.lower()) if has_prefix else None

    if factory is not None:
        try:
            return factory(env_name, num_envs_per_device, kwargs_payload)
        except Exception as exc:
            raise EnvironmentNotFoundError(
                f"Failed to construct environment '{env_name}' with registered prefix '{prefix}'."
            ) from exc

    try:
        base_env, env_params = make_gymnax_env(env_name, **kwargs_payload)
        env = GymnaxToStoa(base_env, env_params)
        env = RecordEpisodeMetrics(env)
        env = AutoResetWrapper(env, next_obs_in_extras=True)
        env = VmapWrapper(env, num_envs=num_envs_per_device)
        return env, env_params
    except Exception as exc:
        raise EnvironmentNotFoundError(
            f"Unable to construct environment '{env_name}' from registry or Gymnax fallback."
        ) from exc