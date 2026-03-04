import jax.numpy as jnp
from gymnax import make as make_gymnax_env
from stoa import AddRNGKey, AutoResetWrapper, RecordEpisodeMetrics
from stoa.core_wrappers.episode_metrics import RecordEpisodeMetricsState
from stoa.core_wrappers.vmap import VmapWrapper
from stoa.core_wrappers.wrapper import Wrapper
from stoa.env_adapters.gymnax import GymnaxToStoa


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
        timestep = timestep.replace(extras=new_extras)
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

        return timestep.replace(observation=new_observation, extras=new_extras)

    def reset(self, rng_key, env_params=None):
        state, timestep = self._env.reset(rng_key, env_params)
        return state, self._normalize_timestep(timestep)

    def step(self, state, action, env_params=None):
        state, timestep = self._env.step(state, action, env_params)
        return state, self._normalize_timestep(timestep)


def make_stoa_env(env_name: str, num_envs_per_device: int):
    if env_name.startswith("rustpool:"):
        task_id = env_name.split(":", 1)[1]
        from rustpool.envpool_api.stoa_wrapper import StoaRustpoolWrapper

        env = StoaRustpoolWrapper(task_id=task_id, num_envs_per_device=num_envs_per_device)
        env = RustpoolObsWrapper(env)
        env = BatchedRecordEpisodeMetrics(env)
        return env, None

    if env_name.startswith("jaxpallet:"):
        preset = env_name.split(":", 1)[1]
        from jaxpallet.stoa_adapter import JaxPalletToStoa

        env = JaxPalletToStoa(preset=preset)
        env = AddRNGKey(env)
        env = RecordEpisodeMetrics(env)
        env = AutoResetWrapper(env, next_obs_in_extras=True)
        env = VmapWrapper(env, num_envs=num_envs_per_device)
        return env, None

    base_env, env_params = make_gymnax_env(env_name)
    env = GymnaxToStoa(base_env, env_params)
    env = RecordEpisodeMetrics(env)
    env = AutoResetWrapper(env, next_obs_in_extras=True)
    env = VmapWrapper(env, num_envs=num_envs_per_device)
    return env, env_params