from gymnax import make as make_gymnax_env
from stoa import AutoResetWrapper, RecordEpisodeMetrics
from stoa.env_adapters.gymnax import GymnaxToStoa


def make_stoa_env(env_name: str):
    base_env, env_params = make_gymnax_env(env_name)
    env = GymnaxToStoa(base_env, env_params)
    env = RecordEpisodeMetrics(env)
    env = AutoResetWrapper(env, next_obs_in_extras=True)
    return env, env_params