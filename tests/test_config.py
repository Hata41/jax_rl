from pathlib import Path

import pytest
from hydra.errors import ConfigCompositionException
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import jax

from jax_rl.configs.config import ExperimentConfig, register_configs
from jax_rl.networks import init_policy_value_params


def _config_dir() -> str:
    return str(Path(__file__).resolve().parents[1] / "config")


def _to_typed_config(cfg) -> ExperimentConfig:
    obj = OmegaConf.to_object(cfg)
    if isinstance(obj, dict) and len(obj) == 1:
        only_value = next(iter(obj.values()))
        if isinstance(only_value, ExperimentConfig):
            return only_value
    assert isinstance(obj, ExperimentConfig)
    return obj


def test_hydra_compose_loads_ppo_train_binpack_and_converts_to_typed_config():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="ppo/train_binpack")

    obj = _to_typed_config(cfg)
    assert obj.system.name == "ppo"


def test_hydra_compose_loads_ppo_train_rlpallet_uld_yaml_and_converts_to_typed_config():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="ppo/train_uldenv")

    obj = _to_typed_config(cfg)
    assert obj.env.env_name == "rlpallet:UldEnv-v2"


def test_hydra_compose_loads_train_alphazero_yaml_and_converts_to_typed_config():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="alphazero/train_binpack")

    obj = _to_typed_config(cfg)
    assert obj.system.name == "alphazero"


def test_hydra_compose_overrides_apply_to_typed_config():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(
            config_name="ppo/train_binpack",
            overrides=[
                "ppo.system.actor_lr=0.005",
                "ppo.arch.num_envs=32",
                "ppo.arch.cuda_visible_devices='0,1'",
                "ppo.env.env_kwargs.max_items=42",
            ],
        )

    obj = _to_typed_config(cfg)
    assert obj.system.actor_lr == pytest.approx(0.005)
    assert obj.arch.num_envs == 32
    assert obj.arch.cuda_visible_devices == "0,1"
    assert obj.env.env_kwargs["max_items"] == 42


def test_hydra_compose_rejects_unknown_override_key():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        with pytest.raises(ConfigCompositionException):
            compose(
                config_name="ppo/train_binpack",
                overrides=["ppo.this_key_does_not_exist=10"],
            )


def test_binpack_network_config_instantiates_without_base_mlp_keys():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="ppo/train_binpack")

    obj = _to_typed_config(cfg)

    params = init_policy_value_params(
        key=jax.random.PRNGKey(0),
        network_config=obj.network,
        obs_dim=10,
        action_dims=40 * 20 * 6,
        ems_feature_dim=6,
        item_feature_dim=3,
    )
    assert params is not None
