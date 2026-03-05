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


def test_hydra_compose_loads_train_yaml_and_converts_to_typed_config():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="train")

    obj = OmegaConf.to_object(cfg)
    assert isinstance(obj, ExperimentConfig)


def test_hydra_compose_overrides_apply_to_typed_config():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(
            config_name="train",
            overrides=["system.actor_lr=0.005", "system.num_envs=32"],
        )

    obj = OmegaConf.to_object(cfg)
    assert isinstance(obj, ExperimentConfig)
    assert obj.system.actor_lr == pytest.approx(0.005)
    assert obj.system.num_envs == 32


def test_hydra_compose_rejects_unknown_override_key():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        with pytest.raises(ConfigCompositionException):
            compose(config_name="train", overrides=["this_key_does_not_exist=10"])


def test_binpack_network_config_instantiates_without_base_mlp_keys():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="train")

    obj = OmegaConf.to_object(cfg)
    assert isinstance(obj, ExperimentConfig)

    params = init_policy_value_params(
        key=jax.random.PRNGKey(0),
        network_config=obj.network,
        obs_dim=10,
        action_dims=40 * 20 * 6,
        ems_feature_dim=6,
        item_feature_dim=3,
    )
    assert params is not None
