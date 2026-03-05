from pathlib import Path

import pytest
from hydra.errors import ConfigCompositionException
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from jax_rl.configs.config import PPOConfig, register_configs


def _config_dir() -> str:
    return str(Path(__file__).resolve().parents[1] / "config")


def test_hydra_compose_loads_train_yaml_and_converts_to_typed_config():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="train")

    obj = OmegaConf.to_object(cfg)
    assert isinstance(obj, PPOConfig)


def test_hydra_compose_overrides_apply_to_typed_config():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(
            config_name="train",
            overrides=["actor_lr=0.005", "num_envs=32"],
        )

    obj = OmegaConf.to_object(cfg)
    assert isinstance(obj, PPOConfig)
    assert obj.actor_lr == pytest.approx(0.005)
    assert obj.num_envs == 32


def test_hydra_compose_rejects_unknown_override_key():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        with pytest.raises(ConfigCompositionException):
            compose(config_name="train", overrides=["this_key_does_not_exist=10"])
