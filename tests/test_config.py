from pathlib import Path
from dataclasses import asdict

import pytest
from hydra.errors import ConfigCompositionException
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import jax

from jax_rl.configs.config import (
    ArchConfig,
    CheckpointConfig,
    EnvConfig,
    ExperimentConfig,
    LoggingConfig,
    SystemConfig,
    register_configs,
)
from jax_rl.networks import init_policy_value_params


def _config_dir() -> str:
    return str(Path(__file__).resolve().parents[1] / "config")


def _to_typed_config(cfg) -> ExperimentConfig:
    obj = OmegaConf.to_object(cfg)
    expected_keys = {
        "env",
        "arch",
        "system",
        "checkpointing",
        "logging",
        "network",
        "evaluations",
    }

    def _coerce(mapping: dict) -> ExperimentConfig:
        payload = dict(mapping)
        if isinstance(payload.get("env"), dict):
            payload["env"] = EnvConfig(**payload["env"])
        if isinstance(payload.get("arch"), dict):
            payload["arch"] = ArchConfig(**payload["arch"])
        if isinstance(payload.get("system"), dict):
            payload["system"] = SystemConfig(**payload["system"])
        if isinstance(payload.get("checkpointing"), dict):
            payload["checkpointing"] = CheckpointConfig(**payload["checkpointing"])
        if isinstance(payload.get("logging"), dict):
            payload["logging"] = LoggingConfig(**payload["logging"])
        return ExperimentConfig(**payload)

    if isinstance(obj, dict) and len(obj) == 1:
        only_value = next(iter(obj.values()))
        if isinstance(only_value, ExperimentConfig):
            return only_value
        if isinstance(only_value, dict):
            return _coerce(only_value)

    if isinstance(obj, dict):
        for root_key in ("uldenv", "binpack", "jaxpallet"):
            if root_key not in obj:
                continue
            root_value = obj[root_key]
            if isinstance(root_value, ExperimentConfig):
                merged = asdict(root_value)
                for key in expected_keys:
                    if key in obj:
                        merged[key] = obj[key]
                return _coerce(merged)
            if isinstance(root_value, dict):
                merged = dict(root_value)
                for key in expected_keys:
                    if key in obj:
                        merged[key] = obj[key]
                return _coerce(merged)

        filtered = {key: value for key, value in obj.items() if key in expected_keys}
        if filtered:
            return _coerce(filtered)
    assert isinstance(obj, ExperimentConfig)
    return obj


def test_hydra_compose_loads_ppo_train_binpack_and_converts_to_typed_config():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="binpack/ppo")

    obj = _to_typed_config(cfg)
    assert obj.system.name == "ppo"


def test_hydra_compose_loads_ppo_train_rlpallet_uld_yaml_and_converts_to_typed_config():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="uldenv/ppo")

    obj = _to_typed_config(cfg)
    assert obj.env.env_name == "rlpallet:UldEnv-v2"


def test_hydra_compose_loads_train_alphazero_yaml_and_converts_to_typed_config():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="binpack/alphazero")

    obj = _to_typed_config(cfg)
    assert obj.system.name == "alphazero"


def test_hydra_compose_overrides_apply_to_typed_config():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(
            config_name="binpack/ppo",
            overrides=[
                "binpack.system.actor_lr=0.005",
                "binpack.arch.num_envs=32",
                "binpack.arch.cuda_visible_devices='0,1'",
                "binpack.env.env_kwargs.max_items=42",
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
                config_name="binpack/ppo",
                overrides=["binpack.this_key_does_not_exist=10"],
            )


def test_binpack_network_config_instantiates_without_base_mlp_keys():
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="binpack/ppo")

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
