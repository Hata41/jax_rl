from jax_rl.configs.config import EnvConfig
from jax_rl.configs.evaluations import resolve_eval_env


def test_resolve_eval_env_accepts_nested_env_dict():
    eval_cfg = {
        "env": {
            "env_name": "rlpallet:UldEnv-v2",
            "env_kwargs": {
                "generator_type": "twophase_physics",
                "nested": {"a": 1, "b": 2},
            },
        },
        "env_kwargs": {
            "nested": {"b": 3},
            "max_items": 50,
        },
    }

    env_name, env_kwargs = resolve_eval_env(
        eval_cfg,
        default_env_name="CartPole-v1",
        default_env_kwargs={"seed_mode": "train", "nested": {"a": 0}},
    )

    assert env_name == "rlpallet:UldEnv-v2"
    assert env_kwargs["seed_mode"] == "train"
    assert env_kwargs["generator_type"] == "twophase_physics"
    assert env_kwargs["max_items"] == 50
    assert env_kwargs["nested"] == {"a": 1, "b": 3}


def test_resolve_eval_env_accepts_envconfig_instance():
    eval_cfg = {
        "env": EnvConfig(
            env_name="rlpallet:UldEnv-v2",
            env_kwargs={"target_groups": 32},
        )
    }

    env_name, env_kwargs = resolve_eval_env(
        eval_cfg,
        default_env_name="CartPole-v1",
        default_env_kwargs={"max_items": 25},
    )

    assert env_name == "rlpallet:UldEnv-v2"
    assert env_kwargs == {"max_items": 25, "target_groups": 32}


def test_resolve_eval_env_keeps_legacy_flat_fields():
    eval_cfg = {
        "env_name": "Acrobot-v1",
        "env_kwargs": {"foo": "bar"},
    }

    env_name, env_kwargs = resolve_eval_env(
        eval_cfg,
        default_env_name="CartPole-v1",
        default_env_kwargs={"base": True},
    )

    assert env_name == "Acrobot-v1"
    assert env_kwargs == {"base": True, "foo": "bar"}
