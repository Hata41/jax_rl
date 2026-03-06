import jax_rl.envs.env as env_module
import pytest

from jax_rl.utils.exceptions import EnvironmentNotFoundError


def test_register_env_decorator_dispatches_dummy_prefix(monkeypatch):
    original_registry = dict(env_module._ENV_REGISTRY)
    calls: list[tuple[str, int, dict]] = []

    def _unexpected_gymnax(_name: str):
        raise AssertionError("Gymnax fallback should not be used for registered prefixes")

    monkeypatch.setattr(env_module, "make_gymnax_env", _unexpected_gymnax)

    try:
        @env_module.register_env("dummy")
        def _dummy_factory(env_name: str, num_envs_per_device: int, env_kwargs: dict):
            calls.append((env_name, num_envs_per_device, dict(env_kwargs)))
            return {"backend": "dummy"}, {"task": env_name}

        env, env_params = env_module.make_stoa_env(
            "dummy:test-v0",
            1,
            env_kwargs={"max_items": 42},
        )

        assert calls == [("dummy:test-v0", 1, {"max_items": 42})]
        assert env == {"backend": "dummy"}
        assert env_params == {"task": "dummy:test-v0"}
    finally:
        env_module._ENV_REGISTRY.clear()
        env_module._ENV_REGISTRY.update(original_registry)


def test_invalid_prefix_raises_environment_not_found_error(monkeypatch):
    def _failing_gymnax(_name: str):
        raise RuntimeError("unknown environment")

    monkeypatch.setattr(env_module, "make_gymnax_env", _failing_gymnax)

    with pytest.raises(EnvironmentNotFoundError):
        env_module.make_stoa_env("invalid_prefix:task", 1)


def test_gymnax_fallback_receives_env_kwargs(monkeypatch):
    captured = {}

    def _failing_gymnax(name: str, **kwargs):
        captured["name"] = name
        captured["kwargs"] = dict(kwargs)
        raise RuntimeError("boom")

    monkeypatch.setattr(env_module, "make_gymnax_env", _failing_gymnax)

    with pytest.raises(EnvironmentNotFoundError):
        env_module.make_stoa_env(
            "CartPole-v1",
            num_envs_per_device=1,
            env_kwargs={"max_items": 50},
        )

    assert captured["name"] == "CartPole-v1"
    assert captured["kwargs"] == {"max_items": 50}


def test_uldenv_v2_forces_max_items_and_episode_steps(monkeypatch):
    original_registry = dict(env_module._ENV_REGISTRY)
    calls: list[tuple[str, int, dict]] = []

    def _unexpected_gymnax(_name: str):
        raise AssertionError("Gymnax fallback should not be used for registered prefixes")

    monkeypatch.setattr(env_module, "make_gymnax_env", _unexpected_gymnax)

    try:
        @env_module.register_env("rlpallet")
        def _dummy_rlpallet_factory(env_name: str, num_envs_per_device: int, env_kwargs: dict):
            calls.append((env_name, num_envs_per_device, dict(env_kwargs)))
            return {"backend": "rlpallet"}, None

        env_module.make_stoa_env(
            "rlpallet:UldEnv-v2",
            1,
            env_kwargs={
                "target_groups": 7,
                "max_mult": 4,
                "max_items": 999,
                "max_episode_steps": 999,
            },
        )

        assert calls == [
            (
                "rlpallet:UldEnv-v2",
                1,
                {
                    "target_groups": 7,
                    "max_mult": 4,
                    "max_items": 28,
                    "max_episode_steps": 28,
                },
            )
        ]
    finally:
        env_module._ENV_REGISTRY.clear()
        env_module._ENV_REGISTRY.update(original_registry)


def test_non_uldenv_env_keeps_original_env_kwargs(monkeypatch):
    original_registry = dict(env_module._ENV_REGISTRY)
    calls: list[tuple[str, int, dict]] = []

    def _unexpected_gymnax(_name: str):
        raise AssertionError("Gymnax fallback should not be used for registered prefixes")

    monkeypatch.setattr(env_module, "make_gymnax_env", _unexpected_gymnax)

    try:
        @env_module.register_env("rlpallet")
        def _dummy_rlpallet_factory(env_name: str, num_envs_per_device: int, env_kwargs: dict):
            calls.append((env_name, num_envs_per_device, dict(env_kwargs)))
            return {"backend": "rlpallet"}, None

        env_module.make_stoa_env(
            "rlpallet:OtherEnv-v1",
            1,
            env_kwargs={
                "target_groups": 7,
                "max_mult": 4,
                "max_items": 999,
                "max_episode_steps": 999,
            },
        )

        assert calls == [
            (
                "rlpallet:OtherEnv-v1",
                1,
                {
                    "target_groups": 7,
                    "max_mult": 4,
                    "max_items": 999,
                    "max_episode_steps": 999,
                },
            )
        ]
    finally:
        env_module._ENV_REGISTRY.clear()
        env_module._ENV_REGISTRY.update(original_registry)
