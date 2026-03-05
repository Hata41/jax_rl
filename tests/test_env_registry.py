import jax_rl.envs.env as env_module
import pytest

from jax_rl.utils.exceptions import EnvironmentNotFoundError


def test_register_env_decorator_dispatches_dummy_prefix(monkeypatch):
    original_registry = dict(env_module._ENV_REGISTRY)
    calls: list[tuple[str, int]] = []

    def _unexpected_gymnax(_name: str):
        raise AssertionError("Gymnax fallback should not be used for registered prefixes")

    monkeypatch.setattr(env_module, "make_gymnax_env", _unexpected_gymnax)

    try:
        @env_module.register_env("dummy")
        def _dummy_factory(env_name: str, num_envs_per_device: int):
            calls.append((env_name, num_envs_per_device))
            return {"backend": "dummy"}, {"task": env_name}

        env, env_params = env_module.make_stoa_env("dummy:test-v0", 1)

        assert calls == [("dummy:test-v0", 1)]
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
