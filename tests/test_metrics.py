import jax
import numpy as np
import importlib
from types import SimpleNamespace

from purejax_ppo.config import PPOConfig
from purejax_ppo.train import (
    _extract_completed_episode_metrics,
    train,
)


def _tiny_config(**overrides):
    num_envs = jax.local_device_count()
    config = PPOConfig(
        env_name="CartPole-v1",
        seed=0,
        total_timesteps=num_envs * 8,
        num_envs=num_envs,
        num_steps=8,
        minibatch_size=num_envs * 8,
        update_epochs=2,
        hidden_size=16,
        hidden_layers=1,
        log_every=1,
        eval_every=1,
        eval_episodes=0,
    )
    return PPOConfig(**{**config.__dict__, **overrides})


def test_sps_calculation_validity(monkeypatch):
    train_module = importlib.import_module("purejax_ppo.train")

    config = _tiny_config(eval_episodes=0)
    captured = {}

    def fake_log_scalar_metrics(_writer, metrics, _step):
        captured.update(metrics)

    time_values = iter([0.0, 1.0, 2.0, 3.0])

    def fake_time():
        return next(time_values)

    monkeypatch.setattr(train_module, "log_scalar_metrics", fake_log_scalar_metrics)
    monkeypatch.setattr(train_module, "time", SimpleNamespace(time=fake_time))

    train(config)

    expected_act_sps = config.num_envs * config.num_steps
    expected_train_sps = config.update_epochs * (config.rollout_batch_size // config.minibatch_size)

    assert captured["act/steps_per_second"] == expected_act_sps
    assert captured["train/steps_per_second"] == expected_train_sps


def test_episode_masking_uses_completed_only():
    rollout_infos = {
        "episode_return": np.array(
            [[10.0, 200.0], [300.0, 400.0]], dtype=np.float32
        ),
        "episode_length": np.array(
            [[5.0, 50.0], [60.0, 70.0]], dtype=np.float32
        ),
        "is_terminal_step": np.array([[False, False], [True, False]], dtype=bool),
    }

    metrics = _extract_completed_episode_metrics(rollout_infos)

    assert metrics["act/episode_return"] == 300.0
    assert metrics["act/episode_length"] == 60.0


def test_metric_prefix_enforcement_with_eval(monkeypatch):
    train_module = importlib.import_module("purejax_ppo.train")

    config = _tiny_config(eval_episodes=1, eval_every=1)
    captured = {}

    def fake_log_scalar_metrics(_writer, metrics, _step):
        captured.update(metrics)

    def fake_evaluate(_params, _config, num_episodes, max_steps_per_episode=1_000):
        return {
            "return_mean": 1.0,
            "return_std": 0.0,
            "return_min": 1.0,
            "return_max": 1.0,
            "episodes": int(num_episodes),
            "steps": 10,
        }

    monkeypatch.setattr(train_module, "log_scalar_metrics", fake_log_scalar_metrics)
    monkeypatch.setattr(train_module, "evaluate", fake_evaluate)

    train(config)

    assert captured
    assert any(key.startswith("eval/") for key in captured)
    assert all(
        key.startswith(("act/", "train/", "eval/", "misc/"))
        for key in captured
    )
