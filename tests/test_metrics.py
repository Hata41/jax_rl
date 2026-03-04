import jax
import numpy as np
import importlib
from types import SimpleNamespace

from jax_rl.config import PPOConfig
from jax_rl.logging import extract_completed_episode_metrics
from jax_rl.train import (
    train,
)
from jax_rl.types import LogEvent


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
    train_module = importlib.import_module("jax_rl.train")

    config = _tiny_config(eval_episodes=0)
    captured = []

    class _CaptureLogger:
        def __init__(self):
            self.sinks = []

        def log_config(self, _config):
            return

        def materialize(self, metrics, event):
            prefix = "" if event is LogEvent.ABSOLUTE else f"{event.value}/"
            return {f"{prefix}{k}": float(np.asarray(v)) for k, v in metrics.items()}

        def log(self, metrics, step, event):
            captured.append((event, step, dict(metrics)))

        def flush(self):
            return

        def close(self):
            return

    time_values = iter([0.0, 1.0, 2.0, 3.0])

    def fake_time():
        return next(time_values)

    monkeypatch.setattr(
        train_module.jaxRL_Logger,
        "from_config",
        staticmethod(lambda _config: _CaptureLogger()),
    )
    monkeypatch.setattr(train_module, "time", SimpleNamespace(time=fake_time))

    train(config)

    flat = {}
    for event, _step, metrics in captured:
        prefix = "" if event is LogEvent.ABSOLUTE else f"{event.value}/"
        flat.update({f"{prefix}{k}": float(np.asarray(v)) for k, v in metrics.items()})

    expected_act_sps = config.num_envs * config.num_steps
    expected_train_sps = config.update_epochs * (config.rollout_batch_size // config.minibatch_size)

    assert flat["act/steps_per_second"] == expected_act_sps
    assert flat["train/steps_per_second"] == expected_train_sps


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

    metrics = extract_completed_episode_metrics(rollout_infos)

    assert metrics["episode_return"] == 300.0
    assert metrics["episode_length"] == 60.0


def test_metric_prefix_enforcement_with_eval(monkeypatch):
    train_module = importlib.import_module("jax_rl.train")

    config = _tiny_config(eval_episodes=1, eval_every=1)
    captured = []

    class _CaptureLogger:
        def __init__(self):
            self.sinks = []

        def log_config(self, _config):
            return

        def materialize(self, metrics, event):
            prefix = "" if event is LogEvent.ABSOLUTE else f"{event.value}/"
            return {f"{prefix}{k}": float(np.asarray(v)) for k, v in metrics.items()}

        def log(self, metrics, step, event):
            captured.append((event, step, dict(metrics)))

        def flush(self):
            return

        def close(self):
            return

    def fake_evaluate(_params, _config, num_episodes, max_steps_per_episode=1_000):
        return {
            "return_mean": 1.0,
            "return_std": 0.0,
            "return_min": 1.0,
            "return_max": 1.0,
            "episodes": int(num_episodes),
            "steps": 10,
        }

    monkeypatch.setattr(
        train_module.jaxRL_Logger,
        "from_config",
        staticmethod(lambda _config: _CaptureLogger()),
    )
    monkeypatch.setattr(train_module, "evaluate", fake_evaluate)

    train(config)

    assert captured
    observed_events = {event for event, _, _ in captured}
    assert LogEvent.ACT in observed_events
    assert LogEvent.TRAIN in observed_events
    assert LogEvent.EVAL in observed_events
    assert LogEvent.ABSOLUTE in observed_events
