import jax
import numpy as np
import importlib
from dataclasses import replace
from types import SimpleNamespace

from jax_rl.configs.config import ArchConfig, EnvConfig, ExperimentConfig, IOConfig, LoggingConfig, SystemConfig
from jax_rl.systems.ppo.eval import EvaluationManager
from jax_rl.systems.ppo.anakin.system import (
    train,
)
from jax_rl.utils.logging import extract_completed_episode_metrics
from jax_rl.utils.types import LogEvent


def _tiny_config(**overrides):
    num_envs = jax.local_device_count()
    config = ExperimentConfig(
        env=EnvConfig(env_name="CartPole-v1", seed=0),
        arch=ArchConfig(
            total_timesteps=num_envs * 8,
            num_envs=num_envs,
            num_steps=8,
        ),
        system=SystemConfig(
            minibatch_size=num_envs * 8,
            update_epochs=2,
        ),
        io=IOConfig(logger=LoggingConfig(log_every=1)),
        evaluations={},
    )

    updated_config = config
    for key, value in overrides.items():
        if key in {"env", "arch", "system", "io"} and isinstance(value, dict):
            nested_config = getattr(updated_config, key)
            updated_config = replace(updated_config, **{key: replace(nested_config, **value)})
            continue
        updated_config = replace(updated_config, **{key: value})

    return updated_config


def test_sps_calculation_validity(monkeypatch):
    train_module = importlib.import_module("jax_rl.systems.ppo.anakin.system")

    config = _tiny_config()
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

    expected_act_sps = config.arch.num_envs * config.arch.num_steps
    expected_train_sps = config.system.update_epochs * (
        config.rollout_batch_size // config.system.minibatch_size
    )

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

    assert set(metrics["episode_return"].keys()) >= {"mean", "min", "max"}
    assert set(metrics["episode_length"].keys()) >= {"mean", "min", "max"}
    assert metrics["episode_return"]["mean"] == 300.0
    assert metrics["episode_return"]["min"] == 300.0
    assert metrics["episode_return"]["max"] == 300.0
    assert metrics["episode_length"]["mean"] == 60.0
    assert metrics["episode_length"]["min"] == 60.0
    assert metrics["episode_length"]["max"] == 60.0


def test_train_console_output_excludes_reward_mean(capsys):
    config = _tiny_config()

    train(config)

    out = capsys.readouterr().out
    assert "Reward mean" not in out


def test_metric_prefix_enforcement_with_eval(monkeypatch):
    train_module = importlib.import_module("jax_rl.systems.ppo.anakin.system")

    config = _tiny_config(
        evaluations={
            "eval_1": {
                "env_name": "CartPole-v1",
                "eval_every": 1,
                "num_episodes": 1,
                "max_steps_per_episode": 16,
                "greedy": True,
            }
        }
    )
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

    class _FakeEvaluator:
        def __init__(self, env_name, num_episodes, max_steps_per_episode, greedy, env_kwargs=None):
            del env_name, max_steps_per_episode, greedy, env_kwargs
            self.num_episodes = int(num_episodes)

        def run(self, replicated_params, seed):
            del replicated_params, seed
            return {
                "return_mean": 1.0,
                "return_std": 0.0,
                "return_min": 1.0,
                "return_max": 1.0,
                "episodes": int(self.num_episodes),
                "steps": 10,
            }

        def close(self):
            return

    monkeypatch.setattr(
        train_module.jaxRL_Logger,
        "from_config",
        staticmethod(lambda _config: _CaptureLogger()),
    )
    monkeypatch.setattr(train_module, "Evaluator", _FakeEvaluator)

    train(config)

    assert captured
    observed_events = {event for event, _, _ in captured}
    assert LogEvent.ACT in observed_events
    assert LogEvent.TRAIN in observed_events
    assert LogEvent.EVAL in observed_events
    assert LogEvent.ABSOLUTE in observed_events


def test_multiple_evaluations_logging(monkeypatch):
    train_module = importlib.import_module("jax_rl.systems.ppo.anakin.system")

    config = _tiny_config(
        evaluations={
            "eval_1": {
                "env_name": "CartPole-v1",
                "eval_every": 1,
                "num_episodes": 1,
            },
            "eval_2": {
                "env_name": "CartPole-v1",
                "eval_every": 1,
                "num_episodes": 1,
                "greedy": False,
            },
        }
    )
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

    class _FakeEvaluator:
        def __init__(self, env_name, num_episodes, max_steps_per_episode, greedy, env_kwargs=None):
            del env_name, max_steps_per_episode, greedy, env_kwargs
            self.num_episodes = int(num_episodes)

        def run(self, replicated_params, seed):
            del replicated_params, seed
            return {
                "return_mean": float(self.num_episodes),
                "return_std": 0.0,
                "return_min": 1.0,
                "return_max": 1.0,
                "episodes": int(self.num_episodes),
                "steps": 10,
            }

        def close(self):
            return

    monkeypatch.setattr(
        train_module.jaxRL_Logger,
        "from_config",
        staticmethod(lambda _config: _CaptureLogger()),
    )
    monkeypatch.setattr(train_module, "Evaluator", _FakeEvaluator)

    train(config)

    eval_payloads = [metrics for event, _, metrics in captured if event is LogEvent.EVAL]
    assert eval_payloads
    merged_eval_metrics = {}
    for payload in eval_payloads:
        merged_eval_metrics.update(payload)

    assert "eval_1/return_mean" in merged_eval_metrics
    assert "eval_2/return_mean" in merged_eval_metrics


def test_spo_metric_prefix_enforcement_with_eval(monkeypatch):
    train_module = importlib.import_module("jax_rl.systems.spo.anakin.system")

    num_envs = max(jax.local_device_count(), 1)
    config = ExperimentConfig(
        env=EnvConfig(env_name="CartPole-v1", seed=0),
        arch=ArchConfig(total_timesteps=num_envs, num_envs=num_envs, num_steps=1),
        system=SystemConfig(name="spo", learner_updates_per_cycle=1),
        io=IOConfig(logger=LoggingConfig(log_every=1, tensorboard_logdir=None)),
        evaluations={
            "eval_1": {
                "env_name": "CartPole-v1",
                "eval_every": 1,
                "num_episodes": 1,
                "max_steps_per_episode": 16,
                "greedy": True,
                "action_selection": "policy",
            }
        },
    )

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

    fake_train_state = SimpleNamespace(
        opt_states=SimpleNamespace(actor_opt_state={"hyperparams": {"learning_rate": 3e-4}}),
        params=SimpleNamespace(actor_online={"params": 1.0}),
    )
    fake_runner_state = SimpleNamespace(
        train_state=fake_train_state,
        key=np.asarray(0),
    )
    fake_system = SimpleNamespace(
        env=None,
        env_params=None,
        actor_optimizer=None,
        critic_optimizer=None,
        dual_optimizer=None,
        is_rustpool=False,
        num_envs_per_device=1,
        buffer_add_fn=None,
        buffer_sample_fn=None,
        runner_state=fake_runner_state,
        checkpointer=SimpleNamespace(
            save=lambda **kwargs: True,
            checkpoint_path_for_step=lambda step: f"/tmp/ckpt/{step}",
        ),
    )

    def _fake_rollout(runner_state):
        return runner_state, (
            None,
            {
                "search_finite": np.asarray(True),
                "invalid_action_rate": np.asarray(0.0),
            },
        )

    def _fake_update(runner_state, _rollout_outputs):
        return runner_state, {
            "loss_is_finite": np.asarray(True),
            "search_finite": np.asarray(True),
            "loss_actor_dual": np.asarray(0.1),
        }

    monkeypatch.setattr(
        train_module.jaxRL_Logger,
        "from_config",
        staticmethod(lambda _config: _CaptureLogger()),
    )
    monkeypatch.setattr(train_module, "build_system", lambda _config, _runner_cls: fake_system)
    monkeypatch.setattr(train_module, "make_spo_steps", lambda **kwargs: (_fake_rollout, _fake_update))
    monkeypatch.setattr(train_module, "unreplicate_tree", lambda x: x)
    monkeypatch.setattr(
        train_module,
        "evaluate_spo",
        lambda **kwargs: {
            "return_mean": 1.0,
            "return_std": 0.0,
            "return_min": 1.0,
            "return_max": 1.0,
            "episodes": 1,
            "steps": 10,
        },
    )

    train_module.train(config)

    assert captured
    observed_events = {event for event, _, _ in captured}
    assert LogEvent.ACT in observed_events
    assert LogEvent.TRAIN in observed_events
    assert LogEvent.EVAL in observed_events
    assert LogEvent.ABSOLUTE in observed_events

    eval_payloads = [metrics for event, _, metrics in captured if event is LogEvent.EVAL]
    assert eval_payloads
    merged_eval_metrics = {}
    for payload in eval_payloads:
        merged_eval_metrics.update(payload)
    assert "eval_1/return_mean" in merged_eval_metrics


def test_alphazero_metric_prefix_enforcement_with_eval(monkeypatch):
    train_module = importlib.import_module("jax_rl.systems.alphazero.anakin.system")

    num_envs = max(jax.local_device_count(), 1)
    config = ExperimentConfig(
        env=EnvConfig(env_name="CartPole-v1", seed=0),
        arch=ArchConfig(total_timesteps=num_envs, num_envs=num_envs, num_steps=1),
        system=SystemConfig(name="alphazero", learner_updates_per_cycle=1),
        io=IOConfig(logger=LoggingConfig(log_every=1, tensorboard_logdir=None)),
        evaluations={
            "eval_1": {
                "env_name": "CartPole-v1",
                "eval_every": 1,
                "num_episodes": 1,
                "max_steps_per_episode": 16,
                "greedy": True,
                "action_selection": "policy",
            }
        },
    )

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

    fake_train_state = SimpleNamespace(
        actor_opt_state={"hyperparams": {"learning_rate": 3e-4}},
        params={"params": 1.0},
    )
    fake_runner_state = SimpleNamespace(
        train_state=fake_train_state,
        key=np.asarray(0),
    )
    fake_system = SimpleNamespace(
        env=None,
        env_params=None,
        actor_optimizer=None,
        critic_optimizer=None,
        is_rustpool=False,
        num_envs_per_device=1,
        buffer_add_fn=None,
        buffer_sample_fn=None,
        runner_state=fake_runner_state,
        checkpointer=SimpleNamespace(
            save=lambda **kwargs: True,
            checkpoint_path_for_step=lambda step: f"/tmp/ckpt/{step}",
        ),
    )

    def _fake_rollout(runner_state):
        return runner_state, (
            None,
            {
                "search_finite": np.asarray(True),
                "invalid_action_rate": np.asarray(0.0),
            },
        )

    def _fake_update(runner_state, _rollout_outputs):
        return runner_state, {
            "loss_is_finite": np.asarray(True),
            "search_finite": np.asarray(True),
            "loss_total": np.asarray(0.1),
        }

    class _FakeEvalManager:
        def __init__(self, **kwargs):
            del kwargs

        def run_if_needed(self, **kwargs):
            del kwargs
            return {
                "eval_1/return_mean": 1.0,
                "eval_1/return_std": 0.0,
                "eval_1/return_min": 1.0,
                "eval_1/return_max": 1.0,
                "eval_1/episodes": 1,
                "eval_1/steps": 10,
                "eval_1/steps_per_second": 10.0,
            }

        def close(self):
            return

    monkeypatch.setattr(
        train_module.jaxRL_Logger,
        "from_config",
        staticmethod(lambda _config: _CaptureLogger()),
    )
    monkeypatch.setattr(train_module, "build_system", lambda _config, _runner_cls: fake_system)
    monkeypatch.setattr(train_module, "make_alphazero_steps", lambda **kwargs: (_fake_rollout, _fake_update))
    monkeypatch.setattr(train_module, "EvaluationManager", _FakeEvalManager)
    monkeypatch.setattr(train_module, "unreplicate_tree", lambda x: x)

    train_module.train(config)

    assert captured
    observed_events = {event for event, _, _ in captured}
    assert LogEvent.ACT in observed_events
    assert LogEvent.TRAIN in observed_events
    assert LogEvent.EVAL in observed_events
    assert LogEvent.ABSOLUTE in observed_events

    eval_payloads = [metrics for event, _, metrics in captured if event is LogEvent.EVAL]
    assert eval_payloads
    merged_eval_metrics = {}
    for payload in eval_payloads:
        merged_eval_metrics.update(payload)
    assert "eval_1/return_mean" in merged_eval_metrics


def test_evaluation_env_kwargs_default_and_override(monkeypatch):
    captured_env_kwargs = {}

    class _FakeEvaluator:
        def __init__(self, env_name, num_episodes, max_steps_per_episode, greedy, env_kwargs=None):
            del env_name, max_steps_per_episode, greedy
            self.num_episodes = int(num_episodes)
            self.env_kwargs = dict(env_kwargs or {})
            label = "eval_default"
            if self.env_kwargs.get("max_items") == 100:
                label = "eval_override"
            captured_env_kwargs[label] = self.env_kwargs

        def run(self, replicated_params, seed):
            del replicated_params, seed
            return {
                "return_mean": float(self.num_episodes),
                "return_std": 0.0,
                "return_min": 1.0,
                "return_max": 1.0,
                "episodes": int(self.num_episodes),
                "steps": 10,
            }

        def close(self):
            return

    manager = EvaluationManager(
        evaluations={
            "eval_default": {
                "env_name": "CartPole-v1",
                "eval_every": 1,
                "num_episodes": 1,
            },
            "eval_override": {
                "env_name": "CartPole-v1",
                "eval_every": 1,
                "num_episodes": 1,
                "env_kwargs": {"max_items": 100},
            },
        },
        default_env_name="CartPole-v1",
        default_env_kwargs={"max_items": 50},
        evaluator_cls=_FakeEvaluator,
    )

    _ = manager.run_if_needed(update_idx=0, params=object(), seed=0)

    manager.close()

    assert captured_env_kwargs["eval_default"] == {"max_items": 50}
    assert captured_env_kwargs["eval_override"] == {"max_items": 100}
