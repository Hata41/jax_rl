from pathlib import Path
from typing import Any
from dataclasses import asdict

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .configs.config import (
    ArchConfig,
    CheckpointConfig,
    EnvConfig,
    ExperimentConfig,
    IOConfig,
    LoggerConfig,
    SystemConfig,
    register_configs,
)
from .configs.evaluations import resolve_eval_env
from .utils.runtime import configure_jax_runtime_defaults

configure_jax_runtime_defaults()

register_configs()

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "config")


def _has_explicit_tensorboard_run_name_override(overrides: list[str] | None) -> bool:
    if not overrides:
        return False
    for override in overrides:
        key = override.split("=", 1)[0].lstrip("+")
        if key == "logging.tensorboard_run_name" or key.endswith(
            ".logging.tensorboard_run_name"
        ):
            return True
    return False


def _sanitize_run_token(value: str) -> str:
    return (
        str(value)
        .replace(":", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def inject_run_id(
    config: ExperimentConfig,
    *,
    preserve_tensorboard_run_name: bool = False,
    run_name_override: str | None = None,
) -> tuple[ExperimentConfig, str]:
    del preserve_tensorboard_run_name, run_name_override

    if getattr(config, "checkpointing", None) is not None:
        config.io.checkpoint = config.checkpointing
        if config.io.name is None and getattr(config.checkpointing, "checkpoint_name", None):
            config.io.name = str(config.checkpointing.checkpoint_name)
    if getattr(config, "logging", None) is not None:
        config.io.logger = config.logging
        if config.io.name is None and getattr(config.logging, "tensorboard_run_name", None):
            config.io.name = str(config.logging.tensorboard_run_name)

    system_name = _sanitize_run_token(str(config.system.name).lower())
    io_name = _sanitize_run_token(str(config.io.name)) if config.io.name else system_name
    checkpoint_path = Path(config.io.checkpoint.checkpoint_dir) / system_name / io_name

    config.io.checkpoint.checkpoint_dir = str(checkpoint_path)
    config.io.name = io_name
    run_id = f"{system_name}_{io_name}"
    return config, run_id


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="binpack/ppo")
def main(cfg: DictConfig) -> None:
    arch_cfg = cfg.get("arch") or {}
    platform = arch_cfg.get("platform") if hasattr(arch_cfg, "get") else None
    cuda_visible_devices = (
        arch_cfg.get("cuda_visible_devices") if hasattr(arch_cfg, "get") else None
    )
    configure_jax_runtime_defaults(
        platform=platform,
        cuda_visible_devices=cuda_visible_devices,
    )

    from .systems.ppo.eval import evaluate as ppo_evaluate

    typed = OmegaConf.to_object(cfg)

    def _coerce_experiment_config(mapping: dict[str, Any]) -> ExperimentConfig:
        payload = dict(mapping)

        env_value = payload.get("env")
        if isinstance(env_value, dict):
            payload["env"] = EnvConfig(**env_value)

        arch_value = payload.get("arch")
        if isinstance(arch_value, dict):
            payload["arch"] = ArchConfig(**arch_value)

        system_value = payload.get("system")
        if isinstance(system_value, dict):
            payload["system"] = SystemConfig(**system_value)

        checkpoint_value = payload.get("checkpointing")
        logging_value = payload.get("logging")
        io_value = payload.get("io")

        if io_value is not None and (checkpoint_value is not None or logging_value is not None):
            raise ValueError(
                "Config defines both legacy keys ('logging'/'checkpointing') and new key ('io'). "
                "Use only 'io'."
            )

        if isinstance(io_value, dict):
            io_payload = dict(io_value)
            checkpoint_payload = io_payload.get("checkpoint")
            if isinstance(checkpoint_payload, dict):
                io_payload["checkpoint"] = CheckpointConfig(**checkpoint_payload)
            logger_payload = io_payload.get("logger")
            if isinstance(logger_payload, dict):
                io_payload["logger"] = LoggerConfig(**logger_payload)
            payload["io"] = IOConfig(**io_payload)
        elif io_value is None and (checkpoint_value is not None or logging_value is not None):
            io_payload: dict[str, Any] = {}
            if isinstance(checkpoint_value, dict):
                io_payload["checkpoint"] = CheckpointConfig(**checkpoint_value)
                legacy_checkpoint_name = checkpoint_value.get("checkpoint_name")
                if legacy_checkpoint_name:
                    io_payload["name"] = str(legacy_checkpoint_name)
            if isinstance(logging_value, dict):
                io_payload["logger"] = LoggerConfig(
                    log_every=int(logging_value.get("log_every", 10)),
                    tensorboard_logdir=logging_value.get("tensorboard_logdir"),
                )
                legacy_tb_name = logging_value.get("tensorboard_run_name")
                if legacy_tb_name and "name" not in io_payload:
                    io_payload["name"] = str(legacy_tb_name)
            payload["io"] = IOConfig(**io_payload)

        return ExperimentConfig(**payload)
    expected_keys = {
        "env",
        "arch",
        "system",
        "io",
        "network",
        "evaluations",
    }

    if isinstance(typed, ExperimentConfig):
        config = typed
    elif isinstance(typed, dict):
        if len(typed) == 1:
            only_value = next(iter(typed.values()))
            if isinstance(only_value, ExperimentConfig):
                config = only_value
            elif isinstance(only_value, dict):
                config = _coerce_experiment_config({str(k): v for k, v in only_value.items()})
            else:
                typed_config: dict[str, Any] = {str(key): value for key, value in typed.items()}
                config = _coerce_experiment_config(typed_config)
        else:
            wrapper_config: ExperimentConfig | None = None
            for root_key in ("uldenv", "binpack", "jaxpallet"):
                if root_key not in typed:
                    continue
                root_value = typed[root_key]
                if isinstance(root_value, ExperimentConfig):
                    merged_full = asdict(root_value)
                    merged = {key: merged_full[key] for key in expected_keys if key in merged_full}
                elif isinstance(root_value, dict):
                    merged = {str(k): v for k, v in root_value.items()}
                else:
                    continue
                for key in expected_keys:
                    if key in typed:
                        merged[key] = typed[key]
                wrapper_config = _coerce_experiment_config(merged)
                break

            if wrapper_config is not None:
                config = wrapper_config
            else:
                typed_config = {str(key): value for key, value in typed.items() if str(key) in expected_keys}
                config = _coerce_experiment_config(typed_config)
    else:
        raise TypeError(
            f"Hydra config did not resolve to ExperimentConfig. Got: {type(typed)}"
        )

    _ = HydraConfig.get().overrides.task
    config, run_id = inject_run_id(config)
    print(f"Starting run: {run_id}")

    system_name = str(config.system.name).lower()
    if system_name == "ppo":
        from .systems.ppo.anakin.system import train as train_fn
    elif system_name == "alphazero":
        from .systems.alphazero.system import train as train_fn
    elif system_name == "spo":
        from .systems.spo.system import train as train_fn
    else:
        raise ValueError(
            f"Unsupported system.name '{config.system.name}'. Expected 'ppo', 'alphazero' or 'spo'."
        )

    output = train_fn(config)
    if output.get("tensorboard_run_dir"):
        print(f"tensorboard_run_dir={output['tensorboard_run_dir']}")
    for eval_name, eval_cfg in (config.evaluations or {}).items():
        eval_cfg = dict(eval_cfg)
        num_episodes = int(eval_cfg.get("num_episodes", 10))
        if num_episodes <= 0:
            continue
        env_name, env_kwargs = resolve_eval_env(
            eval_cfg,
            default_env_name=config.env.env_name,
            default_env_kwargs=config.env.env_kwargs,
        )
        if system_name == "alphazero":
            from .systems.alphazero.eval import evaluate as alphazero_evaluate

            eval_metrics = alphazero_evaluate(
                params=output["params"],
                config=config,
                env_name=env_name,
                seed=config.env.seed,
                num_episodes=num_episodes,
                max_steps_per_episode=int(eval_cfg.get("max_steps_per_episode", 1_000)),
                greedy=bool(eval_cfg.get("greedy", True)),
                env_kwargs=env_kwargs,
                action_selection=str(eval_cfg.get("action_selection", "policy")),
            )
        elif system_name == "spo":
            from .systems.spo.eval import evaluate as spo_evaluate

            eval_metrics = spo_evaluate(
                params=output["params"],
                config=config,
                env_name=env_name,
                seed=config.env.seed,
                num_episodes=num_episodes,
                max_steps_per_episode=int(eval_cfg.get("max_steps_per_episode", 1_000)),
                greedy=bool(eval_cfg.get("greedy", True)),
                env_kwargs=env_kwargs,
                action_selection=str(eval_cfg.get("action_selection", "policy")),
            )
        else:
            eval_metrics = ppo_evaluate(
                params=output["params"],
                env_name=env_name,
                seed=config.env.seed,
                num_episodes=num_episodes,
                max_steps_per_episode=int(eval_cfg.get("max_steps_per_episode", 1_000)),
                greedy=bool(eval_cfg.get("greedy", True)),
                env_kwargs=env_kwargs,
            )
        eval_str = " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sorted(eval_metrics.items())
        )
        print(f"eval[{eval_name}] {eval_str}")


if __name__ == "__main__":
    main()