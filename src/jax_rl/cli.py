from pathlib import Path
import time
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .configs.config import ExperimentConfig, register_configs
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
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_token_source = run_name_override if run_name_override else timestamp
    run_token = _sanitize_run_token(run_token_source)
    safe_env_name = _sanitize_run_token(config.env.env_name)
    run_id = f"{config.system.name}_{safe_env_name}_{run_token}"

    checkpoint_path = (
        Path(config.checkpointing.checkpoint_dir)
        / str(config.system.name)
        / safe_env_name
        / run_token
    )
    checkpoint_name = getattr(config.checkpointing, "checkpoint_name", None)
    if checkpoint_name:
        checkpoint_path = checkpoint_path / _sanitize_run_token(checkpoint_name)

    config.checkpointing.checkpoint_dir = str(checkpoint_path)

    if config.logging.tensorboard_logdir and not preserve_tensorboard_run_name:
        config.logging.tensorboard_run_name = run_id

    return config, run_id


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="ppo/train_binpack")
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

    if isinstance(typed, ExperimentConfig):
        config = typed
    elif isinstance(typed, dict):
        if len(typed) == 1:
            only_value = next(iter(typed.values()))
            if isinstance(only_value, ExperimentConfig):
                config = only_value
            else:
                typed_config: dict[str, Any] = {
                    str(key): value for key, value in typed.items()
                }
                config = ExperimentConfig(**typed_config)
        else:
            typed_config: dict[str, Any] = {str(key): value for key, value in typed.items()}
            config = ExperimentConfig(**typed_config)
    else:
        raise TypeError(
            f"Hydra config did not resolve to ExperimentConfig. Got: {type(typed)}"
        )

    hydra_overrides = HydraConfig.get().overrides.task
    preserve_tb_run_name = _has_explicit_tensorboard_run_name_override(hydra_overrides)
    run_name_override = (
        str(config.logging.tensorboard_run_name) if preserve_tb_run_name else None
    )
    config, run_id = inject_run_id(
        config,
        preserve_tensorboard_run_name=preserve_tb_run_name,
        run_name_override=run_name_override,
    )
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
        env_name = str(eval_cfg.get("env_name", config.env.env_name))
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
                env_kwargs=dict(eval_cfg.get("env_kwargs", config.env.env_kwargs)),
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
                env_kwargs=dict(eval_cfg.get("env_kwargs", config.env.env_kwargs)),
            )
        eval_str = " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sorted(eval_metrics.items())
        )
        print(f"eval[{eval_name}] {eval_str}")


if __name__ == "__main__":
    main()