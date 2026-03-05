from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from .configs.config import ExperimentConfig, register_configs
from .utils.runtime import configure_jax_runtime_defaults

configure_jax_runtime_defaults()

register_configs()

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "config")


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="train")
def main(cfg: DictConfig) -> None:
    system_cfg = cfg.get("system") or {}
    platform = system_cfg.get("platform") if hasattr(system_cfg, "get") else None
    cuda_visible_devices = (
        system_cfg.get("cuda_visible_devices") if hasattr(system_cfg, "get") else None
    )
    configure_jax_runtime_defaults(
        platform=platform,
        cuda_visible_devices=cuda_visible_devices,
    )

    from .systems.ppo.eval import evaluate
    from .systems.ppo.anakin.system import train

    typed = OmegaConf.to_object(cfg)

    if isinstance(typed, ExperimentConfig):
        config = typed
    elif isinstance(typed, dict):
        typed_config: dict[str, Any] = {str(key): value for key, value in typed.items()}
        config = ExperimentConfig(**typed_config)
    else:
        raise TypeError(
            f"Hydra config did not resolve to ExperimentConfig. Got: {type(typed)}"
        )

    output = train(config)
    if output.get("tensorboard_run_dir"):
        print(f"tensorboard_run_dir={output['tensorboard_run_dir']}")
    for eval_name, eval_cfg in (config.evaluations or {}).items():
        eval_cfg = dict(eval_cfg)
        num_episodes = int(eval_cfg.get("num_episodes", 10))
        if num_episodes <= 0:
            continue
        env_name = str(eval_cfg.get("env_name", config.env.env_name))
        eval_metrics = evaluate(
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