import hydra
from omegaconf import DictConfig, OmegaConf

from .configs.config import PPOConfig, register_configs
from .utils.runtime import configure_jax_runtime_defaults

configure_jax_runtime_defaults()

from .systems.ppo.eval import evaluate
from .systems.ppo.anakin.system import train

register_configs()

@hydra.main(version_base=None, config_path="../../config", config_name="train")
def main(cfg: DictConfig) -> None:
    typed = OmegaConf.to_object(cfg)

    if isinstance(typed, PPOConfig):
        config = typed
    elif isinstance(typed, dict):
        config = PPOConfig(**typed)
    else:
        raise TypeError(f"Hydra config did not resolve to PPOConfig. Got: {type(typed)}")

    output = train(config)
    if output.get("tensorboard_run_dir"):
        print(f"tensorboard_run_dir={output['tensorboard_run_dir']}")
    for eval_name, eval_cfg in (config.evaluations or {}).items():
        eval_cfg = dict(eval_cfg)
        num_episodes = int(eval_cfg.get("num_episodes", 10))
        if num_episodes <= 0:
            continue
        env_name = str(eval_cfg.get("env_name", config.env_name))
        eval_metrics = evaluate(
            params=output["params"],
            env_name=env_name,
            seed=config.seed,
            num_episodes=num_episodes,
            max_steps_per_episode=int(eval_cfg.get("max_steps_per_episode", 1_000)),
            greedy=bool(eval_cfg.get("greedy", True)),
        )
        eval_str = " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sorted(eval_metrics.items())
        )
        print(f"eval[{eval_name}] {eval_str}")


if __name__ == "__main__":
    main()