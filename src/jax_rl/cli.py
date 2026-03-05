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
    if config.eval_episodes > 0:
        eval_metrics = evaluate(output["params"], config, num_episodes=config.eval_episodes)
        eval_str = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(eval_metrics.items()))
        print(f"eval {eval_str}")


if __name__ == "__main__":
    main()