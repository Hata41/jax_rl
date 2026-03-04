import argparse
from dataclasses import fields

import yaml

from .config import PPOConfig
from .runtime import configure_jax_runtime_defaults

configure_jax_runtime_defaults()

from .eval import evaluate
from .train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Pure JAX PPO trainer")
    parser.add_argument("--config", default="config/train.yaml")
    return parser.parse_args()


def _normalize_optional_string(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def load_config_file(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if "training" not in raw or not isinstance(raw["training"], dict):
        raise ValueError("Config must define a [training] table.")

    section = dict(raw["training"])

    ppo_fields = {item.name for item in fields(PPOConfig)}
    unknown_keys = sorted(set(section.keys()) - ppo_fields)
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {', '.join(unknown_keys)}")

    for optional_key in ("resume_from", "tensorboard_logdir"):
        if optional_key in section:
            section[optional_key] = _normalize_optional_string(section[optional_key])

    config = PPOConfig(**section)
    return config


def main():
    args = parse_args()
    config = load_config_file(args.config)
    output = train(config)
    if output.get("tensorboard_run_dir"):
        print(f"tensorboard_run_dir={output['tensorboard_run_dir']}")
    if config.eval_episodes > 0:
        eval_metrics = evaluate(output["params"], config, num_episodes=config.eval_episodes)
        eval_str = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(eval_metrics.items()))
        print(f"eval {eval_str}")


if __name__ == "__main__":
    main()