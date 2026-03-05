from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
import jax


@dataclass
class PPOConfig:
    env_name: str = "CartPole-v1"
    seed: int = 0
    total_timesteps: int = 100_000

    num_envs: int = 16
    num_steps: int = 128

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    update_epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5

    hidden_size: int = 64
    hidden_layers: int = 2
    network: dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "jax_rl.networks.PolicyValueModel",
            "hidden_sizes": [64, 64],
        }
    )

    log_every: int = 10
    evaluations: dict[str, dict[str, Any]] = field(default_factory=dict)
    checkpoint_dir: str = "checkpoints"
    save_interval_steps: int = 0
    max_to_keep: int = 1
    keep_period: Optional[int] = None
    resume_from: str | None = None
    tensorboard_logdir: str | None = None
    tensorboard_run_name: str = "default"

    @property
    def rollout_batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // self.rollout_batch_size

    @property
    def local_device_count(self) -> int:
        return jax.local_device_count()


_CONFIGS_REGISTERED = False


def register_configs() -> None:
    """Register structured config schemas for Hydra/OmegaConf."""
    global _CONFIGS_REGISTERED
    if _CONFIGS_REGISTERED:
        return

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=PPOConfig)
    _CONFIGS_REGISTERED = True