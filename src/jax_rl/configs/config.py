from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
import jax


@dataclass
class EnvConfig:
    env_name: str = "CartPole-v1"
    seed: int = 0
    env_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchConfig:
    total_timesteps: int = 100_000
    platform: str | None = None
    cuda_visible_devices: str | None = None
    num_envs: int = 16
    num_steps: int = 25


@dataclass
class SystemConfig:
    name: str = "ppo"

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    optimizer: str = "adam"
    lr_schedule: str = "linear"
    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    update_epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5

    num_simulations: int = 8
    max_depth: int = 4
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    search_method: str = "muzero"
    search_method_kwargs: dict[str, Any] = field(default_factory=dict)

    total_buffer_size: int = 16_384
    total_batch_size: int = 1_024
    sample_sequence_length: int = 1
    period: int = 1
    warmup_steps: int = 0
    learner_updates_per_cycle: int = 1
    tree_memory_budget_mb: int = 512

    mpo_epsilon: float = 0.1
    mpo_epsilon_policy: float = 0.05
    dual_lr: float = 1e-4
    dual_init_log_temperature: float = -2.0
    dual_init_log_alpha: float = -2.0
    target_tau: float = 0.005
    num_particles: int = 8
    search_depth: int = 4
    search_gamma: float = 0.99
    search_gae_lambda: float = 0.95
    spo_resampling_mode: str = "ess"
    spo_resampling_period: int = 1
    spo_ess_threshold: float = 0.5


@dataclass
class CheckpointConfig:
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str | None = None
    transfer_weights_only: bool = False
    save_interval_steps: int = 0
    max_to_keep: int = 1
    keep_period: Optional[int] = None
    resume_from: str | None = None


@dataclass
class LoggingConfig:
    log_every: int = 10
    tensorboard_logdir: str | None = None
    tensorboard_run_name: str = "default"


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    arch: ArchConfig = field(default_factory=ArchConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    network: dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "jax_rl.networks.PolicyValueModel",
            "hidden_sizes": [64, 64],
        }
    )

    evaluations: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def rollout_batch_size(self) -> int:
        return self.arch.num_envs * self.arch.num_steps

    @property
    def num_updates(self) -> int:
        return self.arch.total_timesteps // self.rollout_batch_size

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
    cs.store(name="base_config", node=ExperimentConfig)
    _CONFIGS_REGISTERED = True