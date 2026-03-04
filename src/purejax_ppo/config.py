from dataclasses import dataclass
from typing import Optional

import jax


@dataclass(frozen=True)
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

    log_every: int = 10
    eval_every: int = 10
    eval_episodes: int = 0
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