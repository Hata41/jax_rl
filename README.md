# jax_rl

JAX implementation of PPO using `flax.nnx`, `optax`, and a functional training loop.

- Explicit PRNG key threading
- `jax.lax.scan` rollout/update flow
- Multi-backend environments (`rustpool`, `jaxpallet`, Gymnax fallback)
- Structured Hydra config (`ExperimentConfig`)

## Quick Start

```bash
# install
uv pip install -e .
uv pip install -e .[dev]

# train with default config
uv run jax-rl-train

# run tests
uv run --extra dev pytest -q
```

## Configuration Overview

Root config is `ExperimentConfig` in `src/jax_rl/configs/config.py`.

```yaml
env:
  env_name: rustpool:BinPack-v0
  seed: 0
  env_kwargs: {}

system:
  total_timesteps: 2048
  num_envs: 8
  num_steps: 32
  actor_lr: 0.0003
  critic_lr: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  update_epochs: 1
  minibatch_size: 64
  max_grad_norm: 0.5

checkpointing:
  checkpoint_dir: checkpoints
  save_interval_steps: 0
  max_to_keep: 1
  keep_period: null
  resume_from: null

logging:
  log_every: 1
  tensorboard_logdir: runs_tb
  tensorboard_run_name: default

network:
  _target_: jax_rl.networks.PolicyValueModel
  hidden_sizes: [64, 64]

evaluations: {}
```

Computed properties:

- `rollout_batch_size = system.num_envs * system.num_steps`
- `num_updates = system.total_timesteps // rollout_batch_size`

## Common CLI Overrides

```bash
# training hyperparameters
uv run jax-rl-train system.actor_lr=0.001 system.num_envs=32

# model config
uv run jax-rl-train network.hidden_dim=128

# env routing
uv run jax-rl-train env.env_name=CartPole-v1

# dynamic env kwargs (force-add in Hydra)
uv run jax-rl-train +env.env_kwargs.max_items=50

# hardware selection
uv run jax-rl-train system.platform=cpu
uv run jax-rl-train system.platform=cuda system.cuda_visible_devices='0,1'
```

## Evaluation Profiles

Global env kwargs are inherited by evaluation unless overridden per profile.

```yaml
env:
  env_name: rustpool:BinPack-v0
  env_kwargs:
    max_items: 50

evaluations:
  default_eval:
    env_name: rustpool:BinPack-v0
    eval_every: 1
    num_episodes: 16
  stress_eval:
    env_name: rustpool:BinPack-v0
    eval_every: 1
    num_episodes: 16
    env_kwargs:
      max_items: 100
```

## Environment Routing

`env.env_name` supports:

- `rustpool:<task_id>`
- `jaxpallet:<preset>`
- fallback Gymnax id (e.g. `CartPole-v1`)

Environment constructor signature:

- `make_stoa_env(env_name: str, num_envs_per_device: int, env_kwargs: dict[str, Any] | None = None)`

## TensorBoard

```yaml
logging:
  tensorboard_logdir: runs_tb
  tensorboard_run_name: exp_001
```

```bash
tensorboard --logdir runs_tb
```

## Checkpoint Resume

```yaml
checkpointing:
  checkpoint_dir: checkpoints
  save_interval_steps: 100
  resume_from: checkpoints
```

Training continues up to `system.total_timesteps`.

## Troubleshooting

- `ValueError: num_envs must be divisible by local device count`
  - Set `system.num_envs` to a multiple of `jax.local_device_count()`.
- `ValueError: minibatch_size must divide num_envs * num_steps`
  - Adjust `system.minibatch_size` to divide rollout batch size.
- `TypeError: unexpected keyword argument ...` during env creation
  - Invalid key in `env.env_kwargs` or `evaluations.<name>.env_kwargs`.

## Documentation

- Training loop walkthrough: `src/jax_rl/systems/ppo/anakin/README.md`

## Architecture

`jax_rl` is a modular PPO implementation combining:

- Functional state flow (`TrainState`, explicit PRNG threading)
- JAX transforms (`pmap`, `lax.scan`) for batched rollouts and updates
- `flax.nnx` split parameters (`graphdef`, `state`)
- Separate actor/critic Optax optimizers
- Orbax checkpointing
- Multi-backend environment routing (`rustpool`, `jaxpallet`, Gymnax fallback)

### Runtime Entry

CLI entrypoint: `src/jax_rl/cli.py`

Flow:

1. Compose Hydra config from `config/train.yaml`.
2. Apply runtime env vars early from raw config if provided (`system.platform`, `system.cuda_visible_devices`).
3. Convert composed config to typed `ExperimentConfig`.
4. Run `train(config)` and optional evaluation profiles.

### Config Model

Root config is `ExperimentConfig` with namespaces:

- `env`: `env_name`, `seed`, `env_kwargs`
- `system`: PPO hyperparameters + optional runtime selection
- `checkpointing`: checkpoint settings
- `logging`: logger and tensorboard settings
- root maps: `network`, `evaluations`

### Environment Routing

`make_stoa_env` signature:

- `make_stoa_env(env_name: str, num_envs_per_device: int, env_kwargs: dict[str, Any] | None = None)`

Routing modes:

- `rustpool:<task_id>`
- `jaxpallet:<preset>`
- fallback Gymnax id (e.g. `CartPole-v1`)

`env_kwargs` is forwarded to backend constructors and Gymnax fallback.

### Training Data Flow

Main orchestration in `src/jax_rl/systems/ppo/anakin/system.py`:

1. `build_system(config)` initializes env/model/optimizers/checkpointer and replicates state.
2. `make_ppo_steps(...)` builds pmapped rollout/update functions.
3. Update loop runs rollout, PPO update, optional evaluation, logging, checkpoint save.
4. Returns run summary and final params.

## Public API

Exported from `src/jax_rl/__init__.py`:

- `ExperimentConfig`
- `train`
- `evaluate`
- `Checkpointer`
