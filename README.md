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
  name: ppo
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

arch:
  total_timesteps: 2048
  platform: null
  cuda_visible_devices: null
  num_envs: 8
  num_steps: 32

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

- `rollout_batch_size = arch.num_envs * arch.num_steps`
- `num_updates = arch.total_timesteps // rollout_batch_size`

## Common CLI Overrides

```bash
# training hyperparameters
uv run jax-rl-train system.actor_lr=0.001 arch.num_envs=32

# model config
uv run jax-rl-train network.hidden_dim=128

# env routing
uv run jax-rl-train env.env_name=CartPole-v1

# dynamic env kwargs (force-add in Hydra)
uv run jax-rl-train +env.env_kwargs.max_items=50

# hardware selection
uv run jax-rl-train arch.platform=cpu
uv run jax-rl-train arch.platform=cuda arch.cuda_visible_devices='0,1'
```

## Config Choices Quick Reference

- `system.name`: `ppo` | `alphazero`
- `env.env_name` patterns:
  - `rustpool:<task_id>` (example: `rustpool:BinPack-v0`)
  - `rlpallet:<task_id>` (example: `rlpallet:UldEnv-v2`)
  - `jaxpallet:<preset>` (example: `jaxpallet:PMC-PLD`)
  - Gymnax fallback id (example: `CartPole-v1`)
- `arch.platform`: `null` (auto) | `cpu` | `gpu` | `tpu`
- `arch.cuda_visible_devices`: `null` or a GPU id list string (example: `'0'`, `'0,1'`)
- AlphaZero only:
  - `system.search_method`: `muzero` | `gumbel`
  - `evaluations.<name>.action_selection`: `policy` | `search`

Runtime behavior for outputs:

- Run id is auto-generated as `{system.name}_{safe_env_name}_{run_token}`.
- `run_token` is timestamp by default, or sanitized CLI `logging.tensorboard_run_name` if explicitly set.
- `checkpointing.checkpoint_dir` is auto-expanded to `<base>/<system.name>/<safe_env_name>/<run_token>`.
- Optional `checkpointing.checkpoint_name` adds a final subfolder (example: `best_model`).
- `logging.tensorboard_run_name` is auto-set to run id unless explicitly overridden from CLI.
- `checkpointing.resume_from` stays exactly as provided (never rewritten by run-id injection).

Example CLI override:

- `uv run jax-rl-train checkpointing.checkpoint_name=best_model`

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

Training continues up to `arch.total_timesteps`.

## Troubleshooting

- `ValueError: num_envs must be divisible by local device count`
  - Set `arch.num_envs` to a multiple of `jax.local_device_count()`.
- `ValueError: minibatch_size must divide num_envs * num_steps`
  - Adjust `system.minibatch_size` to divide rollout batch size.
- `TypeError: unexpected keyword argument ...` during env creation
  - Invalid key in `env.env_kwargs` or `evaluations.<name>.env_kwargs`.

## Documentation

- Training loop walkthrough: `src/jax_rl/systems/ppo/anakin/README.md`

## Architecture

`jax_rl` is a modular PPO/SPO/AlphaZero implementation combining:

- Functional state flow (`TrainState`, explicit PRNG threading)
- JAX transforms (`pmap`, `lax.scan`) for batched rollouts and updates
- `flax.nnx` split parameters (`graphdef`, `state`)
- Separate actor/critic Optax optimizers
- Orbax checkpointing
- Multi-backend environment routing (`rustpool`, `jaxpallet`, Gymnax fallback)

## Recent Stability Fixes (rlpallet UldEnv)

Recent fixes addressed hard-to-reproduce invalid-action behavior on `rlpallet:UldEnv-v2`, especially for SPO search:

- rlpallet probe initialization in `jax_rl` is now mask-aware and avoids invalid synthetic probe actions.
- PPO/Modular policy paths include safe all-invalid mask fallback.
- SPO search now applies explicit safe masking in both root and recurrent sampling.
- SPO rollout now re-samples next particle actions from post-resample logits (fixes action/state misalignment).
- SPO recurrent rollout treats non-positive simulated `state_id` as terminal to avoid re-expanding invalid simulated states.

These changes eliminate startup invalid-action spam for PPO/AlphaZero and significantly harden SPO against simulation-side invalid action propagation.

## SPO Evaluation Modes

SPO supports two evaluation action-selection modes in `evaluations.<name>.action_selection`:

- `policy`: direct policy action selection (no SPO search)
- `search`: SPO search-based action selection

Example (`config/spo/train_uldenv.yaml`):

```yaml
evaluations:
  policy_eval:
    action_selection: policy
    env_name: rlpallet:UldEnv-v2
    num_episodes: 32
  search_eval:
    action_selection: search
    env_name: rlpallet:UldEnv-v2
    num_episodes: 32
```

### Runtime Entry

CLI entrypoint: `src/jax_rl/cli.py`

Flow:

1. Compose Hydra config from `config/train.yaml`.
2. Apply runtime env vars early from raw config if provided (`arch.platform`, `arch.cuda_visible_devices`).
3. Convert composed config to typed `ExperimentConfig`.
4. Run `train(config)` and optional evaluation profiles.

### Config Model

Root config is `ExperimentConfig` with namespaces:

- `env`: `env_name`, `seed`, `env_kwargs`
- `arch`: shared architecture/hardware settings (`total_timesteps`, `platform`, `cuda_visible_devices`, `num_envs`, `num_steps`)
- `system`: algorithm hyperparameters
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
