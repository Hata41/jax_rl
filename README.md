# jax_rl

JAX implementation of Proximal Policy Optimization (PPO) with `flax.nnx` networks and a functional training loop:

- Explicit PRNG key threading
- `jax.lax.scan` trajectory collection
- `jax.jit` update step
- Gymnax environments
- Optax optimizer and gradient clipping

## Install

```bash
uv pip install -e .
uv pip install -e .[dev]
```

## Run

```bash
# edit the full training setup in config/train.yaml, then run
uv run jax-rl-train

# explicit config path
uv run jax-rl-train --config config/train.yaml

# examples (set these values inside config/train.yaml):
# save_interval_steps: 50
# checkpoint_dir: checkpoints
# resume_from: checkpoints
# tensorboard_logdir: runs
# tensorboard_run_name: cartpole_baseline
# eval_episodes: 10

# in another terminal
tensorboard --logdir runs
```

## Training Guide

This section is a full reference for launching training reliably across local CPU, `jaxpallet`, and `rustpool` setups.

### 1) Recommended command forms

Use either of these:

```bash
# preferred console script
uv run jax-rl-train --config config/train.yaml

# equivalent module form
uv run python -m jax_rl.cli --config config/train.yaml
```

Do **not** add a `train` subcommand (the CLI is single-command):

```bash
# wrong
uv run python -m jax_rl.cli train --config config/train.yaml
```

### 2) Quick start flow

```bash
# from repo root
uv sync

# optional: dev extras for testing/export
uv sync --extra dev

# launch training with current config file
uv run jax-rl-train --config config/train.yaml
```

### 3) Minimal config profile (fast local smoke runs)

Current defaults in `config/train.yaml` are tuned for quick iteration:

- `env_name: CartPole-v1`
- `total_timesteps: 2048`
- `num_envs: 8`
- `num_steps: 32`
- `minibatch_size: 64`
- `hidden_size: 32`
- `hidden_layers: 1`
- `tensorboard_logdir:` (disabled)

This gives short runs that are useful to validate installation and end-to-end pipeline behavior.

### 4) Environment-specific examples

#### CartPole / Gymnax fallback

```yaml
training:
  env_name: CartPole-v1
  total_timesteps: 200000
  num_envs: 16
  num_steps: 128
  minibatch_size: 256
```

#### JaxPallet

```yaml
training:
  env_name: jaxpallet:PMC-PLD
  total_timesteps: 500000
  num_envs: 8
  num_steps: 64
  minibatch_size: 128
```

#### RustPool

```yaml
training:
  env_name: rustpool:BinPack-v0
  total_timesteps: 500000
  num_envs: 8
  num_steps: 64
  minibatch_size: 128
```

Notes for `rustpool`:

- Observation keys are normalized in the env wrapper to the binpack model contract (`ems_pos`, `item_dims`, `item_mask`, `action_mask`).
- `num_envs` must be divisible by local device count.

### 5) TensorBoard usage

Enable in config:

```yaml
training:
  tensorboard_logdir: runs_tb
  tensorboard_run_name: exp_001
```

Run training, then in another terminal:

```bash
tensorboard --logdir runs_tb
```

If TensorBoard backend initialization fails, training continues and a warning is printed once to stderr.

### 6) Resume from checkpoint

Example config values:

```yaml
training:
  checkpoint_dir: checkpoints
  save_interval_steps: 100
  resume_from: checkpoints
```

Behavior:

- Train state, optimizer state, and RNG key are restored.
- `start_update` is inferred from checkpoint metadata.
- Training continues up to configured `total_timesteps`.

### 7) Interpreting console logs

Console output is event-based and color-coded:

- `ACT` (cyan): rollout/acting metrics
- `TRAIN` (magenta): PPO update metrics
- `EVAL` (green): evaluation metrics
- `ABSOLUTE` (plain): absolute counters like timestep

Format:

```text
TAG - Key: Value | Key: Value
```

### 8) Common launch errors and fixes

- `ModuleNotFoundError: No module named 'purejax_ppo'`
  - Cause: old script entry point.
  - Fix: use current project metadata and run via `uv run jax-rl-train`.

- `error: unrecognized arguments: train`
  - Cause: attempted subcommand mode.
  - Fix: remove `train` token and run `uv run python -m jax_rl.cli --config ...`.

- `CUDA_ERROR_NO_DEVICE` plugin warning on CPU-only machine
  - Project runtime defaults force CPU for project entry points/tests.
  - If you run plain `import jax` outside project entry points, set:

```bash
export JAX_PLATFORMS=cpu
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1
```

- `ValueError: num_envs must be divisible by local device count`
  - Fix: set `num_envs` to a multiple of `jax.local_device_count()`.

- `ValueError: minibatch_size must divide num_envs * num_steps`
  - Fix: choose `minibatch_size` that cleanly divides rollout batch size.

### 9) Suggested run matrix before long experiments

```bash
# 1) quick unit/integration confidence
uv run --extra dev pytest tests/test_integration.py -q

# 2) minimal local training smoke
uv run jax-rl-train --config config/train.yaml

# 3) optional full suite
uv run --extra dev pytest -q
```

## Test

```bash
uv run --extra dev pytest -q

# run ONNX export/equivalence tests only
uv run --extra dev pytest tests/test_onnx_export.py -q
```

## Notes

- Supported action spaces: `Discrete` and `MultiDiscrete`.
- Actor and critic use separate learning rates (`actor_lr`, `critic_lr`).
- The training loop is functional and deterministic under a fixed seed.
- Checkpoints are managed with Orbax (`orbax-checkpoint`) and store train state, PRNG key, metrics, and metadata.

### Telemetry and logging

Training telemetry is centralized through `jaxRL_Logger` (`src/jax_rl/logging.py`) with a multi-sink dispatch pattern.

- Event categories are enforced by `LogEvent`: `ACT`, `TRAIN`, `EVAL`, `ABSOLUTE`, `MISC`.
- Any JAX/NumPy array logged through `logger.log(...)` is auto-summarized by `describe(...)` to scalar stats:
  - `mean`, `std`, `min`, `max`
- Console sink (`ConsoleLogger`) prints one-line logs in the format:
  - `TAG - Key: Value | Key: Value`
- Console colors:
  - `TRAIN`: magenta
  - `EVAL`: green
  - `ACT`: cyan
  - `MISC`: yellow
  - `ABSOLUTE`: plain
- Keys are normalized for readability (e.g. `loss_total` -> `Loss total`).
- TensorBoard sink is enabled only when `tensorboard_logdir` is configured.
- Sink init failures are non-fatal: logger warns once to stderr and continues training.

Minimal usage pattern inside training loop:

```python
logger.log(act_metrics, step, LogEvent.ACT)
logger.log(train_metrics, step, LogEvent.TRAIN)
logger.log({"timestep": step}, step, LogEvent.ABSOLUTE)
```

### Environment routing (`env_name`)

`env_name` now supports three routing modes:

- `rustpool:<task_id>`
  - Uses `StoaRustpoolWrapper(task_id, num_envs_per_device)`.
  - Native pre-batched per device and internally handles resets.
  - No `AutoResetWrapper` and no outer `VmapWrapper` are applied.
  - Rustpool action masks are normalized into `observation["action_mask"]`.

- `jaxpallet:<preset>`
  - Uses `JaxPalletToStoa(preset=...)`.
  - Wrapped as: `RecordEpisodeMetrics -> AutoResetWrapper -> VmapWrapper(num_envs_per_device)`.
  - `VmapWrapper` is intentionally outermost to match rustpool-style batched signatures.

- Fallback (e.g. `CartPole-v1`)
  - Uses gymnax via `GymnaxToStoa`.
  - Wrapped as: `RecordEpisodeMetrics -> AutoResetWrapper -> VmapWrapper(num_envs_per_device)`.

Implementation detail: environment reset/initialization happens inside `jax.pmap` in training to satisfy rustpool device-axis requirements.

## Migration

### Breaking changes (quick checklist)

- Network parameters are no longer plain actor/critic dicts.
- `PolicyValueParams` now stores `graphdef` + `state` (from `flax.nnx.split`).
- `policy_value_apply` now expects `(graphdef, state, obs)`.
- Optimizers are initialized and updated against `params.state`.
- Checkpoint payloads now persist/restore the NNX split model structure.
- ONNX export reads the same split state and merges before tracing.

### Old vs new API

| Area | Old | New |
|---|---|---|
| Model init | `params = init_policy_value_params(...)` | `params = init_policy_value_params(...)` |
| Param layout | `params.actor`, `params.critic` | `params.graphdef`, `params.state` |
| Forward call | `policy_value_apply(params, obs)` | `policy_value_apply(params.graphdef, params.state, obs)` |
| Optimizer init | `optimizer.init(params.actor/critic)` | `optimizer.init(params.state)` |
| Loss signature | `ppo_loss(params, batch, ...)` | `ppo_loss(graphdef, state, batch, ...)` |
| ONNX export source | raw param dicts | split NNX params merged before trace |

If you are upgrading from older configs, replace the single optimizer key:

```yaml
# old
training:
  learning_rate: 0.0003
```

with separate actor/critic learning rates:

```yaml
# new
training:
  actor_lr: 0.0003
  critic_lr: 0.0003
```

### NNX policy/value contract

Networks now use `flax.nnx` modules and are stored as a split pair:

- `graphdef`: static model graph
- `state`: dynamic parameter/state tree updated by Optax

Initialization:

```python
params = init_policy_value_params(
  key,
  obs_dim=4,
  action_dims=2,
  hidden_sizes=(64, 64),
)
# params.graphdef, params.state
```

Forward pass:

```python
dist, values = policy_value_apply(params.graphdef, params.state, obs)
```

ONNX export uses the same contract and merges internally before tracing:

```python
export_model_to_onnx(params, obs_shape=(4,), filepath="policy_value.onnx")
```

To keep `jax2onnx` compatibility, model code is limited to ONNX-safe JAX primitives
used by linear layers and simple activations (e.g. matmul/add/tanh/concat/reshape).