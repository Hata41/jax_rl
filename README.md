# jax_rl

JAX implementation of Proximal Policy Optimization (PPO) with `flax.nnx` networks and a functional training loop:

- Explicit PRNG key threading
- `jax.lax.scan` trajectory collection
- `jax.jit` update step
- Gymnax environments
- Optax optimizer and gradient clipping

## Architecture

This section describes the current architecture of `jax_rl` end-to-end, with emphasis on the functional PPO training loop, environment routing, model construction, checkpointing, logging, and extension points.

### 1) System Overview

`jax_rl` is a modular PPO implementation that combines:

- Functional state flow (`TrainState`, explicit RNG threading)
- JAX transforms (`pmap`, `lax.scan`) for batched rollouts/updates
- `flax.nnx` models split into graph/state for clean serialization
- Optax actor/critic optimizers with independent schedules
- Orbax checkpoint management for resumable training
- Multi-backend environment routing (`rustpool`, `jaxpallet`, Gymnax fallback)

Primary package surface is exported from `src/jax_rl/__init__.py`:

- `PPOConfig`
- `train`
- `evaluate`
- `Checkpointer`

### 2) Architectural Goals

- Preserve deterministic training behavior under fixed seed and config.
- Keep public APIs stable while allowing internal refactoring.
- Isolate backend-specific environment setup behind a uniform constructor.
- Separate acting, learning, evaluation, logging, and persistence concerns.
- Make failures domain-specific through explicit exceptions.

### 3) Runtime Entry and Bootstrapping

#### CLI

The training entry point is in `src/jax_rl/cli.py`:

1. Parse `--config` path.
2. Load YAML `training` section into `PPOConfig`.
3. Execute `train(config)`.
4. Optionally run post-train `evaluate(...)`.

#### Runtime Defaults

`configure_jax_runtime_defaults` in `src/jax_rl/runtime.py` sets CPU runtime defaults when no explicit JAX platform env vars are present and no NVIDIA device is detected.

### 4) Configuration Model

Configuration is centralized in `src/jax_rl/config.py` via immutable `PPOConfig`.

Notable computed properties:

- `rollout_batch_size = num_envs * num_steps`
- `num_updates = total_timesteps // rollout_batch_size`
- `local_device_count = jax.local_device_count()`

Network creation is Hydra-driven through `training.network._target_`, with runtime dimensions injected at construction.

### 5) Core Data Model

Main typed runtime structures are in `src/jax_rl/types.py`:

- `PolicyValueParams(graphdef, state)`
- `TrainState(params, actor_opt_state, critic_opt_state)`
- `RunnerState(train_state, env_state, obs, key)`
- `RolloutBatch` and `FlattenBatch`
- `LogEvent` event categories for telemetry

These types define the canonical payloads exchanged between rollout, update, checkpoint, and evaluation subsystems.

### 6) Module Responsibilities

#### Environment Construction

`src/jax_rl/env.py`

- Provides environment wrappers and normalization adapters.
- Exposes `make_stoa_env(env_name: str, num_envs_per_device: int)`.
- Routes backend creation through a prefix registry.

#### Networks and Distributions

`src/jax_rl/networks.py`

- Observation flattening and action-mask handling.
- Policy/value model definitions (MLP and binpack transformer variants).
- Hydra target resolution via `init_policy_value_params(...)`.
- Distribution abstraction and forward apply function.

#### Rollout Collection

`src/jax_rl/rollout.py`

- Uses `lax.scan` to collect trajectories.
- Computes done/truncated flags and bootstrap values.
- Returns structured rollout batch + auxiliary episode info.

#### PPO Update Step

`src/jax_rl/update.py`

- Computes GAE/returns.
- Flattens dataset and shuffles minibatches.
- Runs epoch/minibatch update loops with gradient computation.
- Splits/merges actor and critic updates from module-prefixed gradients.

#### Losses and Advantages

- PPO objective in `src/jax_rl/losses.py`
- GAE in `src/jax_rl/advantages.py`

#### Training Orchestration

`src/jax_rl/train.py`

- Performs configuration validation and setup.
- Creates env, model, optimizer, logger, and checkpointer.
- Runs rollout/update loop with periodic eval/checkpointing.

#### Evaluation

`src/jax_rl/eval.py`

- Creates single-env evaluation environment.
- Runs deterministic policy mode action selection.
- Aggregates episode return statistics.

#### Checkpointing

`src/jax_rl/checkpoint.py`

- Encapsulates Orbax manager/checkpointer.
- Saves and restores `train_state` + RNG key + metadata.
- Supports explicit path restore and manager-based restore.

#### Logging

`src/jax_rl/logging.py`

- Console and TensorBoard sinks behind `jaxRL_Logger`.
- Metric flattening/stat summarization for arrays.
- Event-prefixed metric materialization.

#### Export

`src/jax_rl/export.py`

- ONNX export through `jax2onnx` using merged graph/state forward wrapper.

### 7) Environment Routing Architecture

`env_name` follows a prefix pattern:

- `rustpool:<task>`
- `jaxpallet:<preset>`
- no prefix (or unmatched prefix) -> Gymnax fallback attempt

Registry components in `src/jax_rl/env.py`:

- `EnvFactory = Callable[[str, int], tuple[Any, Any]]`
- `_ENV_REGISTRY: dict[str, EnvFactory]`
- `@register_env(prefix)` decorator

Built-in registrations:

- `_make_rustpool_env` via `@register_env("rustpool")`
- `_make_jaxpallet_env` via `@register_env("jaxpallet")`

Failure mode:

- If no registered prefix handles `env_name` and Gymnax fallback fails, `EnvironmentNotFoundError` is raised.

#### Adding a New Backend

1. Implement a factory function with signature `(env_name: str, num_envs_per_device: int) -> tuple[env, env_params]`.
2. Register it with `@register_env("mybackend")`.
3. Keep all backend-specific imports inside the factory for optional dependency isolation.
4. Return environment objects matching the training loop expectations.

### 8) Exception Taxonomy

Domain exceptions are defined in `src/jax_rl/exceptions.py`:

- `JaxRLError(Exception)`
- `ConfigDivisibilityError(JaxRLError, ValueError)`
- `NetworkTargetResolutionError(JaxRLError, ValueError)`
- `EnvironmentNotFoundError(JaxRLError, ValueError)`
- `CheckpointRestoreError(JaxRLError, FileNotFoundError)`

Usage guidelines:

- Raise `ConfigDivisibilityError` only for device/batch divisibility checks.
- Raise `NetworkTargetResolutionError` for Hydra target resolution/instantiation failures.
- Raise `EnvironmentNotFoundError` for unresolved env creation paths.
- Raise `CheckpointRestoreError` for restore path/payload validation failures.

### 9) Checkpoint Restore Architecture

`Checkpointer.restore(...)` keeps its public signature and delegates internally:

- `_restore_from_explicit_path(target, timestep, template_items)`
- `_restore_from_manager(timestep, template_items)`

Both return a normalized internal payload consumed by `restore` to emit the stable response contract:

- `step: int`
- `train_state: TrainState`
- `key: Array`
- `metadata: dict`

Validation/coercion helpers:

- `_coerce_train_state(...)`
- `_validate_train_state(...)`

### 10) Training Loop Data Flow

High-level flow in `src/jax_rl/train.py`:

1. Validate rollout/minibatch/device divisibility.
2. Build env via `make_stoa_env`.
3. Infer observation/action dimensions.
4. Initialize model params + optimizers.
5. Optionally restore from checkpoint.
6. Replicate train state across local devices.
7. `pmap` init runner state.
8. For each update:
  - `pmap` rollout step (`collect_rollout`)
  - `pmap` PPO update (`ppo_update`)
  - optional eval
  - structured logging
  - periodic checkpoint save
9. Return final run summary and model params.

### 11) Numerical and Shape Contracts

#### Observation Flattening

`flatten_observation_features(...)` in `src/jax_rl/networks.py`:

- Accepts array or mapping of arrays.
- Excludes `action_mask` from concatenated feature vector.
- Returns `(features, action_mask | None)`.

#### Binpack Logits

`_flatten_binpack_logits(...)` in `src/jax_rl/networks.py`:

- Transforms score tensor from `[B, E, I]` to flattened logits `[B, I * E * R]` where `R` is inferred from action mask dimensionality.

#### PPO Objective

`ppo_loss(...)` in `src/jax_rl/losses.py`:

- Clipped policy ratio objective.
- Clipped value target objective.
- Entropy regularization.
- NaN/Inf guard on reported metrics.

#### GAE

`compute_gae(...)` in `src/jax_rl/advantages.py`:

- Supports truncated-episode handling and optional bootstrap values.
- Uses reverse scan recursion for efficient batched advantage computation.

### 12) Logging and Observability

`jaxRL_Logger` in `src/jax_rl/logging.py`:

- Console sink always active by default.
- TensorBoard sink enabled when configured.
- Array metrics are summarized into scalar stats for non-console sinks.
- Sink failures are non-fatal and warned once.

### 13) Public API Stability Contract

The following signatures are stable and should not be changed without versioned migration:

- `make_stoa_env(env_name: str, num_envs_per_device: int)`
- `Checkpointer.restore(...)`
- `train(config: PPOConfig)`
- `ppo_loss(...)`

Internal refactors should preserve:

- Returned object structure and key names.
- Existing CLI behavior.
- Training loop semantics under equivalent config/seed.

### 14) Testing Strategy and Coverage

Architecture-level checks are covered by tests in `tests`:

- Environment registry and fallback errors in `tests/test_env_registry.py`
- Divisibility exception behavior in `tests/test_scaling.py`
- Network target resolution errors in `tests/test_networks.py`
- Checkpoint roundtrip and restore failure behavior in `tests/test_checkpoint_eval.py`
- Integration pipeline behavior in `tests/test_integration.py`

Recommended verification commands:

- `uv run pytest tests/test_env_registry.py -q`
- `uv run pytest tests/test_scaling.py tests/test_networks.py tests/test_checkpoint_eval.py -q`
- `uv run pytest -q`

### 15) Extension Checklist

When adding a new module or backend:

1. Keep domain errors explicit and use `exceptions.py` taxonomy.
2. Keep optional backend imports local to backend builders.
3. Preserve existing public signatures and payload shapes.
4. Add targeted tests first, then run full regression suite.
5. Update architecture documentation in this README in the same change.

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

### ONNX operator/component coverage reference

For the authoritative list of supported `jax2onnx` operators and `flax.nnx` API coverage, see the `jax2onnx` repository documentation at:

- `docs/user_guide/flax_api_coverage.md`

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