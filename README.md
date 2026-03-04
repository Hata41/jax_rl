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