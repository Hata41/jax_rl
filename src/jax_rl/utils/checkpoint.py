from contextlib import contextmanager
import json
import logging
from pathlib import Path
from typing import Any, TypedDict, cast

import jax
import orbax.checkpoint as ocp

from .exceptions import CheckpointRestoreError
from .logging import format_colored_block
from .types import PolicyValueParams, TrainState


class RestorePayload(TypedDict):
    step: int
    items: dict[str, Any]
    metadata: dict[str, Any]


_ALGO_NAMES = {"ppo", "spo", "alphazero"}


def _sanitize_run_token(value: str) -> str:
    return (
        str(value)
        .replace(":", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def _infer_checkpoint_root_and_env(checkpoint_dir: str, env_name: str) -> tuple[Path, str]:
    path = Path(checkpoint_dir)
    parts = path.parts

    algo_index: int | None = None
    for idx, part in enumerate(parts):
        if part in _ALGO_NAMES and idx + 1 < len(parts):
            algo_index = idx
            break

    if algo_index is None:
        return Path("checkpoints"), _sanitize_run_token(env_name)

    root = Path(parts[0])
    for part in parts[1:algo_index]:
        root /= part
    return root, _sanitize_run_token(env_name)


def resolve_resume_from(
    *,
    checkpoint_dir: str,
    env_name: str,
    resume_from: str,
    source_algo: str,
) -> str:
    """Resolve shorthand checkpoint names to latest run/step path.

    If `resume_from` already points to an existing directory, it is returned unchanged.
    Otherwise, `resume_from` can be:
    - `<name>`: resolved as latest run leaf for `source_algo`
    - `<algo>/<name>`: resolved as latest run leaf for the explicit `algo`
      where algo in {ppo, spo, alphazero}
    - any other multi-segment path: returned unchanged
    """
    target = Path(resume_from).expanduser()
    if target.is_dir():
        return str(target)

    resolved_algo = str(source_algo).lower()
    leaf_name = resume_from
    if len(target.parts) == 2 and target.parts[0].lower() in _ALGO_NAMES:
        resolved_algo = target.parts[0].lower()
        leaf_name = target.parts[1]
    elif len(target.parts) > 1:
        return str(target)

    def _resolve_leaf_under_base(base_dir: Path, leaf: str) -> str | None:
        direct_leaf = base_dir / leaf
        if direct_leaf.is_dir():
            has_numeric_step = any(
                child.is_dir() and child.name.isdigit() for child in direct_leaf.iterdir()
            )
            if has_numeric_step:
                return str(direct_leaf)
            fallback = str(direct_leaf)
        else:
            fallback = None

        candidate_runs = sorted(
            [run_dir for run_dir in base_dir.iterdir() if run_dir.is_dir()],
            key=lambda path: path.name,
            reverse=True,
        )
        for run_dir in candidate_runs:
            leaf_dir = run_dir / leaf
            if not leaf_dir.is_dir():
                continue
            if fallback is None:
                fallback = str(leaf_dir)
            has_numeric_step = any(
                child.is_dir() and child.name.isdigit() for child in leaf_dir.iterdir()
            )
            if has_numeric_step:
                return str(leaf_dir)

        return fallback

    checkpoint_root, env_token = _infer_checkpoint_root_and_env(checkpoint_dir, env_name)
    env_base = checkpoint_root / resolved_algo / env_token
    if env_base.is_dir():
        resolved = _resolve_leaf_under_base(env_base, leaf_name)
        if resolved is not None:
            return resolved

    flat_base = checkpoint_root / resolved_algo
    if flat_base.is_dir():
        resolved = _resolve_leaf_under_base(flat_base, leaf_name)
        if resolved is not None:
            return resolved

    return str(target)


@contextmanager
def _suppress_orbax_startup_logs():
    absl_logger = logging.getLogger("absl")
    orbax_logger = logging.getLogger("orbax")
    previous_absl_level = absl_logger.level
    previous_orbax_level = orbax_logger.level
    previous_absl_verbosity = None

    try:
        try:
            from absl import logging as absl_logging

            previous_absl_verbosity = absl_logging.get_verbosity()
            absl_logging.set_verbosity(absl_logging.WARNING)
        except Exception:
            previous_absl_verbosity = None

        absl_logger.setLevel(logging.WARNING)
        orbax_logger.setLevel(logging.WARNING)
        yield
    finally:
        absl_logger.setLevel(previous_absl_level)
        orbax_logger.setLevel(previous_orbax_level)
        if previous_absl_verbosity is not None:
            try:
                from absl import logging as absl_logging

                absl_logging.set_verbosity(previous_absl_verbosity)
            except Exception:
                pass


class Checkpointer:
    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 1,
        keep_period: int | None = None,
        save_interval_steps: int = 0,
        metadata: dict[str, Any] | None = None,
    ):
        self._checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self._metadata = metadata or {}
        json.dumps(self._metadata)
        self._checkpointer = ocp.PyTreeCheckpointer()
        self._options = ocp.CheckpointManagerOptions(
            create=True,
            max_to_keep=max_to_keep,
            keep_period=keep_period,
            save_interval_steps=max(1, int(save_interval_steps)) if save_interval_steps else 1,
            best_fn=lambda value: value["metric"],
            best_mode="max",
        )
        with _suppress_orbax_startup_logs():
            self._manager = ocp.CheckpointManager(
                str(self._checkpoint_dir),
                self._checkpointer,
                options=self._options,
                metadata=self._metadata,
            )
        metadata_path = self._checkpoint_dir / "metadata" / "_ROOT_METADATA"
        print(
            format_colored_block(
                "CHECKPOINT METADATA",
                {
                    "path": str(metadata_path),
                    "metadata": self._metadata,
                },
            )
        )


    def save(self, timestep: int, train_state: TrainState, key, metric: float) -> bool:
        return bool(
            self._manager.save(
                step=int(timestep),
                items={"train_state": train_state, "key": key},
                metrics={"metric": float(metric)},
            )
        )


    def restore(
        self,
        timestep: int | None = None,
        checkpoint_path: str | None = None,
        template_train_state: Any | None = None,
        template_key=None,
    ) -> dict[str, Any]:
        """Restore training state from an explicit path or this manager's directory."""
        template_items = None
        if template_train_state is not None:
            template_items = {"train_state": template_train_state, "key": template_key}

        if checkpoint_path is not None:
            restored_payload = self._restore_from_explicit_path(
                target=Path(checkpoint_path).expanduser().resolve(),
                timestep=timestep,
                template_items=template_items,
            )
        else:
            restored_payload = self._restore_from_manager(
                timestep=timestep,
                template_items=template_items,
            )

        restored = restored_payload["items"]
        step = restored_payload["step"]
        metadata = restored_payload["metadata"]

        train_state_payload = restored["train_state"]
        if isinstance(template_train_state, TrainState):
            train_state = self._coerce_train_state(train_state_payload)
            self._validate_train_state(train_state)
        else:
            train_state = train_state_payload
        key = restored["key"]

        return {
            "step": int(step),
            "train_state": train_state,
            "key": key,
            "metadata": metadata,
        }

    def _restore_from_explicit_path(
        self,
        target: Path,
        timestep: int | None,
        template_items: dict[str, object] | None,
    ) -> RestorePayload:
        if not target.exists():
            raise CheckpointRestoreError(f"Checkpoint path does not exist: {target}")
        if not target.is_dir():
            raise CheckpointRestoreError(f"Checkpoint path must be a directory: {target}")

        direct_step = target.name.isdigit() and timestep is None
        if direct_step:
            step = int(target.name)
            parent_dir = target.parent
            temp_manager = ocp.CheckpointManager(
                str(parent_dir),
                self._checkpointer,
                options=self._options,
            )
            if template_items is not None:
                restored = temp_manager.restore(step, items=template_items)
            else:
                restored = temp_manager.restore(step)
            metadata = temp_manager.metadata() if hasattr(temp_manager, "metadata") else {}
            restored_items = cast(dict[str, Any], restored)
            metadata_dict = cast(dict[str, Any], metadata)
            return {"step": int(step), "items": restored_items, "metadata": metadata_dict}

        temp_manager = ocp.CheckpointManager(
            str(target),
            self._checkpointer,
            options=self._options,
        )
        step = int(timestep) if timestep is not None else temp_manager.latest_step()
        if step is None:
            raise CheckpointRestoreError(f"No checkpoints found in: {target}")
        if template_items is not None:
            restored = temp_manager.restore(step, items=template_items)
        else:
            restored = temp_manager.restore(step)
        metadata = temp_manager.metadata() if hasattr(temp_manager, "metadata") else {}
        restored_items = cast(dict[str, Any], restored)
        metadata_dict = cast(dict[str, Any], metadata)
        return {"step": int(step), "items": restored_items, "metadata": metadata_dict}

    def _restore_from_manager(
        self,
        timestep: int | None,
        template_items: dict[str, object] | None,
    ) -> RestorePayload:
        step = int(timestep) if timestep is not None else self._manager.latest_step()
        if step is None:
            raise CheckpointRestoreError(
                f"No checkpoints found in manager directory: {self._checkpoint_dir}"
            )
        if template_items is not None:
            restored = self._manager.restore(step, items=template_items)
        else:
            restored = self._manager.restore(step)
        metadata = self._manager.metadata() if hasattr(self._manager, "metadata") else {}
        restored_items = cast(dict[str, Any], restored)
        metadata_dict = cast(dict[str, Any], metadata)
        return {"step": int(step), "items": restored_items, "metadata": metadata_dict}

    def latest_step(self) -> int | None:
        latest = self._manager.latest_step()
        return int(latest) if latest is not None else None

    def all_steps(self) -> tuple[int, ...]:
        return tuple(int(step) for step in self._manager.all_steps())

    def checkpoint_path_for_step(self, step: int) -> str:
        return str(self._checkpoint_dir / str(int(step)))

    @staticmethod
    def _validate_train_state(train_state: Any) -> None:
        if not isinstance(train_state, TrainState):
            raise CheckpointRestoreError(
                "Restored train_state is not a TrainState PyTree; "
                f"got {type(train_state)}"
            )
        if not isinstance(train_state.params, PolicyValueParams):
            raise CheckpointRestoreError(
                "Restored TrainState params do not match expected PolicyValueParams layout"
            )

        params_treedef = jax.tree_util.tree_structure(train_state.params)
        expected_treedef = jax.tree_util.tree_structure(
            PolicyValueParams(
                graphdef=train_state.params.graphdef,
                state=train_state.params.state,
            )
        )
        if params_treedef != expected_treedef:
            raise CheckpointRestoreError("Restored TrainState parameter tree structure does not match")

    @staticmethod
    def _coerce_train_state(payload: Any) -> TrainState:
        if isinstance(payload, TrainState):
            return payload
        if not isinstance(payload, dict):
            raise CheckpointRestoreError(
                "Restored train_state payload must be TrainState or dict; "
                f"got {type(payload)}"
            )
        required = {"params", "actor_opt_state", "critic_opt_state"}
        missing = required - set(payload.keys())
        if missing:
            raise CheckpointRestoreError(
                "Restored train_state payload is missing required keys: "
                + ", ".join(sorted(missing))
            )

        params_payload = payload["params"]
        if isinstance(params_payload, PolicyValueParams):
            params = params_payload
        elif isinstance(params_payload, dict) and {"graphdef", "state"}.issubset(params_payload):
            params = PolicyValueParams(
                graphdef=params_payload["graphdef"],
                state=params_payload["state"],
            )
        else:
            raise CheckpointRestoreError("Restored train_state params do not match expected structure")

        return TrainState(
            params=params,
            actor_opt_state=payload["actor_opt_state"],
            critic_opt_state=payload["critic_opt_state"],
        )