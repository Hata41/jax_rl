import json
from pathlib import Path
from typing import Any, TypedDict, cast

import jax
import orbax.checkpoint as ocp

from .exceptions import CheckpointRestoreError
from .types import PolicyValueParams, TrainState


class RestorePayload(TypedDict):
    step: int
    items: dict[str, Any]
    metadata: dict[str, Any]


class Checkpointer:
    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 1,
        keep_period: int | None = None,
        save_interval_steps: int = 0,
        metadata: dict[str, Any] | None = None,
    ):
        self._checkpoint_dir = Path(checkpoint_dir)
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
        self._manager = ocp.CheckpointManager(
            str(self._checkpoint_dir),
            self._checkpointer,
            options=self._options,
            metadata=self._metadata,
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
        template_train_state: TrainState | None = None,
        template_key=None,
    ) -> dict[str, Any]:
        """Restore training state from an explicit path or this manager's directory."""
        template_items = None
        if template_train_state is not None:
            template_items = {"train_state": template_train_state, "key": template_key}

        if checkpoint_path is not None:
            restored_payload = self._restore_from_explicit_path(
                target=Path(checkpoint_path),
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

        train_state = self._coerce_train_state(restored["train_state"])
        key = restored["key"]

        self._validate_train_state(train_state)

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