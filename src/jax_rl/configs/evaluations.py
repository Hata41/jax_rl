from __future__ import annotations

from typing import Any

from .config import EnvConfig


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(base_value, value)
        else:
            merged[key] = value
    return merged


def resolve_eval_env(
    eval_cfg: dict[str, Any],
    *,
    default_env_name: str,
    default_env_kwargs: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    resolved_env_name = str(default_env_name)
    resolved_env_kwargs: dict[str, Any] = dict(default_env_kwargs or {})

    env_value = eval_cfg.get("env")
    if isinstance(env_value, EnvConfig):
        resolved_env_name = str(env_value.env_name or resolved_env_name)
        resolved_env_kwargs = _deep_merge_dict(
            resolved_env_kwargs,
            dict(env_value.env_kwargs or {}),
        )
    elif isinstance(env_value, dict):
        if "env_name" in env_value:
            resolved_env_name = str(env_value.get("env_name") or resolved_env_name)
        if "env_kwargs" in env_value:
            resolved_env_kwargs = _deep_merge_dict(
                resolved_env_kwargs,
                dict(env_value.get("env_kwargs") or {}),
            )

    if "env_name" in eval_cfg:
        resolved_env_name = str(eval_cfg.get("env_name") or resolved_env_name)

    if "env_kwargs" in eval_cfg:
        resolved_env_kwargs = _deep_merge_dict(
            resolved_env_kwargs,
            dict(eval_cfg.get("env_kwargs") or {}),
        )

    return resolved_env_name, resolved_env_kwargs
