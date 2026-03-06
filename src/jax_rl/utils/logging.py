from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
import json
from numbers import Number
from pathlib import Path
import sys
import time
from typing import Any, Mapping, cast

import jax
import numpy as np
from colorama import Fore, Style, init as colorama_init

from .types import LogEvent


def _to_host_array(value: Any) -> np.ndarray:
    host_value = jax.device_get(value)
    return np.asarray(host_value)


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def describe(value: Any) -> Any:
    array = _to_host_array(value)
    if array.ndim == 0:
        return _to_python_scalar(array.item())

    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "mean": float(np.nanmean(array)),
            "std": float(np.nanstd(array)),
            "min": float(np.nanmin(array)),
            "max": float(np.nanmax(array)),
        }


def _flatten_described_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}

    def _flatten(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                next_prefix = f"{prefix}_{sub_key}" if prefix else str(sub_key)
                _flatten(next_prefix, sub_value)
            return

        described = describe(value)
        if isinstance(described, dict):
            for stat_key, stat_value in described.items():
                flattened[f"{prefix}_{stat_key}"] = float(stat_value)
        else:
            flattened[prefix] = float(described)

    for key, value in metrics.items():
        _flatten(str(key), value)

    return flattened


def _event_tag(event: LogEvent) -> str:
    if event is LogEvent.ABSOLUTE:
        return ""
    return event.value


def _prefix_event_metrics(event: LogEvent, metrics: Mapping[str, float]) -> dict[str, float]:
    tag = _event_tag(event)
    if not tag:
        return dict(metrics)
    return {f"{tag}/{key}": float(value) for key, value in metrics.items()}


def _warn_once(warned: set[str], key: str, message: str):
    if key in warned:
        return
    warned.add(key)
    print(message, file=sys.stderr)


def _normalize_console_key(key: str) -> str:
    return key.replace("_", " ").capitalize()


def _format_console_value(value: float) -> str:
    if not np.isfinite(value):
        return str(value)
    return f"{value:.6g}"


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {str(k): _to_serializable(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value


def _colorize_json(value: Any, indent: int = 0, indent_step: int = 2) -> str:
    if isinstance(value, Mapping):
        items = list(value.items())
        if not items:
            return "{}"
        lines = ["{"]
        for idx, (key, sub_value) in enumerate(items):
            comma = "," if idx < len(items) - 1 else ""
            key_text = f'{Fore.CYAN}{json.dumps(str(key))}{Style.RESET_ALL}'
            value_text = _colorize_json(sub_value, indent=indent + indent_step, indent_step=indent_step)
            lines.append(
                f"{' ' * (indent + indent_step)}{key_text}: {value_text}{comma}"
            )
        lines.append(f"{' ' * indent}}}")
        return "\n".join(lines)

    if isinstance(value, list):
        if not value:
            return "[]"
        lines = ["["]
        for idx, item in enumerate(value):
            comma = "," if idx < len(value) - 1 else ""
            item_text = _colorize_json(item, indent=indent + indent_step, indent_step=indent_step)
            lines.append(f"{' ' * (indent + indent_step)}{item_text}{comma}")
        lines.append(f"{' ' * indent}]")
        return "\n".join(lines)

    if value is None:
        return f"{Style.DIM}null{Style.RESET_ALL}"
    if isinstance(value, bool):
        return f"{Fore.MAGENTA}{str(value).lower()}{Style.RESET_ALL}"
    if isinstance(value, Number):
        return f"{Fore.YELLOW}{value}{Style.RESET_ALL}"
    if isinstance(value, str):
        return f"{Fore.GREEN}{json.dumps(value)}{Style.RESET_ALL}"
    return f"{Fore.GREEN}{json.dumps(str(value))}{Style.RESET_ALL}"


def format_colored_block(title: str, payload: Mapping[str, Any]) -> str:
    serialized = _to_serializable(payload)
    top_border = f"{Fore.BLUE}{Style.BRIGHT}{'=' * 88}{Style.RESET_ALL}"
    title_line = f"{Fore.BLUE}{Style.BRIGHT}{title}{Style.RESET_ALL}"
    body = _colorize_json(serialized)
    return "\n".join((top_border, title_line, top_border, body, top_border))


def extract_learning_rate(actor_opt_state) -> float:
    stack = [actor_opt_state]
    while stack:
        current = stack.pop()
        if hasattr(current, "hyperparams"):
            hyperparams = getattr(current, "hyperparams")
            if isinstance(hyperparams, dict) and "learning_rate" in hyperparams:
                return float(np.asarray(hyperparams["learning_rate"]))
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (tuple, list)):
            stack.extend(current)
    return float("nan")


def extract_completed_episode_metrics(rollout_infos: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    returns = np.asarray(
        rollout_infos.get("returned_episode_returns", rollout_infos.get("episode_return")),
        dtype=np.float32,
    )
    lengths = np.asarray(
        rollout_infos.get("returned_episode_lengths", rollout_infos.get("episode_length")),
        dtype=np.float32,
    )
    completed = np.asarray(
        rollout_infos.get("returned_episode", rollout_infos.get("is_terminal_step")),
        dtype=bool,
    )
    completed_returns = returns[completed]
    if completed_returns.size == 0:
        return {}
    completed_lengths = lengths[completed]
    return {
        "episode_return": describe(completed_returns),
        "episode_length": describe(completed_lengths),
    }


class BaseLogger(ABC):
    @abstractmethod
    def log_stat(self, key: str, value: float, step: int, event: LogEvent) -> None:
        pass

    @abstractmethod
    def log_dict(self, metrics: Mapping[str, Any], step: int, event: LogEvent) -> None:
        pass

    @abstractmethod
    def log_config(self, config: Mapping[str, Any]) -> None:
        pass

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


class ConsoleLogger(BaseLogger):
    _EVENT_COLORS = {
        LogEvent.TRAIN: Fore.MAGENTA,
        LogEvent.EVAL: Fore.GREEN,
        LogEvent.ACT: Fore.CYAN,
        LogEvent.MISC: Fore.YELLOW,
        LogEvent.ABSOLUTE: "",
    }

    def __init__(self, stream=None):
        colorama_init(autoreset=False)
        self._stream = stream if stream is not None else sys.stdout

    def log_stat(self, key: str, value: float, step: int, event: LogEvent) -> None:
        self.log_dict({key: value}, step=step, event=event)

    def log_dict(self, metrics: Mapping[str, Any], step: int, event: LogEvent) -> None:
        del step
        if not metrics:
            return
        color = self._EVENT_COLORS[event]
        tag = event.name
        key_width = 25
        lines = [tag]

        for key, raw_value in sorted(metrics.items()):
            display_key = _normalize_console_key(key)
            scalar = float("nan")

            if isinstance(raw_value, Mapping):
                stats = {str(k): float(v) for k, v in raw_value.items()}
            else:
                described = describe(raw_value)
                if isinstance(described, dict):
                    stats = {str(k): float(v) for k, v in described.items()}
                else:
                    stats = None
                    scalar = float(described)

            if stats is None:
                value_text = _format_console_value(scalar)
            else:
                mean = stats.get("mean")
                min_value = stats.get("min")
                max_value = stats.get("max")
                if mean is not None and min_value is not None and max_value is not None:
                    value_text = (
                        f"{_format_console_value(mean)} "
                        f"(Min: {_format_console_value(min_value)}, Max: {_format_console_value(max_value)})"
                    )
                else:
                    value_text = ", ".join(
                        f"{_normalize_console_key(stat_key)}: {_format_console_value(stat_value)}"
                        for stat_key, stat_value in sorted(stats.items())
                    )

            lines.append(f"{display_key + ':':<{key_width}} {value_text}")

        line = "\n".join(lines)
        if color:
            line = f"{color}{line}{Style.RESET_ALL}"
        self._stream.write(f"{line}\n")
        self._stream.flush()

    def log_config(self, config: Mapping[str, Any]) -> None:
        if not config:
            return
        rendered = format_colored_block("RUN CONFIG", config)
        self._stream.write(f"{rendered}\n")
        self._stream.flush()


class TensorBoardLogger(BaseLogger):
    def __init__(self, log_dir: str):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        self._event_cls = Event
        self._summary_cls = Summary
        self._writer = EventFileWriter(log_dir)

    def _to_tag(self, key: str, event: LogEvent) -> str:
        prefix = _event_tag(event)
        if not prefix:
            return key
        return f"{prefix}/{key}"

    def log_stat(self, key: str, value: float, step: int, event: LogEvent) -> None:
        summary_cls = cast(Any, self._summary_cls)
        event_cls = cast(Any, self._event_cls)
        summary = summary_cls(
            value=[summary_cls.Value(tag=self._to_tag(key, event), simple_value=float(value))]
        )
        logged_event = event_cls(wall_time=time.time(), step=int(step), summary=summary)
        self._writer.add_event(logged_event)

    def log_dict(self, metrics: Mapping[str, float], step: int, event: LogEvent) -> None:
        for key, value in sorted(metrics.items()):
            self.log_stat(key, value, step=step, event=event)

    def log_config(self, config: Mapping[str, Any]) -> None:
        del config

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class jaxRL_Logger:
    def __init__(self, sinks: list[BaseLogger] | None = None):
        self._sinks = list(sinks or [])
        self._warnings: set[str] = set()

    @property
    def sinks(self) -> list[BaseLogger]:
        return list(self._sinks)

    @classmethod
    def from_config(cls, config) -> "jaxRL_Logger":
        sinks: list[BaseLogger] = [ConsoleLogger()]
        warned: set[str] = set()

        logger_cfg = getattr(getattr(config, "io", None), "logger", None)
        run_name = getattr(getattr(config, "io", None), "name", None)
        if logger_cfg is not None and getattr(logger_cfg, "tensorboard_logdir", None):
            resolved_name = str(run_name) if run_name else str(config.system.name).lower()
            run_dir = Path(logger_cfg.tensorboard_logdir) / resolved_name
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                sinks.append(TensorBoardLogger(str(run_dir)))
            except Exception as exc:
                _warn_once(
                    warned,
                    "tensorboard_init",
                    f"[jaxRL_Logger] Warning: failed to initialize TensorBoardLogger ({exc}). Continuing without it.",
                )

        instance = cls(sinks=sinks)
        instance._warnings.update(warned)
        return instance

    def _dispatch(self, method_name: str, *args, **kwargs) -> None:
        for sink in self._sinks:
            sink_name = type(sink).__name__
            try:
                getattr(sink, method_name)(*args, **kwargs)
            except Exception as exc:
                _warn_once(
                    self._warnings,
                    f"dispatch_{sink_name}_{method_name}",
                    f"[jaxRL_Logger] Warning: sink {sink_name}.{method_name} failed ({exc}).",
                )

    def materialize(self, metrics: Mapping[str, Any], event: LogEvent) -> dict[str, float]:
        flattened = _flatten_described_metrics(metrics)
        return _prefix_event_metrics(event, flattened)

    def log(self, metrics: Mapping[str, Any], step: int, event: LogEvent) -> None:
        flattened = _flatten_described_metrics(metrics)
        for sink in self._sinks:
            sink_name = type(sink).__name__
            payload = metrics if isinstance(sink, ConsoleLogger) else flattened
            try:
                sink.log_dict(payload, step, event)
            except Exception as exc:
                _warn_once(
                    self._warnings,
                    f"dispatch_{sink_name}_log_dict",
                    f"[jaxRL_Logger] Warning: sink {sink_name}.log_dict failed ({exc}).",
                )

    def log_stat(self, key: str, value: Any, step: int, event: LogEvent) -> None:
        described = describe(value)
        if isinstance(described, dict):
            self.log({key: value}, step=step, event=event)
            return
        scalar = float(described)
        self._dispatch("log_stat", key, scalar, step, event)

    def log_config(self, config: Any) -> None:
        if is_dataclass(config) and not isinstance(config, type):
            config_data = _to_serializable(config)
        elif isinstance(config, Mapping):
            config_data = _to_serializable(config)
        else:
            config_data = {"config": str(config)}
        self._dispatch("log_config", config_data)

    def flush(self) -> None:
        self._dispatch("flush")

    def close(self) -> None:
        self._dispatch("close")
