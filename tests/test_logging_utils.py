from io import StringIO
from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
from colorama import Fore

from jax_rl.config import PPOConfig
from jax_rl.logging import BaseLogger, ConsoleLogger, describe, jaxRL_Logger
from jax_rl.types import LogEvent


def test_describe_array_stats_with_nans():
    values = jnp.array([[1.0, 2.0, jnp.nan], [3.0, 4.0, 5.0]], dtype=jnp.float32)
    stats = describe(values)

    assert set(stats.keys()) == {"mean", "std", "min", "max"}
    np_values = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]], dtype=np.float32)
    assert stats["mean"] == float(np.nanmean(np_values))
    assert stats["std"] == float(np.nanstd(np_values))
    assert stats["min"] == float(np.nanmin(np_values))
    assert stats["max"] == float(np.nanmax(np_values))


def test_console_logger_train_has_color_codes():
    stream = StringIO()
    logger = ConsoleLogger(stream=stream)

    logger.log_dict({"loss_total": 1.23, "done_fraction": 0.5}, step=1, event=LogEvent.TRAIN)

    output = stream.getvalue()
    assert Fore.MAGENTA in output
    assert "TRAIN" in output
    assert "Loss total:" in output


def test_console_logger_tabular_alignment_for_described_metrics():
    stream = StringIO()
    logger = ConsoleLogger(stream=stream)

    logger.log_dict(
        {
            "episode_return": {"mean": 10.0, "min": 7.0, "max": 12.0},
            "episode_length": {"mean": 5.0, "min": 4.0, "max": 6.0},
        },
        step=1,
        event=LogEvent.ABSOLUTE,
    )

    lines = [line for line in stream.getvalue().splitlines() if line]
    assert lines[0] == "ABSOLUTE"
    assert lines[1].startswith("Episode length:")
    assert lines[2].startswith("Episode return:")
    assert "(Min: 4, Max: 6)" in lines[1]
    assert "(Min: 7, Max: 12)" in lines[2]

    first_value_column = lines[1].index("5 ")
    second_value_column = lines[2].index("10 ")
    assert first_value_column == second_value_column


class _MockSink(BaseLogger):
    def __init__(self):
        self.log_dict_mock = Mock()

    def log_stat(self, key: str, value: float, step: int, event: LogEvent) -> None:
        return

    def log_dict(self, metrics, step: int, event: LogEvent) -> None:
        self.log_dict_mock(metrics, step, event)

    def log_config(self, config) -> None:
        return


def test_jaxrl_logger_dispatches_to_all_sinks():
    sink_one = _MockSink()
    sink_two = _MockSink()
    logger = jaxRL_Logger(sinks=[sink_one, sink_two])

    logger.log({"reward": 1.0}, step=10, event=LogEvent.ACT)

    sink_one.log_dict_mock.assert_called_once()
    sink_two.log_dict_mock.assert_called_once()


def test_logger_config_toggle_disables_tensorboard_sink():
    config = PPOConfig(tensorboard_logdir=None)
    logger = jaxRL_Logger.from_config(config)

    sink_names = {type(sink).__name__ for sink in logger.sinks}
    assert "TensorBoardLogger" not in sink_names
