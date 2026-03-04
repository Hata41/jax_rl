from __future__ import annotations

from pathlib import Path
import time

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class TensorBoardWriter:
    def __init__(self, log_dir: str):
        self._writer = EventFileWriter(log_dir)

    def add_scalar(self, tag: str, scalar_value: float, global_step: int):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=float(scalar_value))])
        event = Event(wall_time=time.time(), step=int(global_step), summary=summary)
        self._writer.add_event(event)

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()


def create_tensorboard_writer(logdir: str | None, run_name: str):
    if not logdir:
        return None

    run_dir = Path(logdir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return TensorBoardWriter(str(run_dir))


def log_scalar_metrics(writer, metrics: dict[str, float], step: int):
    if writer is None:
        return
    for key, value in sorted(metrics.items()):
        writer.add_scalar(key, float(value), step)