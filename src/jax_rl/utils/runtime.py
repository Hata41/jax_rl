import os
import time
from contextlib import contextmanager
from pathlib import Path


def configure_jax_runtime_defaults(
    platform: str | None = None,
    cuda_visible_devices: str | None = None,
) -> None:
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    if platform is not None:
        os.environ["JAX_PLATFORMS"] = str(platform)

    if os.environ.get("JAX_PLATFORMS") or os.environ.get("JAX_PLATFORM_NAME"):
        return

    has_nvidia = any(
        Path(path).exists()
        for path in (
            "/dev/nvidiactl",
            "/dev/nvidia0",
            "/proc/driver/nvidia/version",
        )
    )
    if not has_nvidia:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ.setdefault("JAX_SKIP_CUDA_CONSTRAINTS_CHECK", "1")


def safe_steps_per_second(work_units: float, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0.0:
        return float("nan")
    return float(work_units / elapsed_seconds)


class PhaseTimer:
    def __init__(self, now_fn=None):
        self._now = now_fn or time.time
        self._elapsed_seconds: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str):
        start = self._now()
        try:
            yield
        finally:
            self._elapsed_seconds[name] = self._now() - start

    def elapsed(self, name: str) -> float:
        return float(self._elapsed_seconds.get(name, 0.0))

    def steps_per_second(self, name: str, work_units: float) -> float:
        return safe_steps_per_second(work_units, self.elapsed(name))