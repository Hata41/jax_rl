import os
from pathlib import Path


def configure_jax_runtime_defaults() -> None:
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