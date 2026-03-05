"""Domain-specific exceptions for jax_rl."""


class JaxRLError(Exception):
    """Base exception for all jax_rl domain errors."""


class ConfigDivisibilityError(JaxRLError, ValueError):
    """Raised when rollout/minibatch/env counts are not device-divisible."""


class NetworkTargetResolutionError(JaxRLError, ValueError):
    """Raised when Hydra network target config cannot be resolved or instantiated."""


class EnvironmentNotFoundError(JaxRLError, ValueError):
    """Raised when no environment backend can build the requested environment."""


class CheckpointRestoreError(JaxRLError, FileNotFoundError):
    """Raised when restoring checkpoints fails due to missing/invalid checkpoint data."""
