import numpy as np

from ..networks import flatten_observation_features


def space_flat_dim(space) -> int:
    shape = getattr(space, "shape", None)
    if shape is not None:
        return int(np.prod(shape))

    spaces = getattr(space, "spaces", None)
    if isinstance(spaces, dict):
        return int(
            sum(
                space_flat_dim(subspace)
                for key, subspace in spaces.items()
                if key != "action_mask"
            )
        )
    if isinstance(space, dict):
        return int(
            sum(
                space_flat_dim(subspace)
                for key, subspace in space.items()
                if key != "action_mask"
            )
        )

    if isinstance(space, tuple) and len(space) >= 1:
        shape = space[0]
        if isinstance(shape, (tuple, list)):
            return int(np.prod(shape))
    if isinstance(space, list):
        return int(np.prod(space))

    if hasattr(space, "generate_value"):
        sample_obs = space.generate_value()
        flat_obs, _ = flatten_observation_features(sample_obs, batch_ndim=0)
        return int(np.prod(flat_obs.shape))

    raise ValueError(f"Unsupported observation space for flat dim inference: {type(space)}")


def space_feature_dim(obs_space, key: str, default: int) -> int:
    spaces = getattr(obs_space, "spaces", None)
    if isinstance(spaces, dict) and key in spaces:
        leaf = spaces[key]
        shape = getattr(leaf, "shape", None)
        if shape is not None:
            return int(shape[-1])

    if isinstance(obs_space, dict) and key in obs_space:
        leaf = obs_space[key]
        if isinstance(leaf, tuple) and len(leaf) >= 1 and isinstance(leaf[0], (tuple, list)):
            return int(leaf[0][-1])
    return int(default)
