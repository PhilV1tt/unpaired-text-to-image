'''to easily have each configuration of the disentanglement_dataset'''

import numpy as np

FACTORS = {
    "object_color": ["white", "green", "red", "blue", "brown", "olive"],
    "object_shape": ["cone", "cube", "cylinder", "hexagonal prism", "pyramid", "sphere"],
    "object_size": ["small", "large"],
    "camera_height": ["top", "center", "bottom"],
    "background_color": ["purple", "sea green", "salmon"],
}
SHAPE = tuple(len(values) for values in FACTORS.values())
N_CONFIGS = int(np.prod(SHAPE))


def check(config: tuple[int, ...]) -> None:
    """Raise if a config has the wrong length, a non-integer index or an index outside its factor's range."""
    if len(config) != len(SHAPE):
        raise ValueError(f"config {config} has {len(config)} coordinates, expected {len(SHAPE)}")
    for name, a, n in zip(FACTORS, config, SHAPE, strict=True):
        if isinstance(a, bool) or not isinstance(a, (int, np.integer)):
            raise TypeError(f"config {config}: {name} = {a!r} is a {type(a).__name__}, expected int")
        if not 0 <= a < n:
            raise ValueError(f"config {config}: {name} = {a} is outside [0, {n - 1}]")


def encode(config: tuple[int, ...]) -> int:
    """Value indices (one per factor) -> dataset index."""
    check(config)
    return int(np.ravel_multi_index(config, SHAPE))


def decode(index: int) -> tuple[int, ...]:
    """Dataset index -> value indices (one per factor)."""
    if not 0 <= index < N_CONFIGS:
        raise ValueError(f"index {index} is outside [0, {N_CONFIGS - 1}]")
    return tuple(int(a) for a in np.unravel_index(index, SHAPE))


def caption(config: tuple[int, ...]) -> str:
    check(config)
    color, shape, size, camera, background = (
        values[a] for values, a in zip(FACTORS.values(), config, strict=True)
    )
    return f"a {size} {color} {shape}, camera at the {camera}, on a {background} background"
