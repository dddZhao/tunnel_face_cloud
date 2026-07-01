from __future__ import annotations

import numpy as np


def voxel_downsample_placeholder(xyz: np.ndarray, voxel_size_m: float | None) -> np.ndarray:
    """Return input points for phase 1.

    The phase-1 pipeline keeps the exported tunnel-face crop unchanged for
    deliverable outputs. Analysis functions perform their own reproducible
    sampling where needed.
    """
    return xyz if voxel_size_m is None else xyz
