from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np


@dataclass(slots=True)
class PointCloudData:
    """XYZ point cloud plus source metadata.

    Points are stored as an ``(N, 3)`` float64 array in meters.
    """

    xyz: np.ndarray
    path: Path
    las_header: Any | None = None
    dimensions: list[str] | None = None


def read_laz(path: str | Path) -> PointCloudData:
    """Read LAS/LAZ with laspy and return XYZ only."""
    p = Path(path).expanduser()
    las = laspy.read(p)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)
    dims = [d.name for d in las.point_format.dimensions]
    return PointCloudData(xyz=xyz, path=p, las_header=las.header, dimensions=dims)


def write_laz_xyz(path: str | Path, xyz: np.ndarray) -> None:
    """Write XYZ-only LAZ output without modifying source files."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float64)
    header = laspy.LasHeader(point_format=3, version="1.2")
    if len(xyz):
        mins = xyz.min(axis=0)
    else:
        mins = np.zeros(3)
    header.offsets = mins
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.write(p)


def has_rgb(dimensions: list[str] | None) -> bool:
    dims = set(dimensions or [])
    return {"red", "green", "blue"}.issubset(dims)


def has_intensity(dimensions: list[str] | None) -> bool:
    return "intensity" in set(dimensions or [])


def transform_points(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 left-multiplied homogeneous transform to row-stored points."""
    pts = np.asarray(xyz, dtype=np.float64)
    tfm = np.asarray(transform, dtype=np.float64)
    if tfm.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {tfm.shape}")
    if not np.isfinite(tfm).all():
        raise ValueError("Transform contains NaN or Inf.")
    if not np.isfinite(pts).all():
        raise ValueError("Input points contain NaN or Inf.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out_xyz = pts @ tfm[:3, :3].T + tfm[:3, 3]
    if not np.isfinite(out_xyz).all():
        raise ValueError("Transformed points contain NaN or Inf.")
    return out_xyz
