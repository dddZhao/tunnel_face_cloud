from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class PlaneFit:
    normal: np.ndarray
    d: float
    center: np.ndarray
    rmse: float
    median_abs_distance: float
    q95_abs_distance: float
    inlier_ratio: float
    distances: np.ndarray
    inlier_mask: np.ndarray

    def equation(self) -> list[float]:
        return [float(self.normal[0]), float(self.normal[1]), float(self.normal[2]), float(self.d)]


def fit_plane_svd(xyz: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit a plane by total least squares and return unit normal, d, center."""
    center = np.median(xyz, axis=0)
    centered = xyz - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    d = -float(normal @ center)
    return normal, d, center


def point_plane_distances(xyz: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.einsum("ij,j->i", xyz, normal) + d


def ransac_plane(
    xyz: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
    max_points: int | None = None,
) -> PlaneFit:
    """Robust plane fit using RANSAC followed by SVD refinement on inliers."""
    pts = np.asarray(xyz, dtype=np.float64)
    if len(pts) < 3:
        raise ValueError("At least three points are required for plane fitting.")
    rng = np.random.default_rng(seed)
    if max_points and len(pts) > max_points:
        pts = pts[rng.choice(len(pts), size=max_points, replace=False)]
    best_mask: np.ndarray | None = None
    best_count = -1
    for _ in range(iterations):
        ids = rng.choice(len(pts), size=3, replace=False)
        a, b, c = pts[ids]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        d = -float(n @ a)
        with np.errstate(over="ignore", invalid="ignore"):
            distances = point_plane_distances(pts, n, d)
        if not np.isfinite(distances).all():
            continue
        mask = np.abs(distances) <= threshold
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
    if best_mask is None or best_count < 3:
        normal, d, center = fit_plane_svd(pts)
        mask = np.ones(len(pts), dtype=bool)
    else:
        normal, d, center = fit_plane_svd(pts[best_mask])
        mask = np.abs(point_plane_distances(pts, normal, d)) <= threshold
    distances = point_plane_distances(pts, normal, d)
    absd = np.abs(distances)
    return PlaneFit(
        normal=normal,
        d=d,
        center=center,
        rmse=float(np.sqrt(np.mean(distances[mask] ** 2))),
        median_abs_distance=float(np.median(absd)),
        q95_abs_distance=float(np.percentile(absd, 95)),
        inlier_ratio=float(mask.mean()),
        distances=distances,
        inlier_mask=mask,
    )


def plane_to_json_dict(fit: PlaneFit) -> dict[str, Any]:
    return {
        "plane_equation_ax_by_cz_d": fit.equation(),
        "unit_normal": fit.normal.tolist(),
        "plane_center": fit.center.tolist(),
        "rmse_m": fit.rmse,
        "median_abs_distance_m": fit.median_abs_distance,
        "q95_abs_distance_m": fit.q95_abs_distance,
        "inlier_ratio": fit.inlier_ratio,
    }
