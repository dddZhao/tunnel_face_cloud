from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .io import PointCloudData, has_intensity, has_rgb, read_laz


def _sample_indices(n: int, sample_size: int, seed: int) -> np.ndarray:
    if n <= sample_size:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=sample_size, replace=False))


def nearest_neighbor_stats(xyz: np.ndarray, sample_size: int, seed: int) -> dict[str, Any]:
    """Compute reproducible nearest-neighbor distance statistics on a sample."""
    idx = _sample_indices(len(xyz), sample_size, seed)
    sample = xyz[idx]
    if len(sample) < 2:
        return {"sample_size": len(sample)}
    tree = cKDTree(xyz)
    dist, _ = tree.query(sample, k=2, workers=-1)
    nn = dist[:, 1]
    q = np.percentile(nn, [5, 25, 50, 75, 95])
    return {
        "sample_size": int(len(sample)),
        "mean_m": float(np.mean(nn)),
        "median_m": float(np.median(nn)),
        "q05_m": float(q[0]),
        "q25_m": float(q[1]),
        "q75_m": float(q[3]),
        "q95_m": float(q[4]),
    }


def inspect_cloud(
    cycle: dict[str, Any],
    output_dir: Path,
    quality_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Inspect one cloud and write JSON plus bbox preview."""
    cloud = read_laz(cycle["pointcloud"])
    xyz = cloud.xyz
    finite_mask = np.isfinite(xyz).all(axis=1)
    mins = np.nanmin(xyz, axis=0)
    maxs = np.nanmax(xyz, axis=0)
    bbox_size = maxs - mins
    center = (mins + maxs) / 2.0
    rounded = np.round(xyz[finite_mask], int(quality_cfg.get("duplicate_round_decimals", 6)))
    duplicate_count = int(len(rounded) - len(np.unique(rounded, axis=0)))
    nn = nearest_neighbor_stats(
        xyz[finite_mask],
        int(quality_cfg.get("sample_size", 50000)),
        seed,
    )
    median_nn = float(nn.get("median_m", math.nan))
    bbox_volume = float(np.prod(np.maximum(bbox_size, 1e-12)))
    global_density = float(len(xyz) / bbox_volume)
    recommended_voxel = median_nn * 3.0 if math.isfinite(median_nn) else math.nan
    recommended_plane_threshold = median_nn * 2.5 if math.isfinite(median_nn) else math.nan
    unit_hint = "meter_likely" if np.nanmax(bbox_size) < 1000 else "check_unit_or_coordinates"

    result = {
        "cycle_id": cycle["id"],
        "file_path": str(cloud.path),
        "file_size_bytes": int(Path(cloud.path).stat().st_size),
        "point_count": int(len(xyz)),
        "xyz_min": mins.tolist(),
        "xyz_max": maxs.tolist(),
        "bbox_size": bbox_size.tolist(),
        "bbox_center": center.tolist(),
        "has_rgb": has_rgb(cloud.dimensions),
        "has_intensity": has_intensity(cloud.dimensions),
        "las_dimensions": cloud.dimensions,
        "has_nan_or_inf": bool(np.any(~finite_mask)),
        "duplicate_point_count": duplicate_count,
        "duplicate_point_ratio": float(duplicate_count / max(len(xyz), 1)),
        "unit_hint": unit_hint,
        "nearest_neighbor": nn,
        "global_bbox_density_points_per_m3": global_density,
        "recommended_voxel_size_m": recommended_voxel,
        "recommended_plane_threshold_m": recommended_plane_threshold,
        "recommended_normal_radius_m": median_nn * 6.0 if math.isfinite(median_nn) else math.nan,
        "recommended_roughness_radii_m": [
            median_nn * 3.0,
            median_nn * 6.0,
            median_nn * 12.0,
        ]
        if math.isfinite(median_nn)
        else [],
        "obvious_outlier_hint": bool(np.nanmax(bbox_size) > 100 * np.nanmedian(np.maximum(bbox_size, 1e-12))),
    }
    (output_dir / f"{cycle['id']}_quality.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_bbox(xyz, output_dir / f"{cycle['id']}_bbox.png", cycle["id"], seed)
    return result


def plot_bbox(xyz: np.ndarray, path: Path, title: str, seed: int) -> None:
    idx = _sample_indices(len(xyz), min(20000, len(xyz)), seed)
    sample = xyz[idx]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=0.2, alpha=0.4)
    ax.set_title(f"{title} sampled point cloud and bbox")
    ax.set_xlabel("raw X")
    ax.set_ylabel("raw Y")
    ax.set_zlabel("raw Z")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def inspect_dataset(cfg: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    """Inspect all configured cycles and write summary CSV/Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    qcfg = cfg.get("processing", {}).get("quality", {})
    seed = int(cfg.get("random_seed", 20260701))
    rows = [inspect_cloud(cycle, output_dir, qcfg, seed + i) for i, cycle in enumerate(cfg["cycles"])]
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    write_quality_report(rows, output_dir / "quality_report.md")
    return summary


def write_quality_report(rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Point Cloud Quality Report", ""]
    for row in rows:
        nn = row["nearest_neighbor"]
        lines += [
            f"## {row['cycle_id']}",
            f"- File: `{row['file_path']}`",
            f"- Points: {row['point_count']}",
            f"- BBox size m: {row['bbox_size']}",
            f"- RGB: {row['has_rgb']}; Intensity field: {row['has_intensity']}",
            f"- NaN/Inf: {row['has_nan_or_inf']}",
            f"- Duplicate ratio: {row['duplicate_point_ratio']:.6f}",
            f"- NN median m: {nn.get('median_m')}",
            f"- Recommended voxel m: {row['recommended_voxel_size_m']}",
            f"- Recommended plane threshold m: {row['recommended_plane_threshold_m']}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
