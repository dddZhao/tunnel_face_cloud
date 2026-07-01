from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .io import read_laz
from .plane_fitting import PlaneFit, plane_to_json_dict, point_plane_distances, ransac_plane
from .quality import nearest_neighbor_stats
from .roi import apply_roi


def robust_centroid(xyz: np.ndarray, trim_ratio: float = 0.1) -> np.ndarray:
    """Compute a coordinate-wise trimmed mean robust center."""
    if len(xyz) == 0:
        raise ValueError("Cannot compute center of an empty point set.")
    lo = np.quantile(xyz, trim_ratio, axis=0)
    hi = np.quantile(xyz, 1.0 - trim_ratio, axis=0)
    mask = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    if mask.sum() < max(10, len(xyz) * 0.1):
        return np.median(xyz, axis=0)
    return np.mean(xyz[mask], axis=0)


def project_point_to_plane(point: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    dist = float(point @ normal + d)
    return point - dist * normal


def estimate_axes(
    plane_normal: np.ndarray,
    vertical_hint: np.ndarray,
    advance_sign: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    """Estimate orthonormal raw-basis tunnel axes ex, ey, ez."""
    warnings: list[str] = []
    ex = np.asarray(plane_normal, dtype=float) * float(advance_sign)
    ex = ex / np.linalg.norm(ex)
    zh = np.asarray(vertical_hint, dtype=float)
    zh = zh / np.linalg.norm(zh)
    z_proj = zh - float(zh @ ex) * ex
    z_norm = np.linalg.norm(z_proj)
    if z_norm < 1e-6:
        warnings.append("vertical_hint_is_parallel_to_tunnel_axis")
        raise ValueError("Vertical hint is nearly parallel to estimated tunnel axis.")
    ez = z_proj / z_norm
    ey = np.cross(ez, ex)
    ey = ey / np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    ez = ez / np.linalg.norm(ez)
    vertical_confidence = float(z_norm)
    return ex, ey, ez, warnings, vertical_confidence


def rotation_raw_to_tunnel(ex: np.ndarray, ey: np.ndarray, ez: np.ndarray) -> np.ndarray:
    """Rows are tunnel axes expressed in raw coordinates."""
    return np.vstack([ex, ey, ez])


def make_transform(rotation: np.ndarray, center_raw: np.ndarray, station_m: float) -> np.ndarray:
    """Create raw-to-tunnel transform using column-vector left multiplication."""
    t = -rotation @ center_raw + np.array([station_m, 0.0, 0.0])
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = t
    return T


def estimate_cycle_frame(
    cycle: dict[str, Any],
    cfg: dict[str, Any],
    output_dir: Path,
    station_m: float,
    seed: int,
) -> dict[str, Any]:
    """Estimate plane, center, axes, and transform for one cycle."""
    cloud = read_laz(cycle["pointcloud"])
    face = apply_roi(cloud.xyz, cycle.get("face_roi"))
    if len(face) < 50:
        raise ValueError(f"{cycle['id']} face ROI has too few points: {len(face)}")
    pcfg = cfg.get("processing", {})
    q = nearest_neighbor_stats(face, int(pcfg.get("quality", {}).get("sample_size", 50000)), seed)
    threshold = pcfg.get("face_plane", {}).get("distance_threshold_m")
    if threshold is None:
        threshold = float(q.get("median_m", 0.01)) * float(
            pcfg.get("face_plane", {}).get("distance_threshold_factor", 2.5)
        )
    plane = ransac_plane(
        face,
        threshold=float(threshold),
        iterations=int(pcfg.get("face_plane", {}).get("num_iterations", 3000)),
        seed=seed,
        max_points=int(pcfg.get("face_plane", {}).get("max_points", 100000)),
    )
    center_method = pcfg.get("frame", {}).get("center_method", "projected_robust_centroid")
    if center_method == "manual_center" and pcfg.get("frame", {}).get("manual_center") is not None:
        center = np.asarray(pcfg["frame"]["manual_center"], dtype=float)
    elif center_method == "bbox_center":
        center = (face.min(axis=0) + face.max(axis=0)) / 2.0
    else:
        center = robust_centroid(face)
        if center_method == "projected_robust_centroid":
            center = project_point_to_plane(center, plane.normal, plane.d)
    vh = np.asarray(cycle.get("vertical_hint", {}).get("axis", [0.0, 0.0, 1.0]), dtype=float)
    sign = float(cycle.get("advance_direction", {}).get("sign", 1.0))
    ex, ey, ez, warnings, vertical_confidence = estimate_axes(plane.normal, vh, sign)
    R = rotation_raw_to_tunnel(ex, ey, ez)
    T = make_transform(R, center, station_m)
    angle_normal_to_x = math.degrees(math.acos(float(np.clip(abs(plane.normal @ ex), -1.0, 1.0))))
    frame = {
        "cycle_id": cycle["id"],
        "station_m": float(station_m),
        "face_point_count": int(len(face)),
        "face_center_raw": center.tolist(),
        "face_normal_raw": plane.normal.tolist(),
        "tunnel_axis_raw": ex.tolist(),
        "lateral_axis_raw": ey.tolist(),
        "vertical_axis_raw": ez.tolist(),
        "rotation_matrix_raw_to_tunnel": R.tolist(),
        "transform_raw_to_tunnel": T.tolist(),
        "transform_tunnel_to_raw": np.linalg.inv(T).tolist(),
        "plane": plane_to_json_dict(plane),
        "plane_fit_rmse_m": plane.rmse,
        "axis_uncertainty_deg": angle_normal_to_x,
        "vertical_uncertainty_deg": math.degrees(math.acos(float(np.clip(vertical_confidence, -1.0, 1.0)))),
        "center_uncertainty_m": float(plane.q95_abs_distance),
        "frame_confidence": float(max(0.0, min(1.0, plane.inlier_ratio * vertical_confidence))),
        "warnings": warnings,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{cycle['id']}_frame.json").write_text(
        json.dumps(frame, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{cycle['id']}_face_plane.json").write_text(
        json.dumps(plane_to_json_dict(plane), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_residual_hist(plane, output_dir / f"{cycle['id']}_plane_residual_hist.png", cycle["id"])
    plot_frame_preview(face, plane, center, ex, ey, ez, output_dir / f"{cycle['id']}_frame_preview.png", cycle["id"], seed)
    return frame


def plot_residual_hist(plane: PlaneFit, path: Path, cycle_id: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(plane.distances, bins=80, color="#4c78a8", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"{cycle_id} plane residuals")
    ax.set_xlabel("signed distance to fitted plane (m)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_frame_preview(
    face: np.ndarray,
    plane: PlaneFit,
    center: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    path: Path,
    cycle_id: str,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    sample = face if len(face) <= 25000 else face[rng.choice(len(face), size=25000, replace=False)]
    dist = point_plane_distances(sample, plane.normal, plane.d)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c=dist, s=0.3, cmap="coolwarm", alpha=0.55)
    scale = max(np.ptp(sample, axis=0).max() * 0.15, 0.5)
    for vec, color, label in [(ex, "red", "X"), (ey, "green", "Y"), (ez, "blue", "Z")]:
        ax.quiver(center[0], center[1], center[2], vec[0], vec[1], vec[2], length=scale, color=color, linewidth=2)
        ax.text(*(center + vec * scale * 1.1), label, color=color)
    ax.scatter([center[0]], [center[1]], [center[2]], c="black", s=25)
    ax.set_title(f"{cycle_id} frame preview")
    ax.set_xlabel("raw X")
    ax.set_ylabel("raw Y")
    ax.set_zlabel("raw Z")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="plane residual (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def estimate_frames(cfg: dict[str, Any], output_dir: Path, stations: dict[str, float]) -> list[dict[str, Any]]:
    seed = int(cfg.get("random_seed", 20260701))
    frames = []
    for i, cycle in enumerate(cfg["cycles"]):
        frames.append(estimate_cycle_frame(cycle, cfg, output_dir, stations[cycle["id"]], seed + i))
    return frames
