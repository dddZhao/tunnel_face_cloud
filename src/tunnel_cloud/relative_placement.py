from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .io import read_laz, transform_points, write_laz_xyz
from .roi import apply_roi


def load_frame(frame_dir: Path, cycle_id: str) -> dict[str, Any]:
    path = frame_dir / f"{cycle_id}_frame.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing frame file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def place_cycles(cfg: dict[str, Any], frame_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    """Transform configured cycles to the unified tunnel coordinate system."""
    output_dir.mkdir(parents=True, exist_ok=True)
    transform_dir = output_dir / "transforms"
    transform_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    placed_samples: list[tuple[str, np.ndarray, np.ndarray]] = []
    for cycle in cfg["cycles"]:
        frame = load_frame(frame_dir, cycle["id"])
        T = np.asarray(frame["transform_raw_to_tunnel"], dtype=float)
        Tinv = np.asarray(frame["transform_tunnel_to_raw"], dtype=float)
        cloud = read_laz(cycle["pointcloud"])
        rel = transform_points(cloud.xyz, T)
        face = apply_roi(cloud.xyz, cycle.get("face_roi"))
        face_rel = transform_points(face, T)
        write_laz_xyz(output_dir / f"{cycle['id']}_relative.laz", rel)
        write_laz_xyz(output_dir / f"{cycle['id']}_face_relative.laz", face_rel)
        np.save(transform_dir / f"{cycle['id']}_raw_to_tunnel.npy", T)
        np.save(transform_dir / f"{cycle['id']}_tunnel_to_raw.npy", Tinv)
        np.savetxt(transform_dir / f"{cycle['id']}_raw_to_tunnel.txt", T, fmt="%.10f")
        center = np.asarray(frame["face_center_raw"], dtype=float)
        center_rel = transform_points(center[None, :], T)[0]
        rows.append(
            {
                "cycle_id": cycle["id"],
                "station_m": frame["station_m"],
                "center_x_m": center_rel[0],
                "center_y_m": center_rel[1],
                "center_z_m": center_rel[2],
                "plane_fit_rmse_m": frame["plane_fit_rmse_m"],
                "frame_confidence": frame["frame_confidence"],
                "advance_distance_uncertainty_m": 0.0,
            }
        )
        sample = rel if len(rel) <= 25000 else rel[np.linspace(0, len(rel) - 1, 25000).astype(int)]
        placed_samples.append((cycle["id"], sample, center_rel))
    write_summary(rows, output_dir / "placement_summary.csv")
    plot_multi_cycle(placed_samples, output_dir / "multi_cycle_overview.png")
    return rows


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_multi_cycle(samples: list[tuple[str, np.ndarray, np.ndarray]], path: Path) -> None:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(samples), 1)))
    centers = []
    for color, (cycle_id, pts, center) in zip(colors, samples):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.25, alpha=0.35, color=color, label=cycle_id)
        ax.scatter([center[0]], [center[1]], [center[2]], color=color, s=45, marker="x")
        centers.append(center)
    if centers:
        centers_arr = np.vstack(centers)
        ax.plot(centers_arr[:, 0], centers_arr[:, 1], centers_arr[:, 2], color="black", linewidth=1.5)
    ax.set_title("Multi-cycle relative placement")
    ax.set_xlabel("Tunnel X advance (m)")
    ax.set_ylabel("Tunnel Y lateral (m)")
    ax.set_zlabel("Tunnel Z vertical (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
