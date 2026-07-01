from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .io import read_laz
from .quality import nearest_neighbor_stats


def compute_basic_features(points: np.ndarray, seed: int, sample_size: int = 120000) -> dict[str, np.ndarray | float]:
    """Compute first-stage geometry features from XYZ only."""
    rng = np.random.default_rng(seed)
    pts = points
    if len(points) > sample_size:
        pts = points[rng.choice(len(points), size=sample_size, replace=False)]
    nn = nearest_neighbor_stats(pts, min(50000, len(pts)), seed)
    d = float(nn.get("median_m", 0.01))
    tree = cKDTree(pts)
    dist, _ = tree.query(pts, k=2, workers=-1)
    nn_dist = dist[:, 1]
    depth = pts[:, 0] - np.median(pts[:, 0])
    radii = np.array([3 * d, 6 * d, 12 * d])
    roughness = []
    density = []
    for radius in radii:
        neighborhoods = tree.query_ball_point(pts, r=float(radius), workers=-1)
        rvals = np.empty(len(pts), dtype=float)
        dvals = np.empty(len(pts), dtype=float)
        volume = 4.0 / 3.0 * np.pi * radius**3
        for i, ids in enumerate(neighborhoods):
            local = pts[ids]
            dvals[i] = len(ids) / volume
            if len(local) < 5:
                rvals[i] = np.nan
                continue
            cov = np.cov((local - local.mean(axis=0)).T)
            eig = np.linalg.eigvalsh(cov)
            rvals[i] = np.sqrt(max(eig[0], 0.0))
        roughness.append(rvals)
        density.append(dvals)
    return {
        "points": pts,
        "nearest_neighbor_m": nn_dist,
        "signed_depth_m": depth,
        "radii_m": radii,
        "roughness": np.vstack(roughness),
        "local_density": np.vstack(density),
        "nn_median_m": d,
    }


def generate_features(cfg: dict[str, Any], placement_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("random_seed", 20260701))
    rows = []
    for i, cycle in enumerate(cfg["cycles"]):
        cloud = read_laz(placement_dir / f"{cycle['id']}_face_relative.laz")
        feats = compute_basic_features(
            cloud.xyz,
            seed + i,
            int(cfg.get("processing", {}).get("features", {}).get("sample_size", 120000)),
        )
        np.savez_compressed(output_dir / f"{cycle['id']}_face_features.npz", **feats)
        rough = feats["roughness"]
        rows.append(
            {
                "cycle_id": cycle["id"],
                "sample_points": len(feats["points"]),
                "nn_median_m": feats["nn_median_m"],
                "roughness_r1_median_m": float(np.nanmedian(rough[0])),
                "roughness_r2_median_m": float(np.nanmedian(rough[1])),
                "roughness_r3_median_m": float(np.nanmedian(rough[2])),
            }
        )
        plot_feature(feats["points"], rough[0], output_dir / f"{cycle['id']}_roughness_r1.png", f"{cycle['id']} roughness r1")
        plot_feature(feats["points"], feats["signed_depth_m"], output_dir / f"{cycle['id']}_depth_map.png", f"{cycle['id']} signed depth")
    pd.DataFrame(rows).to_csv(output_dir / "feature_statistics.csv", index=False)


def plot_feature(points: np.ndarray, values: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(points[:, 1], points[:, 2], c=values, s=0.5, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Tunnel Y (m)")
    ax.set_ylabel("Tunnel Z (m)")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
