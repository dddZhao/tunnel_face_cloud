from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .io import read_laz


@dataclass(slots=True)
class FaceProjector:
    """Project tunnel-coordinate points to Y-Z image coordinates."""

    y_min: float
    z_min: float
    pixel_size_m: float
    width: int
    height: int

    def point_to_pixel(self, point_xyz: np.ndarray) -> tuple[int, int]:
        y = float(point_xyz[1])
        z = float(point_xyz[2])
        u = int(np.floor((y - self.y_min) / self.pixel_size_m))
        v = int(np.floor((z - self.z_min) / self.pixel_size_m))
        return u, self.height - 1 - v

    def pixel_depth_to_point(self, u: int, v: int, depth: float) -> np.ndarray:
        y = self.y_min + (u + 0.5) * self.pixel_size_m
        z = self.z_min + (self.height - 1 - v + 0.5) * self.pixel_size_m
        return np.array([depth, y, z], dtype=float)

    def metadata(self, cycle_id: str) -> dict[str, Any]:
        return {
            "cycle_id": cycle_id,
            "image_width": self.width,
            "image_height": self.height,
            "pixel_size_m": self.pixel_size_m,
            "y_min": self.y_min,
            "y_max": self.y_min + self.width * self.pixel_size_m,
            "z_min": self.z_min,
            "z_max": self.z_min + self.height * self.pixel_size_m,
            "depth_axis": "X",
            "horizontal_axis": "Y",
            "vertical_axis": "Z",
        }


def make_projector(points: np.ndarray, cfg: dict[str, Any]) -> FaceProjector:
    pcfg = cfg.get("processing", {}).get("projection", {})
    yz_min = points[:, 1:3].min(axis=0)
    yz_max = points[:, 1:3].max(axis=0)
    span = yz_max - yz_min
    max_size = int(pcfg.get("max_image_size", 2048))
    pixel = pcfg.get("pixel_size_m", "auto")
    if pixel == "auto":
        pixel_size = max(float(np.max(span) / max_size), float(pcfg.get("min_pixel_size_m", 0.005)))
    else:
        pixel_size = float(pixel)
    width = int(np.ceil(span[0] / pixel_size)) + 1
    height = int(np.ceil(span[1] / pixel_size)) + 1
    return FaceProjector(float(yz_min[0]), float(yz_min[1]), pixel_size, width, height)


def project_points(points: np.ndarray, projector: FaceProjector) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.full((projector.height, projector.width), np.nan, dtype=np.float32)
    density = np.zeros((projector.height, projector.width), dtype=np.uint32)
    for p in points:
        u, v = projector.point_to_pixel(p)
        if 0 <= u < projector.width and 0 <= v < projector.height:
            density[v, u] += 1
            if np.isnan(depth[v, u]) or p[0] > depth[v, u]:
                depth[v, u] = p[0]
    mask = density > 0
    return depth, density, mask


def save_float_tiff(path: Path, arr: np.ndarray) -> None:
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img.save(path)


def generate_projections(cfg: dict[str, Any], placement_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for cycle in cfg["cycles"]:
        cloud = read_laz(placement_dir / f"{cycle['id']}_face_relative.laz")
        projector = make_projector(cloud.xyz, cfg)
        depth, density, mask = project_points(cloud.xyz, projector)
        save_float_tiff(output_dir / f"{cycle['id']}_depth.tif", depth)
        save_preview(depth, output_dir / f"{cycle['id']}_depth_preview.png", "Depth X (m)", cmap="magma")
        save_preview(density.astype(float), output_dir / f"{cycle['id']}_density.png", "Point density", cmap="viridis")
        Image.fromarray((mask.astype(np.uint8) * 255)).save(output_dir / f"{cycle['id']}_valid_mask.png")
        (output_dir / f"{cycle['id']}_projection.json").write_text(
            json.dumps(projector.metadata(cycle["id"]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def save_preview(arr: np.ndarray, path: Path, title: str, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(arr, cmap=cmap)
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
