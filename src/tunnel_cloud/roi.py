from __future__ import annotations

from typing import Any

import numpy as np

from .io import read_laz


def apply_roi(xyz: np.ndarray, roi_cfg: dict[str, Any] | None) -> np.ndarray:
    """Return points selected by a configured ROI.

    Supported modes: full_cloud, none, bbox, external_laz.
    """
    cfg = roi_cfg or {"method": "full_cloud"}
    method = cfg.get("method", "full_cloud")
    if method in {"full_cloud", "none"}:
        return xyz
    if method == "bbox":
        lo = np.asarray(cfg["min"], dtype=float)
        hi = np.asarray(cfg["max"], dtype=float)
        mask = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        return xyz[mask]
    if method in {"external_laz", "external_file"}:
        return read_laz(cfg["path"]).xyz
    raise ValueError(f"Unsupported ROI method: {method}")
