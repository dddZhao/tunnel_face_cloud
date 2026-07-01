from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and attach its directory."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    cfg["_config_path"] = str(config_path)
    cfg["_config_dir"] = str(config_path.parent)
    return cfg


def output_root(cfg: dict[str, Any]) -> Path:
    """Return the output root, resolved relative to the config file."""
    root = Path(cfg.get("output_root", "outputs"))
    if not root.is_absolute():
        config_dir = Path(cfg["_config_dir"])
        project_dir = config_dir.parent if config_dir.name == "configs" else config_dir
        root = project_dir / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cycle_distance_positions(cfg: dict[str, Any]) -> dict[str, float]:
    """Compute cumulative advance positions from distance_from_previous_m."""
    s = 0.0
    positions: dict[str, float] = {}
    for cycle in cfg["cycles"]:
        s += float(cycle.get("distance_from_previous_m", 0.0))
        positions[cycle["id"]] = s
    return positions
