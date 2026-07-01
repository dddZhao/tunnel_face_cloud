from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_yz_scatter(points: np.ndarray, values: np.ndarray, path: Path, title: str) -> None:
    """Save a simple scientific Y-Z projection scatter plot."""
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
