from __future__ import annotations

from typing import Iterable

import numpy as np


def adjacent_station_errors(stations: Iterable[float], known_distances: Iterable[float]) -> list[float]:
    """Return axial placement errors for adjacent cycles."""
    s = list(stations)
    d = list(known_distances)
    return [float((s[i] - s[i - 1]) - d[i]) for i in range(1, min(len(s), len(d)))]
