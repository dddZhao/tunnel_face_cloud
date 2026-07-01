import numpy as np

from tunnel_cloud.config import cycle_distance_positions
from tunnel_cloud.frame_estimation import make_transform
from tunnel_cloud.io import transform_points


def test_cumulative_positions_are_exact():
    cfg = {
        "cycles": [
            {"id": "C01", "distance_from_previous_m": 0.0},
            {"id": "C02", "distance_from_previous_m": 2.0},
            {"id": "C03", "distance_from_previous_m": 2.1},
        ]
    }
    assert cycle_distance_positions(cfg) == {"C01": 0.0, "C02": 2.0, "C03": 4.1}


def test_transform_places_center_at_station():
    R = np.eye(3)
    center = np.array([10.0, 20.0, 30.0])
    T = make_transform(R, center, station_m=5.2)
    out = transform_points(center[None, :], T)[0]
    assert np.allclose(out, [5.2, 0.0, 0.0])
