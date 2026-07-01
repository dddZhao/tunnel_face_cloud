import numpy as np

from tunnel_cloud.projection import FaceProjector


def test_projection_roundtrip_with_pixel_center_tolerance():
    projector = FaceProjector(y_min=-1.0, z_min=2.0, pixel_size_m=0.01, width=500, height=400)
    point = np.array([5.2, 0.234, 3.456])
    u, v = projector.point_to_pixel(point)
    recovered = projector.pixel_depth_to_point(u, v, point[0])
    assert abs(recovered[0] - point[0]) < 1e-12
    assert abs(recovered[1] - point[1]) <= projector.pixel_size_m
    assert abs(recovered[2] - point[2]) <= projector.pixel_size_m
