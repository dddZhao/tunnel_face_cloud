import numpy as np

from tunnel_cloud.plane_fitting import ransac_plane


def test_ransac_plane_with_outliers():
    rng = np.random.default_rng(42)
    xy = rng.uniform(-5, 5, size=(1000, 2))
    z = 0.2 * xy[:, 0] - 0.1 * xy[:, 1] + 3.0 + rng.normal(0, 0.01, size=1000)
    pts = np.column_stack([xy, z])
    outliers = rng.uniform(-10, 10, size=(100, 3))
    cloud = np.vstack([pts, outliers])
    fit = ransac_plane(cloud, threshold=0.05, iterations=500, seed=1)
    expected = np.array([-0.2, 0.1, 1.0])
    expected = expected / np.linalg.norm(expected)
    assert abs(float(fit.normal @ expected)) > 0.999
    assert fit.rmse < 0.03
    assert fit.inlier_ratio > 0.85
