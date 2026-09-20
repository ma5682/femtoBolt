from __future__ import annotations

import unittest

import numpy as np

from femto.geometry.pose_geometry import (
    Intrinsics,
    Landmark2D,
    back_project_pixel,
    depth_to_points,
    landmarks_to_3d,
    masked_depth_residual_mm,
    robust_depth_mm,
)


class GeometryTests(unittest.TestCase):
    def test_back_projection_at_principal_point(self):
        intr = Intrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
        point = back_project_pixel(320, 240, 1000.0, intr)
        np.testing.assert_allclose(point, [0.0, 0.0, 1.0])

    def test_foreground_cluster_is_selected(self):
        depth = np.full((11, 11), 1500.0, dtype=np.float32)
        depth[3:8, 3:8] = 800.0
        depth[5, 5] = 0.0
        value, count = robust_depth_mm(depth, 5, 5, radius=5)
        self.assertGreaterEqual(count, 3)
        self.assertAlmostEqual(value, 800.0)

    def test_landmark_depth_and_surface_use_same_intrinsics(self):
        intr = Intrinsics(5, 5, 4.0, 4.0, 2.0, 2.0)
        depth = np.full((5, 5), 1000.0, dtype=np.float32)
        joints, pixels, _ = landmarks_to_3d(
            [Landmark2D(0.5, 0.5)], depth, intr, depth_radius=1
        )
        points, pixel_list = depth_to_points(depth, intr, stride=1)
        selected = points[np.all(pixel_list == pixels[0], axis=1)][0]
        np.testing.assert_allclose(joints[0], selected)

    def test_depth_residual(self):
        one = np.full((32, 32), 800.0, dtype=np.float32)
        two = np.full((32, 32), 812.0, dtype=np.float32)
        mask = np.ones((32, 32), dtype=np.uint8)
        value = masked_depth_residual_mm(one, two, mask, mask, stride=2)
        self.assertEqual(value, 12.0)


if __name__ == "__main__":
    unittest.main()
