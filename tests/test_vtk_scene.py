"""Display copies preserve world coordinates, topology IDs and route breaks."""
from types import SimpleNamespace
import unittest

import numpy as np

from femto.viewer.vtk_scene import scene_snapshot


class VTKSceneTests(unittest.TestCase):
    def setUp(self):
        self.mesh = SimpleNamespace(vertices=np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]]),
                                    triangles=np.array([[0, 1, 2], [1, 3, 2]]),
                                    vertex_colors=np.ones((4, 3)))
        self.model = SimpleNamespace(mesh=self.mesh)

    def test_original_ids_axes_and_source_ownership(self):
        scene = scene_snapshot(self.model, {'triangle_indices': [1]})
        np.testing.assert_array_equal(scene.vertices, self.mesh.vertices)
        np.testing.assert_array_equal(scene.triangles[scene.region_ids], [[1, 3, 2]])
        self.assertFalse(np.shares_memory(scene.vertices, self.mesh.vertices))
        self.assertFalse(scene.vertices.flags.writeable)
        self.assertTrue(self.mesh.vertices.flags.writeable)
        self.assertEqual(scene.colors[0].tolist(), [255, 255, 255])

    def test_disconnected_route_and_missing_coordinates_never_bridge(self):
        p = lambda x: {'xyz_m': [x, 0, 1]}
        scene = scene_snapshot(self.model, plan={'segments': [[p(0), p(1), {}, p(3), p(4)], [p(7), p(8)]]})
        self.assertEqual([run[:, 0].tolist() for run in scene.route_runs], [[0, 1], [3, 4], [7, 8]])

    def test_reject_invalid_region_and_coordinate_convention(self):
        for ids in ([2], [-1], [.5]):
            with self.assertRaises(ValueError):
                scene_snapshot(self.model, {'triangle_indices': ids})
        with self.assertRaises(ValueError):
            scene_snapshot(self.model, plan={'unit': 'mm'})

    def test_vertex_only_geometry(self):
        self.mesh.triangles = []
        scene = scene_snapshot(self.model)
        self.assertEqual(scene.triangles.shape, (0, 3))
        self.assertEqual(len(scene.vertices), 4)


if __name__ == '__main__':
    unittest.main()
