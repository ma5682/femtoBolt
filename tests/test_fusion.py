import unittest

import numpy as np
import open3d as o3d
import trimesh

from fusion import fill_small_mesh_holes


def boundary_edges(mesh: o3d.geometry.TriangleMesh) -> int:
    tri = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        process=False,
    )
    counts = np.bincount(tri.edges_unique_inverse)
    return int((counts == 1).sum())


class FusionTests(unittest.TestCase):
    def test_small_internal_hole_is_filled_but_outer_boundary_stays_open(self) -> None:
        spacing = 0.004
        vertices = np.array(
            [[x * spacing, y * spacing, 0.0] for y in range(4) for x in range(4)],
            dtype=np.float64,
        )
        faces = []
        for y in range(3):
            for x in range(3):
                if (x, y) == (1, 1):
                    continue
                a = y * 4 + x
                b = a + 1
                c = a + 4
                d = c + 1
                faces.extend(((a, b, d), (a, d, c)))
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))

        repaired = fill_small_mesh_holes(mesh, maximum_radius_m=0.006)

        self.assertGreater(len(repaired.triangles), len(mesh.triangles))
        self.assertEqual(boundary_edges(repaired), 12)
        self.assertFalse(repaired.is_watertight())
        self.assertTrue(repaired.is_edge_manifold(allow_boundary_edges=True))


if __name__ == "__main__":
    unittest.main()
