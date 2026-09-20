"""Nearest triangle intersection in the stored metre-based world frame."""
import numpy as np
import open3d as o3d


class SurfacePicker:
    def __init__(self, mesh):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def pick(self, near, far):
        near, far = np.asarray(near, float).reshape(3), np.asarray(far, float).reshape(3)
        length = np.linalg.norm(far - near)
        if not np.isfinite([*near, *far, length]).all() or length <= 0:
            raise ValueError('Invalid picking ray')
        direction = (far - near) / length
        result = self.scene.cast_rays(o3d.core.Tensor(np.r_[near, direction][None], dtype=o3d.core.Dtype.Float32))
        distance = float(result['t_hit'].numpy()[0])
        if not np.isfinite(distance) or distance < 0 or distance > length:
            return None
        return {'point_world_m': (near + direction * distance).tolist(),
                'triangle_index': int(result['primitive_ids'].numpy()[0]),
                'coordinate_frame': 'world_rdf', 'unit': 'metre',
                'source': 'nearest_mesh_triangle_intersection', 'scale_validation': 'not_physically_validated'}
