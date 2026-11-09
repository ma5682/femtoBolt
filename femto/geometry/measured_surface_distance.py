"""Measured triangle witnesses for a bounded, one-way surface distance check."""
import numpy as np


def same_surface_support(mesh, scene, points, reference_points, reference_faces,
                         tolerance_mm, check=lambda: None):
    """Keep only witnesses on the reference hit's edge-connected component.

    Open3D proposes triangle IDs and barycentric coordinates. Distances are
    recomputed against convex points on the original float64 triangles, so an
    approximate nearest query cannot manufacture a positive distance result.
    This is pointwise support, not a two-way Hausdorff or coverage certificate.
    """
    import open3d as o3d
    from femto.geometry.triangle_components import triangle_components
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    points, reference_points = np.asarray(points, float), np.asarray(reference_points, float)
    reference_faces = np.array(reference_faces, dtype=np.int64, copy=True)
    if (points.ndim != 2 or points.shape[1] != 3 or reference_points.shape != points.shape
            or reference_faces.shape != (len(points),) or not np.isfinite(points).all()
            or not np.isfinite(reference_points).all() or not 0 < tolerance_mm <= 2):
        raise ValueError('实测表面距离输入无效')
    check()
    components = triangle_components(mesh)
    if components is None:
        components = mesh.cluster_connected_triangles()
    labels = np.asarray(components[0])
    check()

    def witnesses(query):
        ids_parts, distance_parts = [], []
        for start in range(0, len(query), 32768):
            check(); part = query[start:start+32768]
            result = scene.compute_closest_points(o3d.core.Tensor(part.astype(np.float32)))
            ids = result['primitive_ids'].numpy().astype(np.int64)
            uv = result['primitive_uvs'].numpy().astype(float)
            valid = (ids >= 0) & (ids < len(faces)) & np.isfinite(uv).all(axis=1)
            weights = np.maximum(np.c_[1-uv.sum(axis=1), uv], 0.)
            weights /= weights.sum(axis=1, keepdims=True)
            distance = np.full(len(part), np.inf)
            triangles = vertices[faces[ids[valid]]]
            closest = np.einsum('ij,ijk->ik', weights[valid], triangles)
            distance[valid] = np.linalg.norm(part[valid]-closest, axis=1)
            ids[~valid] = -1
            ids_parts.append(ids);distance_parts.append(distance)
        return np.concatenate(ids_parts), np.concatenate(distance_parts)

    if not len(points):return np.empty(0, bool), np.empty(0), dict(component_mismatches=0)
    candidate_faces, distances = witnesses(points)
    missing = (reference_faces < 0) | (reference_faces >= len(faces))
    if missing.any():
        ids, residual = witnesses(reference_points[missing])
        # Exact-ray retries may lack an ID; bind only an actual triangle at that
        # verified source intersection, never a neighbour across a positive gap.
        reference_faces[missing] = np.where(residual+1e-9 <= 1e-6, ids, -1)
    valid = (reference_faces >= 0) & (candidate_faces >= 0)
    same = np.zeros(len(points), bool)
    same[valid] = labels[reference_faces[valid]] == labels[candidate_faces[valid]]
    check()
    return (same & (distances+1e-9 <= tolerance_mm/1000), distances*1000,
            dict(component_mismatches=int((valid & ~same).sum()),
                 unresolved_reference_faces=int((reference_faces < 0).sum())))
