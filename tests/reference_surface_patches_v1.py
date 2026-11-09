"""Connected metric cells with measured representatives and preserved boundaries.

This creates an explicitly approximate planning surface, never capture evidence.
Every output vertex is an input vertex. The full source surface is retained by
the caller. No disconnected parts are joined merely because they share a cell.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
import open3d as o3d

VERSION = 'connected-measured-surface-cells-v1'
CELL_MM = 3.
MAX_DISPLACEMENT_MM = 5.


def _edges(faces):
    return np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)


def simplify(mesh, check=lambda: None, *, cell_mm=CELL_MM):
    """Refine offending cells until edge incidence and boundary identity hold.

    The maximum vertex displacement bounds the affine map on every source
    triangle, including collapsed triangles. It is not a physical accuracy or
    continuous surface coverage certificate. Raw-depth checks remain required.
    """
    if type(cell_mm) not in (int, float) or not np.isfinite(cell_mm) or not 1 <= cell_mm <= 5:
        raise ValueError('规划表面分区尺寸应为 1–5 mm')
    check()
    v = np.asarray(mesh.vertices); f = np.asarray(mesh.triangles)
    if not len(f) or not np.isfinite(v).all():raise ValueError('规划分区需要有效的原始表面')
    record = dict(schema=VERSION, cell_mm=float(cell_mm), max_displacement_limit_mm=MAX_DISPLACEMENT_MM,
                  input_vertices=len(v), input_triangles=len(f), representative='nearest_measured_vertex_to_cell_centroid',
                  preserve_source_boundary=True, merge_disconnected_cells=False,
                  orientation_rule='preserve_reference_ray_side_not_noisy_per_pixel_normals',
                  physical_accuracy_verified=False, execution_ready=False)
    if len(f) < 20000:
        record.update(status='already_sparse', output_vertices=len(v), output_triangles=len(f),
                      max_vertex_displacement_mm=0., refinement_passes=0)
        return deepcopy(mesh), record
    edges, counts = np.unique(_edges(f), axis=0, return_counts=True)
    protected = np.unique(edges[counts != 2])
    locked = np.zeros(len(v), bool); locked[protected] = True
    # Tangential cells in the fixed reference camera XY plane avoid creating
    # artificial zigzags at depth-quantization bin boundaries. The full XYZ
    # displacement and orientation checks below refine steep/curved cells.
    cells = np.floor(v[:, :2] / (cell_mm / 1000)).astype(np.int64)
    inside = (cells[edges[:, 0]] == cells[edges[:, 1]]).all(axis=1)
    inside &= ~(locked[edges[:, 0]] | locked[edges[:, 1]])
    links = edges[inside]
    labels = np.arange(len(v))
    for _ in range(64):
        check()
        a, b = labels[links[:, 0]], labels[links[:, 1]]
        different = a != b
        if not different.any():break
        low, high = np.minimum(a[different], b[different]), np.maximum(a[different], b[different])
        np.minimum.at(labels, high, low)
        for _ in range(6):labels = labels[labels]
    else:raise ValueError('规划分区连通性计算超限，原模型保留')
    source_boundary = edges[counts == 1]
    original_side = np.einsum('ij,ij->i', np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), v[f[:, 0]])
    for iteration in range(12):
        check()
        keys = labels.copy(); keys[locked] = np.arange(len(v))[locked] + len(v)
        _, inverse = np.unique(keys, return_inverse=True)
        count = np.bincount(inverse)
        centre = np.column_stack([np.bincount(inverse, weights=v[:, axis]) / count for axis in range(3)])
        distance = np.sum((v - centre[inverse]) ** 2, axis=1)
        order = np.lexsort((np.arange(len(v)), distance, inverse))
        _, first = np.unique(inverse[order], return_index=True)
        representatives = order[first]
        mapped = inverse[f]
        keep = (mapped[:, 0] != mapped[:, 1]) & (mapped[:, 1] != mapped[:, 2]) & (mapped[:, 2] != mapped[:, 0])
        candidate = mapped[keep]
        _, unique = np.unique(np.sort(candidate, axis=1), axis=0, return_index=True)
        candidate = candidate[np.sort(unique)]
        coarse_edges, incidence = np.unique(_edges(candidate), axis=0, return_counts=True)
        bad = list(np.unique(coarse_edges[incidence > 2]))
        # A cell containing a loop/hole/closed component cannot be contracted
        # to a representative vertex. Refine it instead of deleting topology.
        internal_edges = inverse[edges[:, 0]] == inverse[edges[:, 1]]
        internal_faces = (mapped[:, 0] == mapped[:, 1]) & (mapped[:, 1] == mapped[:, 2])
        characteristic = (count - np.bincount(inverse[edges[internal_edges, 0]], minlength=len(count))
                          + np.bincount(mapped[internal_faces, 0], minlength=len(count)))
        bad.extend(np.flatnonzero(characteristic != 1))
        # Any new boundary would mean a hole was introduced by collapsing cells.
        boundary = coarse_edges[incidence == 1]
        expected = inverse[source_boundary]
        expected_keys = np.sort(expected, axis=1)[:, 0] * len(representatives) + np.sort(expected, axis=1)[:, 1]
        boundary_keys = boundary[:, 0] * len(representatives) + boundary[:, 1]
        bad.extend(boundary[~np.isin(boundary_keys, expected_keys)].ravel())
        bad.extend(expected[~np.isin(expected_keys, boundary_keys)].ravel())
        # Preserve reference-ray orientation; per-pixel noisy normals may change
        # on the approximate surface. The full source-depth guard is separate.
        changed = v[representatives[mapped[keep]]]
        normals = np.cross(changed[:, 1] - changed[:, 0], changed[:, 2] - changed[:, 0])
        reversed_faces = np.einsum('ij,ij->i', normals, changed[:, 0]) * original_side[keep] <= 0
        bad.extend(mapped[keep][reversed_faces].ravel())
        displacement = np.linalg.norm(v - v[representatives[inverse]], axis=1)
        bad.extend(inverse[displacement > MAX_DISPLACEMENT_MM / 1000])
        bad = np.unique(bad).astype(np.int64)
        if not len(bad):break
        newly_locked = np.isin(inverse, bad)
        if not np.any(newly_locked & ~locked):
            raise ValueError('原始表面包含无法保持的拓扑，请检查模型；未生成简化表面')
        locked |= newly_locked
    else:raise ValueError('规划分区细化超过预算；原始模型与选区保留')
    check()
    result = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v[representatives]),
                                     o3d.utility.Vector3iVector(candidate))
    if mesh.has_vertex_colors():result.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors)[representatives])
    result.compute_vertex_normals()
    record.update(status='simplified', output_vertices=len(result.vertices), output_triangles=len(result.triangles),
                  max_vertex_displacement_mm=float(displacement.max(initial=0.) * 1000),
                  refinement_passes=iteration + 1, protected_boundary_vertices=len(protected))
    check()
    return result, record
