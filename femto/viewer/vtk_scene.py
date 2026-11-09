"""Numeric display snapshots. World RDF metres are retained without conversion.

No Open3D, VTK or Qt dependency: source geometry remains owned by the model.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SceneSnapshot:
    vertices: np.ndarray
    triangles: np.ndarray
    colors: np.ndarray | None
    region_ids: np.ndarray
    route_runs: tuple
    source_triangle_ids: np.ndarray


def scene_snapshot(model, region=None, plan=None, *, selection_only=False):
    mesh = model.mesh
    vertices = np.array(mesh.vertices, dtype=np.float64, copy=True)
    raw_triangles = np.asarray(mesh.triangles)
    if raw_triangles.size and not np.issubdtype(raw_triangles.dtype, np.integer):
        raise ValueError('Display triangle indices must be integers')
    triangles = np.array(raw_triangles, dtype=np.int64, copy=True)
    if vertices.size == 0:
        vertices = vertices.reshape(0, 3)
    if triangles.size == 0:
        triangles = triangles.reshape(0, 3)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Display vertices must be finite N x 3 world coordinates')
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError('Display triangles must be N x 3')
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError('Display triangle index outside original mesh')
    colors = np.asarray(getattr(mesh, 'vertex_colors', []))
    colors = (np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8)
              if colors.shape == vertices.shape and np.isfinite(colors).all() else None)
    ids = np.asarray((region or {}).get('triangle_indices', []))
    if ids.size and (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)):
        raise ValueError('Region triangle IDs must be integers')
    ids = np.array(ids, dtype=np.int64, copy=True).reshape(-1)
    if ids.size and (ids.min() < 0 or ids.max() >= len(triangles)):
        raise ValueError('Region triangle index outside original mesh')
    source_ids = np.arange(len(triangles), dtype=np.int64)
    if selection_only:
        if region is None:
            vertices, triangles, colors, ids = np.empty((0,3)), np.empty((0,3),np.int64), None, np.empty(0,np.int64)
            source_ids = ids.copy()
            plan = None
        else:
            from femto.viewer.selected_surface import selected_surface
            vertices, triangles, colors, source_ids = selected_surface(vertices, triangles, colors, ids, region)
            ids = np.arange(len(triangles), dtype=np.int64)
    runs = []
    if plan is not None:
        if plan.get('unit', 'metre') != 'metre' or plan.get('coordinate_frame', 'world_rdf') != 'world_rdf':
            raise ValueError('Display route requires world RDF metres')
        for chain in plan.get('segments', []):
            # Each chain remains independent; missing coordinates terminate a run.
            run = []
            for point in chain:
                xyz = np.asarray(point.get('xyz_m', []), dtype=np.float64)
                if xyz.shape != (3,) or not np.isfinite(xyz).all():
                    if len(run) >= 2:
                        runs.append(np.array(run))
                    run = []
                else:
                    run.append(xyz.copy())
            if len(run) >= 2:
                runs.append(np.array(run))
    for value in (vertices, triangles, colors, ids, source_ids, *runs):
        if value is not None:
            value.setflags(write=False)
    return SceneSnapshot(vertices, triangles, colors, ids, tuple(runs), source_ids)
