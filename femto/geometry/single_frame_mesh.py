"""Pixel-adjacent, calibrated single-frame surface with original RGB texture."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import trimesh

from femto.geometry.fusion import _native_cache_path
from femto.io_utils import write_json


def build_pixel_mesh(depth_mm, color_bgr, intrinsic, distortion, T_WC):
    """Connect only measured neighboring pixels; no resampling or hole filling.

    Each triangle must satisfy all three depth edges. The diagonal with the
    smaller depth jump is chosen per quad; each triangle is then independently
    validated, retaining the valid half of a quad bordering an unknown pixel.
    """
    height, width = depth_mm.shape
    if color_bgr.shape != (height, width, 3) or (height, width) != (intrinsic.height, intrinsic.width):
        raise ValueError("Aligned RGB-D dimensions must match color calibration")
    if "BROWN_CONRADY_K6" not in str(distortion.get("model", "")):
        raise ValueError("Pixel mesh requires RGB BROWN_CONRADY_K6 calibration")
    valid = np.isfinite(depth_mm) & (depth_mm >= 250) & (depth_mm <= 2210)
    rows, cols = np.nonzero(valid)
    if not len(rows):
        raise ValueError("No measured pixels in 0.25–2.21 m")
    matrix = np.array([[intrinsic.fx, 0, intrinsic.cx], [0, intrinsic.fy, intrinsic.cy], [0, 0, 1]], float)
    coeff = np.array([distortion[k] for k in ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')])
    rays = cv2.undistortPoints(np.column_stack((cols, rows)).astype(float).reshape(-1, 1, 2), matrix, coeff).reshape(-1, 2)
    z = depth_mm[rows, cols].astype(float) / 1000
    xyz = np.column_stack((rays * z[:, None], z))
    finite = np.isfinite(xyz).all(axis=1)
    xyz, rows, cols, z = xyz[finite], rows[finite], cols[finite], z[finite]
    if not len(xyz):
        raise ValueError("No finite calibrated points")
    ids = np.full((height, width), -1, np.int32)
    ids[rows, cols] = np.arange(len(rows), dtype=np.int32)
    a, b, c, d = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel(), ids[1:, :-1].ravel(), ids[1:, 1:].ravel()
    def diagonal_cost(i, j):
        return np.where((i >= 0) & (j >= 0), np.abs(z[i] - z[j]), np.inf)
    use_ad = diagonal_cost(a, d) < diagonal_cost(b, c)
    triangles = np.concatenate((
        np.column_stack((a[~use_ad], c[~use_ad], b[~use_ad])),
        np.column_stack((b[~use_ad], c[~use_ad], d[~use_ad])),
        np.column_stack((a[use_ad], d[use_ad], b[use_ad])),
        np.column_stack((a[use_ad], c[use_ad], d[use_ad])),
    ))
    candidates = len(triangles)
    triangles = triangles[np.all(triangles >= 0, axis=1)]
    measured_candidates = len(triangles)
    keep = np.ones(len(triangles), bool)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        i, j = triangles[:, first], triangles[:, second]
        # 10 mm floor / 2% relative gate is a continuity heuristic, not sensor accuracy.
        limit = np.maximum(.010, .02 * np.minimum(z[i], z[j]))
        keep &= np.abs(z[i] - z[j]) <= limit
        keep &= np.linalg.norm(xyz[i] - xyz[j], axis=1) <= np.sqrt(2) * limit
    triangles = triangles[keep]
    if not len(triangles):
        raise ValueError("No pixel-adjacent triangle passed measured-depth continuity")
    used, inverse = np.unique(triangles, return_inverse=True)
    triangles = inverse.reshape(-1, 3).astype(np.int32)
    vertices = xyz[used] @ T_WC[:3, :3].T + T_WC[:3, 3]
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    mesh.vertex_colors = o3d.utility.Vector3dVector(color_bgr[rows[used], cols[used], ::-1] / 255.)
    mesh.compute_vertex_normals()
    uv = np.column_stack(((cols[used] + .5) / width, 1 - (rows[used] + .5) / height))
    report = {
        "method": "single_frame_pixel_adjacency", "frame_count": 1,
        "unit": "metre", "coordinate_system": "World RDF; T_WC maps camera to world",
        "depth_range_mm_inclusive": [250, 2210],
        "depth_edge_limit_m": "max(0.010, 0.02 * min(endpoint_depths_m))",
        "spatial_edge_limit_m": "sqrt(2) * depth_edge_limit_m",
        "candidate_triangles": candidates,
        "rejected_missing_depth_triangles": candidates - measured_candidates,
        "rejected_discontinuity_triangles": measured_candidates - len(triangles),
        "vertices": len(vertices), "triangles": len(triangles),
        "retained_points_without_triangle": len(xyz) - len(used),
        "smoothing": False, "hole_filling": False, "temporal_merge": False,
        "uv_convention": "pixel centers: u=(column+0.5)/width, v=1-(row+0.5)/height",
        "limitations": ["Depth continuity is heuristic; noise and small depth steps can remain.",
                        "RGB sharpness is texture detail, not added depth resolution or verified accuracy."]}
    return mesh, uv, report


def export_pixel_mesh(mesh, uv, source_rgb: Path, output: Path, report: dict):
    """PLY carries vertex colors; OBJ+MTL and GLB carry the original RGB texture."""
    stem = 'scene_surface_mesh_m'
    ply, obj, glb = (output / (stem + ext) for ext in ('.ply', '.obj', '.glb'))
    texture = output / ('texture_rgb' + source_rgb.suffix.lower())
    shutil.copy2(source_rgb, texture)
    uv_path = output / 'mesh_vertex_uv.npy'
    np.save(uv_path, uv)
    cached = _native_cache_path(ply)
    try:
        if not o3d.io.write_triangle_mesh(str(cached), mesh, write_ascii=False):
            raise RuntimeError("Failed to write pixel mesh PLY")
        shutil.move(str(cached), ply)
    finally:
        cached.unlink(missing_ok=True)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    mtl = output / (stem + '.mtl')
    mtl.write_text('newmtl original_rgb\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd ' + texture.name + '\n', encoding='ascii')
    with obj.open('w', encoding='ascii', newline='\n') as handle:
        handle.write(f'# metres; World RDF\nmtllib {mtl.name}\n')
        np.savetxt(handle, vertices, fmt='v %.9f %.9f %.9f')
        np.savetxt(handle, uv, fmt='vt %.9f %.9f')
        handle.write('usemtl original_rgb\n')
        for start in range(0, len(faces), 100000):
            indices = np.repeat(faces[start:start + 100000].astype(np.int64) + 1, 2, axis=1)
            np.savetxt(handle, indices, fmt='f %d/%d %d/%d %d/%d')
    with Image.open(texture) as image:
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=image.copy(),
                                                       metallicFactor=0, roughnessFactor=1, doubleSided=True)
    visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False).export(glb)
    report = {**report, 'texture_source': str(source_rgb.resolve()),
              'texture_file': texture.name, 'ply_appearance': 'vertex_color_fallback',
              'obj_appearance': 'original_rgb_texture; distribute OBJ, MTL and texture together',
              'glb_appearance': 'embedded_rgb_texture'}
    write_json(output / 'mesh_report.json', report)
    return {"ply": str(ply), "obj": str(obj), "glb": str(glb), "mtl": str(mtl),
            "texture": str(texture), "vertex_uv": str(uv_path), "report": report}
