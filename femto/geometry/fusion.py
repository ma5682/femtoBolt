"""Finite-burst RGB-D fusion and export helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import warnings
from typing import Sequence

import cv2
import numpy as np
import open3d as o3d
import trimesh

from femto.geometry.pose_geometry import Intrinsics, depth_to_points
from femto.io_utils import write_json


NATIVE_IO_CACHE = Path(os.environ["LOCALAPPDATA"]) / "FemtoPatientPose" / "native_io"
NATIVE_IO_CACHE.mkdir(parents=True, exist_ok=True)


def _native_cache_path(final_path: Path) -> Path:
    token = hashlib.sha1(str(final_path).encode("utf-8")).hexdigest()[:12]
    return NATIVE_IO_CACHE / f"{token}_{final_path.name}"


def write_point_cloud_safe(path: Path, cloud: o3d.geometry.PointCloud) -> bool:
    """Use an ASCII-only temporary path for Open3D's Windows native writer."""
    temporary = _native_cache_path(path)
    temporary.unlink(missing_ok=True)
    ok = o3d.io.write_point_cloud(str(temporary), cloud, write_ascii=False, compressed=False)
    if ok and temporary.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temporary), str(path))
        return True
    return False


def read_geometry_safe(path: Path, *, mesh: bool):
    """Copy a Chinese-path geometry to ASCII cache before native Open3D read."""
    temporary = _native_cache_path(path)
    shutil.copy2(path, temporary)
    try:
        return (
            o3d.io.read_triangle_mesh(str(temporary))
            if mesh
            else o3d.io.read_point_cloud(str(temporary))
        )
    finally:
        temporary.unlink(missing_ok=True)


def temporal_median_depth(
    depths_mm: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    minimum_support: int,
    chunk_rows: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a per-pixel median without stacking every full frame at once."""
    if not depths_mm or len(depths_mm) != len(masks):
        raise ValueError("depth and mask frame lists must be non-empty and equal")
    shape = depths_mm[0].shape
    if any(item.shape != shape for item in depths_mm) or any(item.shape != shape for item in masks):
        raise ValueError("all depth and mask frames must have one pixel grid")
    height, width = shape
    fused = np.zeros(shape, dtype=np.float32)
    support = np.zeros(shape, dtype=np.uint16)
    fused_mask = np.zeros(shape, dtype=np.uint8)
    for y0 in range(0, height, chunk_rows):
        y1 = min(height, y0 + chunk_rows)
        stack = np.stack(
            [
                np.where(mask[y0:y1].astype(bool), depth[y0:y1], np.nan)
                for depth, mask in zip(depths_mm, masks)
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        valid = np.isfinite(stack) & (stack > 0)
        count = valid.sum(axis=0)
        support[y0:y1] = count.astype(np.uint16)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(np.where(valid, stack, np.nan), axis=0)
        accepted = count >= minimum_support
        fused[y0:y1][accepted] = median[accepted]
        fused_mask[y0:y1][accepted] = 255
    return fused, fused_mask, support


def median_joints(joint_frames_m: Sequence[np.ndarray], minimum_support: int) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack(joint_frames_m, axis=0).astype(np.float64)
    support = np.isfinite(stack).all(axis=2).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        points = np.nanmedian(stack, axis=0)
    points[support < minimum_support] = np.nan
    return points, support


def create_surface_cloud(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    color_bgr: np.ndarray,
    intr: Intrinsics,
    *,
    stride: int = 2,
    voxel_m: float = 0.003,
) -> o3d.geometry.PointCloud:
    points, pixels = depth_to_points(depth_mm, intr, mask=mask, stride=stride)
    cloud = o3d.geometry.PointCloud()
    if len(points) == 0:
        return cloud
    colors = color_bgr[pixels[:, 1], pixels[:, 0], ::-1].astype(np.float64) / 255.0
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if voxel_m > 0:
        cloud = cloud.voxel_down_sample(voxel_m)
    if len(cloud.points) >= 100:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    return cloud


def _scaled_rgbd(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    intr: Intrinsics,
    scale: float,
) -> tuple[o3d.geometry.RGBDImage, o3d.camera.PinholeCameraIntrinsic]:
    masked_depth = np.where(mask.astype(bool), depth_mm, 0.0)
    height, width = masked_depth.shape
    if scale != 1.0:
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        color_bgr = cv2.resize(color_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
        masked_depth = cv2.resize(
            masked_depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
    else:
        new_width, new_height = width, height
    color_rgb = np.ascontiguousarray(color_bgr[:, :, ::-1])
    depth_u16 = np.rint(np.clip(masked_depth, 0, 65535)).astype(np.uint16)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_rgb),
        o3d.geometry.Image(depth_u16),
        depth_scale=1000.0,
        depth_trunc=3.86,
        convert_rgb_to_intensity=False,
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        new_width,
        new_height,
        intr.fx * scale,
        intr.fy * scale,
        intr.cx * scale,
        intr.cy * scale,
    )
    return rgbd, intrinsic


def build_fixed_camera_tsdf(
    colors_bgr: Sequence[np.ndarray],
    depths_mm: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    intr: Intrinsics,
    *,
    voxel_length_m: float = 0.005,
    sdf_trunc_m: float = 0.02,
    image_scale: float = 0.5,
    minimum_component_triangles: int = 50,
) -> o3d.geometry.TriangleMesh:
    """Fuse a finite burst whose camera pose is fixed for every frame."""
    if not (len(colors_bgr) == len(depths_mm) == len(masks)) or not depths_mm:
        raise ValueError("TSDF input lists must be non-empty and equal")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length_m,
        sdf_trunc=sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    extrinsic_world_to_camera = np.eye(4, dtype=np.float64)
    for color, depth, mask in zip(colors_bgr, depths_mm, masks):
        rgbd, intrinsic = _scaled_rgbd(color, depth, mask, intr, image_scale)
        volume.integrate(rgbd, intrinsic, extrinsic_world_to_camera)
    mesh = volume.extract_triangle_mesh()
    if len(mesh.triangles):
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels_array = np.asarray(labels)
        counts_array = np.asarray(counts)
        if len(counts_array):
            keep_components = counts_array >= max(1, minimum_component_triangles)
            if not keep_components.any():
                keep_components[int(np.argmax(counts_array))] = True
            mesh.remove_triangles_by_mask(~keep_components[labels_array])
            mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
    return mesh


def fill_small_mesh_holes(
    mesh: o3d.geometry.TriangleMesh, maximum_radius_m: float
) -> o3d.geometry.TriangleMesh:
    """Fill bounded internal holes while leaving the large silhouette open."""
    if maximum_radius_m <= 0.0 or not len(mesh.triangles):
        return o3d.geometry.TriangleMesh(mesh)
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    filled = tensor_mesh.fill_holes(maximum_radius_m).to_legacy()
    filled.remove_degenerate_triangles()
    filled.remove_duplicated_triangles()
    filled.remove_duplicated_vertices()
    filled.remove_non_manifold_edges()
    filled.remove_unreferenced_vertices()

    # Trimesh additionally closes isolated triangle/quad micro-holes. It adds
    # faces only, so the vertex colours generated by TSDF stay aligned.
    vertices = np.asarray(filled.vertices)
    faces = np.asarray(filled.triangles)
    colors = np.asarray(filled.vertex_colors)
    repaired_tm = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy(), process=False)
    trimesh.repair.fill_holes(repaired_tm)
    trimesh.repair.fix_winding(repaired_tm)
    trimesh.repair.fix_normals(repaired_tm, multibody=True)
    repaired = o3d.geometry.TriangleMesh()
    repaired.vertices = o3d.utility.Vector3dVector(np.asarray(repaired_tm.vertices))
    repaired.triangles = o3d.utility.Vector3iVector(np.asarray(repaired_tm.faces))
    if colors.shape == vertices.shape:
        repaired.vertex_colors = o3d.utility.Vector3dVector(colors)
    repaired.remove_non_manifold_edges()
    repaired.remove_degenerate_triangles()
    repaired.remove_duplicated_triangles()
    repaired.remove_duplicated_vertices()
    repaired.remove_unreferenced_vertices()
    repaired.compute_vertex_normals()
    return repaired


def export_mesh_formats(
    mesh: o3d.geometry.TriangleMesh,
    output_dir: Path,
    *,
    stem: str = "patient_surface_tsdf_m",
    report_name: str = "mesh_report.json",
    note: str = "Top-view visible surface is normally open; no large-hole closure is applied.",
    write_obj_glb: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    ply = output_dir / f"{stem}.ply"
    obj = output_dir / f"{stem}.obj"
    glb = output_dir / f"{stem}.glb"
    cached_ply = _native_cache_path(ply)
    cached_obj = _native_cache_path(obj)
    cached_ply.unlink(missing_ok=True)
    cached_obj.unlink(missing_ok=True)
    if not o3d.io.write_triangle_mesh(str(cached_ply), mesh, write_ascii=False):
        raise RuntimeError("Open3D failed to write TSDF PLY")
    shutil.move(str(cached_ply), str(ply))
    if write_obj_glb:
        if not o3d.io.write_triangle_mesh(str(cached_obj), mesh, write_ascii=False):
            raise RuntimeError("Open3D failed to write TSDF OBJ")
        shutil.move(str(cached_obj), str(obj))
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    visual = None
    if colors.shape == vertices.shape:
        visual = np.column_stack(
            (np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8), np.full(len(colors), 255, np.uint8))
        )
    tri = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=visual, process=False)
    if write_obj_glb:
        tri.export(glb)
    edge_face_counts = np.bincount(tri.edges_unique_inverse)
    _, open3d_component_counts, _ = mesh.cluster_connected_triangles()
    component_faces = sorted(
        (int(value) for value in np.asarray(open3d_component_counts)), reverse=True
    )
    # Trimesh's watertight flag is useful here; the top-view patient surface is
    # expected to remain open at its outer silhouette.
    report = {
        "vertices": int(len(vertices)),
        "triangles": int(len(faces)),
        "watertight": bool(tri.is_watertight),
        "winding_consistent": bool(tri.is_winding_consistent),
        "boundary_edges": int((edge_face_counts == 1).sum()),
        "nonmanifold_edges": int((edge_face_counts > 2).sum()),
        "connected_components": len(component_faces),
        "largest_component_triangles": component_faces[0] if component_faces else 0,
        "note": note,
    }
    write_json(output_dir / report_name, report)
    return {"ply": str(ply), "obj": str(obj), "glb": str(glb), "report": report,
            "intermediate_obj_glb_written": bool(write_obj_glb)}
