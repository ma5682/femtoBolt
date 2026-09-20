"""Validated multi-view RGB-D point-cloud and TSDF reconstruction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any
from types import SimpleNamespace

import cv2
import numpy as np
import open3d as o3d

from femto.geometry.fusion import export_mesh_formats, write_point_cloud_safe
from femto.io_utils import imread_unicode as _read_image, imwrite_unicode, load_json, load_jsonl
from femto.dataset.scene_dataset_validator import resolve_dataset
from femto.geometry.scene_gaussian import export_gaussian_initialization
from femto.geometry.single_frame_quality import clean_depth, compare_geometry, evidence_manifest
from femto.geometry.single_frame_mesh import build_pixel_mesh, export_pixel_mesh
from femto.geometry.multiview_texture import export_multiview_texture
from femto.depth.depth_quality import process_depth_quality


from femto.paths import REPO_ROOT as ROOT
from femto.geometry.rgbd_frames import distortion_maps as _distortion_maps, load_poses as _load_poses, prepare_rgbd as _prepare_rgbd
DEFAULT_DATASETS = ROOT / "datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse a validated moving-camera Femto RGB-D dataset")
    parser.add_argument("dataset", nargs="?", type=Path)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-scale", type=float, default=1.0)
    parser.add_argument("--voxel-m", type=float, default=0.005)
    parser.add_argument("--sdf-trunc-m", type=float, default=0.02)
    parser.add_argument("--depth-min-m", type=float, default=0.5)
    parser.add_argument("--depth-max-m", type=float, default=3.86)
    parser.add_argument("--point-stride", type=int, default=1)
    return parser.parse_args()


def _camera_trajectory(poses: list[np.ndarray]) -> o3d.geometry.LineSet:
    centers = np.asarray([pose[:3, 3] for pose in poses], dtype=np.float64)
    lines = np.column_stack((np.arange(len(centers) - 1), np.arange(1, len(centers)))) if len(centers) > 1 else np.empty((0, 2), int)
    trajectory = o3d.geometry.LineSet()
    trajectory.points = o3d.utility.Vector3dVector(centers)
    trajectory.lines = o3d.utility.Vector2iVector(lines.astype(np.int32))
    trajectory.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.25, 0.08], (len(lines), 1)))
    return trajectory


def _write_line_set_ascii(path: Path, trajectory: o3d.geometry.LineSet) -> None:
    """Write a tiny line-set PLY without invoking Unicode-sensitive native I/O."""
    points = np.asarray(trajectory.points, dtype=np.float64)
    lines = np.asarray(trajectory.lines, dtype=np.int64)
    colors = np.asarray(trajectory.colors, dtype=np.float64)
    with path.open("w", encoding="ascii", newline="\n") as output:
        output.write("ply\nformat ascii 1.0\n")
        output.write(f"element vertex {len(points)}\n")
        output.write("property float x\nproperty float y\nproperty float z\n")
        output.write(f"element edge {len(lines)}\n")
        output.write("property int vertex1\nproperty int vertex2\n")
        output.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point in points:
            output.write(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")
        for index, line in enumerate(lines):
            color = colors[index] if colors.shape == (len(lines), 3) else np.array([1.0, 0.25, 0.08])
            rgb = np.rint(np.clip(color, 0.0, 1.0) * 255.0).astype(np.uint8)
            output.write(f"{line[0]} {line[1]} {rgb[0]} {rgb[1]} {rgb[2]}\n")


def _cloud_mesh_consistency(cloud,mesh):
    """Internal layer-consistency diagnostics; never a physical-accuracy claim."""
    points=np.asarray(cloud.points);normals=np.asarray(cloud.normals)
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distances=scene.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()
    tree=o3d.geometry.KDTreeFlann(cloud);second=0
    for index,point in enumerate(points):
        count,ids,_=tree.search_radius_vector_3d(point,.01)
        if count>1 and np.any(np.abs((points[np.asarray(ids)]-point)@normals[index])>.004):second+=1
    return dict(fused_to_mesh_median_mm=float(np.median(distances)*1000),
        fused_to_mesh_p90_mm=float(np.percentile(distances,90)*1000),
        fused_to_mesh_p95_mm=float(np.percentile(distances,95)*1000),
        points_without_mesh_support_over_50mm=int((distances>.05).sum()),
        second_sheet_fraction=float(second/max(1,len(points))),
        interpretation='Internal consistency of multi-view observation union versus supported TSDF mesh; not physical error.')


def _prepare_single_frame_evidence(dataset,output,frame,T_WC,color,depth,filled_depth,
                                   depth_confidence,temporal_input):
    from femto.geometry.depth_cloud import depth_cloud

    evidence_manifest(dataset, frame, output / "source_evidence.json")
    intr = SimpleNamespace(**load_json(dataset / "calibration" / "color_intrinsic.json"))
    distortion = load_json(dataset / "calibration" / "distortion.json")["rgb"]
    raw_evidence_depth=_read_image(dataset/frame['reference_raw_depth'],cv2.IMREAD_UNCHANGED) if temporal_input else depth
    raw_cloud = depth_cloud(raw_evidence_depth, color, intr, distortion, 250, 2210)
    raw_cloud.transform(T_WC)
    depth, cleaning_report = clean_depth(depth)
    cleaned_cloud = depth_cloud(depth, color, intr, distortion, 250, 2210)
    cleaned_cloud.transform(T_WC)
    if not len(cleaned_cloud.points):
        raise ValueError("No valid depth survived single-frame cleanup")
    imwrite_unicode(output / "cleaned_depth_mm.png", depth)
    raw_cloud_path = output / "scene_raw_points_m.ply"
    if not write_point_cloud_safe(raw_cloud_path, raw_cloud):
        raise RuntimeError("Failed to save uncleaned single-frame point cloud")
    from femto.geometry.depth_confidence_cloud import build_confident_depth_cloud, write_confident_point_cloud
    if temporal_input:
        repaired_cloud,point_confidence=depth_cloud(filled_depth,color,intr,distortion,250,2210,
                                        confidence=depth_confidence,temporal_confidence=True)
    else:
        repaired_cloud, point_confidence = build_confident_depth_cloud(
            _read_image(dataset / frame['depth'], cv2.IMREAD_UNCHANGED),
            filled_depth, depth_confidence, color, intr, distortion, 250, 2210)
    repaired_cloud.transform(T_WC)
    repaired_cloud_path = output / "derived" / "scene_hole_filled_points_m.ply"
    repaired_cloud_path.parent.mkdir(parents=True, exist_ok=True)
    write_confident_point_cloud(repaired_cloud_path, repaired_cloud, point_confidence,temporal_confidence=temporal_input)
    mesh, vertex_uv, pixel_mesh_report = build_pixel_mesh(depth, color, intr, distortion, T_WC)
    return (depth,cleaning_report,cleaned_cloud,intr,distortion,raw_cloud,raw_cloud_path,
            repaired_cloud_path,mesh,vertex_uv,pixel_mesh_report)


def _frame_metrics(frame,T_WC,depth,filtered_depth,camera_point_count,accumulated,
                   accumulated_before,voxel_m,single_frame_quality,per_frame_metrics):
    accumulated_before_downsample=None;accumulated_downsampled=False
    # Bound RAM for long 1080p captures while keeping raw RGB-D immutable.
    if not single_frame_quality and len(accumulated.points) > 1_000_000:
        accumulated_before_downsample=int(len(accumulated.points));accumulated_downsampled=True
        accumulated = accumulated.voxel_down_sample(voxel_m)
    edge=np.zeros(filtered_depth.shape,bool)
    valid_h=(filtered_depth[:,1:]>0)&(filtered_depth[:,:-1]>0)
    valid_v=(filtered_depth[1:]>0)&(filtered_depth[:-1]>0)
    horizontal_jump=valid_h&(np.abs(filtered_depth[:,1:].astype(np.int32)-filtered_depth[:,:-1].astype(np.int32))>30)
    vertical_jump=valid_v&(np.abs(filtered_depth[1:].astype(np.int32)-filtered_depth[:-1].astype(np.int32))>30)
    edge[:,1:] |= horizontal_jump
    edge[:,:-1] |= horizontal_jump
    edge[1:] |= vertical_jump
    edge[:-1] |= vertical_jump
    per_frame_metrics.append({
        "frame": int(frame["frame"]),
        "valid_depth_ratio": float(np.count_nonzero(filtered_depth) / filtered_depth.size),
        "camera_center_m": T_WC[:3, 3].tolist(),
        "remapped_valid_minus_raw_valid": int(np.count_nonzero(filtered_depth)-np.count_nonzero(depth)),
        "depth_jump_edge_pixels_over_30mm": int(edge.sum()),
        "camera_point_count": camera_point_count,
        "accumulated_point_count_before_add": accumulated_before,
        "accumulated_point_count_before_downsample": accumulated_before_downsample,
        "accumulated_voxel_downsample_applied": accumulated_downsampled,
        "accumulated_point_count_after_frame": int(len(accumulated.points)) if not single_frame_quality else None,
    })
    return accumulated


def _integrate_frame(dataset,output,frame,T_WC,color,depth,filled_depth,depth_confidence,
                     map_x,map_y,K,image_scale,depth_min_m,depth_max_m,local_measured,
                     local_prepared_depths,integrations_per_frame,volume,point_stride,
                     ordinal,accumulated,initial_display_subset):
    integration_depth=depth
    if local_measured:
        from femto.local.local_temporal_support import filtered_depth
        integration_depth=filtered_depth(dataset,frame,depth)
    rgbd, intrinsic, _, filtered_depth = _prepare_rgbd(
        color, integration_depth, map_x, map_y, K, image_scale, depth_min_m, depth_max_m
    )
    if local_prepared_depths is not None:
        local_prepared_depths.append(filtered_depth)
    repaired_only = np.where(
        depth_confidence == np.float32(0.3), filled_depth, 0).astype(np.uint16)
    # Legacy Open3D accepts unit observations only. Integer 10:3
    # repetitions implement the exact relative 1.0:0.3 confidence;
    # sampling 30% would still give each accepted repair full weight.
    repetitions=10 if np.any(repaired_only) else 1
    integrations_per_frame.append(repetitions)
    for _ in range(repetitions):
        volume.integrate(rgbd, intrinsic, np.linalg.inv(T_WC))
    camera_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic, project_valid_depth_only=True
    )
    if point_stride > 1 and len(camera_cloud.points):
        camera_cloud = camera_cloud.select_by_index(list(range(0, len(camera_cloud.points), point_stride)))
    camera_cloud.transform(T_WC)
    camera_point_count=int(len(camera_cloud.points))
    if local_measured and ordinal == 0:
        initial_cloud=camera_cloud
        if frame.get('local_temporal_support'):
            initial_rgbd,initial_intrinsic,_,_=_prepare_rgbd(color,depth,map_x,map_y,K,image_scale,depth_min_m,depth_max_m)
            initial_cloud=o3d.geometry.PointCloud.create_from_rgbd_image(initial_rgbd,initial_intrinsic).transform(T_WC)
        # A row-major valid-pixel cloud has gaps where depth is invalid.  A
        # linear stride of 16 is the bounded display equivalent of an
        # approximate 4x4 image stride and is never used for measurement.
        initial_display_subset=initial_cloud.select_by_index(list(range(0,len(initial_cloud.points),16)))
        if not write_point_cloud_safe(output / 'initial_observed_points_m.ply', initial_display_subset):
            raise RuntimeError('Initial observed cloud write failed')
    accumulated_before=int(len(accumulated.points));accumulated += camera_cloud
    if not local_measured and np.any(repaired_only):
        repaired_rgbd, repaired_intrinsic, _, _ = _prepare_rgbd(
            color, repaired_only, map_x, map_y, K, image_scale, depth_min_m, depth_max_m
        )
        for _ in range(3):
            volume.integrate(repaired_rgbd, repaired_intrinsic, np.linalg.inv(T_WC))
    return filtered_depth,accumulated,initial_display_subset,camera_point_count,accumulated_before


def reconstruct_scene(
    dataset: Path,
    output: Path,
    *,
    image_scale: float = 1.0,
    voxel_m: float = 0.005,
    sdf_trunc_m: float = 0.02,
    depth_min_m: float = 0.5,
    depth_max_m: float = 3.86,
    point_stride: int = 1,
    single_frame_quality: bool = False,
    local_post_component_filter: bool = True,
) -> dict[str, Any]:
    dataset = dataset.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    if not 0.1 <= image_scale <= 1.0:
        raise ValueError("image_scale must be in [0.1, 1.0]")
    if voxel_m <= 0 or sdf_trunc_m <= voxel_m or point_stride < 1:
        raise ValueError("Invalid voxel, SDF truncation or point stride")
    manifest = load_json(dataset / "manifest.json")
    frames = load_jsonl(dataset / "frames.jsonl")
    if manifest.get("schema") != "femto-full-scene-rgbd-v1" or not frames:
        raise ValueError("Input is not a non-empty femto-full-scene-rgbd-v1 dataset")
    poses = _load_poses(dataset, frames)
    if single_frame_quality:
        if len(frames) != 1 or not np.allclose([depth_min_m, depth_max_m], [.25, 2.21]):
            raise ValueError("Single-frame workflow requires exactly one frame and 0.25–2.21 m")
        if image_scale != 1.0 or point_stride != 1:
            raise ValueError("Single-frame workflow requires full image scale and point stride 1")
    map_x, map_y, K, expected_width, expected_height = _distortion_maps(dataset)

    local_measured = bool(manifest.get('local_capture', {}).get('measured_only'))
    temporal_input=bool(manifest.get('temporal_fusion'))
    if temporal_input and not single_frame_quality:
        raise ValueError('Phase 1 prepares temporal TSDF inputs only; continuous-weight integrate is deferred to phase 2. Legacy non-temporal TSDF remains available.')
    output.mkdir(parents=True, exist_ok=False)
    volume = None if single_frame_quality else o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_m,
        sdf_trunc=sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    accumulated = o3d.geometry.PointCloud()
    per_frame_metrics: list[dict[str, Any]] = []
    depth_quality_frames: list[dict[str, Any]] = []
    local_prepared_depths = [] if local_measured and image_scale == 1.0 else None
    integrations_per_frame = []
    initial_display_subset = None
    remap_x,remap_y=np.meshgrid(np.arange(expected_width),np.arange(expected_height))
    remap_shift=np.hypot(map_x-remap_x,map_y-remap_y)
    remap_valid=np.isfinite(map_x)&np.isfinite(map_y)&(map_x>=0)&(map_x<expected_width)&(map_y>=0)&(map_y<expected_height)
    source_pixels=np.rint(np.c_[map_x[remap_valid],map_y[remap_valid]]).astype(np.int64)
    unique_source=len(np.unique(source_pixels[:,1]*expected_width+source_pixels[:,0])) if len(source_pixels) else 0
    distortion_diagnostics=dict(new_K=K.tolist(),maximum_displacement_px=float(np.max(remap_shift[remap_valid])),
        median_displacement_px=float(np.median(remap_shift[remap_valid])),
        nearest_neighbour_duplicate_pixels=int(remap_valid.sum()-unique_source),
        nearest_neighbour_unsampled_source_pixels=int(expected_width*expected_height-unique_source),
        nearest_neighbour_unmapped_target_pixels=int(expected_width*expected_height-remap_valid.sum()),
        nearest_neighbour_unsampled_due_to_duplicate_pixels=int(remap_valid.sum()-unique_source),
        nearest_neighbour_affected_fraction=float((expected_width*expected_height-unique_source)/(expected_width*expected_height)))
    for ordinal, (frame, T_WC) in enumerate(zip(frames, poses)):
        camera_point_count=None;accumulated_before=None;accumulated_before_downsample=None;accumulated_downsampled=False
        color = _read_image(dataset / frame["rgb"], cv2.IMREAD_COLOR)
        depth = _read_image(dataset / frame["depth"], cv2.IMREAD_UNCHANGED)
        if color.shape[:2] != (expected_height, expected_width) or depth.shape != (expected_height, expected_width):
            raise ValueError(f"Frame {frame['frame']} dimensions do not match calibration")
        if depth.dtype != np.uint16:
            raise ValueError(f"Frame {frame['frame']} depth is not uint16 millimetres")
        from femto.depth.temporal_depth_layer import prepared_depth
        filled_depth, depth_confidence, depth_quality, depth_quality_paths = prepared_depth(
            dataset,frame,depth,
            depth_min_m * 1000.0,
            depth_max_m * 1000.0,
            output / "derived" / "depth_quality" / f"frame_{ordinal:06d}",
            rgb_bgr=color,
            keep_artifacts=not local_measured,
        )
        if local_measured:
            # Preserve the quality artifacts, but never integrate inferred repairs.
            filled_depth = depth.copy()
            depth_confidence = np.where(depth > 0, 1.0, 0.0).astype(np.float32)
        depth_quality_frames.append({
            "frame": int(frame["frame"]),
            "original_invalid_rate": depth_quality["original_invalid_rate"],
            "filled_invalid_rate": depth_quality["filled_invalid_rate"],
            "repaired_pixels": depth_quality["repaired_pixels"],
            "policy": depth_quality["policy"],
            "artifacts": depth_quality_paths,
        })
        if temporal_input:
            # Explicit float-weight TSDF inputs; the default single-view mesh
            # retains the existing pixel-adjacency reconstruction behavior.
            input_rgbd,input_intrinsic,_,input_depth=_prepare_rgbd(
                color,filled_depth,map_x,map_y,K,image_scale,depth_min_m,depth_max_m)
            input_conf=cv2.remap(depth_confidence,map_x,map_y,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
            if input_conf.shape!=input_depth.shape:
                input_conf=cv2.resize(input_conf,(input_depth.shape[1],input_depth.shape[0]),interpolation=cv2.INTER_NEAREST)
            input_conf=np.where(input_depth>0,input_conf,0).astype(np.float32)
            input_dir=output/'derived'/'tsdf_input'; input_dir.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(input_dir/f'{ordinal:06d}.npz',depth_m=np.asarray(input_rgbd.depth),
                                confidence=input_conf,rgb=np.asarray(input_rgbd.color),
                                intrinsic=input_intrinsic.intrinsic_matrix,T_CW=np.linalg.inv(T_WC))
        if single_frame_quality:
            (depth,cleaning_report,cleaned_cloud,intr,distortion,raw_cloud,raw_cloud_path,
             repaired_cloud_path,mesh,vertex_uv,pixel_mesh_report) = _prepare_single_frame_evidence(
                dataset,output,frame,T_WC,color,depth,filled_depth,depth_confidence,temporal_input)
            filtered_depth = depth
        else:
            (filtered_depth,accumulated,initial_display_subset,camera_point_count,
             accumulated_before) = _integrate_frame(dataset,output,frame,T_WC,color,depth,
                filled_depth,depth_confidence,map_x,map_y,K,image_scale,depth_min_m,depth_max_m,
                local_measured,local_prepared_depths,integrations_per_frame,volume,point_stride,
                ordinal,accumulated,initial_display_subset)
        accumulated = _frame_metrics(frame,T_WC,depth,filtered_depth,camera_point_count,
            accumulated,accumulated_before,voxel_m,single_frame_quality,per_frame_metrics)

    geometry = _finalize_geometry(SimpleNamespace(
        dataset=dataset,output=output,frames=frames,poses=poses,voxel_m=voxel_m,
        single_frame_quality=single_frame_quality,accumulated=accumulated,
        cleaned_cloud=cleaned_cloud if single_frame_quality else None,temporal_input=temporal_input,
        depth=depth,color=color,intr=intr if single_frame_quality else None,
        distortion=distortion if single_frame_quality else None,depth_confidence=depth_confidence,
        volume=volume,mesh=mesh if single_frame_quality else None,local_measured=local_measured,
        map_x=map_x,map_y=map_y,K=K,expected_width=expected_width,
        expected_height=expected_height,depth_min_m=depth_min_m,depth_max_m=depth_max_m,
        local_prepared_depths=local_prepared_depths,
        local_post_component_filter=local_post_component_filter,manifest=manifest,
        initial_display_subset=initial_display_subset,vertex_uv=vertex_uv if single_frame_quality else None,
        pixel_mesh_report=pixel_mesh_report if single_frame_quality else None,
        raw_cloud=raw_cloud if single_frame_quality else None,
        cleaning_report=cleaning_report if single_frame_quality else None))
    cloud,mesh,cloud_path=geometry.cloud,geometry.mesh,geometry.cloud_path
    local_support,mesh_exports=geometry.local_support,geometry.mesh_exports
    quality_path,trajectory_path,gaussian=geometry.quality_path,geometry.trajectory_path,geometry.gaussian
    return _write_reconstruction_report(SimpleNamespace(
        dataset=dataset,output=output,frames=frames,image_scale=image_scale,voxel_m=voxel_m,
        sdf_trunc_m=sdf_trunc_m,depth_min_m=depth_min_m,depth_max_m=depth_max_m,
        point_stride=point_stride,single_frame_quality=single_frame_quality,
        distortion_diagnostics=distortion_diagnostics,cloud_path=cloud_path,
        mesh_exports=mesh_exports,trajectory_path=trajectory_path,gaussian=gaussian,
        cloud=cloud,mesh=mesh,per_frame_metrics=per_frame_metrics,
        depth_quality_frames=depth_quality_frames,integrations_per_frame=integrations_per_frame,
        local_measured=local_measured,manifest=manifest,local_support=local_support,
        initial_display_subset=initial_display_subset,temporal_input=temporal_input,
        raw_cloud_path=raw_cloud_path if single_frame_quality else None,
        quality_path=quality_path,repaired_cloud_path=repaired_cloud_path if single_frame_quality else None,
        cleaning_report=cleaning_report if single_frame_quality else None,
        raw_cloud=raw_cloud if single_frame_quality else None))


def _finalize_geometry(s):
    cloud = s.cleaned_cloud if s.single_frame_quality else s.accumulated.voxel_down_sample(s.voxel_m)
    if not s.single_frame_quality and len(cloud.points) >= 100:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    if not len(cloud.points):
        shutil.rmtree(s.output)
        raise ValueError("No valid point survived reconstruction")
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=s.voxel_m * 4.0, max_nn=30))
    cloud.orient_normals_towards_camera_location(s.poses[0][:3, 3])
    cloud_path = s.output / "scene_fused_points_m.ply"
    if s.temporal_input and s.single_frame_quality:
        from femto.geometry.depth_cloud import depth_cloud
        from femto.geometry.depth_confidence_cloud import write_confident_point_cloud
        _,clean_conf=depth_cloud(s.depth,s.color,s.intr,s.distortion,250,2210,
                                confidence=np.where(s.depth>0,s.depth_confidence,0),temporal_confidence=True)
        write_confident_point_cloud(cloud_path,cloud,clean_conf,temporal_confidence=True)
    elif not write_point_cloud_safe(cloud_path, cloud):
        shutil.rmtree(s.output)
        raise RuntimeError("Open3D failed to write the fused point cloud")
    mesh = s.mesh
    if not s.single_frame_quality:
        mesh = s.volume.extract_triangle_mesh()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        if len(mesh.triangles):
            labels, counts, _ = mesh.cluster_connected_triangles()
            counts_array = np.asarray(counts)
            labels_array = np.asarray(labels)
            if len(counts_array):
                keep = counts_array >= max(30, int(counts_array.max() * 0.002))
                mesh.remove_triangles_by_mask(~keep[labels_array])
                mesh.remove_unreferenced_vertices()
            mesh.compute_vertex_normals()
    if not len(mesh.triangles):
        shutil.rmtree(s.output)
        raise ValueError("TSDF extraction produced an empty mesh")
    local_support = None
    if s.local_measured:
        from femto.local.local_geometry import retain_observed_triangles
        local_support = retain_observed_triangles(mesh, s.dataset, s.frames, s.poses,
            (s.map_x, s.map_y, s.K, s.expected_width, s.expected_height), s.depth_min_m, s.depth_max_m,
            prepared_depths=s.local_prepared_depths)
        if s.local_post_component_filter:
            from femto.local.local_geometry import filter_small_components
            local_support['post_retain_component_filter']=filter_small_components(mesh)
            local_support['post_retain_component_filter']['applied']=True
        else:
            local_support['post_retain_component_filter']=dict(applied=False,
                reason='candidate_requires_same-source_comparison_before_default_adoption')
        if not len(mesh.triangles):
            raise ValueError('No measured-supported annotation surface remains')
        (s.output / 'local_geometry.json').write_text(json.dumps(dict(
            **s.manifest['local_capture'], support=local_support, coordinate_frame='world_rdf',
            unit='metre', transform_direction='T_W0_Ct', inferred_repairs_integrated=False,
            display_artifacts=dict(initial_observed_point_cloud='initial_observed_points_m.ply',
                role='display_subset_not_measurement',linear_point_stride=16,
                approximate_pixel_stride=4,point_count=int(len(s.initial_display_subset.points)))),
            ensure_ascii=False, indent=2), encoding='utf-8')
    surface_note = (
        "Single-view measured TSDF surface; small observed gaps may remain and unobserved regions are not invented."
        if len(s.poses) == 1
        else "Multi-view measured TSDF surface; unobserved large regions are intentionally not invented."
    )
    if s.single_frame_quality:
        mesh_exports = export_pixel_mesh(mesh, s.vertex_uv, s.dataset / s.frames[0]["rgb"], s.output, s.pixel_mesh_report)
    else:
        mesh_exports = export_mesh_formats(
            mesh, s.output, stem="scene_tsdf_mesh_m", report_name="mesh_report.json", note=surface_note,
            write_obj_glb=len(s.frames)<=1,
        )
        if len(s.frames) > 1:
            mesh_exports = export_multiview_texture(mesh, s.dataset, s.frames, s.poses, s.output, mesh_exports)
    quality_path = s.output / "single_frame_quality.json"
    if s.single_frame_quality:
        compare_geometry(s.raw_cloud, cloud, mesh, s.cleaning_report, quality_path)
    trajectory = _camera_trajectory(s.poses)
    trajectory_path = s.output / "camera_trajectory.ply"
    _write_line_set_ascii(trajectory_path, trajectory)
    gaussian = export_gaussian_initialization(
        cloud, s.output / "gaussian", voxel_m=max(s.voxel_m * 1.5, 0.008), source_dataset=str(s.dataset)
    )
    return SimpleNamespace(cloud=cloud,mesh=mesh,cloud_path=cloud_path,local_support=local_support,
        mesh_exports=mesh_exports,quality_path=quality_path,trajectory_path=trajectory_path,gaussian=gaussian)


def _write_reconstruction_report(s):
    dataset,output,frames=s.dataset,s.output,s.frames
    single_frame_quality,local_measured,temporal_input=s.single_frame_quality,s.local_measured,s.temporal_input
    cloud,mesh,manifest=s.cloud,s.mesh,s.manifest
    report = {
        "schema": "femto-scene-reconstruction-v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": "complete_geometry_gaussian_init_untrained",
        "source_dataset": str(dataset),
        "frame_count": len(frames),
        "parameters": {
            "image_scale": s.image_scale,
            "voxel_m": None if single_frame_quality else s.voxel_m,
            "sdf_trunc_m": None if single_frame_quality else s.sdf_trunc_m,
            "depth_min_m": s.depth_min_m,
            "depth_max_m": s.depth_max_m,
            "point_stride": s.point_stride,
            "single_frame_quality": single_frame_quality,
            "mesh_method": "single_frame_pixel_adjacency" if single_frame_quality else "tsdf",
            "distortion_remap": s.distortion_diagnostics,
        },
        "outputs": {
            "point_cloud": str(s.cloud_path),
            "mesh": s.mesh_exports,
            "camera_trajectory": str(s.trajectory_path),
            "gaussian_initialization": s.gaussian,
        },
        "metrics": {
            "fused_point_count": int(len(cloud.points)),
            "mesh_vertices": int(len(mesh.vertices)),
            "mesh_triangles": int(len(mesh.triangles)),
            "valid_depth_ratio_mean": float(np.mean([item["valid_depth_ratio"] for item in s.per_frame_metrics])),
        },
        "frames": s.per_frame_metrics,
        "depth_quality": {
            "enabled": True,
            "frames": s.depth_quality_frames,
            "integrations_per_frame": s.integrations_per_frame,
            "filled_depth_used": bool(not local_measured and any(value==10 for value in s.integrations_per_frame)),
            "tsdf_confidence_policy": (
                "Exact relative weights 1.0:0.3 via 10 original and 3 repaired unit integrations per frame; increased TSDF runtime."
                if not single_frame_quality else
                "Single-frame original mesh remains observation-only; a separate derived hole-filled point cloud is exported."
            ),
        },
        "limitations": [
            "Geometry quality depends on calibrated intrinsics and validated T_WC poses.",
            "Gaussian output is initialization-only and is not a trained photoreal 3DGS model.",
        ],
    }
    if single_frame_quality:
        report["outputs"].update({
            "raw_point_cloud": str(s.raw_cloud_path), "cleaned_point_cloud": str(s.cloud_path),
            "cleaned_depth": str(output / "cleaned_depth_mm.png"),
            "source_evidence": str(output / "source_evidence.json"),
            "quality_report": str(s.quality_path),
            "hole_filled_point_cloud": str(s.repaired_cloud_path),
        })
        report["cleaning"] = s.cleaning_report
        report["coordinate_system"] = "World RDF; T_WC maps camera to world; metres"
        report["metrics"]["raw_point_count"] = len(s.raw_cloud.points)
        report["metrics"]["cleaned_point_count"] = len(cloud.points)
    report_path = output / "reconstruction_report.json"
    if local_measured:
        report['metrics'].update(_cloud_mesh_consistency(cloud,mesh))
        report['metrics']['fused_point_cloud_role']='multi_view_observation_union_display_and_diagnostics'
        report['metrics']['measurement_geometry_default']='tsdf_mesh'
        report['local_capture'] = dict(**manifest['local_capture'], support=s.local_support)
        report['outputs']['initial_observed_point_cloud'] = str(output/'initial_observed_points_m.ply')
        report['outputs']['local_geometry'] = str(output/'local_geometry.json')
        report['depth_quality']['tsdf_confidence_policy'] = (
            'Local capture: raw observed depth only; inferred repairs excluded. '
            'Mesh annotations restricted by raw-depth support; physical accuracy pending.')
        report['depth_quality']['local_measured_repairs_never_integrated'] = True
        report['outputs']['initial_observed_point_cloud_role']='display_subset_not_measurement'
        report['outputs']['initial_observed_point_cloud_linear_stride']=16
        report['outputs']['initial_observed_point_cloud_approximate_pixel_stride']=4
        report['metrics']['initial_display_point_count']=int(len(s.initial_display_subset.points))
        first=frames[0]
        report['outputs']['initial_observed_source_sha256']={key:hashlib.sha256((dataset/first[key]).read_bytes()).hexdigest()
            for key in ('rgb','depth','pose')}
    if temporal_input:
        report['depth_quality']['tsdf_confidence_policy']='Phase 1: continuous confidence saved in TSDF inputs; integrate unchanged and temporal TSDF execution deferred to phase 2'
        report['temporal_fusion']=manifest['temporal_fusion']
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    try:
        dataset = resolve_dataset(args.dataset, args.datasets_root)
        output = (args.output or (dataset / "derived" / "reconstruction")).resolve()
        report = reconstruct_scene(
            dataset,
            output,
            image_scale=args.image_scale,
            voxel_m=args.voxel_m,
            sdf_trunc_m=args.sdf_trunc_m,
            depth_min_m=args.depth_min_m,
            depth_max_m=args.depth_max_m,
            point_stride=args.point_stride,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": "complete", "output": str(output), "metrics": report["metrics"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
