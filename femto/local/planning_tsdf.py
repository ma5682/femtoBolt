"""A separate, measured-only TSDF planning surface for any local frame count.

The producer keeps the full measurement mesh and RGB-D evidence. This builder
reuses its exact calibrated selection and admitted poses, without editing them.
The planning adapter must additionally guard against the full source surface.
"""
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from femto.dataset.reconstruction_inputs import snapshot, verify
from femto.geometry.local_region import RegionDepthFilter
from femto.geometry.rgbd_frames import distortion_maps, load_poses, prepare_depth, rgbd_from_prepared
from femto.io_utils import imread_unicode, load_jsonl
from femto.local.local_geometry import filter_small_components, retain_observed_triangles
from femto.local.local_temporal_support import filtered_depth
from femto.operation_diagnostics import diagnosed

VERSION = 'local-measured-tsdf-planning-v1'
LEGACY_PROFILE = 'local-tsdf-5mm-full-reference-adaptive-sampling-v1'
PROFILE = 'local-tsdf-5mm-measured-surface-distance-v2'
VOXEL_M = .005
SDF_TRUNC_M = .020
MAX_PREPARED_BYTES = 512 * 1024**2


@diagnosed('planning.tsdf_surface')
def build(request, scope, minimum, maximum, check=lambda: None, progress=lambda text: None):
    """Return a new mesh and replay metadata; never publish to the source."""
    check()
    evidence = scope['input_provenance']
    dataset = Path(scope['source_dataset']).resolve()
    verify(evidence, check)
    if snapshot(dataset, check) != evidence:
        raise ValueError('TSDF 规划输入与原测量模型的来源不一致，请重新选区重建')
    frames = load_jsonl(dataset / 'frames.jsonl')
    poses = load_poses(dataset, frames)
    maps = distortion_maps(dataset)
    mx, my, k, width, height = maps
    if width * height * 2 * len(frames) > MAX_PREPARED_BYTES:
        raise ValueError('TSDF 规划深度准备超过 512 MiB 预算，原模型保留')
    region = RegionDepthFilter(request, dataset, minimum, maximum, Path(request).parent)
    # read_request verifies RGB-D, calibration and mask; also bind the producer's
    # complete scope before preparing an independently derived surface.
    if region.record != scope:
        raise ValueError('TSDF 规划选区与原测量模型的重建范围不一致')
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_M, sdf_trunc=SDF_TRUNC_M,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    used_frames, used_poses, depths, integrations = [], [], [], []
    expected = {row['frame']: row for row in scope['frames']}
    for frame, pose in zip(frames, poses):
        check()
        raw = imread_unicode(dataset / frame['depth'], cv2.IMREAD_UNCHANGED)
        if raw.shape != (height, width) or raw.dtype != np.uint16:
            raise ValueError('TSDF 规划深度必须是标定尺寸的 uint16 毫米图像')
        depth = prepare_depth(filtered_depth(dataset, frame, raw), mx, my, minimum, maximum)
        depth = region.filter(depth, pose, k, frame['frame'])
        pixels = int(np.count_nonzero(depth))
        original = expected.get(frame['frame'], {})
        if (original.get('surface_input_pixels') != pixels
                or original.get('used_for_surface') != bool(pixels)):
            raise ValueError('TSDF 规划实际贡献与原测量模型不一致')
        if not pixels:
            continue
        color = imread_unicode(dataset / frame['rgb'], cv2.IMREAD_COLOR)
        if color.shape != (height, width, 3):
            raise ValueError('TSDF 规划 RGB 与标定尺寸不一致')
        color = cv2.remap(color, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rgbd, intrinsic, _, _ = rgbd_from_prepared(color, depth, k, maximum)
        progress(f'生成统一规划表面：融合第 {len(used_frames)+1} 个有效视角')
        check()
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
        check()
        used_frames.append(frame); used_poses.append(pose); depths.append(depth)
        integrations.append(dict(frame=frame['frame'], passes=1, measured_pixels=pixels))
    if not used_frames or len(used_frames) != scope['contributing_frame_count']:
        raise ValueError('TSDF 规划没有匹配的实测贡献帧，原模型保留')
    mesh = volume.extract_triangle_mesh()
    del volume
    check()
    if len(mesh.triangles) > 4_000_000:
        raise ValueError('TSDF 规划表面超过 400 万三角面预算')
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    initial_components = filter_small_components(mesh)
    check()
    support = retain_observed_triangles(mesh, dataset, used_frames, used_poses, maps,
        minimum, maximum, prepared_depths=depths)
    check()
    support['post_retain_component_filter'] = filter_small_components(mesh)
    if not len(mesh.triangles):
        raise ValueError('TSDF 规划没有受实测支持的表面，原模型保留')
    mesh.compute_vertex_normals()
    verify(evidence, check)
    record = dict(schema=VERSION, voxel_m=VOXEL_M, sdf_trunc_m=SDF_TRUNC_M,
        input_frame_count=len(frames), contributing_frame_count=len(used_frames),
        integrations=integrations, input_provenance_sha256=evidence['sha256'],
        mask_sha256=scope['mask_sha256'], support_margin_mm=scope['support_margin_mm'],
        coordinate_frame='world_rdf', unit='metre', transform_direction='T_W0_Ct',
        integration_policy='one_pass_per_actual_contributor_no_inferred_depth',
        output_vertices=len(mesh.vertices), output_triangles=len(mesh.triangles),
        initial_components=initial_components, observed_support=support,
        physical_accuracy_verified=False, execution_ready=False)
    check()
    return mesh, record


def checked(record, scope):
    """Validate the saved producer contract without requiring external RGB-D."""
    if (not isinstance(record, dict) or not isinstance(scope, dict)
            or not isinstance(scope.get('frames'), list)
            or any(not isinstance(row, dict) for row in scope['frames'])):
        raise ValueError('TSDF 规划表面的实际贡献帧记录无效')
    rows = record.get('integrations', [])
    expected = [row for row in scope.get('frames', []) if row.get('used_for_surface')]
    if (record.get('schema') != VERSION or record.get('voxel_m') != VOXEL_M
            or record.get('sdf_trunc_m') != SDF_TRUNC_M
            or record.get('input_frame_count') != scope.get('input_frame_count')
            or record.get('contributing_frame_count') != scope.get('contributing_frame_count')
            or record.get('input_provenance_sha256') != scope.get('input_provenance', {}).get('sha256')
            or record.get('mask_sha256') != scope.get('mask_sha256')
            or record.get('support_margin_mm') != scope.get('support_margin_mm')
            or record.get('coordinate_frame') != 'world_rdf' or record.get('unit') != 'metre'
            or record.get('transform_direction') != 'T_W0_Ct'
            or record.get('integration_policy') != 'one_pass_per_actual_contributor_no_inferred_depth'
            or not expected or rows != [dict(frame=r['frame'], passes=1,
                measured_pixels=r['surface_input_pixels']) for r in expected]):
        raise ValueError('TSDF 规划表面的版本、参数或实际贡献帧绑定不一致')
    return record
