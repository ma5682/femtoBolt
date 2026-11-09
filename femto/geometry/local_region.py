"""Immutable reference selection and pose-aware, observation-only TSDF support.

Masks use the same full-size undistorted RGB pixels as FixedRGBModel. The
requested target is never intersected with depth validity; only integration is.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from femto.geometry.rgbd_frames import distortion_maps, load_poses
from femto.io_utils import imread_unicode, imwrite_unicode, load_json_object, load_jsonl, sha256_file, write_json_strict
from femto.operation_diagnostics import diagnosed

SCHEMA = 'femto-region-first-reconstruction-v2'
LEGACY_SCHEMA = 'femto-region-first-reconstruction-v1'
DEFAULT_MARGIN_MM = 0.0  # New requests model only the physician's painted selection.
DEPTH_CONSISTENCY_M = .01  # Same reference/mesh consistency limit as FixedProjection.


@dataclass
class RegionReference:
    dataset: Path
    rgb: np.ndarray
    depth: np.ndarray
    k: np.ndarray
    pose: np.ndarray
    binding: dict
    frame_count: int = 0


@diagnosed('selection.reference_load')
def load_reference(dataset):
    dataset = Path(dataset).resolve()
    manifest = load_json_object(dataset/'manifest.json')
    local = manifest.get('local_capture', {})
    rows = load_jsonl(dataset/'frames.jsonl')
    if not local.get('measured_only') or not rows:
        raise ValueError('局部选区重建需要至少一个已接纳的 RGB-D 视角及有效位姿')
    poses = load_poses(dataset, rows)
    index = next(i for i, row in enumerate(rows) if row['frame'] == local['reference_frame_id'])
    if index != 0 or not np.allclose(poses[index], np.eye(4), rtol=0, atol=1e-8):
        raise ValueError('固定参考帧必须对应原始 W0，不能更换参考位姿')
    mx, my, k, w, h = distortion_maps(dataset)
    if min(w, h) <= 0 or w*h > 8_388_608:
        raise ValueError('参考图尺寸超出预算')
    row = rows[index]
    rgb = imread_unicode(dataset/row['rgb'], cv2.IMREAD_COLOR)
    depth = imread_unicode(dataset/row['depth'], cv2.IMREAD_UNCHANGED)
    if rgb.shape != (h,w,3) or depth.shape != (h,w) or depth.dtype != np.uint16:
        raise ValueError('参考 RGB-D 与标定尺寸不一致')
    binding = dict(session_id=local['session_id'], reference_frame_id=row['frame'],
        image_size=[w,h], undistorted_intrinsic=k.tolist(),
        rgb_sha256=sha256_file(dataset/row['rgb']), depth_sha256=sha256_file(dataset/row['depth']),
        calibration_sha256={name: sha256_file(dataset/'calibration'/name)
                            for name in ('color_intrinsic.json','distortion.json')})
    return RegionReference(dataset, cv2.remap(rgb,mx,my,cv2.INTER_LINEAR)[:,:,::-1].copy(),
        cv2.remap(depth,mx,my,cv2.INTER_NEAREST), k, poses[index], binding,len(rows))


def save_request(reference, mask, destination, margin_mm=DEFAULT_MARGIN_MM):
    mask = np.asarray(mask)
    if mask.shape != reference.depth.shape or not np.isin(mask, [0,1]).all() or not mask.any():
        raise ValueError('请先在固定参考图上涂抹工作区域')
    if not np.isfinite(margin_mm) or not (margin_mm == 0 or 25 <= margin_mm <= 200):
        raise ValueError('仅选区重建使用 0 mm；旧周边模式须为 25–200 mm')
    # Detect a changed source before recording an apparently valid selection.
    if load_reference(reference.dataset).binding != reference.binding:
        raise ValueError('参考来源已改变，请重新读取')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    imwrite_unicode(destination/'requested_mask.png', mask.astype(np.uint8))
    record = dict(schema=SCHEMA if margin_mm == 0 else LEGACY_SCHEMA,
        source_dataset=str(reference.dataset), reference=reference.binding,
        mask_sha256=sha256_file(destination/'requested_mask.png'), selected_pixels=int(mask.sum()),
        support_margin_mm=float(margin_mm), depth_consistency_m=DEPTH_CONSISTENCY_M,
        mask_frame='undistorted_reference_rgb', raw_rgbd_cropped=False,
        geometry_scope='selected_surface_only' if margin_mm == 0 else 'selected_surface_and_observed_surroundings',
        missing_reference_depth_policy='retain_intent_exclude_unverified_geometry',
        execution_ready=False)
    write_json_strict(destination/'region_request.json',record)
    return destination/'region_request.json'


def read_request(path, dataset=None):
    path = Path(path)
    record = load_json_object(path)
    if record.get('schema') not in (SCHEMA, LEGACY_SCHEMA):
        raise ValueError('未知的局部重建选区版本')
    reference = load_reference(dataset or record['source_dataset'])
    if reference.binding != record['reference']:
        raise ValueError('选区与实际重建参考 RGB-D / 标定 / 会话不一致')
    mask_path = path.parent/'requested_mask.png'
    if sha256_file(mask_path) != record['mask_sha256']:
        raise ValueError('重建选区文件已改变')
    mask = imread_unicode(mask_path, cv2.IMREAD_UNCHANGED)
    margin = record['support_margin_mm']
    if (mask.shape != reference.depth.shape or not np.isin(mask,[0,1]).all()
            or not mask.any() or int(mask.sum()) != record['selected_pixels']
            or not np.isfinite(margin)
            or not (margin == 0 and record['schema'] == SCHEMA
                    and record.get('geometry_scope') == 'selected_surface_only'
                    or 25 <= margin <= 200 and record['schema'] == LEGACY_SCHEMA)
            or record.get('depth_consistency_m') != DEPTH_CONSISTENCY_M):
        raise ValueError('重建选区或周边保留带无效')
    return reference, mask, record


class RegionDepthFilter:
    def __init__(self, request, dataset, minimum, maximum, output):
        self.reference, self.mask, self.record = read_request(request, dataset)
        self.minimum, self.maximum = minimum, maximum
        self.distance = (None if self.record['support_margin_mm'] == 0 else
                         cv2.distanceTransform((~self.mask.astype(bool)).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
        self.output = Path(output)
        self.metrics = []

    @diagnosed('selection.map_depth')
    def filter(self, depth, pose, k, frame_id):
        """Backproject each view, transform to locked reference, reject occlusion.

        Projection is chunked to bound peak memory. Missing reference depth is
        deliberately unresolved, even if another camera sees a deeper surface.
        """
        ref = self.reference
        if depth.shape != ref.depth.shape or not np.allclose(k,ref.k,rtol=0,atol=1e-8):
            raise ValueError('局部重建必须使用原尺寸去畸变参考')
        output = np.zeros_like(depth)
        transform = np.linalg.inv(ref.pose) @ pose
        h,w = depth.shape
        focal = min(k[0,0],k[1,1])
        # Exact equality only: even a small actual camera motion must use the
        # pose-aware path. In this case projection returns the same pixel.
        if (np.array_equal(ref.pose, np.eye(4)) and np.array_equal(pose, ref.pose)
                and np.array_equal(k, ref.k)):
            z = depth.astype(float)/1000
            rz = ref.depth.astype(float)/1000
            nearby = (self.mask.astype(bool) if self.record['support_margin_mm'] == 0 else
                      self.distance*z/focal <= self.record['support_margin_mm']/1000)
            keep = ((depth >= self.minimum*1000)&(depth <= self.maximum*1000)
                    &(rz >= self.minimum)&(rz <= self.maximum)
                    &(np.abs(z-rz) <= DEPTH_CONSISTENCY_M)
                    &nearby)
            output[keep] = depth[keep]
            self._record(depth,output,frame_id)
            return output
        for top in range(0,h,64):
            block = depth[top:top+64]
            y,x = np.nonzero((block >= self.minimum*1000)&(block <= self.maximum*1000))
            y = y+top
            z = depth[y,x].astype(float)/1000
            points = np.c_[(x-k[0,2])*z/k[0,0], (y-k[1,2])*z/k[1,1], z]
            points = points @ transform[:3,:3].T + transform[:3,3]
            good = np.isfinite(points).all(axis=1)&(points[:,2]>0)
            ids = np.flatnonzero(good)
            p = points[ids]
            uv = np.rint(p[:,:2]/p[:,2,None]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]).astype(np.int64)
            inside = (uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            ids,uv = ids[inside],uv[inside]
            rz = ref.depth[uv[:,1],uv[:,0]].astype(float)/1000
            consistent = (rz>=self.minimum)&(rz<=self.maximum)&(np.abs(points[ids,2]-rz)<=DEPTH_CONSISTENCY_M)
            nearby = (self.mask[uv[:,1],uv[:,0]].astype(bool) if self.record['support_margin_mm'] == 0 else
                      self.distance[uv[:,1],uv[:,0]]*points[ids,2]/focal <= self.record['support_margin_mm']/1000)
            ids = ids[consistent & nearby]
            output[y[ids],x[ids]] = depth[y[ids],x[ids]]
        self._record(depth,output,frame_id)
        return output

    def _record(self, depth, output, frame_id):
        retained=int(np.count_nonzero(output))
        self.metrics.append(dict(frame=int(frame_id), input_valid_pixels=int(np.count_nonzero(depth)),
            integrated_pixels=retained, local_retained_pixels=retained,
            used_for_surface=bool(retained), exclusion_reason=None if retained else 'no_observed_surface_in_selection_and_margin'))

    def save(self):
        record = dict(self.record, frames=self.metrics,
            reference_missing_selected_pixels=int(np.count_nonzero(self.mask.astype(bool)&
                ~((self.reference.depth>=self.minimum*1000)&(self.reference.depth<=self.maximum*1000)))))
        imwrite_unicode(self.output/'requested_mask.png',self.mask)
        # Also persisted in local_geometry for portable project provenance.
        write_json_strict(self.output/'region_request.json',record)
        return record


def model_selection(model):
    """Read original intent without silently clipping it to the reconstructed mesh."""
    # An explicitly derived planning version requires a new selection; its
    # original region-first scope describes geometry provenance, not a mask in
    # the new pixel grid. Verify that lineage before declining automatic replay.
    derived=(getattr(model,'view_reference',None) or {}).get('derivation') or {}
    if derived.get('schema')=='reduced-planning-reference-v1':
        lineage=getattr(model,'source_lineage',None) or {}
        if (derived.get('selection_policy')!='new_selection_required_no_automatic_roi_transfer'
                or lineage.get('schema')!='reduced-planning-lineage-v1'
                or lineage.get('derivation')!=derived
                or lineage.get('original_view_reference')!=derived.get('source_view_reference')
                or derived.get('source_geometry_sha256')!=model.store.digest):
            raise ValueError('派生规划参考缺少可核验的新选区来源')
        scope=(getattr(model.store,'local_context',None) or {}).get('reconstruction_scope')
        if scope is not None:
            original=derived['source_view_reference']
            for key in ('session_id','reference_frame_id','rgb_sha256','depth_sha256','image_size','undistorted_intrinsic'):
                if scope['reference'][key]!=original.get(key):
                    raise ValueError('派生参考与原局部重建来源不一致')
        return None
    reconstruction = getattr(model,'reconstruction',None)
    if reconstruction is None:
        return None
    path = Path(reconstruction)/'region_request.json'
    if not path.is_file():
        if (getattr(model.store,'local_context',None) or {}).get('reconstruction_scope'):
            raise ValueError('局部模型缺少原始选区文件，不能默认改成空选区')
        return None
    record = load_json_object(path)
    if record.get('schema') not in (SCHEMA, LEGACY_SCHEMA):
        raise ValueError('未知的局部重建选区版本')
    if record != (getattr(model.store,'local_context',None) or {}).get('reconstruction_scope'):
        raise ValueError('局部重建选区与模型记录的重建范围不一致')
    for key in ('session_id','reference_frame_id','rgb_sha256','depth_sha256','image_size','undistorted_intrinsic'):
        if record['reference'][key] != model.view_reference[key]:
            raise ValueError('局部重建选区与规划参考不一致')
    mask_path = path.parent/'requested_mask.png'
    if sha256_file(mask_path) != record['mask_sha256']:
        raise ValueError('局部重建目标意图已改变')
    mask = imread_unicode(mask_path,cv2.IMREAD_UNCHANGED)
    if mask.shape != model.rgb.shape[:2] or not np.isin(mask,[0,1]).all() or int(mask.sum()) != record['selected_pixels']:
        raise ValueError('局部重建目标意图无效')
    return mask
