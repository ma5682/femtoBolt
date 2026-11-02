"""Bound reference-pixel tasks and exact before/after stage comparisons.

No robot actions, estimated 3D targets or unknown-area percentages are invented.
The class map is the existing mesh/reference-depth diagnosis, not full-surface
completeness. The unchanged selected support domain is compared pixel by pixel.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from femto.annotation.reference_quality import LABELS


SCHEMA = 'femto-scan-feedback-v1'
ACTION = {
    2: ('inspect_reconstruction', '参考帧有深度但缺少网格；先检查重建与过滤，再决定是否补采。'),
    3: ('acquire_observation', '缺少网格与参考深度；在保持跟踪的条件下调整视角并停稳补采。'),
    4: ('inspect_depth_range', '网格表面超出有效深度范围；检查距离与模型。'),
    5: ('inspect_conflict', '网格与参考深度冲突；检查目标是否移动、配准及重建。'),
    6: ('inspect_reference', '缺少有效源图像像素；核对原始扫描范围与标定。'),
    7: ('inspect_reference_depth', '网格有支持但首帧深度未知；保留待确认，检查新观测来源。'),
}
NO_GAIN_LIMIT = 2  # Resource stopping advice; does not certify completeness.


def target_key(roi, proof, reference):
    value = dict(scan_roi=roi, reference_provenance=proof, view_reference=reference)
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def build_feedback(classes, binding, previous=None, *, raw_mapping=None):
    """Return JSON tasks plus comparison counters; never mutate diagnosis maps."""
    required=('target_key','stage_id','session_id','reference_frame_id','model_sha256')
    if any(binding.get(key) is None for key in required):
        raise ValueError('Missing feedback source binding')
    result = dict(schema=SCHEMA, binding=dict(binding), coordinate_frame='world_rdf',
                  unit='metre', transform_direction='T_W0_Ct',
                  scope='bound_reference_image_pixel_support', execution_ready=False,
                  physical_accuracy_verified=False, status='cannot_determine',
                  tasks=[], comparison=None, stagnant_rounds=0, stop_reason=None)
    if classes is None:
        return result
    values=np.asarray(classes)
    if values.ndim!=2 or values.size>8_388_608 or not np.isin(values,range(8)).all():
        raise ValueError('Invalid bounded reference diagnosis map')
    selected=values!=0
    counts={str(code):int((values==code).sum()) for code in LABELS}
    total=int(selected.sum()); good=counts['1']; pending=total-good
    result.update(counts=counts,requested_pixels=total,supported_pixels=good,pending_pixels=pending)
    result['status']=('reference_support_met' if total and pending==0 else
                      'needs_review_or_acquisition' if good else 'cannot_determine')
    unmapped=0
    if raw_mapping is not None:
        counts_raw=[raw_mapping.get(key) for key in
                    ('raw_requested_pixels','represented_raw_pixels','unmapped_raw_target_pixels')]
        if (any(type(n) is not int or n<0 for n in counts_raw)
                or counts_raw[0]!=counts_raw[1]+counts_raw[2]):
            raise ValueError('Invalid original target sampling account')
        result['raw_target_mapping']=dict(raw_mapping)
        unmapped=counts_raw[2]
        if unmapped:
            result['status']='needs_review_or_acquisition' if good else 'cannot_determine'
            result['tasks'].append(dict(task_id=f"{binding['stage_id']}:raw_mapping",
                reason_code=6,reason='原扫描框部分像素无法映射到参考检查图',
                action='inspect_reference',suggestion='核对扫描框与标定；这些原目标像素保持未知，不能按检查图全绿判定采够。',
                priority='normal',pixel_count=unmapped,
                reference_pixel_bounds=raw_mapping['raw_pixel_bounds'],
                bounds_meaning='original_scan_rectangle_contains_unrepresented_pixels',
                pixel_frame='raw_aligned_rgb',source_binding=dict(binding),
                position_world_m=None,candidate_observation_pose=None))
    for code,(action,note) in ACTION.items():
        yy,xx=np.nonzero(values==code)
        if not len(xx):continue
        # This bounding box locates a reason group, not a filled rectangle or a
        # guessed connected surface. The exact per-pixel map remains authoritative.
        result['tasks'].append(dict(task_id=f"{binding['stage_id']}:{code}",
            reason_code=code,reason=LABELS[code],action=action,suggestion=note,
            priority='normal',pixel_count=int(len(xx)),
            reference_pixel_bounds=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1],
            bounds_meaning='bounding_box_of_reason_pixels_not_a_filled_target',
            pixel_frame='undistorted_reference_rgb',source_binding=dict(binding),
            position_world_m=None,candidate_observation_pose=None))
    if previous is not None:
        report,old_map=previous
        if report.get('binding',{}).get('target_key')!=binding['target_key']:
            raise ValueError('Cannot compare a different original target/reference')
        if old_map is None:
            result['comparison_unavailable_reason']='previous_diagnosis_unavailable'
            return result
        old=np.asarray(old_map)
        if old.shape!=values.shape or not np.array_equal(old!=0,selected):
            raise ValueError('Original target domain changed; comparison rejected')
        improved=int(((old!=1)&(values==1)&selected).sum())
        degraded=int(((old==1)&(values!=1)).sum())
        result['comparison']=dict(previous_stage_id=report['binding']['stage_id'],
            improved_pixels=improved,degraded_pixels=degraded,
            unchanged_supported_pixels=int(((old==1)&(values==1)).sum()),
            remaining_pending_pixels=pending,
            interpretation='reference_pixel_evidence_change_not_physical_surface_gain')
        result['stagnant_rounds']=(int(report.get('stagnant_rounds',0))+1
                                   if improved==0 and (pending or unmapped) else 0)
        if result['stagnant_rounds']>=NO_GAIN_LIMIT:
            result['stop_reason']='repeated_no_observed_reference_support_gain'
    return result
