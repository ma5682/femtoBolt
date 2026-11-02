"""Versioned task/result/verification boundary for future scanning providers.

An envelope passing validation is still not a frame accepted for fusion.
Providers never publish models or bypass the existing RGB-D/tracking gate.
"""
from __future__ import annotations

import copy
import re
import numpy as np

from femto.dataset.scene_dataset_validator import validate_pose

TASK_SCHEMA='femto-acquisition-task-batch-v1'
RESULT_SCHEMA='femto-acquisition-observation-v1'
VERIFY_SCHEMA='femto-acquisition-verification-v1'
HASH=re.compile(r'^[0-9a-f]{64}$')


def task_batch(feedback,calibration_sha256):
    binding=copy.deepcopy(feedback['binding'])
    if (not isinstance(calibration_sha256,str) or not HASH.fullmatch(calibration_sha256) or not binding.get('stage_id')
            or not binding.get('session_id') or type(binding.get('reference_frame_id')) is not int):
        raise ValueError('Invalid acquisition task source')
    return dict(schema=TASK_SCHEMA,task_batch_id=binding['stage_id'],source_binding=binding,
        calibration_sha256=calibration_sha256,tasks=copy.deepcopy(feedback.get('tasks',[])),
        coordinate_frame='world_rdf',unit='metre',transform_direction='T_W0_Ct',
        requires_capture_gate=True,execution_ready=False,physical_accuracy_verified=False)


def validate_observation(packet,task,*,current_binding,calibration_sha256):
    if (not isinstance(packet,dict) or not isinstance(task,dict)
            or task.get('schema')!=TASK_SCHEMA or packet.get('schema')!=RESULT_SCHEMA
            or packet.get('task_batch_id')!=task.get('task_batch_id')
            or task.get('source_binding')!=current_binding
            or packet.get('source_binding')!=current_binding
            or packet.get('calibration_sha256')!=calibration_sha256
            or task.get('calibration_sha256')!=calibration_sha256
            or packet.get('coordinate_frame')!='world_rdf' or packet.get('unit')!='metre'
            or packet.get('transform_direction')!='T_W0_Ct'
            or packet.get('accepted_for_fusion') is not False):
        raise ValueError('Stale or mismatched acquisition observation envelope')
    status=packet.get('execution_status')
    if status not in ('received','failed','cancelled'):
        raise ValueError('Unknown acquisition execution status')
    if status=='received':
        rgbd=packet.get('rgbd')
        if not isinstance(rgbd,dict):raise ValueError('Missing RGB-D observation')
        times=[rgbd.get(key) for key in ('capture_timestamp_us','color_timestamp_us','depth_timestamp_us')]
        if (any(type(n) is not int or n<0 for n in times)
                or abs(times[1]-times[2])>5000
                or any(not isinstance(rgbd.get(key),str) or not rgbd[key] for key in ('rgb','depth'))):
            raise ValueError('Missing or unsynchronized full RGB-D evidence')
        pose=packet.get('T_W0_Ct')
        if pose is not None and validate_pose(pose):raise ValueError('Invalid observation pose')
    clean=copy.deepcopy(packet)
    clean.update(accepted_for_fusion=False,requires_capture_gate=True)
    return clean


def verification(task,observation,record,*,current_binding,calibration_sha256):
    packet=validate_observation(observation,task,current_binding=current_binding,
                                calibration_sha256=calibration_sha256)
    accepted=False;reason=packet['execution_status']
    if packet['execution_status']=='received':
        rgbd=packet['rgbd']
        if (not isinstance(record,dict) or record.get('session_id')!=current_binding['session_id']
                or record.get('reference_frame_id')!=current_binding['reference_frame_id']
                or record.get('calibration_sha256')!=calibration_sha256
                or record.get('capture_timestamp_us')!=rgbd['capture_timestamp_us']
                or record.get('color_timestamp_us')!=rgbd['color_timestamp_us']
                or record.get('depth_timestamp_us')!=rgbd['depth_timestamp_us']):
            raise ValueError('Observation is not associated with this capture gate record')
        if packet.get('T_W0_Ct') is not None and not np.array_equal(
                np.asarray(packet['T_W0_Ct']),np.asarray(record.get('T_WC'))):
            raise ValueError('Observation pose differs from associated capture gate pose')
        accepted=(record.get('selected_for_fusion') is True and record.get('state')=='tracking'
                  and not validate_pose(record.get('T_WC'))
                  and record.get('quality',{}).get('usable') is True
                  and record.get('continuity',{}).get('current_frame_verified') is True)
        receipt=record.get('raw_write_receipt')
        committed=isinstance(receipt,dict) and all(receipt.get(key)==record.get(key) for key in
            ('frame_id','capture_timestamp_us','color_timestamp_us','depth_timestamp_us'))
        accepted=accepted and committed
        reason='accepted_by_capture_gate' if accepted else record.get('rejection_reason','not_selected_by_capture_gate')
    return dict(schema=VERIFY_SCHEMA,task_batch_id=task['task_batch_id'],
        source_binding=copy.deepcopy(current_binding),calibration_sha256=calibration_sha256,
        accepted_for_fusion=accepted,reason=reason,
        frame_id=record.get('frame_id') if isinstance(record,dict) else None,
        T_W0_Ct=copy.deepcopy(record['T_WC']) if accepted else None,
        coordinate_frame='world_rdf',unit='metre',transform_direction='T_W0_Ct',
        target_improvement=None,remaining_tasks=None,
        improvement_meaning='requires_next_stage_same_target_comparison',
        execution_ready=False,physical_accuracy_verified=False)
