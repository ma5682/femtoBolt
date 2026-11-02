"""Associate manual supplement requests with immutable raw receipts and task versions."""
from __future__ import annotations

import copy
from pathlib import Path

from femto.capture.scan_exchange import RESULT_SCHEMA,task_batch,verification
from femto.io_utils import append_jsonl


def current_tasks(capture):
    manager=getattr(capture,'checkpoints',None)
    if manager is None:return None
    with manager.lock:
        feedback=copy.deepcopy(manager.last_feedback[0]) if manager.last_feedback else None
    if feedback is None:return None
    return task_batch(feedback,capture.calibration['sha256'])


def record_manual_observation(capture,task,record):
    """Called only after raw commit and the internal admission decision."""
    if task is None:return None
    frame_id=record['frame_id']
    packet=dict(schema=RESULT_SCHEMA,task_batch_id=task['task_batch_id'],
        source_binding=copy.deepcopy(task['source_binding']),
        calibration_sha256=capture.calibration['sha256'],
        coordinate_frame='world_rdf',unit='metre',transform_direction='T_W0_Ct',
        accepted_for_fusion=False,execution_status='received',provider='manual_camera',
        requested_task_ids=[item['task_id'] for item in task['tasks']],
        request_meaning='manual_view_for_current_batch_not_claimed_resolution_of_each_task',
        T_W0_Ct=copy.deepcopy(record.get('T_WC')),
        rgbd=dict(rgb=str(capture.session/'rgb'/f'{frame_id:06d}.png'),
                  depth=str(capture.session/'depth'/f'{frame_id:06d}.png'),
                  capture_timestamp_us=record['capture_timestamp_us'],
                  color_timestamp_us=record['color_timestamp_us'],
                  depth_timestamp_us=record['depth_timestamp_us']))
    for key in ('rgb','depth'):
        if not Path(packet['rgbd'][key]).is_file():
            raise ValueError('Manual observation is missing committed raw RGB-D')
    # Binding is frozen at request time. A later stage does not retroactively
    # associate this observation with a different task/model revision.
    checked=verification(task,packet,record,current_binding=task['source_binding'],
                         calibration_sha256=capture.calibration['sha256'])
    row=dict(schema='femto-manual-supplement-exchange-v1',task_batch=task,
             observation=packet,verification=checked)
    append_jsonl(capture.session/'derived/manual_supplement_exchanges.jsonl',row)
    return checked
