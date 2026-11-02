"""Snapshot exact gap evidence before navigating to an independent acquisition."""
from pathlib import Path

import numpy as np

from femto.dataset.gap_followup import SCHEMA, STATE, checked, digest
from femto.dataset.task_context import checked_task
from femto.planning.planning_revision import array_record, digest as plan_digest
from femto.planning.surface_trajectory import canonical_json_bytes


def prepare(model,plan,mask,reason,pixel_reason,output,check=lambda:None):
    from femto.planning.planning_io import staging_directory,publish
    task=checked_task(getattr(model,'task_context',None))
    if task is None:raise ValueError('缺少可核验的任务信息')
    from femto.planning.planning_revision import require_current
    require_current(model,plan['planning_context']['revision']['inputs']['selection'],
                    plan['treatment_head'],plan['planning_context']['revision'])
    mask=np.asarray(mask)
    if mask.dtype!=np.bool_ or mask.shape!=model.projection.depth.shape or not mask.any():
        raise ValueError('缺口范围与当前参考图不一致')
    context=plan['planning_context'];yy,xx=np.nonzero(mask);output=Path(output).resolve()
    request=dict(schema=SCHEMA,task_id=task['task_id'],origin_target_id=context['revision']['target_id'],
        origin_computation_sha256=context['revision']['sha256'],origin_plan_sha256=plan_digest(plan),
        origin_geometry_sha256=plan['plan_binding']['geometry_sha256'],mask_sha256=array_record(mask)['sha256'],
        pixel_count=int(mask.sum()),reference_bounds=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1],
        reason=reason,pixel_reason=pixel_reason,evidence_directory=str(output),next_state=STATE)
    request['sha256']=digest(request);checked(request,task['task_id'])
    payload=canonical_json_bytes(plan)
    if len(payload)>100*1024**2:raise ValueError('来源方案超过缺口证据快照预算')
    if output.exists():raise FileExistsError('缺口证据必须使用新目录')
    output.parent.mkdir(parents=True,exist_ok=True)
    with staging_directory(output.parent,'.gap-followup-') as stage:
        check();np.savez_compressed(stage/'reference_gap.npz',mask=mask)
        (stage/'source_plan.json').write_bytes(payload)
        (stage/'request.json').write_bytes(canonical_json_bytes(request))
        publish(lambda:stage.rename(output),check,None)
    return request
