"""Final-model confirmation bound to immutable reconstruction content.

Older captures without the new requirement retain their original semantics.
Stage checkpoints are always rejected by planning, including direct file opens.
Confirmation never changes the acquisition quality or pending-target diagnosis.
"""
from __future__ import annotations

from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path

from femto.io_utils import load_json_object, sha256_file, write_json

SCHEMA = 'femto-acquisition-completion-v1'
ASSESSMENT = {'reference_support_met','needs_review_or_acquisition','cannot_determine'}


def _read(path,limit=1024**2):
    if path.stat().st_size>limit:raise ValueError('采集确认资产超过读取预算。')
    return load_json_object(path)


def result_root(value):
    path=Path(value).resolve()
    return path.parent if path.name=='reconstruction' else path


def read_completion(value, *, check=lambda:None, require_confirmed=False):
    root=result_root(value)
    report_path=root/'reconstruction/reconstruction_report.json'
    if not report_path.is_file():
        return None
    check()
    report=_read(report_path,10*1024**2)
    context=report.get('local_capture') or {}
    if context.get('capture_scope')=='stage_checkpoint':
        raise ValueError('阶段模型不能进入规划，请先结束采集并确认最终模型。')
    summary_path=root/'demo_summary.json'
    summary=load_json_object(summary_path) if summary_path.is_file() else {}
    path=root/'acquisition_completion.json'
    required=summary.get('acquisition_completion_required') is True
    if not path.is_file():
        if required:raise ValueError('缺少最终采集核验与确认记录。')
        return None
    record=_read(path)
    binding=record.get('binding') or {}
    mesh=root/'reconstruction/scene_tsdf_mesh_m.ply'
    if (record.get('schema')!=SCHEMA or record.get('assessment_status') not in ASSESSMENT
            or type(record.get('confirmed_for_planning')) is not bool
            or record.get('feedback',{}).get('binding')!=binding
            or record.get('feedback',{}).get('status')!=record.get('assessment_status')
            or record.get('execution_ready') is not False
            or record.get('physical_accuracy_verified') is not False
            or binding.get('session_id')!=context.get('session_id')
            or type(binding.get('reference_frame_id')) is not int
            or binding['reference_frame_id']!=context.get('reference_frame_id')
            or Path(report['outputs']['mesh']['ply']).resolve()!=mesh
            or binding.get('model_sha256')!=sha256_file(mesh,check)
            or record.get('reconstruction_report_sha256')!=sha256_file(report_path,check)):
        raise ValueError('最终采集确认与模型来源或版本不一致。')
    expected_end={'reference_support_met':'reference_support_met',
                  'needs_review_or_acquisition':'manual_end_pending',
                  'cannot_determine':'manual_end_unknown'}[record['assessment_status']]
    if record.get('end_status')!=expected_end:
        raise ValueError('人工结束不能改写采集质量结论。')
    if require_confirmed:
        publication=load_json_object(root/'publication.json')
        if (record['confirmed_for_planning'] is not True
                or publication.get('status')!='published'
                or publication.get('session_id')!=binding['session_id']
                or Path(publication.get('result','')).resolve()!=root):
            raise ValueError('请先检查并确认最终模型，再进入工作区域规划。')
    check()
    return record


def confirm_for_planning(value, expected_binding, *, check=lambda:None):
    root=result_root(value)
    record=read_completion(root,check=check)
    if record is None:raise ValueError('没有可确认的最终采集核验记录。')
    if record['binding']!=expected_binding:
        raise ValueError('确认请求属于另一个模型版本。')
    publication=load_json_object(root/'publication.json')
    if (publication.get('status')!='published'
            or publication.get('session_id')!=expected_binding['session_id']
            or Path(publication.get('result','')).resolve()!=root):
        raise ValueError('最终模型尚未成功发布，不能确认。')
    check()
    confirmed=dict(record,confirmed_for_planning=True,
        confirmed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(root/'acquisition_completion.json',confirmed,allow_nan=False,
               durable=True,unique_temporary=True,check=check,max_bytes=1024**2)
    return confirmed


def portable_completion(record,geometry_digest,reference):
    """Preserve source quality in a portable plan without claiming fresh acquisition."""
    wrapper=dict(schema='femto-portable-acquisition-completion-v1',
        geometry_digest=geometry_digest,reference=copy.deepcopy(reference),
        completion=copy.deepcopy(record),meaning='saved_acquisition_assessment_not_current_patient_check')
    wrapper['completion_sha256']=hashlib.sha256(json.dumps(record,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
    validate_portable_completion(wrapper,geometry_digest,reference)
    return wrapper


def validate_portable_completion(wrapper,geometry_digest,reference):
    if not isinstance(wrapper,dict):raise ValueError('Invalid portable acquisition assessment')
    record=wrapper.get('completion')
    if not isinstance(record,dict) or not isinstance(reference,dict):
        raise ValueError('Invalid portable acquisition assessment')
    binding=record.get('binding',{})
    if not isinstance(binding,dict) or not isinstance(record.get('feedback'),dict):
        raise ValueError('Invalid portable acquisition binding')
    if (not isinstance(record,dict) or wrapper.get('schema')!='femto-portable-acquisition-completion-v1'
            or wrapper.get('geometry_digest')!=geometry_digest or wrapper.get('reference')!=reference
            or record.get('schema')!=SCHEMA or record.get('assessment_status') not in ASSESSMENT
            or record.get('confirmed_for_planning') is not True
            or record.get('execution_ready') is not False or record.get('physical_accuracy_verified') is not False
            or binding.get('session_id')!=reference.get('session_id')
            or binding.get('reference_frame_id')!=reference.get('reference_frame_id')
            or record.get('feedback',{}).get('binding')!=binding
            or record.get('feedback',{}).get('status')!=record.get('assessment_status')):
        raise ValueError('Portable acquisition assessment does not match model/reference')
    digest=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
    if wrapper.get('completion_sha256')!=digest:
        raise ValueError('Portable acquisition assessment content changed')
    end={'reference_support_met':'reference_support_met','needs_review_or_acquisition':'manual_end_pending',
         'cannot_determine':'manual_end_unknown'}[record['assessment_status']]
    if record.get('end_status')!=end:raise ValueError('Saved manual end cannot hide unknown acquisition')
    return copy.deepcopy(record)
