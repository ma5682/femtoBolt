"""Read-only local-session catalogue. Never updates current-session/latest markers."""
from __future__ import annotations
from typing import Any, Callable

import json
import math
from pathlib import Path
import time
from femto.io_utils import load_json_object, write_json
from femto.local.local_processing import result_quality, empty_report


def deleted_records(outputs: str | Path) -> Any:
    path=Path(outputs)/'local_history_deleted.json'
    if not path.exists():return {}
    data=load_json_object(path)
    if data.get('schema')!='local-history-deleted-v1' or not isinstance(data.get('records'),dict):
        raise ValueError('Invalid history deletion index')
    return data['records']


def set_records_deleted(rows: Any,datasets: Any,outputs: str | Path,deleted: bool=True) -> None:
    """Atomically change catalogue visibility for one or more records.

    This never moves or erases capture/model evidence.  Validating the whole
    batch before writing prevents a partial multi-record delete if one row is
    outside the allowed history roots.
    """
    rows=list(rows)
    if not rows:return
    for row in rows:
        key=Path(row['key']).resolve()
        root=Path(outputs if row.get('result') else datasets).resolve()
        pattern='complete_demo_*' if row.get('result') else 'scene_*_local_*'
        if key.parent!=root or not key.match(pattern):
            raise ValueError('Record is outside the history directory')
    records=deleted_records(outputs)
    if deleted:
        now=time.time()
        for row in rows:
            key=str(Path(row['key']).resolve())
            records[key]={'name':row['name'],'deleted_at':now}
    else:
        for row in rows:records.pop(str(Path(row['key']).resolve()),None)
    root=Path(outputs);root.mkdir(parents=True,exist_ok=True)
    write_json(
        root/'local_history_deleted.json',
        {'schema':'local-history-deleted-v1','records':records},
        unique_temporary=True,
        temporary_prefix='.history-',
    )


def set_record_deleted(row: Any,datasets: Any,outputs: Any,deleted: bool=True) -> None:
    """Compatibility wrapper for a single catalogue record."""
    set_records_deleted([row],datasets,outputs,deleted)


def publication_status(result: str | Path,session_id: str) -> Any:
    """Legacy results have no receipt; new results require explicit publication."""
    result=Path(result).resolve();receipt=result/'publication.json'
    summary_path=result/'demo_summary.json'
    summary=load_json_object(summary_path) if summary_path.is_file() else {}
    if not receipt.is_file():
        return 'missing' if summary.get('publication_required') is True else 'legacy_not_recorded'
    data=load_json_object(receipt)
    if (data.get('schema')!='femto-local-publication-v1' or data.get('session_id')!=session_id or
            Path(data.get('result','')).resolve()!=result or
            (summary.get('publication_required') is True and summary.get('publication_id')!=session_id)):
        return 'invalid'
    return data.get('status') if data.get('status') in ('pending','published','cancelled','failed') else 'invalid'


def fixed_reference_eligibility(reconstruction: str | Path,metadata: Any,report: Any,check: Callable[..., Any]=lambda:None) -> tuple[bool, str | None, list[Any], list[Any]]:
    """Read-only eligibility. The launched viewer still validates actual geometry."""
    from femto.annotation.scene_annotations import AnnotationStore
    reconstruction=Path(reconstruction)
    dataset=Path(report['source_dataset'])
    required=[reconstruction/'local_geometry.json',reconstruction/'reconstruction_report.json',
              dataset/'manifest.json',dataset/'frames.jsonl',dataset/'calibration/color_intrinsic.json',
              dataset/'calibration/distortion.json',Path(metadata['tracking_report'])]
    missing=[str(p) for p in required if not p.is_file()]
    if missing:return False,'missing_source_files',required,missing
    frames=[]
    with (dataset/'frames.jsonl').open(encoding='utf-8-sig') as stream:
        for line in stream:
            check()
            if line.strip():frames.append(json.loads(line))
            if len(frames)>10000:return False,'source_budget',required,[]
    if not frames:return False,'missing_reference',required,[]
    first=frames[0]
    required.extend(dataset/first[key] for key in ('rgb','depth'))
    required.extend(dataset/frame['pose'] for frame in frames)
    proof=metadata.get('reference_provenance')
    if isinstance(proof,dict) and isinstance(proof.get('calibration'),dict):
        for name in proof['calibration'].get('files',[]):
            if isinstance(name,str):required.append(dataset/'calibration'/name)
    missing=[str(p) for p in required if not p.is_file()]
    if missing:return False,'missing_source_files',required,missing
    check()
    reference=AnnotationStore.source_origin_reference(reconstruction/'scene_tsdf_mesh_m.ply',metadata)
    if reference['kind']!='initial_color_camera':return False,'origin_unverified',required,[]
    intrinsic=load_json_object(dataset/'calibration/color_intrinsic.json')
    distortion=load_json_object(dataset/'calibration/distortion.json')['rgb']
    if ('BROWN_CONRADY_K6' not in str(distortion.get('model','')) or
            any(not math.isfinite(float(intrinsic[key])) for key in ('fx','fy','cx','cy','width','height')) or
            min(float(intrinsic['fx']),float(intrinsic['fy']),float(intrinsic['width']),float(intrinsic['height']))<=0 or
            any(not math.isfinite(float(distortion[key])) for key in ('k1','k2','p1','p2','k3','k4','k5','k6'))):
        return False,'invalid_calibration',required,[]
    return True,None,required,[]


def comparison_source_files(report: Any,raw_source: str | Path,check: Callable[..., Any]=lambda:None) -> Any:
    """Cheap file availability only; the comparison worker resolves online poses."""
    dataset=Path(report['source_dataset']);raw_source=Path(raw_source)
    files=[dataset/'manifest.json',dataset/'frames.jsonl',raw_source/'manifest.json',raw_source/'frames.jsonl']
    count=0
    with (dataset/'frames.jsonl').open(encoding='utf-8-sig') as stream:
        for line in stream:
            check()
            if not line.strip():continue
            count+=1
            if count>10000:raise ValueError('Comparison source exceeds history preflight budget')
            frame=json.loads(line)
            files.extend(dataset/frame[key] for key in ('rgb','depth','pose'))
    return sorted({str(path) for path in files})


def scan_history(datasets: str | Path,outputs: str | Path,*,cancel: Any=None,seconds: float=30) -> Any:
    deleted=deleted_records(outputs)
    deadline=time.monotonic()+seconds
    def check():
        if cancel is not None and cancel.is_set():raise RuntimeError('History scan cancelled')
        if time.monotonic()>deadline:raise TimeoutError('History scan exceeded its 30 s budget')
    rows=[];completed_sources=set()
    for result in Path(outputs).glob('complete_demo_*'):
        check()
        reconstruction=result/'reconstruction'
        context=reconstruction/'local_geometry.json'
        if not context.is_file():continue
        row=dict(key=str(result.resolve()),name=result.name,result=str(result.resolve()),source=None,
                 session_id=None,state='incomplete',count=None,error=None,fixed=False,mesh=None,report=None,
                 fixed_reason='missing_reference',fixed_files=[],missing_source_files=[],compare=False,comparison_files=[])
        try:
            metadata=load_json_object(context)
            report=load_json_object(reconstruction/'reconstruction_report.json')
            row['session_id']=metadata.get('session_id')
            row['count']=report.get('frame_count')
            dataset=Path(report['source_dataset'])
            tracking=Path(metadata['tracking_report'])
            source=tracking.parent.parent
            row['source']=str(source.resolve())
            mesh=reconstruction/'scene_tsdf_mesh_m.ply'
            if mesh.is_file():row['mesh']=str(mesh.resolve());row['state']='model'
            try:row['publication_status']=publication_status(result,row['session_id'])
            except (OSError,ValueError,KeyError,TypeError):row['publication_status']='invalid'
            if row['mesh'] and row['publication_status'] not in ('published','legacy_not_recorded'):
                row['state']='unpublished';row['fixed_reason']='unpublished_result'
            if row['state']=='model':completed_sources.add(str(source.resolve()))
            if row['state']=='model':
                fixed,reason,required,missing=fixed_reference_eligibility(reconstruction,metadata,report,check)
                if (result/'publication.json').is_file():required.append(result/'publication.json')
                row.update(fixed=fixed,fixed_reason=reason,fixed_files=[str(p) for p in required],missing_source_files=missing)
                if fixed:
                    row['fixed_file_versions']={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in required}
                    row['comparison_files']=comparison_source_files(report,source,check)
                    row['compare']=all(Path(name).is_file() for name in row['comparison_files'])
        except (OSError,ValueError,KeyError,TypeError,StopIteration) as error:row['error']=str(error)
        row['quality']=result_quality(result)
        for report_path in (result/'processing_report.html',result/'result_quality.html',result/'demo_report.html',reconstruction/'reconstruction_report.json'):
            if report_path.is_file():row['report']=str(report_path.resolve());break
        row['time']=result.stat().st_mtime
        rows.append(row)
    for source in Path(datasets).glob('scene_*_local_*'):
        check()
        if not source.is_dir() or str(source.resolve()) in completed_sources:continue
        row=dict(key=str(source.resolve()),name=source.name,result=None,source=str(source.resolve()),
                 session_id=None,state='recorded',count=None,error=None,fixed=False,mesh=None,report=None,time=source.stat().st_mtime)
        try:
            report_path=source/'derived/local_capture_report.json'
            failure=source/'derived/local_failure.json'
            if report_path.is_file():
                report=load_json_object(report_path);row['session_id']=report.get('session_id')
                row['count']=len(report.get('accepted_frames',[]));row['report']=str(report_path.resolve())
                row['state']='lost' if report.get('state')=='lost' else 'recorded'
            if failure.is_file():
                report=load_json_object(failure);row['report']=str(failure.resolve())
                row['state']='failed';row['error']=report.get('error');row['session_id']=report.get('session_id',row['session_id'])
        except (OSError,ValueError,KeyError,TypeError) as error:row['error']=str(error);row['state']='incomplete'
        row['quality']=empty_report()
        if row['state']=='failed':
            try:
                processing=report.get('processing_report')
                if processing:
                    quality=load_json_object(processing)
                    if quality.get('source_session_id') != row['session_id']:
                        raise ValueError('Processing session mismatch')
                    row['quality']=quality
                    detail=Path(processing).with_suffix('.html')
                    if detail.is_file():row['report']=str(detail)
            except (OSError,ValueError,TypeError) as error:row['quality']['read_error']=str(error)
        rows.append(row)
    return sorted((r for r in rows if r['key'] not in deleted),key=lambda r:(r['time'],r['name']),reverse=True)


def _source_files_unchanged(row):
    versions=row.get('fixed_file_versions',{})
    try:
        for name in row.get('fixed_files',[]):
            path=Path(name)
            if not path.is_file():return False
            if name in versions:
                stat=path.stat()
                if (stat.st_size,stat.st_mtime_ns)!=tuple(versions[name]):return False
        return True
    except OSError:return False


def existing_actions(row: Any) -> dict[str, Any]:
    fixed=bool(row.get('fixed') and row.get('mesh') and Path(row['mesh']).is_file() and _source_files_unchanged(row))
    return dict(fixed=fixed,
                compare=bool(fixed and row.get('compare') and row.get('state')=='model'
                             and all(Path(name).is_file() for name in row.get('comparison_files',[]))),
                mesh=bool(row.get('mesh') and Path(row['mesh']).is_file()),
                report=bool(row.get('report') and Path(row['report']).is_file()),
                folder=bool((row.get('result') or row.get('source')) and Path(row.get('result') or row['source']).is_dir()))
