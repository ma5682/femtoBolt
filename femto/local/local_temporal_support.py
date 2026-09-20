"""Five-frame stationary support masks; selected depth values never averaged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import cv2
import numpy as np
import time
from femto.io_utils import imread_unicode, imwrite_unicode, write_json, write_jsonl
from femto.temporal.scene_temporal_fusion import audit_temporal_motion
from femto.tracking.local_stability import window_valid,RULE as WINDOW_RULE
from femto.dataset.local_reference import CALIBRATION_FILES,calibration_identity,validate_reference_provenance

MASK_SOURCE_SCHEMA='femto-local-temporal-mask-source-v1'
PIXEL_GRID='raw_depth_aligned_to_color_before_undistortion'


def _json(path):
    value=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(value,dict):raise ValueError('Temporal provenance must be an object')
    return value


def _rows(path):
    rows=[json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if any(not isinstance(row,dict) for row in rows):raise ValueError('Invalid temporal frame records')
    return rows


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def _values_sha(depth):
    if not isinstance(depth,np.ndarray) or depth.dtype!=np.uint16 or depth.ndim!=2:
        raise ValueError('Temporal depth must be the original uint16 millimetre image')
    return hashlib.sha256(np.ascontiguousarray(depth,dtype='<u2').tobytes()).hexdigest()


def _unique(rows,key,value):
    found=[row for row in rows if row.get(key)==value]
    if len(found)!=1:raise ValueError('Temporal source frame is missing or duplicated')
    return found[0]


def _timestamps(frame):
    result={key:frame.get(key) for key in ('camera_timestamp_us','color_timestamp_us','depth_timestamp_us')}
    if any(type(value) is not int or value<0 for value in result.values()):
        raise ValueError('Temporal frame timestamps are missing or invalid')
    if abs(result['color_timestamp_us']-result['depth_timestamp_us'])>5000:
        raise ValueError('Temporal frame RGB-D is not synchronized')
    return result


def _source_context(dataset,frame):
    """Read the actual dataset, selected frame and fixed-reference proof."""
    dataset=Path(dataset).resolve();manifest=_json(dataset/'manifest.json')
    local=manifest.get('local_capture',{})
    session=local.get('session_id');reference=local.get('reference_frame_id')
    if (not local.get('measured_only') or not isinstance(session,str) or not session
            or type(reference) is not int or reference<0 or type(frame.get('frame')) is not int):
        raise ValueError('Temporal mask lacks a trusted session/reference binding')
    rows=_rows(dataset/'frames.jsonl')
    actual=_unique(rows,'frame',frame['frame'])
    times=_timestamps(actual)
    if _timestamps(frame)!=times:
        raise ValueError('Temporal selected frame timestamp mismatch')
    for key in ('rgb','depth','pose'):
        if (dataset/actual[key]).resolve()!=(dataset/frame[key]).resolve():
            raise ValueError('Temporal selected frame source path mismatch')
    first=_unique(rows,'frame',reference)
    proof=validate_reference_provenance(local.get('reference_provenance'),dataset,first,_json(dataset/first['pose']))
    pose=_json(dataset/actual['pose'])
    if (pose.get('session_id')!=session or pose.get('reference_frame_id')!=reference
            or pose.get('capture_timestamp_us')!=times['camera_timestamp_us']
            or pose.get('reference_provenance')!=proof):
        raise ValueError('Temporal selected pose/session/reference mismatch')
    calibration=calibration_identity(dataset)
    return dict(dataset=dataset,manifest=manifest,local=local,frame=actual,
                session_id=session,reference_frame_id=reference,timestamps=times,
                calibration_sha256=calibration['sha256'],
                reference_provenance_sha256=hashlib.sha256(json.dumps(proof,sort_keys=True,
                    separators=(',',':')).encode('utf-8')).hexdigest())


def _window_evidence(context,source_frames):
    """Resolve five real source frames; no depth/RGB re-audit or SDK access."""
    if (not isinstance(source_frames,list) or len(source_frames)!=5
            or any(type(index) is not int for index in source_frames)
            or len(set(source_frames))!=5 or context['frame']['frame'] not in source_frames):
        raise ValueError('Temporal mask source window/frame mismatch')
    dataset=context['dataset'];local=context['local'];frame=context['frame']
    raw=Path(local['tracking_report']).resolve().parent.parent
    records=_rows(raw/'derived/live_tracking.jsonl')
    window=stationary_window(records,frame['frame'])
    if window is None or [r['frame_id'] for r in window[0]]!=source_frames:
        raise ValueError('Temporal mask window no longer matches trusted tracking records')
    raw_rows=_rows(raw/'frames.jsonl')
    required={dataset/'manifest.json',dataset/'frames.jsonl',raw/'frames.jsonl',raw/'derived/live_tracking.jsonl'}
    for name in CALIBRATION_FILES:
        current=dataset/'calibration'/name;original=raw/'calibration'/name
        if _sha(current)!=_sha(original):raise ValueError('Temporal raw calibration mismatch')
        required.update((current,original))
    for record in window[0]:
        if record.get('session_id')!=context['session_id'] or record.get('reference_frame_id')!=context['reference_frame_id']:
            raise ValueError('Temporal tracking window belongs to another session/reference')
        row=_unique(raw_rows,'frame',record['frame_id'])
        times=_timestamps(row)
        if (times['camera_timestamp_us']!=record.get('capture_timestamp_us')
                or times['color_timestamp_us']!=record.get('color_timestamp_us')
                or times['depth_timestamp_us']!=record.get('depth_timestamp_us')):
            raise ValueError('Temporal RGB-D timestamp provenance mismatch')
        for key in ('rgb','depth'):required.add((raw/row[key]).resolve())
        if record['frame_id']==frame['frame']:
            if times!=context['timestamps']:raise ValueError('Temporal selected/raw frame timestamps differ')
            for key in ('rgb','depth'):
                if (raw/row[key]).resolve()!=(dataset/frame[key]).resolve():
                    raise ValueError('Temporal selected frame is not its original RGB-D source')
    return {path.resolve() for path in required}


def _image_binding(context,depth):
    dataset=context['dataset'];frame=context['frame']
    return dict(selected_frame_id=frame['frame'],session_id=context['session_id'],
        reference_frame_id=context['reference_frame_id'],timestamps=context['timestamps'],
        calibration_sha256=context['calibration_sha256'],
        reference_provenance_sha256=context['reference_provenance_sha256'],pixel_grid=PIXEL_GRID,
        rgb=dict(path=str((dataset/frame['rgb']).resolve()),sha256=_sha(dataset/frame['rgb'])),
        depth=dict(path=str((dataset/frame['depth']).resolve()),sha256=_sha(dataset/frame['depth']),
            values_sha256=_values_sha(depth),shape=list(depth.shape),dtype='uint16_le',unit='millimetre'))


def _verify_hashes(evidence,required):
    if not isinstance(evidence,dict):raise ValueError('Temporal source hashes are missing')
    for path in required:
        expected=evidence.get(str(path.resolve()))
        if not isinstance(expected,str) or len(expected)!=64 or _sha(path)!=expected:
            raise ValueError('Temporal source file hash mismatch or missing source evidence: '+str(path))


def _validate_mask_source(dataset,frame,depth,spec,mask_path):
    context=_source_context(dataset,frame)
    source_frames=spec.get('source_frames')
    binding=spec.get('source_binding')
    if binding is not None:
        if not isinstance(binding,dict) or binding.get('schema')!=MASK_SOURCE_SCHEMA:
            raise ValueError('Unsupported temporal mask source binding')
        expected=_image_binding(context,depth)
        if any(binding.get(key)!=value for key,value in expected.items()):
            raise ValueError('Temporal mask/source RGB-D, frame, session, time or calibration mismatch')
        source=Path(binding['source_dataset']).resolve()
        source_rows=_rows(source/'frames.jsonl')
        source_context=_source_context(source,_unique(source_rows,'frame',frame['frame']))
        if any(source_context[key]!=context[key] for key in
               ('session_id','reference_frame_id','timestamps','calibration_sha256','reference_provenance_sha256')):
            raise ValueError('Temporal source dataset binding mismatch')
        required=_window_evidence(source_context,source_frames)
        _verify_hashes(binding.get('source_sha256'),required)
        return
    # Old masks may be used only if their original report can reconstruct the
    # source binding. Existing model viewing never calls this reconstruction gate.
    try:
        report_path=Path(context['local']['temporal_support_report']).resolve()
        report=_json(report_path)
        if report.get('schema')!='femto-local-temporal-support-v1' or report.get('source_unchanged') is not True:
            raise ValueError('Legacy temporal report is missing verified source evidence')
        row=_unique(report.get('frames',[]),'frame',frame['frame'])
        if row.get('applied') is not True or row.get('source_frames')!=source_frames:
            raise ValueError('Legacy temporal report frame/window mismatch')
        if mask_path!=(report_path.parent/f"support_{frame['frame']:06}.png").resolve():
            raise ValueError('Legacy temporal mask path does not match its applied frame')
        source=Path(report['source_dataset']).resolve()
        source_frame=_unique(_rows(source/'frames.jsonl'),'frame',frame['frame'])
        source_context=_source_context(source,source_frame)
        if _image_binding(source_context,depth)!=_image_binding(context,depth):
            raise ValueError('Legacy temporal source dataset differs from the actual frame')
        required=_window_evidence(source_context,source_frames)
        _verify_hashes(report.get('source_sha256'),required)
        decoded=imread_unicode(context['dataset']/frame['depth'],cv2.IMREAD_UNCHANGED)
        if decoded.dtype!=np.uint16 or not np.array_equal(decoded,depth):
            raise ValueError('Legacy temporal depth input differs from source pixels')
    except (KeyError,OSError,TypeError,ValueError) as error:
        raise ValueError('Legacy temporal mask has insufficient or mismatched source provenance; '
                         'rebuild masks in a new derived directory: '+str(error)) from error


def support_mask(depths,reference_index,motion_mask,minimum=250,maximum=2210):
    if len(depths)!=5 or not 0<=reference_index<5:raise ValueError('Five frames required')
    shape=depths[0].shape
    if any(d.shape!=shape or d.dtype!=np.uint16 for d in depths) or motion_mask.shape!=shape:
        raise ValueError('Depth/motion grids mismatch')
    keep=np.zeros(shape,bool)
    for y in range(0,shape[0],48):
        values=np.stack([d[y:y+48] for d in depths]).astype(np.float32)
        valid=(values>=minimum)&(values<=maximum)&(values>0)
        ref=values[reference_index]
        # Four of five actual observations must support the selected depth.
        support=(valid&(np.abs(values-ref)<=10)).sum(axis=0)
        keep[y:y+48]=valid[reference_index]&(support>=4)&~motion_mask[y:y+48]
    return keep


def stationary_window(records,frame_id):
    from femto.tracking.local_stability import find_window
    return find_window(records, frame_id)


def filtered_depth(dataset,frame,depth):
    spec=frame.get('local_temporal_support')
    if not spec:return depth
    path=(Path(dataset)/spec['mask']).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('Temporal mask hash mismatch')
    mask=imread_unicode(path,cv2.IMREAD_UNCHANGED)
    if mask.dtype!=np.uint8 or mask.shape!=depth.shape or not np.isin(mask,[0,255]).all():raise ValueError('Invalid support mask')
    _values_sha(depth)
    _validate_mask_source(dataset,frame,depth,spec,path)
    return np.where(mask>0,depth,0).astype(np.uint16)


def prepare(dataset,output):
    dataset,output=Path(dataset).resolve(),Path(output).resolve()
    if output==dataset or dataset in output.parents:raise ValueError('Output cannot modify source dataset')
    manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8-sig'))
    local=manifest.get('local_capture',{})
    if not local.get('measured_only'):raise ValueError('Local measured geometry required')
    raw=Path(local['tracking_report']).parent.parent
    records=[json.loads(s) for s in (raw/'derived/live_tracking.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    raw_rows={r['frame']:r for r in (json.loads(s) for s in (raw/'frames.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip())}
    frames=[json.loads(s) for s in (dataset/'frames.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    started=time.perf_counter()
    output.mkdir(parents=True,exist_ok=False);rows=[];hashes={}
    def digest(path):
        key=str(path.resolve());value=_sha(path)
        if key in hashes and hashes[key]!=value:raise ValueError('Source changed during temporal checks')
        hashes[key]=value
    digest(raw/'derived/live_tracking.jsonl');digest(raw/'frames.jsonl');digest(dataset/'frames.jsonl');digest(dataset/'manifest.json')
    for path in (dataset/'calibration').glob('*.json'):
        original=raw/'calibration'/path.name
        digest(path);digest(original)
        if path.read_bytes()!=original.read_bytes():raise ValueError('Temporal calibration mismatch')
    from femto.tracking.local_stability import candidate_status,all_stationary_windows
    accepted_ids={frame['frame'] for frame in frames}
    all_windows=all_stationary_windows(records)
    for item in all_windows:item['contains_accepted_frame']=bool(accepted_ids.intersection(item['source_frame_ids']))
    record_indices={r.get('frame_id'):i for i,r in enumerate(records)}
    for frame in frames:
        row=dict(frame=frame['frame'],applied=False,window_found=False,reason='no_stationary_five_frame_window');rows.append(row)
        if frame['frame'] in record_indices:
            row['candidate_status']=candidate_status(records,record_indices[frame['frame']])
        window=stationary_window(records,frame['frame'])
        if window is None:continue
        group,reference_index=window;images=[];depths=[]
        row['window_found']=True
        for r in group:
            raw_row=raw_rows[r['frame_id']]
            if any(raw_row.get(raw_key)!=r.get(track_key) for raw_key,track_key in (
                    ('camera_timestamp_us','capture_timestamp_us'),('color_timestamp_us','color_timestamp_us'),('depth_timestamp_us','depth_timestamp_us'))):
                raise ValueError('Temporal RGB-D timestamp provenance mismatch')
            for folder,target,flags in [('rgb',images,cv2.IMREAD_COLOR),('depth',depths,cv2.IMREAD_UNCHANGED)]:
                path=raw/raw_row[folder];digest(path);target.append(imread_unicode(path,flags))
        selected_depth=dataset/frame['depth'];selected_rgb=dataset/frame['rgb'];digest(selected_depth);digest(selected_rgb)
        if not np.array_equal(depths[reference_index],imread_unicode(selected_depth,cv2.IMREAD_UNCHANGED)) or not np.array_equal(images[reference_index],imread_unicode(selected_rgb,cv2.IMREAD_COLOR)):
            raise ValueError('Selected frame differs from stationary window reference')
        motion_mask,motion=audit_temporal_motion(images);row.update(source_frames=[r['frame_id'] for r in group],motion=motion)
        if not motion['passed']:row['reason']='image_motion_rejected';continue
        keep=support_mask(depths,reference_index,motion_mask)
        reference=depths[reference_index];valid=(reference>=250)&(reference<=2210)
        retained=float(keep.sum()/max(1,valid.sum()))
        row.update(retained_ratio=retained,valid_pixels=int(valid.sum()),removed_pixels=int((valid&~keep).sum()))
        if retained<.8:row['reason']='less_than_80_percent_retained';continue
        context=_source_context(dataset,frame)
        required=_window_evidence(context,row['source_frames'])
        for source_path in required:digest(source_path)
        binding=dict(schema=MASK_SOURCE_SCHEMA,source_dataset=str(dataset),**_image_binding(context,reference),
            source_sha256={str(p):hashes[str(p)] for p in sorted(required)})
        path=output/f"support_{frame['frame']:06}.png";imwrite_unicode(path,keep.astype(np.uint8)*255)
        frame['local_temporal_support']=dict(mask=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_frames=row['source_frames'],rule='4_of_5_within_10mm_of_selected_raw_depth',source_binding=binding)
        row.update(applied=True,reason=None)
    report=dict(schema='femto-local-temporal-support-v1',source_dataset=str(dataset),frames=rows,
        applied_frames=sum(r['applied'] for r in rows),
        windows_found=sum('source_frames' in r for r in rows),
        all_stationary_windows=all_windows,
        stationary_windows_total=len(all_windows),
        stationary_windows_containing_accepted=sum(bool(r['contains_accepted_frame']) for r in all_windows),
        window_miss_interpretation=('stationary_windows_exist_but_do_not_contain_accepted_frames'
            if all_windows and not any(r['contains_accepted_frame'] for r in all_windows) else
            'no_session_stationary_window' if not all_windows else 'accepted_frame_window_available'),
        window_rule=WINDOW_RULE,elapsed_s=time.perf_counter()-started,source_sha256=hashes,
        policy='Binary rejection mask only. Retained raw values unchanged; no interpolation, median replacement or new depth.',
        physical_accuracy_verified=False)
    report['source_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    if not report['source_unchanged']:raise ValueError('Source changed during temporal checks')
    report_path=output/'temporal_support_report.json';write_json(report_path, report)
    if not report['applied_frames']:return dataset,report
    derived=output/'dataset';derived.mkdir();shutil.copytree(dataset/'calibration',derived/'calibration');(derived/'pose').mkdir()
    for i,frame in enumerate(frames):
        pose=dataset/frame['pose'];frame['pose']=f'pose/{i:06}.json';shutil.copyfile(pose,derived/frame['pose'])
        for key in ('rgb','depth'):frame[key]=os.path.relpath((dataset/frame[key]).resolve(),derived).replace('\\','/')
    write_jsonl(derived/'frames.jsonl', frames, ensure_ascii=True, compact=False)
    manifest['local_capture']['temporal_support_report']=str(report_path)
    write_json(derived/'manifest.json', manifest)
    shutil.copyfile(dataset/'pose_quality_report.json',derived/'pose_quality_report.json')
    return derived,report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('output');a=p.parse_args()
    dataset,report=prepare(a.dataset,a.output);print(json.dumps(dict(dataset=str(dataset),applied_frames=report['applied_frames'],frames=report['frames']),ensure_ascii=False,indent=2))
