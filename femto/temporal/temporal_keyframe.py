"""Quality -> hole filling -> confidence -> temporal depth -> keyframe dataset."""
from datetime import datetime
from pathlib import Path
import json
import shutil
from time import perf_counter
import numpy as np
import cv2

from femto.depth.depth_quality import process_depth_quality
from femto.io_utils import imread_unicode, imwrite_unicode, write_json, write_jsonl
from femto.depth.temporal_depth_fusion import confidence_weighted_median
from femto.temporal.scene_temporal_fusion import audit_temporal_motion, _hash, _child
from femto.depth.temporal_depth_layer import load_temporal_confidence, prepared_depth  # noqa: F401  (re-export)


def assess_temporal_scene(raw, rgb, low, high):
    mask=np.zeros(raw[0].shape,bool)
    if len(raw)==1:
        return mask,dict(scene_motion_detected=False,static_evidence_available=False)
    rgb_mask,rgb_report=audit_temporal_motion(rgb)
    changes=np.zeros(raw[0].shape,np.uint8)
    for first,last in zip(raw,raw[1:]):
        both=(first>=low)&(first<=high)&(first>0)&(last>=low)&(last<=high)&(last>0)
        changes+=both&(np.abs(first.astype(np.int32)-last.astype(np.int32))>50)
    depth_mask=changes>=max(2,int(np.ceil(.3*(len(raw)-1))))
    mask=rgb_mask|depth_mask
    global_rejected=not rgb_report['passed']
    if global_rejected: mask[:]=True
    return mask,dict(scene_motion_detected=bool(mask.any()),global_motion_or_uncertain=global_rejected,
                     rgb=rgb_report,depth_change_limit_mm=50,depth_motion_rate=float(depth_mask.mean()),
                     excluded_pixel_rate=float(mask.mean()),weight_policy='suspect pixels receive zero weight',
                     limitation='Depth/RGB changes are conservative motion hints, not material identification')


def fuse_quality_frames(raw, filled, confidences, low, high, motion_mask=None):
    n=len(raw)
    if n not in (1,5,10,15) or len(filled)!=n or len(confidences)!=n:
        raise ValueError('Expected 1/5/10/15 corresponding quality frames')
    shape=raw[0].shape
    for r,d,c in zip(raw,filled,confidences):
        if r.shape!=shape or d.shape!=shape or c.shape!=shape or r.dtype!=np.uint16 or d.dtype!=np.uint16:
            raise ValueError('Temporal input grids/types mismatch')
        if not np.isfinite(c).all() or np.any((c<0)|(c>1)):
            raise ValueError('Confidence must be finite in [0,1]')
        measured=(r>0)&(r>=low)&(r<=high)
        if np.any(d[measured]!=r[measured]) or np.any(c[~measured]>.300001):
            raise ValueError('Quality stage changed real observations or promoted repairs')
    out=np.zeros(shape,np.uint16); final=np.zeros(shape,np.float32)
    maps={name:np.zeros(shape,dtype) for name,dtype in (
        ('source_count',np.uint8),('original_support_count',np.uint8),('filled_support_count',np.uint8),
        ('temporal_variance',np.float32),('temporal_stability',np.float32),('depth_quality_confidence',np.float32),
        ('selected_original',np.uint8))}
    motion=np.zeros(shape,bool) if motion_mask is None else np.asarray(motion_mask,bool)
    if motion.shape!=shape: raise ValueError('Motion mask dimensions mismatch')
    removed=0
    for y in range(0,shape[0],48):
        sl=slice(y,y+48)
        r=np.stack([a[sl] for a in raw]); d=np.stack([a[sl] for a in filled]).astype(np.float32)
        c=np.stack([a[sl] for a in confidences]).astype(np.float32)
        original=(r>0)&(r>=low)&(r<=high)
        valid=(d>0)&(d>=low)&(d<=high)&(c>0)
        weights=np.where(valid,c,0)
        median=confidence_weighted_median(d,weights)
        dev=np.abs(d-median)
        inliers=valid&(dev<=30)
        iw=np.where(inliers,c,0)
        value=confidence_weighted_median(d,iw)
        count=inliers.sum(axis=0); orig=(inliers&original).sum(axis=0); repairs=(inliers&~original).sum(axis=0)
        purity=iw.sum(axis=0)/np.maximum(weights.sum(axis=0),1e-8)
        mad=confidence_weighted_median(dev,weights)
        selected_original=np.any(inliers&original&(d==value),axis=0)
        # The median's source confidence is capped by provenance. More repaired
        # frames can increase support, never turn an estimate into a measurement.
        base=np.max(np.where(inliers&(d==value),c,0),axis=0)
        base=np.minimum(base,iw.sum(axis=0)/np.maximum(count,1))
        base=np.where(selected_original,base,np.minimum(base,.3))
        stability=(count/n)*purity*np.exp(-mad/15)
        required=1 if n==1 else max(3,int(np.ceil(.6*n)))
        accepted=(count>=required)&(purity>=.8)&(mad<=15)&~motion[sl]
        # With any real observations present, require real temporal support to
        # approve a measured surface; repairs cannot outvote a contradictory depth.
        any_real=np.any(original,axis=0)
        accepted &= ~any_real | ((orig>=required)&selected_original)
        if n==1: accepted=(count==1)&~motion[sl]
        mean=(weights*d).sum(axis=0)/np.maximum(weights.sum(axis=0),1e-8)
        variance=(weights*(d-mean)**2).sum(axis=0)/np.maximum(weights.sum(axis=0),1e-8)
        out[sl]=np.where(accepted,value,0).astype(np.uint16)
        final[sl]=np.where(accepted,base*stability,0)
        for name,a in (('source_count',count),('original_support_count',orig),('filled_support_count',repairs),
                       ('temporal_variance',variance),('temporal_stability',stability),
                       ('depth_quality_confidence',base),('selected_original',selected_original)):
            maps[name][sl]=a
        removed+=int(np.any(valid&~inliers,axis=0).sum())
    report=dict(frame_count=n,before_invalid_rate=float(((raw[0]==0)|(raw[0]<low)|(raw[0]>high)).mean()),
                after_invalid_rate=float((out==0).mean()),stable_pixel_rate=float((final>=.8).mean()),
                removed_outlier_pixels=removed,average_confidence=float(final[out>0].mean()) if out.any() else 0,
                confidence_formula='min(selected source confidence, mean inlier confidence) * support fraction * inlier weight ratio * exp(-MAD/15)',
                point_count=int((out>0).sum()),single_frame_stability_evidence=n>1)
    return out,final,maps,report


def build_temporal_keyframe(source, progress=lambda text:None):
    source=Path(source).resolve(); started=perf_counter()
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('parameters',{}).get('fixed_camera') is not True: raise ValueError('Fixed camera required')
    records=[json.loads(l) for l in (source/'frames.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    n=len(records)
    if n not in (1,5,10,15): raise ValueError('Incomplete quality capture')
    stamps=[r['camera_timestamp_us'] for r in records]
    if any(b<=a for a,b in zip(stamps,stamps[1:])): raise ValueError('Duplicate/nonmonotonic frames')
    if any(abs(r.get('rgb_depth_timestamp_delta_us',0))>5000 for r in records): raise ValueError('RGB-D sync >5ms')
    if len({json.dumps(r.get('aligned_intrinsics'),sort_keys=True) for r in records})>1: raise ValueError('Calibration changed')
    paths=[_child(source,r[k]) for r in records for k in ('rgb','depth')]
    paths += [source/'manifest.json',source/'frames.jsonl',*sorted((source/'calibration').glob('*.json'))]
    hashes={str(p.relative_to(source)):_hash(p) for p in paths}
    low,high=np.asarray(manifest['camera']['depth_range_m'])*1000
    raw=[imread_unicode(_child(source,r['depth']),cv2.IMREAD_UNCHANGED) for r in records]
    rgb=[imread_unicode(_child(source,r['rgb']),cv2.IMREAD_COLOR) for r in records]
    intr=json.loads((source/'calibration/color_intrinsic.json').read_text(encoding='utf-8'))
    for r in records:
        recorded=r.get('aligned_intrinsics')
        if recorded is not None and any(recorded.get(k)!=intr[k] for k in ('width','height','fx','fy','cx','cy')):
            raise ValueError('Frame aligned intrinsics differ from saved calibration')
    shape=(intr['height'],intr['width'])
    if any(d.shape!=shape or d.dtype!=np.uint16 for d in raw) or any(c.shape!=(*shape,3) for c in rgb): raise ValueError('Calibration grid mismatch')
    output=source/'derived'/'temporal_fusion'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    progress('quality')
    filled=[]; confidence=[]
    for i,(d,c) in enumerate(zip(raw,rgb)):
        f,w,_,_=process_depth_quality(d,low,high,output/'per_frame'/f'{i:03d}',rgb_bgr=c)
        filled.append(f); confidence.append(w)
    progress('fusion')
    motion,check=assess_temporal_scene(raw,rgb,low,high)
    write_json(output/'motion_report.json',check)
    imwrite_unicode(output/'scene_motion_mask.png',motion.astype(np.uint8)*255)
    fused,weights,maps,report=fuse_quality_frames(raw,filled,confidence,low,high,motion)
    imwrite_unicode(output/'fused_depth_mm.png',fused)
    imwrite_unicode(output/'fused_confidence.png',np.rint(weights*255).astype(np.uint8))
    np.save(output/'final_confidence.npy',weights,allow_pickle=False)
    np.save(output/'fused_confidence.npy',weights,allow_pickle=False)
    for name,a in maps.items(): np.save(output/(name+'.npy'),a,allow_pickle=False)
    report['scene_motion_detected']=check['scene_motion_detected']
    report['motion']=check
    from femto.depth.temporal_depth_layer import TemporalDepthLayer
    layer=TemporalDepthLayer(fused,weights,maps['source_count'],maps['temporal_variance'],
                             maps['original_support_count'],maps['filled_support_count'],maps['selected_original']).validate()
    if not fused.any():
        report.update(status='rejected_no_trusted_depth',source_sha256=hashes)
        write_json(output/'temporal_report.json',report)
        raise ValueError(f'No trusted temporal depth remains; scene_motion_detected={check["scene_motion_detected"]}: {output}')
    # Reuse the existing keyframe selector for RGB selection and metadata. Its
    # copied raw depth remains a separate observation, never overwritten.
    from femto.temporal.scene_keyframe import build_fixed_keyframe_dataset
    dataset,kreport=build_fixed_keyframe_dataset(source,output/'keyframe')
    frame=json.loads((dataset/'frames.jsonl').read_text())
    raw_reference=frame['depth']
    shutil.copy2(output/'fused_depth_mm.png',dataset/'depth/fused_depth_mm.png')
    np.save(dataset/'final_confidence.npy',weights,allow_pickle=False)
    observed=np.where(maps['selected_original']&(fused>0),fused,0).astype(np.uint16)
    imwrite_unicode(dataset/'depth/temporal_observed_mm.png',observed)
    frame.update(depth='depth/fused_depth_mm.png',depth_confidence='final_confidence.npy',
                 original_depth='depth/temporal_observed_mm.png',reference_raw_depth=raw_reference)
    write_jsonl(dataset/'frames.jsonl', [frame], compact=False)
    meta=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    meta['temporal_fusion']=dict(version=1,report=str(output/'temporal_report.json'),confidence='final_confidence.npy',
                               source_sha256=hashes,reference_raw_depth=raw_reference,source_count=n)
    meta['files']['depth']=frame['depth']
    write_json(dataset/'manifest.json',meta)
    kreport['temporal_fusion']=meta['temporal_fusion']
    kreport['limitations']=['RGB is the selected reference frame; depth is a fixed-grid temporal estimate.',
                             'Original selected depth is retained separately; no independent geometry ground truth.',
                             'Only the fixed camera visible surfaces are available.']
    write_json(dataset/'keyframe_report.json',kreport)
    report.update(status='completed',source=str(source),source_sha256=hashes,
                  source_unchanged=all(_hash(source/p)==s for p,s in hashes.items()),
                  selected_rgb_source=next(r['rgb'] for r in records if r['frame']==kreport['selected_source_frame']),
                  elapsed_ms=(perf_counter()-started)*1000,coordinate_system='K6 camera RDF metres; fixed camera local identity')
    report.update(rate_unit='fraction 0..1',variance_unit='mm^2; all valid weighted samples',
                  confidence_storage='final_confidence.npy float32 is authoritative; PNG is rounded 0..255 visualization',
                  support_count_semantics='inliers within 30 mm, including rejected pixels for auditing',
                  temporal_observation_scope='selected real temporal measurements, not independent ground truth',
                  input_array_memory_mib=sum(a.nbytes for a in raw+rgb+filled+confidence)/2**20)
    if not report['source_unchanged']: raise RuntimeError('Source evidence changed')
    write_json(output/'temporal_report.json',report)
    progress('cloud')
    from types import SimpleNamespace
    from femto.geometry.depth_cloud import depth_cloud
    from femto.geometry.depth_confidence_cloud import write_confident_point_cloud, build_confident_depth_cloud
    selected_rgb=imread_unicode(dataset/frame['rgb'],cv2.IMREAD_COLOR)
    distortion=json.loads((dataset/'calibration/distortion.json').read_text())['rgb']
    cloud,c=build_confident_depth_cloud(raw[0],fused,weights,selected_rgb,SimpleNamespace(**intr),distortion,
                                       low,high,temporal_layer=layer)
    write_confident_point_cloud(output/'fused_pointcloud.ply',cloud,c,temporal_confidence=True)
    report['elapsed_ms']=(perf_counter()-started)*1000
    report['average_processing_ms_per_frame']=report['elapsed_ms']/n
    write_json(output/'temporal_report.json',report)
    return dataset,report

