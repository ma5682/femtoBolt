"""Independent fixed-camera capture/replay mode; does not modify legacy model markers."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from time import perf_counter, monotonic
from types import SimpleNamespace

import cv2
import numpy as np

from scene_capture import imwrite_unicode, json_dump, write_calibration
from scene_static_fusion import _read
from scene_pose_estimator import assess_rgb_motion_pair
from temporal_depth_fusion import fuse_temporal_depth
from depth_quality import process_depth_quality, DepthQualityConfig
from scene_segmented_capture import depth_cloud
from depth_confidence_cloud import write_confident_point_cloud


def _child(root, name):
    path = (root/name).resolve(); path.relative_to(root.resolve())
    return path


def _hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def audit_temporal_motion(images):
    """Background flow gate plus conservative full-resolution local RGB mask."""
    pairs = list(dict.fromkeys([(i-1,i) for i in range(1,len(images))]+[(0,i) for i in range(1,len(images))]))
    reports = []
    mask = np.zeros(images[0].shape[:2],bool)
    for a,b in pairs:
        first,last = images[a],images[b]
        report = assess_rgb_motion_pair(first,last)
        reports.append(dict(pair=[a,b],**report))
        delta = np.max(np.abs(first.astype(np.int16)-last.astype(np.int16)),axis=2)
        # Conservative exclusion also covers moving foreground missed by the
        # background flow. Exposure changes may reject a static scene safely.
        mask |= cv2.dilate((delta>35).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    passed = all(p.get('reliable') and p.get('appears_fixed') and
                 p.get('median_background_displacement_px',float('inf'))<=1.0 for p in reports)
    return mask,dict(passed=passed,pairs=reports,local_rgb_motion_rate=float(mask.mean()),
                     maximum_background_displacement_px=1.0,
                     limitation='Flow and RGB difference are motion hints, not proof of static transparent surfaces')


def build_temporal_dataset(source: Path, *, frame_count=5, start=0, mode='fast'):
    started = perf_counter(); source=Path(source).resolve()
    if frame_count not in (5,10,15) or mode not in ('fast','quality') or start<0:
        raise ValueError('Choose 5/10/15 frames, fast/quality, nonnegative start')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema')!='femto-full-scene-rgbd-v1' or manifest.get('parameters',{}).get('fixed_camera') is not True:
        raise ValueError('Temporal fusion requires an explicitly fixed-camera dataset')
    records=[json.loads(line) for line in (source/'frames.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    frames=records[start:start+frame_count]
    if len(frames)!=frame_count:
        raise ValueError('Not enough distinct captured frames')
    if len({f['depth'] for f in frames})!=frame_count or len({f['rgb'] for f in frames})!=frame_count:
        raise ValueError('Repeated source files cannot increase temporal confidence')
    timestamps=[f.get('camera_timestamp_us') for f in frames]
    if any(t is None for t in timestamps) or any(b<=a for a,b in zip(timestamps,timestamps[1:])):
        raise ValueError('Distinct increasing camera timestamps are required')
    if any(abs(f.get('rgb_depth_timestamp_delta_us',0))>5000 for f in frames):
        raise ValueError('RGB-D synchronization exceeds 5 ms')
    calibration_ids=[json.dumps(f.get('aligned_intrinsics'),sort_keys=True) for f in frames]
    if len(set(calibration_ids))>1:
        raise ValueError('Aligned calibration changed within the burst')
    depth_paths=[_child(source,f['depth']) for f in frames]
    rgb_paths=[_child(source,f['rgb']) for f in frames]
    evidence=[source/'manifest.json',source/'frames.jsonl',*depth_paths,*rgb_paths,*sorted((source/'calibration').glob('*.json'))]
    hashes={str(p.relative_to(source)):_hash(p) for p in evidence}
    intr=SimpleNamespace(**json.loads((source/'calibration/color_intrinsic.json').read_text(encoding='utf-8')))
    distortion=json.loads((source/'calibration/distortion.json').read_text(encoding='utf-8'))['rgb']
    low,high=np.asarray(manifest['camera']['depth_range_m'])*1000
    depths=[_read(p,cv2.IMREAD_UNCHANGED) for p in depth_paths]
    images=[_read(p,cv2.IMREAD_COLOR) for p in rgb_paths]
    shape=(intr.height,intr.width)
    if any(d.shape!=shape or d.dtype!=np.uint16 for d in depths) or any(c.shape!=(*shape,3) for c in images):
        raise ValueError('RGB-D/calibration grids changed')
    output=source/'derived'/('temporal_fusion_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    motion_mask,motion=audit_temporal_motion(images)
    imwrite_unicode(output/'motion_mask.png',motion_mask.astype(np.uint8)*255)
    json_dump(output/'motion_report.json',motion)
    if not motion['passed']:
        json_dump(output/'temporal_report.json',dict(status='rejected_motion_or_insufficient_static_evidence',
                  frame_count=frame_count,source=str(source),motion=motion,source_sha256=hashes,
                  source_unchanged=all(_hash(source/p)==sha for p,sha in hashes.items())))
        return output
    per_frame=[]
    for i,(depth,color) in enumerate(zip(depths,images)):
        _,_,quality,paths=process_depth_quality(depth,low,high,output/'per_frame'/f'{i:03d}',
                                              rgb_bgr=color,config=DepthQualityConfig(mode=mode))
        per_frame.append(dict(frame=frames[i]['frame'],quality_report=paths['quality_report'],
                              original_invalid_rate=quality['original_invalid_rate'],
                              repaired_pixels=quality['repaired_pixels'],processing_ms=quality['processing_ms']))
    fused,confidence,maps,report=fuse_temporal_depth(depths,low,high,motion_mask=motion_mask)
    # Reference RGB is a single explicit original frame; motion-affected pixels
    # are excluded from geometry. Never mix colors from different time instants.
    reference=images[0]
    cloud,weights=depth_cloud(fused,reference,intr,distortion,low,high,
                              confidence=confidence,temporal_confidence=True)
    write_confident_point_cloud(output/'fused_pointcloud.ply',cloud,weights,temporal_confidence=True)
    imwrite_unicode(output/'fused_depth_mm.png',fused)
    np.save(output/'temporal_confidence.npy',confidence,allow_pickle=False)
    imwrite_unicode(output/'temporal_confidence.png',np.rint(confidence*255).astype(np.uint8))
    for name,array in maps.items(): np.save(output/(name+'.npy'),array,allow_pickle=False)
    debug=np.full((*shape,3),(30,30,230),np.uint8)
    debug[confidence>0]=(0,200,200); debug[confidence>=.8]=(0,220,0)
    imwrite_unicode(output/'temporal_debug.png',debug)
    raw_preview=reference.copy(); raw_preview[(depths[0]==0)|(depths[0]<low)|(depths[0]>high)]=(0,0,255)
    fused_preview=reference.copy(); fused_preview[fused==0]=(0,0,255)
    comparison=np.concatenate([cv2.resize(a,(640,360),interpolation=cv2.INTER_NEAREST) for a in (raw_preview,fused_preview)],axis=1)
    imwrite_unicode(output/'comparison.png',comparison)
    unchanged=all(_hash(source/p)==sha for p,sha in hashes.items())
    report.update(status='completed',source=str(source),source_sha256=hashes,source_unchanged=unchanged,
                  frame_ordinals=list(range(start,start+frame_count)),timestamps_us=timestamps,
                  selected_rgb_source=str(rgb_paths[0]),per_frame=per_frame,motion=motion,
                  coordinate_system='K6 color camera RDF metres; no world pose created',
                  mode=mode,elapsed_ms=(perf_counter()-started)*1000,
                  input_buffer_bytes=sum(d.nbytes+c.nbytes for d,c in zip(depths,images)),
                  downstream_policy='Independent temporal PLY only; continuous confidence not automatically passed to legacy TSDF/3DGS',
                  memory_note='input_buffer_bytes is measured array storage, not process peak RSS')
    # Windows process peak working set includes imports and other allocations.
    try:
        import psutil
        memory=psutil.Process().memory_info()
        report['process_peak_working_set_bytes']=getattr(memory,'peak_wset',None)
        report['process_rss_bytes']=memory.rss
    except ImportError:
        report['process_peak_working_set_bytes']=None
    json_dump(output/'temporal_report.json',report)
    if not unchanged: raise RuntimeError('Original evidence changed')
    return output


def capture_temporal_frames(root: Path, frame_count=5):
    """Bounded RAM burst; quality and image encoding run AFTER camera closes."""
    from orbbec_rgbd import OrbbecAlignedCamera
    if frame_count not in (5,10,15): raise ValueError('Choose 5/10/15 frames')
    output=Path(root)/('temporal_capture_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    frames=[]; frame_intrinsics=[]
    with OrbbecAlignedCamera(depth_mode='wide_detail',color_size=(1920,1080),fps=15) as camera:
        deadline=monotonic()+20
        last_timestamp=None
        while len(frames)<frame_count and monotonic()<deadline:
            frame=camera.read(1000)
            if frame is None or (last_timestamp is not None and frame.timestamp_us<=last_timestamp): continue
            # Own copies: SDK buffers may be reused on the next read.
            frame.depth_mm=frame.depth_mm.copy(); frame.color_bgr=frame.color_bgr.copy()
            frames.append(frame); last_timestamp=frame.timestamp_us
            frame_intrinsics.append(camera.intrinsics.to_dict())
        intrinsics=camera.intrinsics
        metadata=dict(camera.metadata)
        write_calibration(output,camera,intrinsics)
    records=[]
    for i,frame in enumerate(frames):
        rgb=f'raw/frame_{i+1:03d}_rgb.png'; depth=f'raw/frame_{i+1:03d}_depth_mm.png'
        imwrite_unicode(output/rgb,frame.color_bgr)
        imwrite_unicode(output/depth,np.rint(np.clip(frame.depth_mm,0,65535)).astype(np.uint16))
        records.append(dict(frame=i,rgb=rgb,depth=depth,camera_timestamp_us=frame.timestamp_us,
                            color_camera_timestamp_us=frame.color_timestamp_us,depth_camera_timestamp_us=frame.depth_timestamp_us,
                            rgb_depth_timestamp_delta_us=frame.color_timestamp_us-frame.depth_timestamp_us,
                            camera_frame_index=frame.frame_index,frame_metadata=frame.frame_metadata,
                            aligned_intrinsics=frame_intrinsics[i],pose_valid=False))
    manifest=dict(schema='femto-full-scene-rgbd-v1',frame_count=len(frames),requested_frame_count=frame_count,camera=metadata,
                  status='captured' if len(frames)==frame_count else 'incomplete_capture',
                  parameters=dict(fixed_camera=True,processing_mode='temporal',depth_min_mm=250,depth_max_mm=2210),
                  pose=dict(valid=False,source='not_estimated; fixed-camera temporal pixels only'),
                  capture_policy='bounded burst; no quality computation/PNG encoding in frame read loop')
    json_dump(output/'manifest.json',manifest)
    json_dump(output/'metadata.json',dict(**manifest,timestamps_us=[f.timestamp_us for f in frames]))
    (output/'frames.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    if len(frames)!=frame_count: raise RuntimeError(f'Capture timeout; partial raw evidence saved at {output}')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',nargs='?',type=Path)
    parser.add_argument('--capture',action='store_true',help='Camera and scene must stay fixed; close other camera owners first')
    parser.add_argument('--mode',choices=('fast','quality'),default='fast')
    parser.add_argument('--frames',type=int,choices=(5,10,15))
    parser.add_argument('--start',type=int,default=0)
    args=parser.parse_args(); count=args.frames or (5 if args.mode=='fast' else 15)
    if args.capture and args.source: parser.error('Use source replay OR --capture')
    if not args.capture and args.source is None: parser.error('Provide a fixed-camera dataset or --capture')
    source=capture_temporal_frames(Path(__file__).resolve().parent/'datasets',count) if args.capture else args.source
    output=build_temporal_dataset(source,frame_count=count,start=args.start,mode=args.mode)
    print(output)
    report=json.loads((output/'temporal_report.json').read_text(encoding='utf-8'))
    if report['status']!='completed': raise SystemExit('Temporal fusion rejected: see motion_report.json')


if __name__=='__main__': main()
