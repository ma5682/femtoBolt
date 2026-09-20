"""Replay/synthetic temporal comparisons; never label synthetic data as captures."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/ scripts import the femto package from the repository root
from femto.depth.depth_quality import assess_and_fill_depth
from femto.io_utils import imread_unicode as _read, write_json as json_dump
from femto.temporal.temporal_keyframe import fuse_quality_frames, assess_temporal_scene
from femto.capture.scene_segmented_capture import depth_cloud


def compare_batches(depths, images, intr, distortion, low, high, modes, *, repeats):
    results=[]
    for n in modes:
        batches=[]; clouds=[]; fused_results=[]; started=perf_counter()
        for batch in range(repeats):
            raw=depths[batch*n:(batch+1)*n]; rgb=images[batch*n:(batch+1)*n]
            if len(raw)!=n: raise ValueError('Distinct frames required; no frame repetition')
            filled=[]; confidences=[]
            for d,c in zip(raw,rgb):
                f,_,w,_=assess_and_fill_depth(d,low,high,rgb_bgr=c)
                filled.append(f); confidences.append(w)
            motion,check=assess_temporal_scene(raw,rgb,low,high)
            fused,c,m,report=fuse_quality_frames(raw,filled,confidences,low,high,motion)
            holes=cv2.connectedComponents((fused==0).astype(np.uint8),connectivity=8)[0]-1
            cloud,_=depth_cloud(fused,rgb[0],SimpleNamespace(**intr),distortion,low,high,
                                 confidence=c,temporal_confidence=True)
            clouds.append(cloud); fused_results.append(fused)
            batches.append(dict(invalid_rate=report['after_invalid_rate'],point_count=report['point_count'],
                                average_confidence=report['average_confidence'],hole_components=holes,
                                scene_motion_detected=check['scene_motion_detected']))
        chamfers=[]; differences=[]; edge_differences=[]
        for a,b,da,db in zip(clouds,clouds[1:],fused_results,fused_results[1:]):
            if len(a.points) and len(b.points):
                chamfers.append(500*(np.mean(a.compute_point_cloud_distance(b))+np.mean(b.compute_point_cloud_distance(a))))
            common=(da>0)&(db>0)
            edge=cv2.dilate((np.abs(np.diff(depths[0].astype(float),axis=1,append=depths[0][:,-1:]))>50).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            diff=np.abs(da.astype(float)-db.astype(float))
            if common.any(): differences.append(float(diff[common].mean()))
            if (common&edge).any(): edge_differences.append(float(diff[common&edge].mean()))
        results.append(dict(frame_count=n,batch_count=repeats,batches=batches,
                            mean_invalid_rate=float(np.mean([b['invalid_rate'] for b in batches])),
                            mean_point_count=float(np.mean([b['point_count'] for b in batches])),
                            mean_confidence=float(np.mean([b['average_confidence'] for b in batches])),
                            repeated_chamfer_mm=chamfers,repeated_common_depth_mae_mm=differences,
                            repeated_edge_depth_mae_mm=edge_differences,
                            average_batch_ms=(perf_counter()-started)*1000/repeats,
                            array_memory_mib=sum(a.nbytes for a in raw+rgb+filled+confidences)/2**20))
    return results


def run(source,output):
    rng=np.random.default_rng(20260910); shape=(192,256)
    truth=np.full(shape,1000,np.uint16); truth[:,128:]=1400
    rgb=rng.integers(0,256,(*shape,3),np.uint8)
    depths=[]
    for i in range(30):
        d=(truth.astype(int)+rng.integers(-4,5,shape)).astype(np.uint16)
        d[rng.random(shape)<.015]=0
        d[10:18,10:18]=0  # Persistent transparent-style no return.
        d[30:38,30:38]=700 if i%2 else 1700  # Unstable reflection-style region.
        depths.append(d)
    intr=dict(width=256,height=192,fx=200,fy=200,cx=128,cy=96)
    distortion=dict(model='BROWN_CONRADY_K6',**{k:0 for k in ('k1','k2','k3','k4','k5','k6','p1','p2')})
    report=dict(synthetic=compare_batches(depths,[rgb]*30,intr,distortion,250,2210,(1,5,10),repeats=3),
                scope='Synthetic material-like patterns are not physical material validation; Chamfer is symmetric mean nearest distance without ICP.',
                memory_scope='RGB/depth/quality array memory only; not full process peak RSS',
                live_fps_validation='pending camera-attached test',tsdf_weight_validation='deferred phase 2')
    if source:
        source=Path(source)
        frames=[json.loads(l) for l in (source/'frames.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        raw=[_read(source/r['depth'],cv2.IMREAD_UNCHANGED) for r in frames]
        rgb=[_read(source/r['rgb'],cv2.IMREAD_COLOR) for r in frames]
        intr=json.loads((source/'calibration/color_intrinsic.json').read_text(encoding='utf-8'))
        dist=json.loads((source/'calibration/distortion.json').read_text(encoding='utf-8'))['rgb']
        manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
        low,high=np.asarray(manifest['camera']['depth_range_m'])*1000
        modes=tuple(n for n in (1,5,10) if n<=len(raw))
        report['real_source']=str(source.resolve())
        report['real']=compare_batches(raw,rgb,intr,dist,low,high,modes,repeats=1)
        report['real_repeated_capture_validation']='pending independent bursts; source has '+str(len(raw))+' frames'
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    json_dump(output/'comparison.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path); parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(); result=run(args.source,args.output)
    print(json.dumps(result,ensure_ascii=True,indent=2))
