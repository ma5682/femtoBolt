"""Persist reproducible synthetic material-pattern comparison and timing, not physical acceptance."""
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter
import json

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # tools/ scripts import repository-root modules
from io_utils import imwrite_unicode, write_json as json_dump
from temporal_depth_fusion import fuse_temporal_depth


def run():
    output=REPO_ROOT/'outputs'/('temporal_validation_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir()
    results=[]
    for index,name in enumerate(('white_table','computer_shell','black_object','phone_glass','transparent_object')):
        rng=np.random.default_rng(20260909+index)
        h,w=192,256
        yy,xx=np.indices((h,w)); truth=(1000+xx//20).astype(np.uint16)
        depths=[]
        for i in range(15):
            d=np.clip(truth.astype(np.int32)+rng.integers(-3,4,(h,w)),0,65535).astype(np.uint16)
            d[rng.random((h,w))<(.04 if name=='black_object' else .01)]=0
            if name=='phone_glass': d[60:120,80:170]=800 if i%2 else 1600
            if name=='transparent_object': d[60:120,80:170]=0
            depths.append(d)
        panels=[]
        for n in (1,5,10,15):
            if n==1:
                fused=depths[0]; confidence=(fused>0).astype(np.float32)
                report=dict(frame_count=1,after_temporal_invalid_rate=float((fused==0).mean()),point_count=int((fused>0).sum()),
                            mean_final_confidence=float(confidence[fused>0].mean()),fusion_ms=0)
            else:
                fused,confidence,_,report=fuse_temporal_depth(depths[:n],250,2210)
            compare=fused>0
            if name in ('phone_glass','transparent_object'): compare[60:120,80:170]=False
            report['known_synthetic_surface_mae_mm']=float(np.abs(fused[compare].astype(float)-truth[compare]).mean())
            report['fixture']=name
            if n>1 and name in ('phone_glass','transparent_object'):
                assert not fused[60:120,80:170].any()
            results.append(report)
            image=np.zeros((h,w,3),np.uint8)
            image[:]=(0,0,230); image[fused>0]=(0,200,0)
            image[(confidence>0)&(confidence<.8)]=(0,220,220)
            image=cv2.copyMakeBorder(image,25,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(image,f'{n} frames | {report["after_temporal_invalid_rate"]:.1%} invalid',(4,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            panels.append(image)
        imwrite_unicode(output/(name+'.png'),np.concatenate(panels,axis=1))
    timing=[]
    for n in (5,10,15):
        rng=np.random.default_rng(7)
        depths=[]
        for i in range(n):
            depth=np.full((1080,1920),1000,np.uint16)
            depth[rng.random(depth.shape)<.0074]=0
            depths.append(depth)
        times=[]
        for repeat in range(3):
            _,_,_,r=fuse_temporal_depth(depths,250,2210)
            times.append(r['fusion_ms'])
        timing.append(dict(frame_count=n,mean_fusion_ms=float(np.mean(times)),samples_ms=times,
                           depth_input_bytes=sum(d.nbytes for d in depths)))
    try:
        import psutil
        memory=psutil.Process().memory_info()
        memory_report=dict(process_peak_working_set_bytes=getattr(memory,'peak_wset',None),process_rss_bytes=memory.rss)
    except ImportError:
        memory_report=dict(process_peak_working_set_bytes=None)
    json_dump(output/'validation.json',dict(data_kind='synthetic',physical_material_acceptance=False,memory=memory_report,
               comparisons=results,performance=timing,performance_scope='fusion only; no SDK, RGB motion, per-frame quality, disk IO, XYZ'))
    print(output)
    print(json.dumps(timing,indent=2))
    return output


if __name__=='__main__': run()
