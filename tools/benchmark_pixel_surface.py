"""Compare a saved pre-change mesh/local-surface implementation with current code.

Run from the repository root. Baseline Python files must be explicitly saved
before editing, under --baseline with their repository-relative paths. Inputs
are synthetic RGB-D; timings exclude capture, publication and route planning.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
from tempfile import TemporaryDirectory
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from femto.geometry import single_frame_mesh as current_mesh
from femto.pipeline import local_single_surface as current_surface


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def arrays(mesh):
    return [np.asarray(getattr(mesh,key)) for key in ('vertices','triangles','vertex_colors','vertex_normals')]


def equivalent(old,new):
    for left,right in zip(arrays(old[0])+[old[1]],arrays(new[0])+[new[1]]):
        np.testing.assert_array_equal(left,right)
    for key,value in old[2].items():
        if new[2][key]!=value:raise AssertionError(f'Report changed: {key}')


def median_pair(before,after,repeats):
    # Warm imports/caches once, then alternate order to limit temporal bias.
    before();after();times=[[],[]]
    for repeat in range(repeats):
        for index in ((0,1) if repeat%2==0 else (1,0)):
            gc.collect();start=time.perf_counter()
            result=(before,after)[index]()
            times[index].append(time.perf_counter()-start);del result
    return [round(statistics.median(rows)*1000,3) for rows in times]


def peak(call):
    gc.collect();tracemalloc.start()
    result=call();_,maximum=tracemalloc.get_traced_memory()
    tracemalloc.stop();del result
    return round(maximum/1024**2,2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    old_mesh=load(args.baseline/'femto/geometry/single_frame_mesh.py','baseline_pixel_mesh')
    old_surface=load(args.baseline/'femto/pipeline/local_single_surface.py','baseline_local_surface')
    old_surface.build_pixel_mesh=old_mesh.build_pixel_mesh
    h,w=720,1280
    y,x=np.indices((h,w),dtype=np.float32)
    depth=(700+(x//80)%6+(y//90)%4).astype(np.uint16)
    color=np.stack(((x*13)%255,(y*17)%255,((x+y)*7)%255),axis=-1).astype(np.uint8)
    intrinsic=SimpleNamespace(width=w,height=h,fx=900.,fy=910.,cx=(w-1)/2,cy=(h-1)/2)
    k=np.array([[intrinsic.fx,0,intrinsic.cx],[0,intrinsic.fy,intrinsic.cy],[0,0,1.]])
    dist=dict(model='BROWN_CONRADY_K6',**dict.fromkeys(('k1','k2','p1','p2','k3','k4','k5','k6'),0.))
    pose=np.eye(4);pose[:3,:3]=cv2.Rodrigues(np.array([.03,-.04,.1]))[0];pose[:3,3]=[.01,.02,.03]
    rows=[]
    with TemporaryDirectory() as temp:
        for name,height,width in (('small_region',120,160),('medium_region',360,480),('full_image',h,w)):
            mask=np.zeros((h,w),bool);top=(h-height)//2;left=(w-width)//2
            mask[top:top+height,left:left+width]=True
            # Interior hole, depth discontinuity, and one disconnected observation.
            mask[h//2-4:h//2+4,w//2-4:w//2+4]=False
            mask[3,3]=True
            measured=np.where(mask,depth,0).astype(np.uint16)
            measured[h//2:h//2+8,w//2+12:w//2+16]+=np.uint16(80)
            # Reapply mask after the synthetic step.
            measured[~mask]=0
            old=old_mesh.build_pixel_mesh(measured,color,intrinsic,dist,pose)
            new=current_mesh.build_pixel_mesh(measured,color,intrinsic,dist,pose)
            equivalent(old,new)
            item=dict(name=name,image_size=[w,h],valid_pixels=int(np.count_nonzero(measured)),
                      mesh_arrays_and_original_report_fields_identical=True,
                      full_image_candidates=old[2]['candidate_triangles'],
                      generated_candidates=new[2]['generated_candidate_triangles'])
            del old,new
            before=lambda:old_mesh.build_pixel_mesh(measured,color,intrinsic,dist,pose)
            after=lambda:current_mesh.build_pixel_mesh(measured,color,intrinsic,dist,pose)
            item['mesh_ms_before_after']=median_pair(before,after,args.repeats)
            item['mesh_python_numpy_peak_MiB_before_after']=[peak(before),peak(after)]
            filter_=SimpleNamespace(filter=lambda d,*_:np.where(measured>0,d,0).astype(np.uint16))
            old_path=Path(temp)/'before';new_path=Path(temp)/'after'
            old_path.mkdir(exist_ok=True);new_path.mkdir(exist_ok=True)
            before=lambda:old_surface.prepare_surface(color,depth,(x,y,k),pose,.25,2.21,filter_,0,old_path)
            after=lambda:current_surface.prepare_surface(color,depth,(x,y,k),pose,.25,2.21,filter_,0,new_path)
            old=before();new=after()
            for a,b in zip(arrays(old[1]),arrays(new[1])):np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(old[3],new[3])
            np.testing.assert_array_equal(old[0].colors,new[0].colors)
            np.testing.assert_allclose(old[0].points,new[0].points,rtol=0,atol=1.2e-7)
            item['max_cloud_coordinate_difference_m']=float(np.max(np.abs(np.asarray(old[0].points)-np.asarray(new[0].points))))
            del old,new
            item['local_preparation_ms_before_after']=median_pair(before,after,args.repeats)
            rows.append(item);print(json.dumps(item),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(schema='pixel-surface-comparison-v1',repeats=args.repeats,
        scope='Synthetic local pixel preparation only; no fusion, publication, normals of cloud, route or save-plan timing',
        memory_scope='tracemalloc Python/NumPy peak; excludes untracked native allocations',results=rows),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
