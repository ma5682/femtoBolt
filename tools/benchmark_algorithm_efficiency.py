"""Alternating offline comparisons against the saved pre-change Python sources."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import open3d as o3d

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from femto.local import local_pose_refinement as current_metrics
from femto.geometry.local_region import RegionDepthFilter
from femto.planning import target_surface
from femto.planning.mesh_slice_planner import _resample
from femto.planning.route_sampling import decision, CURVATURE_POLICY


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def paired(before,after,repeats):
    before();after();times={'before':[],'after':[]}
    for i in range(repeats):
        for name,call in ([('before',before),('after',after)] if i%2==0 else [('after',after),('before',before)]):
            start=time.perf_counter();call();times[name].append(time.perf_counter()-start)
    return dict(samples_s=times,median_s={name:statistics.median(values) for name,values in times.items()})


def run(baseline,repeats):
    old_metrics=load('old_metric_benchmark',baseline/'femto/local/local_pose_refinement.py')
    old_region=load('old_region_benchmark',baseline/'femto/geometry/local_region.py')
    old_target=load('old_target_benchmark',baseline/'femto/planning/target_surface.py')
    rng=np.random.default_rng(241102);points=rng.normal(0,.06,(10000,3));points[:,2]+=.6
    colors=rng.random(points.shape)
    source=o3d.geometry.PointCloud();source.points=o3d.utility.Vector3dVector(points);source.colors=o3d.utility.Vector3dVector(colors)
    target=o3d.geometry.PointCloud(source);target.translate([.001,-.002,0])
    transforms=[np.eye(4) for _ in range(4)]
    for i,t in enumerate(transforms):t[0,3]=i*.0005
    def metrics(module):
        index=(module.MetricIndex(target) if hasattr(module,'MetricIndex') else o3d.geometry.KDTreeFlann(target))
        return [module.metrics(source,target,t,_target_tree=index) for t in transforms]
    assert metrics(old_metrics)==metrics(current_metrics)
    report={'schema':'algorithm-efficiency-comparison-v1','synthetic':True,'repeats':repeats,
            'metrics_10000_points_4_queries_index_included':paired(lambda:metrics(old_metrics),lambda:metrics(current_metrics),repeats)}
    h,w=720,1280
    refdepth=rng.integers(250,2200,(h,w),dtype=np.uint16);refdepth[300:320,500:530]=0
    k=np.array([[900.,0,w/2+.25],[0,900.,h/2+.25],[0,0,1.]])
    args=dict(reference=SimpleNamespace(depth=refdepth,k=k,pose=np.eye(4)),
              minimum=.25,maximum=2.21,distance=rng.random((h,w)).astype(np.float32)*160,
              record={'support_margin_mm':40.})
    old=old_region.RegionDepthFilter.__new__(old_region.RegionDepthFilter);old.__dict__.update(args,metrics=[])
    new=RegionDepthFilter.__new__(RegionDepthFilter);new.__dict__.update(args,metrics=[])
    a=lambda:old.filter(refdepth,np.eye(4),k,0)
    b=lambda:new.filter(refdepth,np.eye(4),k,0)
    assert np.array_equal(a(),b());report['reference_filter_1280x720']=paired(a,b,repeats)
    polygons=[]
    for i in range(32):
        x=i/32;uv=np.array([[x,0],[x+.4/32,0],[x+.4/32,1],[x,1.]])
        polygons.append((i,None,np.c_[uv,np.ones(4)],uv))
    def occlusion(module):
        return module._resolve_pixel_occlusion(polygons,0,0,0,{'pair_checks':0},module.TargetBudget(),
            np.zeros((32,3,3)),np.eye(3),lambda a,b:False,lambda:None)
    assert occlusion(old_target)==occlusion(target_surface)
    report['occlusion_32_separated_clipped_polygons']=paired(lambda:occlusion(old_target),lambda:occlusion(target_surface),repeats)
    import trimesh
    mesh=trimesh.Trimesh(vertices=[[0,0,.5],[.045,0,.5],[.045,.045,.5],[0,.045,.5]],faces=[[0,1,2],[0,2,3]],process=False)
    old_decision=decision(mesh);new_decision=decision(mesh,CURVATURE_POLICY)
    samples={}
    for name,points in [('smooth',np.array([[0,0,0],[.02,0,0],[.04,0,0]])),
                        ('bend',np.array([[0,0,0],[.02,0,0],[.02,.02,0]]))]:
        old_sample=_resample(points,[0,1],old_decision['sample_step_mm']/1000,preserve_face_crossings=True,face_local_spacing=True)
        new_sample=_resample(points,[0,1],new_decision['sample_step_mm']/1000,preserve_face_crossings=True,face_local_spacing=True,sampling_decision=new_decision)
        samples[name]={'before_points':len(old_sample),'after_points':len(new_sample)}
    report['sampling']=samples
    report['checks']={'metric_values_exact':True,'reference_pixels_exact':True,'occlusion_outcome_and_pair_count_exact':True,
                      'raw_files_used':False,'route_sampling_changes_version':True}
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(current_metrics.__file__),Path(target_surface.__file__),ROOT/'femto/geometry/local_region.py',ROOT/'femto/planning/route_sampling.py']}
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=5);args=parser.parse_args()
    if not 1<=args.repeats<=20:parser.error('repeats must be 1..20')
    result=run(args.baseline.resolve(),args.repeats)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v.get('median_s',v) for k,v in result.items() if isinstance(v,dict) and k not in ('source_sha256',)},ensure_ascii=False))
