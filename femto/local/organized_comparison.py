"""Read-only comparison of organized RGB-D candidates against current ICP.

This runner never publishes a model, rewrites poses, changes latest indices, or
supplies planning evidence. Every registration method uses the same spatially
disjoint fit/check blocks and the existing production adoption checks.
"""
from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
import open3d as o3d

from femto.geometry.organized_surface import (VERSION,point_map,integral_normals,edge_labels,
    normal_space_indices,MISSING_BOUNDARY,OCCLUDING,OCCLUDED,HIGH_CURVATURE)
from femto.geometry.rgbd_frames import distortion_maps,prepare_rgbd_arrays,load_poses
from femto.io_utils import imread_unicode,load_json_object,load_jsonl,sha256_file,write_json_strict
from femto.local.local_pose_refinement import solve,metrics,stable_shape,approve
from femto.local.organized_registration import solve_projective
from femto.tracking.moving_quality import rotation_deg

SCHEMA='organized-rgbd-comparison-v1'


class ComparisonCancelled(Exception):
    pass


def _cloud(points,colors, *, voxel=None,cap=60000):
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.colors=o3d.utility.Vector3dVector(colors)
    if voxel:cloud=cloud.voxel_down_sample(voxel)
    return cloud.uniform_down_sample(max(1,(len(cloud.points)+cap-1)//cap))


def _angular(a,b):
    good=np.isfinite(a).all(axis=-1)&np.isfinite(b).all(axis=-1)
    if not good.any():return dict(count=0,median_deg=None,p95_deg=None)
    angle=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',a[good],b[good]),-1,1)))
    return dict(count=int(good.sum()),median_deg=float(np.median(angle)),p95_deg=float(np.percentile(angle,95)))


def analytic_normal_comparison(width=320,height=240):
    """Plane/sphere ground truth with explicitly injected 0.2 mm Gaussian noise."""
    k=np.array([[float(width),0,(width-1)/2],[0,float(width),(height-1)/2],[0,0,1.]])
    y,x=np.indices((height,width),dtype=float)
    u=(x-k[0,2])/k[0,0];v=(y-k[1,2])/k[1,1]
    plane=.7/(1-.2*u+.1*v)
    plane_normal=np.array([.2,-.1,-1.]);plane_normal/=np.linalg.norm(plane_normal)
    denominator=1+u*u+v*v;discriminant=1.05**2-denominator*(1.05**2-.4**2)
    sphere=np.where(discriminant>0,(1.05-np.sqrt(np.maximum(discriminant,0)))/denominator,np.nan)
    rows=[]
    for name,depth in (('plane',plane),('sphere',sphere)):
        actual=point_map(depth*1000,k)
        truth=(np.broadcast_to(plane_normal,actual.shape).copy() if name=='plane'
               else (actual-np.array([0,0,1.05]))/.4)
        runs=[];previous=None
        for seed in (10,11,12):
            measured=depth*1000+np.random.default_rng(seed).normal(0,.2,depth.shape)
            points=point_map(measured,k);valid=np.isfinite(points).all(axis=-1)
            started=time.perf_counter();normal,good=integral_normals(points);elapsed=time.perf_counter()-started
            cloud=_cloud(points[valid],np.zeros((int(valid.sum()),3)),cap=int(valid.sum())+1)
            started=time.perf_counter()
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.02,max_nn=30))
            cloud.orient_normals_towards_camera_location([0,0,0]);kd_elapsed=time.perf_counter()-started
            kd=np.full_like(points,np.nan);kd[valid]=np.asarray(cloud.normals)
            runs.append(dict(seed=seed,valid_normal_fraction=float(good.sum()/max(1,valid.sum())),
                integral_error_vs_truth=_angular(normal,truth),kd_error_vs_truth=_angular(kd,truth),
                kd_error_on_integral_support=_angular(kd[good],truth[good]),
                integral_noise_repeat_difference=_angular(normal,previous) if previous is not None else None,
                integral_s=elapsed,kd_s=kd_elapsed))
            previous=normal
        rows.append(dict(surface=name,width=width,height=height,injected_noise_sigma_mm=.2,runs=runs))
    return rows


def _prepare(dataset,frame,maps,output,scale,check):
    check();started=time.perf_counter()
    color=imread_unicode(dataset/frame['rgb'],cv2.IMREAD_COLOR)
    depth=imread_unicode(dataset/frame['depth'],cv2.IMREAD_UNCHANGED)
    if depth.dtype!=np.uint16:raise ValueError('Expected raw uint16 millimetre depth')
    mx,my,k,w,h=maps
    if depth.shape!=(h,w) or color.shape!=(h,w,3):raise ValueError('RGB-D/calibration shape mismatch')
    bgr,measured,scaled=prepare_rgbd_arrays(color,depth,mx,my,k,scale,.25,2.21)
    points=point_map(measured,scaled);valid=np.isfinite(points).all(axis=-1)
    colors=bgr[...,::-1]/255.
    decode_s=time.perf_counter()-started
    started=time.perf_counter();normal,good=integral_normals(points);integral_s=time.perf_counter()-started
    cloud=_cloud(points[valid],colors[valid],cap=int(valid.sum())+1)
    started=time.perf_counter()
    if len(cloud.points):
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.02,max_nn=30))
        cloud.orient_normals_towards_camera_location([0,0,0])
    kd_s=time.perf_counter()-started
    kd=np.full_like(points,np.nan);kd[valid]=np.asarray(cloud.normals)
    started=time.perf_counter();labels=edge_labels(points,normal);edge_s=time.perf_counter()-started
    # Reserve full spatial blocks, not merely alternating points whose local
    # normal support would silently reuse the check sample. No fit normal can
    # span missing/check blocks because integral_normals requires complete windows.
    y,x=np.indices(valid.shape);fit=valid&(((x//32+y//32)%2)==0);held=valid&~fit
    fit_points=points.copy();fit_points[~fit]=np.nan
    started=time.perf_counter();fit_normal,fit_good=integral_normals(fit_points)
    fit_normal_s=time.perf_counter()-started
    fit_cloud=_cloud(points[fit],colors[fit],voxel=.003)
    check_cloud=_cloud(points[held],colors[held],voxel=.005,cap=10000)
    candidates=fit_points[fit_good];candidate_colors=colors[fit_good]
    started=time.perf_counter();sample=normal_space_indices(fit_normal[fit_good],6000)
    sampled=_cloud(candidates[sample],candidate_colors[sample])
    sampling_s=time.perf_counter()-started
    uniform=candidates[np.linspace(0,len(candidates)-1,min(12000,len(candidates)),dtype=int)] if len(candidates) else candidates
    artifact=output/f'frame_{int(frame["frame"]):06d}_diagnostics.npz'
    np.savez_compressed(artifact,normals=normal,normal_valid=good,edge_labels=labels,
        fit_mask=fit,held_out_mask=held)
    row=dict(frame=frame['frame'],image_size=[measured.shape[1],measured.shape[0]],valid_points=int(valid.sum()),
        normal_valid_points=int(good.sum()),normal_coverage=float(good.sum()/max(1,valid.sum())),
        normal_difference_vs_kd=_angular(normal[valid],kd[valid]),
        timing_s=dict(decode_calibrate_backproject=decode_s,integral_normals=integral_s,kd_normals=kd_s,
                      edge_classification=edge_s,fit_integral_normals=fit_normal_s,normal_space_sampling=sampling_s),
        edges={name:int(np.count_nonzero(labels&flag)) for name,flag in
            (('missing_boundary',MISSING_BOUNDARY),('occluding',OCCLUDING),('occluded',OCCLUDED),('angular_bend',HIGH_CURVATURE))},
        fit_points=len(fit_cloud.points),held_out_points=len(check_cloud.points),sampled_points=len(sampled.points),
        artifact=str(artifact),artifact_sha256=sha256_file(artifact))
    return dict(points=fit_points,normals=fit_normal,k=scaled,fit=fit_cloud,held=check_cloud,
                sampled=sampled,projective_source=uniform),row


def _pair(method,source,target,initial,local,check):
    check();started=time.perf_counter()
    row=dict(method=method,would_pass_existing_gate=False)
    if not stable_shape(source['fit']) or not stable_shape(target['fit']):
        row.update(reasons=['insufficient_3d_structure'],elapsed_s=time.perf_counter()-started);return row
    try:
        if method=='projective':
            forward,forward_detail=solve_projective(source['projective_source'],target['points'],target['normals'],target['k'],initial,check=check)
            reverse,reverse_detail=solve_projective(target['projective_source'],source['points'],source['normals'],source['k'],np.linalg.inv(initial),check=check)
            row['solver_details']=dict(forward=forward_detail,reverse=reverse_detail)
        else:
            key='fit' if method=='baseline_icp' else 'sampled'
            forward=solve(source[key],target[key],initial)
            check();reverse=solve(target[key],source[key],np.linalg.inv(initial))
        solve_s=time.perf_counter()-started
        before=dict(forward=metrics(source['held'],target['held'],initial),
                    reverse=metrics(target['held'],source['held'],np.linalg.inv(initial)))
        after=dict(forward=metrics(source['held'],target['held'],forward),
                   reverse=metrics(target['held'],source['held'],np.linalg.inv(forward)))
        reasons=approve(initial,forward,reverse,before,after,local.get('local_workspace_radius_m'))
        if (local.get('local_workspace_rotation_deg') is not None
                and rotation_deg(forward)>float(local['local_workspace_rotation_deg'])+1e-9):
            reasons.append('outside_local_rotation')
        row.update(candidate_T_reference_source=forward.tolist(),before=before,after=after,
                   reasons=sorted(set(reasons)),would_pass_existing_gate=not reasons,
                   solve_s=solve_s,validation_s=time.perf_counter()-started-solve_s)
    except (ValueError,RuntimeError) as error:
        row.update(reasons=[str(error)])
    row['elapsed_s']=time.perf_counter()-started
    return row


def compare(dataset,output, *, scale=.5,check=lambda:None,progress=lambda text:None):
    original_check=check
    def check():
        try:original_check()
        except Exception as error:raise ComparisonCancelled(str(error)) from error
    dataset=Path(dataset).resolve();output=Path(output).resolve()
    if output==dataset or dataset in output.parents:raise ValueError('Comparison output must be outside the source dataset')
    if output.exists():raise FileExistsError(output)
    if not .25<=scale<=1:raise ValueError('Experiment scale must be between .25 and 1')
    manifest=load_json_object(dataset/'manifest.json');frames=load_jsonl(dataset/'frames.jsonl')
    if not 1<=len(frames)<=20:raise ValueError('Experiment needs 1 to 20 posed RGB-D frames')
    poses=load_poses(dataset,frames)
    evidence={dataset/'manifest.json',dataset/'frames.jsonl'}
    evidence.update(p for p in (dataset/'calibration').iterdir() if p.is_file())
    evidence.update((dataset/frame[key]).resolve() for frame in frames for key in ('rgb','depth','pose'))
    source_hashes={str(p):sha256_file(p) for p in evidence}
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema=SCHEMA,algorithm_version=VERSION,source_dataset=str(dataset),source_sha256=source_hashes,
        state='running',published=False,default_pipeline_changed=False,scale=scale,frames=[],pairs=[],
        fit_validation_split='disjoint 32x32 image blocks; fit normal windows cannot span held-out pixels',
        parameters=dict(normal_radius_px=3,normal_depth_gate_m=.02,edge_depth_gate_m=.03,angular_bend_deg=25,
                        normal_space_samples=6000,projective_source_cap=12000,depth_range_m=[.25,2.21]),
        limitations=['Internal sample consistency; no physical/clinical/robot accuracy certification.',
                     'Normal difference versus KD is a comparison, not ground truth.',
                     'Experimental split differs from production alternating-point split.',
                     'No pose/model/route is adopted or published by this runner.'])
    path=output/'comparison.json'
    write_json_strict(path,report)
    try:
        views=[];maps=distortion_maps(dataset)
        for frame in frames:
            progress(f'prepare frame {frame["frame"]}')
            view,row=_prepare(dataset,frame,maps,output,scale,check);views.append(view);report['frames'].append(row)
            write_json_strict(path,report)
        local=manifest.get('local_capture',{})
        for i in range(1,len(frames)):
            initial=np.linalg.inv(poses[0])@poses[i]
            pair=dict(frame=frames[i]['frame'],initial_T_reference_source=initial.tolist(),methods=[])
            for method in ('baseline_icp','projective','normal_space_icp'):
                progress(f'{method}: frame {frames[i]["frame"]}')
                pair['methods'].append(_pair(method,views[i],views[0],initial,local,check))
            report['pairs'].append(pair);write_json_strict(path,report)
        report['all_view_decisions']={method:dict(
            would_adopt_all_views=bool(report['pairs']) and all(next(r for r in p['methods'] if r['method']==method)['would_pass_existing_gate'] for p in report['pairs']),
            failed_view_keeps_entire_original_pose_set=True,published=False)
            for method in ('baseline_icp','projective','normal_space_icp')}
        report['source_unchanged']=all(sha256_file(Path(p))==digest for p,digest in source_hashes.items())
        if not report['source_unchanged']:raise ValueError('Source changed during experiment')
        report['state']='complete';write_json_strict(path,report)
        return report
    except Exception as error:
        report.update(state='cancelled' if isinstance(error,ComparisonCancelled) else 'failed',error=str(error))
        write_json_strict(path,report);raise
