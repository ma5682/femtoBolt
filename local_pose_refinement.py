"""Conservative offline refinement against fixed W0; immutable source dataset."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import open3d as o3d
import cv2
from io_utils import imread_unicode as _read_image
from scene_reconstruction import _distortion_maps,_prepare_rgbd,_load_poses
from scene_dataset_validator import validate_pose
from moving_quality import rotation_deg,cycle_consistency


def metrics(source,target,transform):
    moved=copy.deepcopy(source).transform(transform)
    tree=o3d.geometry.KDTreeFlann(target)
    points=np.asarray(moved.points);target_colors=np.asarray(target.colors)
    colors=np.asarray(moved.colors);dist=[];errors=[]
    for i,p in enumerate(points):
        count,ids,d=tree.search_knn_vector_3d(p,1)
        if count:
            dist.append(np.sqrt(d[0]))
            if d[0] <= .015**2 and len(colors)==len(points) and len(target_colors)==len(target.points):
                errors.append(float(np.mean(np.abs(colors[i]-target_colors[ids[0]]))))
    d=np.asarray(dist)
    return dict(median_m=float(np.median(d)) if len(d) else None,
                p90_m=float(np.percentile(d,90)) if len(d) else None,
                overlap=float(np.mean(d<=.015)) if len(d) else 0,
                color_error=float(np.median(errors)) if errors else None)


def stable_shape(cloud):
    if len(cloud.points)<100:return False
    eigen=np.linalg.eigvalsh(np.cov(np.asarray(cloud.points).T))
    return bool(eigen[-1]>0 and eigen[0]/eigen[-1]>=.002)


def noise_floor(cloud):
    """Same-frame disjoint-sample consistency; still includes shared sensor noise."""
    a=cloud.select_by_index(list(range(0,len(cloud.points),2)))
    b=cloud.select_by_index(list(range(1,len(cloud.points),2)))
    return metrics(a,b,np.eye(4)) if len(a.points) and len(b.points) else {}


def solve(source,target,initial):
    transform=initial.copy()
    for voxel,gate in ((.012,.03),(.006,.015),(.003,.008)):
        a=source.voxel_down_sample(voxel);b=target.voxel_down_sample(voxel)
        b.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3,max_nn=40))
        loss=o3d.pipelines.registration.TukeyLoss(k=gate/2)
        result=o3d.pipelines.registration.registration_icp(a,b,gate,transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
        transform=result.transformation
        if validate_pose(transform):raise ValueError('Invalid refined transform')
    return transform


def approve(initial,candidate,reverse,before,after,radius=None):
    reasons=[]
    if validate_pose(candidate):return ['invalid_pose']
    correction=np.linalg.inv(initial)@candidate
    if np.linalg.norm(correction[:3,3])>.005 or rotation_deg(correction)>1:reasons.append('correction_exceeds_5mm_or_1deg')
    cycle=cycle_consistency(candidate,reverse)
    if not cycle['passed'] or np.linalg.norm((candidate@reverse)[:3,3])>.002 or rotation_deg(candidate@reverse)>.3:
        reasons.append('bidirectional_disagreement')
    if radius is not None and np.linalg.norm(candidate[:3,3])>radius:reasons.append('outside_workspace')
    for direction in ('forward','reverse'):
        a,b=before[direction],after[direction]
        if any(v is None or not np.isfinite(v) for v in list(a.values())+list(b.values())):
            reasons.append('missing_validation');continue
        if b['overlap']<.6 or b['overlap']<a['overlap']-.01:reasons.append('overlap_degraded')
        if b['median_m']>a['median_m']*.95:reasons.append('insufficient_heldout_improvement')
        if b['p90_m']>a['p90_m']+.0005:reasons.append('tail_error_degraded')
        if b['color_error']>a['color_error']+.01:reasons.append('color_alignment_degraded')
    return sorted(set(reasons))


def refine(dataset,output):
    dataset,output=Path(dataset).resolve(),Path(output).resolve()
    if output==dataset or dataset in output.parents:raise ValueError('Output must be outside source dataset')
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8-sig'))
    local=manifest.get('local_capture',{})
    if not local.get('measured_only'):raise ValueError('Only local observed-depth datasets supported')
    frames=[json.loads(s) for s in (dataset/'frames.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    poses=_load_poses(dataset,frames)
    if not np.allclose(poses[0],np.eye(4),rtol=0,atol=1e-9):raise ValueError('Reference must remain identity')
    if len(frames)>20:
        report=dict(schema='femto-local-refinement-v1',accepted=False,reason='more_than_20_views_budget',source_dataset=str(dataset),frames=[])
        (output/'refinement_report.json').write_text(json.dumps(report),encoding='utf-8')
        return dataset,report
    evidence=set(dataset/'calibration'/p.name for p in (dataset/'calibration').iterdir() if p.is_file())
    evidence.update([dataset/'manifest.json',dataset/'frames.jsonl'])
    for f in frames:evidence.update((dataset/f[k]).resolve() for k in ('rgb','depth','pose'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}
    mx,my,k,_,_=_distortion_maps(dataset)
    clouds=[];checks=[]
    for frame in frames:
        rgb=_read_image(dataset/frame['rgb'],cv2.IMREAD_COLOR);depth=_read_image(dataset/frame['depth'],cv2.IMREAD_UNCHANGED)
        rgbd,intrinsic,_,_=_prepare_rgbd(rgb,depth,mx,my,k,.5,.25,2.21)
        full=o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,intrinsic)
        # Disjoint point samples for fitting and comparison; both still share sensor noise.
        fit=full.select_by_index(list(range(0,len(full.points),2))).voxel_down_sample(.003)
        check=full.select_by_index(list(range(1,len(full.points),2))).voxel_down_sample(.005)
        clouds.append(fit.uniform_down_sample(max(1,(len(fit.points)+59999)//60000)))
        checks.append(check.uniform_down_sample(max(1,(len(check.points)+9999)//10000)))
    rows=[];newposes=[p.copy() for p in poses]
    for i in range(1,len(frames)):
        row=dict(frame=frames[i]['frame'],accepted=False,reasons=[])
        if not stable_shape(clouds[0]) or not stable_shape(clouds[i]):row['reasons']=['insufficient_3d_structure']
        else:
            try:
                t=solve(clouds[i],clouds[0],poses[i]);reverse=solve(clouds[0],clouds[i],np.linalg.inv(poses[i]))
                before=dict(forward=metrics(checks[i],checks[0],poses[i]),reverse=metrics(checks[0],checks[i],np.linalg.inv(poses[i])))
                after=dict(forward=metrics(checks[i],checks[0],t),reverse=metrics(checks[0],checks[i],np.linalg.inv(t)))
                reasons=approve(poses[i],t,reverse,before,after,local.get('local_workspace_radius_m'))
                if (local.get('local_workspace_rotation_deg') is not None and
                        rotation_deg(t)>float(local['local_workspace_rotation_deg'])+1e-9):
                    reasons.append('outside_local_rotation')
                correction=np.linalg.inv(poses[i])@t
                floors=dict(source=noise_floor(checks[i]),target=noise_floor(checks[0]))
                diagnostic=[]
                if 'insufficient_heldout_improvement' in reasons:
                    medians=[value.get('median_m') for value in floors.values() if value.get('median_m') is not None]
                    after_medians=[value.get('median_m') for value in after.values() if value.get('median_m') is not None]
                    if medians and after_medians and min(after_medians)-max(medians)<.001:
                        diagnostic.append('heldout_at_noise_floor')
                    else:diagnostic.append('insufficient_heldout_improvement_above_noise_floor')
                row.update(before=before,after=after,candidate_T_WC=t.tolist(),reasons=sorted(set(reasons)),
                    diagnostic_reasons=diagnostic,noise_floor=floors,
                    online_pose_translation_mm=float(np.linalg.norm(poses[i][:3,3])*1000),
                    online_pose_rotation_deg=rotation_deg(poses[i]),
                    correction_mm=float(np.linalg.norm(correction[:3,3])*1000),
                    correction_deg=rotation_deg(correction),
                    only_correction_limit_rejected=(set(reasons)=={'correction_exceeds_5mm_or_1deg'}),accepted=not reasons)
                if not reasons:newposes[i]=t
            except (ValueError,RuntimeError) as exc:row['reasons']=[str(exc)]
        rows.append(row)
    # Publish only an all-view consistent candidate; no mixture of optimized and old views.
    accepted=bool(rows) and all(r['accepted'] for r in rows)
    report=dict(schema='femto-local-refinement-v1',source_dataset=str(dataset),accepted=accepted,
        reference_fixed=True,method='multiscale robust point-to-plane ICP to first view',frames=rows,
        source_sha256=hashes,limitations=['Internal held-out sample consistency only, not physical accuracy.',
        'No temporal filtering, pose graph or relocalization. Online dataset retained when any view fails.'])
    report['adoption_policy']='all_views_must_pass; one rejected frame keeps the entire online pose set'
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report['source_unchanged']=unchanged
    if not unchanged:raise ValueError('Source changed during refinement')
    report_path=output/'refinement_report.json'
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if not accepted:return dataset,report
    derived=output/'dataset';derived.mkdir();(derived/'pose').mkdir();shutil.copytree(dataset/'calibration',derived/'calibration')
    import os
    for i,frame in enumerate(frames):
        oldpose=dataset/frame['pose'];record=json.loads(oldpose.read_text(encoding='utf-8-sig'))
        record['T_WC']=newposes[i].tolist()
        if i:record.update(pose_source='offline_robust_icp_research_grade',initial_pose_source=record.get('pose_source'),initial_T_WC=poses[i].tolist())
        frame['pose']=f'pose/{i:06}.json'
        (derived/frame['pose']).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        for key in ('rgb','depth'):frame[key]=os.path.relpath((dataset/frame[key]).resolve(),derived).replace('\\','/')
    (derived/'frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames),encoding='utf-8')
    manifest['local_capture']['refinement_report']=str(report_path)
    manifest['pose']=dict(valid=True,source='offline_robust_icp_research_grade')
    manifest['refinement_source_dataset']=str(dataset)
    (derived/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    quality=json.loads((dataset/'pose_quality_report.json').read_text(encoding='utf-8-sig'))
    from scene_to_nerfstudio import pose_diversity
    quality['translation_span_m'],quality['rotation_span_deg']=pose_diversity(newposes)
    quality['refinement_report']=str(report_path)
    quality['limitations']=quality.get('limitations',[])+report['limitations']
    (derived/'pose_quality_report.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding='utf-8')
    return derived,report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('output');a=p.parse_args()
    dataset,report=refine(a.dataset,a.output);print(json.dumps(dict(dataset=str(dataset),accepted=report['accepted'],frames=report['frames']),ensure_ascii=False,indent=2))
