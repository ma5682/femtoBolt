"""Conservative offline refinement against fixed W0; immutable source dataset."""
from __future__ import annotations
from typing import Any

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import open3d as o3d
import cv2
from femto.io_utils import imread_unicode, write_json, write_json_strict, write_jsonl
from femto.geometry.rgbd_frames import distortion_maps, prepare_rgbd, load_poses
from femto.dataset.scene_dataset_validator import validate_pose
from femto.tracking.moving_quality import rotation_deg,cycle_consistency
from femto.operation_diagnostics import diagnosed


class MetricIndex:
    """Per-operation float64 batch index; legacy tie ordering stays authoritative."""
    def __init__(self, target, legacy=None):
        self.target, self.legacy = target, legacy
        points = np.asarray(target.points)
        self.count = len(points)
        self.index = None
        if self.count and np.isfinite(points).all():
            self.index = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(points))
            self.index.knn_index()

    def query(self, points):
        if not self.count or not len(points):
            return np.empty(0, np.int64), np.empty(0), np.empty(0, np.int64)
        if self.index is None or not np.isfinite(points).all():
            uncertain = np.arange(len(points))
            ids = np.full(len(points), -1, np.int64); squared = np.full(len(points), np.inf)
        else:
            neighbors, distances = self.index.knn_search(o3d.core.Tensor(points), min(2, self.count))
            ids = neighbors.numpy()[:, 0].copy(); squared = distances.numpy()[:, 0].copy()
            # A different equal-distance point can change the color gate. Use
            # the original tree for ties and numerically indistinguishable pairs.
            uncertain = (np.flatnonzero(distances.numpy()[:, 1]-squared <=
                         32*np.finfo(float).eps*np.maximum(1., squared))
                         if self.count > 1 else np.empty(0, np.int64))
        if len(uncertain):
            if self.legacy is None:self.legacy = o3d.geometry.KDTreeFlann(self.target)
            for i in uncertain:
                count, old_ids, old_d = self.legacy.search_knn_vector_3d(points[i], 1)
                if count:ids[i], squared[i] = old_ids[0], old_d[0]
        valid = np.flatnonzero(ids >= 0)
        return ids[valid], squared[valid], valid


def metrics(source: o3d.geometry.PointCloud,target: o3d.geometry.PointCloud,transform: np.ndarray,
            *, _target_tree=None) -> dict[str, float | None]:
    xyz = np.asarray(source.points)
    # Match Eigen's scalar expression order, including homogeneous division,
    # without copying colors/normals or allocating a second PointCloud.
    moved = xyz[:, 0, None]*transform[:3, 0]+xyz[:, 1, None]*transform[:3, 1]+xyz[:, 2, None]*transform[:3, 2]+transform[:3, 3]
    moved /= (xyz[:, 0]*transform[3, 0]+xyz[:, 1]*transform[3, 1]+xyz[:, 2]*transform[3, 2]+transform[3, 3])[:, None]
    tree = _target_tree if isinstance(_target_tree, MetricIndex) else MetricIndex(target, _target_tree)
    nearest, squared, rows = tree.query(moved)
    colors, target_colors = np.asarray(source.colors), np.asarray(target.colors)
    has_colors = len(colors)==len(xyz) and len(target_colors)==len(target.points)
    pairs = squared <= .015**2
    errors = (np.mean(np.abs(colors[rows[pairs]]-target_colors[nearest[pairs]]),axis=1)
              if has_colors else np.empty(0,dtype=np.float64))
    d=np.sqrt(squared)
    return dict(median_m=float(np.median(d)) if len(d) else None,
                p90_m=float(np.percentile(d,90)) if len(d) else None,
                overlap=float(np.mean(d<=.015)) if len(d) else 0,
                color_error=float(np.median(errors)) if len(errors) else None)


def stable_shape(cloud: o3d.geometry.PointCloud) -> bool:
    if len(cloud.points)<100:return False
    eigen=np.linalg.eigvalsh(np.cov(np.asarray(cloud.points).T))
    return bool(eigen[-1]>0 and eigen[0]/eigen[-1]>=.002)


def noise_floor(cloud: o3d.geometry.PointCloud) -> dict[str, float | None]:
    """Same-frame disjoint-sample consistency; still includes shared sensor noise."""
    a=cloud.select_by_index(list(range(0,len(cloud.points),2)))
    b=cloud.select_by_index(list(range(1,len(cloud.points),2)))
    return metrics(a,b,np.eye(4)) if len(a.points) and len(b.points) else {}


def solve(source: o3d.geometry.PointCloud,target: o3d.geometry.PointCloud,initial: np.ndarray,
          *, _reference_scales=None, _reference_is_target=True, _other_scales=None) -> np.ndarray:
    transform=initial.copy()
    for voxel,gate in ((.012,.03),(.006,.015),(.003,.008)):
        if _reference_scales is None:
            a=source.voxel_down_sample(voxel);b=target.voxel_down_sample(voxel)
            b.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3,max_nn=40))
        elif _reference_is_target:
            a=(source.voxel_down_sample(voxel) if _other_scales is None else _other_scales[voxel])
            b=_reference_scales[voxel]
        else:
            a=_reference_scales[voxel]
            b=(target.voxel_down_sample(voxel) if _other_scales is None else _other_scales[voxel])
            b.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3,max_nn=40))
        loss=o3d.pipelines.registration.TukeyLoss(k=gate/2)
        result=o3d.pipelines.registration.registration_icp(a,b,gate,transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
        transform=result.transformation
        if validate_pose(transform):raise ValueError('Invalid refined transform')
    return transform


def approve(initial: np.ndarray,candidate: np.ndarray,reverse: np.ndarray,before: dict[str, Any],after: dict[str, Any],radius: float | None=None) -> list[str]:
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


@diagnosed('local.pose_refinement')
def refine(dataset: str | Path,output: str | Path) -> tuple[Path, dict[str, Any]]:
    dataset,output=Path(dataset).resolve(),Path(output).resolve()
    if output==dataset or dataset in output.parents:raise ValueError('Output must be outside source dataset')
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8-sig'))
    local=manifest.get('local_capture',{})
    if not local.get('measured_only'):raise ValueError('Only local observed-depth datasets supported')
    frames=[json.loads(s) for s in (dataset/'frames.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    poses=load_poses(dataset,frames)
    if not np.allclose(poses[0],np.eye(4),rtol=0,atol=1e-9):raise ValueError('Reference must remain identity')
    if len(frames)>20:
        report=dict(schema='femto-local-refinement-v1',accepted=False,reason='more_than_20_views_budget',source_dataset=str(dataset),frames=[])
        write_json(output/'refinement_report.json', report, indent=None, ensure_ascii=True)
        return dataset,report
    evidence=set(dataset/'calibration'/p.name for p in (dataset/'calibration').iterdir() if p.is_file())
    evidence.update([dataset/'manifest.json',dataset/'frames.jsonl'])
    for f in frames:evidence.update((dataset/f[k]).resolve() for k in ('rgb','depth','pose'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}
    mx,my,k,_,_=distortion_maps(dataset)
    clouds=[];checks=[]
    for frame in frames:
        rgb=imread_unicode(dataset/frame['rgb'],cv2.IMREAD_COLOR);depth=imread_unicode(dataset/frame['depth'],cv2.IMREAD_UNCHANGED)
        rgbd,intrinsic,_,_=prepare_rgbd(rgb,depth,mx,my,k,.5,.25,2.21)
        full=o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,intrinsic)
        # Disjoint point samples for fitting and comparison; both still share sensor noise.
        fit=full.select_by_index(list(range(0,len(full.points),2))).voxel_down_sample(.003)
        check=full.select_by_index(list(range(1,len(full.points),2))).voxel_down_sample(.005)
        clouds.append(fit.uniform_down_sample(max(1,(len(fit.points)+59999)//60000)))
        checks.append(check.uniform_down_sample(max(1,(len(check.points)+9999)//10000)))
    # Per-call preparation only: the fixed reference is never transformed by
    # ICP. All candidate/held-out metrics and adoption checks still run.
    reference_scales=None;reference_tree=None;reference_floor=None
    rows=[];newposes=[p.copy() for p in poses]
    for i in range(1,len(frames)):
        row=dict(frame=frames[i]['frame'],accepted=False,reasons=[])
        if not stable_shape(clouds[0]) or not stable_shape(clouds[i]):row['reasons']=['insufficient_3d_structure']
        else:
            try:
                if reference_scales is None:
                    prepared_scales={}
                    for voxel in (.012,.006,.003):
                        prepared=clouds[0].voxel_down_sample(voxel)
                        prepared.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3,max_nn=40))
                        prepared_scales[voxel]=prepared
                    reference_tree=MetricIndex(checks[0])
                    reference_scales=prepared_scales
                current_tree=MetricIndex(checks[i])
                # One view is the forward source and reverse target. Reuse its
                # identical voxel samples only within this bidirectional pair;
                # reverse target normals are still estimated by solve as before.
                current_scales={voxel:clouds[i].voxel_down_sample(voxel) for voxel in (.012,.006,.003)}
                t=solve(clouds[i],clouds[0],poses[i],_reference_scales=reference_scales,
                        _other_scales=current_scales)
                reverse=solve(clouds[0],clouds[i],np.linalg.inv(poses[i]),
                              _reference_scales=reference_scales,_reference_is_target=False,
                              _other_scales=current_scales)
                before=dict(forward=metrics(checks[i],checks[0],poses[i],_target_tree=reference_tree),
                            reverse=metrics(checks[0],checks[i],np.linalg.inv(poses[i]),_target_tree=current_tree))
                after=dict(forward=metrics(checks[i],checks[0],t,_target_tree=reference_tree),
                           reverse=metrics(checks[0],checks[i],np.linalg.inv(t),_target_tree=current_tree))
                reasons=approve(poses[i],t,reverse,before,after,local.get('local_workspace_radius_m'))
                if (local.get('local_workspace_rotation_deg') is not None and
                        rotation_deg(t)>float(local['local_workspace_rotation_deg'])+1e-9):
                    reasons.append('outside_local_rotation')
                correction=np.linalg.inv(poses[i])@t
                if reference_floor is None:reference_floor=noise_floor(checks[0])
                floors=dict(source=noise_floor(checks[i]),target=copy.deepcopy(reference_floor))
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
    write_json_strict(report_path, report)
    if not accepted:return dataset,report
    derived=output/'dataset';derived.mkdir();(derived/'pose').mkdir();shutil.copytree(dataset/'calibration',derived/'calibration')
    import os
    for i,frame in enumerate(frames):
        oldpose=dataset/frame['pose'];record=json.loads(oldpose.read_text(encoding='utf-8-sig'))
        record['T_WC']=newposes[i].tolist()
        if i:record.update(pose_source='offline_robust_icp_research_grade',initial_pose_source=record.get('pose_source'),initial_T_WC=poses[i].tolist())
        frame['pose']=f'pose/{i:06}.json'
        write_json(derived/frame['pose'], record)
        for key in ('rgb','depth'):frame[key]=os.path.relpath((dataset/frame[key]).resolve(),derived).replace('\\','/')
    write_jsonl(derived/'frames.jsonl', frames, ensure_ascii=True, compact=False)
    manifest['local_capture']['refinement_report']=str(report_path)
    manifest['pose']=dict(valid=True,source='offline_robust_icp_research_grade')
    manifest['refinement_source_dataset']=str(dataset)
    write_json(derived/'manifest.json', manifest)
    quality=json.loads((dataset/'pose_quality_report.json').read_text(encoding='utf-8-sig'))
    from femto.tracking.moving_quality import pose_diversity
    quality['translation_span_m'],quality['rotation_span_deg']=pose_diversity(newposes)
    quality['refinement_report']=str(report_path)
    quality['limitations']=quality.get('limitations',[])+report['limitations']
    write_json(derived/'pose_quality_report.json', quality)
    return derived,report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('output');a=p.parse_args()
    dataset,report=refine(a.dataset,a.output);print(json.dumps(dict(dataset=str(dataset),accepted=report['accepted'],frames=report['frames']),ensure_ascii=False,indent=2))
