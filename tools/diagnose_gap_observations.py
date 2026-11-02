"""Trace fixed missing-surface pixels through accepted raw depth and recorded poses."""
import argparse
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from femto.io_utils import load_json,load_jsonl,imread_unicode,write_json_strict
from femto.geometry.rgbd_frames import distortion_maps,load_poses
from femto.local.local_temporal_support import filtered_depth


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();audit=args.audit.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError('Use a new diagnostic directory')
    output.mkdir(parents=True,exist_ok=False)
    report=load_json(audit/'report.json');source=Path(report['source'])
    recorded=load_json(source/'reconstruction_report.json');dataset=Path(recorded['source_dataset'])
    frames=load_jsonl(dataset/'frames.jsonl');poses=load_poses(dataset,frames)
    mx,my,k,width,height=distortion_maps(dataset)
    with np.load(audit/'gap_pixels.npz') as data:
        pixels=data['pixels_xy'].copy();depth=data['reference_depth_m'].copy();cause=data['cause'].copy()
    expected=np.asarray(report['view_reference']['undistorted_intrinsic'])
    if not np.array_equal(k,expected):raise ValueError('Audit and accepted-frame reference calibration differ')
    xyz=np.c_[(pixels-[k[0,2],k[1,2]])/[k[0,0],k[1,1]]*depth[:,None],depth]
    parameters=recorded['parameters'];minimum,maximum=parameters['depth_min_m'],parameters['depth_max_m']
    support=np.zeros(len(depth),np.uint16);conflict=np.zeros(len(depth),np.uint16);rows=[]
    source_hashes={};arrays={}
    for ordinal,(frame,pose) in enumerate(zip(frames,poses)):
        path=(dataset/frame['depth']).resolve();pose_path=(dataset/frame['pose']).resolve()
        source_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        source_hashes[str(pose_path)]=hashlib.sha256(pose_path.read_bytes()).hexdigest()
        for current in (path,pose_path):
            previous=report['source_hashes'].get(str(current))
            if previous is None or previous!=source_hashes[str(current)]:
                raise ValueError('Accepted depth or pose differs from the stage audit source')
        raw=imread_unicode(path,cv2.IMREAD_UNCHANGED)
        if raw.dtype!=np.uint16 or raw.shape!=(height,width):raise ValueError('Invalid source depth')
        prepared=filtered_depth(dataset,frame,raw)
        prepared=cv2.remap(prepared,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        inverse=np.linalg.inv(pose);camera=xyz@inverse[:3,:3].T+inverse[:3,3];z=camera[:,2]
        pixel=np.rint(camera[:,:2]/np.maximum(z[:,None],1e-12)*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]).astype(np.int64)
        inside=(z>0)&(pixel[:,0]>=0)&(pixel[:,0]<width)&(pixel[:,1]>=0)&(pixel[:,1]<height)
        observed=prepared[np.clip(pixel[:,1],0,height-1),np.clip(pixel[:,0],0,width-1)]/1000.
        valid=inside&(observed>=minimum)&(observed<=maximum)
        residual=np.where(valid,(observed-z)*1000,np.nan)
        agrees=valid&(np.abs(residual)<=10.)
        differs=valid&~agrees;support+=agrees;conflict+=differs
        arrays[f'residual_mm_{ordinal}']=residual
        rows.append(dict(frame_id=frame['frame'],in_view_pixels=int(inside.sum()),
            prepared_depth_valid_pixels=int(valid.sum()),reference_point_consistent_pixels=int(agrees.sum()),
            conflicting_pixels=int(differs.sum()),camera_to_world=pose.tolist()))
    summary=[]
    names={0:'no_hit_at_tsdf_extraction',1:'topology_cleanup',2:'initial_component_filter',3:'measured_support_filter',4:'final_component_filter'}
    for code,name in names.items():
        selected=cause==code
        summary.append(dict(cause=name,pixels=int(selected.sum()),
            accepted_view_agreement_histogram={str(n):int((selected&(support==n)).sum()) for n in range(len(frames)+1)},
            pixels_with_conflicting_view=int((selected&(conflict>0)).sum())))
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in source_hashes.items())
    np.savez_compressed(output/'observation_evidence.npz',pixels_xy=pixels,cause=cause,
        consistent_accepted_views=support,conflicting_accepted_views=conflict,**arrays)
    result=dict(schema='femto-gap-observation-trace-v1',source_audit=str(audit),source_dataset=str(dataset),
        accepted_frame_count=len(frames),frames=rows,stage_groups=summary,source_hashes=source_hashes,
        source_unchanged=unchanged,depth_consistency_gate_mm=10,
        interpretation='recorded_pose_depth_consistency_of_reference_points_not_independent_pose_accuracy_or_TSDF_voxel_weight',
        next_steps='Separate extraction-scale effects from inconsistent observations; do not restore rejected triangles solely because point samples agree.',
        execution_ready=False,physical_accuracy_verified=False)
    write_json_strict(output/'report.json',result)
    print(summary,flush=True)
    if not unchanged:raise RuntimeError('Input depth or pose changed')


if __name__=='__main__':main()
