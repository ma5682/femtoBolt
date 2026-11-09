"""Deterministic ray-cast curved-surface fixture for offline RGB-D experiments."""
from pathlib import Path

import numpy as np
import open3d as o3d

from femto.geometry.organized_surface import point_map
from femto.io_utils import write_json_strict,write_jsonl,imwrite_unicode


def generate(output, *, width=320,height=240,frames=3):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if width<128 or height<96 or not 1<=frames<=8:raise ValueError('Invalid synthetic fixture size')
    for child in ('rgb','depth','pose','calibration','ground_truth'):(output/child).mkdir(parents=True,exist_ok=False)
    k=np.array([[width*.9,0,(width-1)/2],[0,width*.9,(height-1)/2],[0,0,1.]])
    y,x=np.indices((height,width),dtype=float)
    depth=700+40*np.sin(x/23)+30*np.cos(y/19)+15*np.sin((x+y)/17)
    points=point_map(depth,k).reshape(-1,3)
    ids=np.arange(width*height).reshape(height,width)
    a,b,c,d=ids[:-1,:-1].ravel(),ids[:-1,1:].ravel(),ids[1:,:-1].ravel(),ids[1:,1:].ravel()
    faces=np.r_[np.c_[a,c,b],np.c_[b,c,d]]
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(points),o3d.utility.Vector3iVector(faces))
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rows=[];truth=[]
    for index in range(frames):
        actual=np.eye(4);actual[:3,3]=[(index%3)*.008,(index//3)*.008,0]
        rays=scene.create_rays_pinhole(o3d.core.Tensor(k),o3d.core.Tensor(np.linalg.inv(actual)),width,height)
        answer=scene.cast_rays(rays);distance=answer['t_hit'].numpy()
        valid=np.isfinite(distance)
        observed=np.where(valid,np.rint(distance*1000),0).astype(np.uint16)
        ray=rays.numpy().astype(float)
        world=ray[...,:3]+ray[...,3:]*np.where(valid,distance,0)[...,None]
        color=np.stack((.5+.3*np.sin(world[...,0]*60),.5+.3*np.cos(world[...,1]*65),
                        .5+.3*np.sin((world[...,0]+world[...,1])*40)),axis=-1)
        bgr=np.rint(np.clip(color[...,::-1],0,1)*255).astype(np.uint8);bgr[~valid]=0
        online=actual.copy()
        if index:online[:3,3]+=[.0015,-.0008,.0005]
        stem=f'{index:06d}'
        imwrite_unicode(output/f'rgb/{stem}.png',bgr);imwrite_unicode(output/f'depth/{stem}.png',observed)
        write_json_strict(output/f'pose/{stem}.json',dict(pose_valid=True,T_WC=online.tolist(),pose_source='synthetic_perturbed'))
        rows.append(dict(frame=index,rgb=f'rgb/{stem}.png',depth=f'depth/{stem}.png',pose=f'pose/{stem}.json'))
        truth.append(actual.tolist())
    write_jsonl(output/'frames.jsonl',rows)
    write_json_strict(output/'calibration/color_intrinsic.json',dict(width=width,height=height,
        fx=k[0,0],fy=k[1,1],cx=k[0,2],cy=k[1,2]))
    write_json_strict(output/'calibration/distortion.json',dict(rgb=dict(model='BROWN_CONRADY_K6',
        **dict.fromkeys(('k1','k2','p1','p2','k3','k4','k5','k6'),0.))))
    write_json_strict(output/'manifest.json',dict(schema='femto-full-scene-rgbd-v1',frame_count=frames,
        synthetic=True,pose=dict(valid=True,source='synthetic_perturbed'),
        local_capture=dict(measured_only=True,session_id='organized_fixture',reference_frame_id=0,
                           local_workspace_radius_m=.03,local_workspace_rotation_deg=5)))
    write_json_strict(output/'pose_quality_report.json',dict(synthetic=True,limitations=['Ray-cast fixture only.']))
    write_json_strict(output/'ground_truth/poses.json',dict(T_WC=truth,note='Ray-cast observations; not real patient data.'))
    return output
