"""Experimental projective point-to-plane registration in camera RDF metres.

An initial measured pose is required. These candidates must still pass the
production bidirectional/held-out checks; convergence alone is not acceptance.
"""
from __future__ import annotations

import cv2
import numpy as np

from femto.dataset.scene_dataset_validator import validate_pose

VERSION = 'projective-refinement-experiment-v1'


def correspondences(source, target, normals, k, transform, *, gate_m=.015):
    source=np.asarray(source,dtype=float)
    target=np.asarray(target,dtype=float);normals=np.asarray(normals,dtype=float)
    k=np.asarray(k,dtype=float);transform=np.asarray(transform,dtype=float)
    if (source.ndim!=2 or source.shape[1]!=3 or target.ndim!=3 or target.shape[-1]!=3
            or normals.shape!=target.shape or k.shape!=(3,3) or not np.isfinite(k).all()
            or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1])
            or k[0,1]!=0 or k[1,0]!=0 or validate_pose(transform)
            or not np.isfinite(gate_m) or gate_m<=0):
        raise ValueError('Invalid projective correspondence input')
    moved=source@transform[:3,:3].T+transform[:3,3]
    ids=np.flatnonzero(np.isfinite(moved).all(axis=1)&(moved[:,2]>0))
    p=moved[ids]
    uv=p[:,:2]/p[:,2,None]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
    h,w=target.shape[:2]
    inside=np.isfinite(uv).all(axis=1)&(uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1)
    ids=ids[inside];uv=np.rint(uv[inside]).astype(int)
    q=target[uv[:,1],uv[:,0]];n=normals[uv[:,1],uv[:,0]]
    length=np.linalg.norm(n,axis=1)
    residual=moved[ids]-q
    good=(np.isfinite(q).all(axis=1)&(q[:,2]>0)&np.isfinite(n).all(axis=1)&(length>1e-12)
          &(np.abs(residual[:,2])<=gate_m)&(np.linalg.norm(residual,axis=1)<=gate_m))
    ids=ids[good];q=q[good];n=n[good]/length[good,None];uv=uv[good]
    # A target pixel cannot acquire arbitrarily many source votes.
    pixel=uv[:,1]*w+uv[:,0]
    distance=np.linalg.norm(moved[ids]-q,axis=1)
    order=np.lexsort((ids,distance,pixel))
    unique=np.r_[True,np.diff(pixel[order])!=0] if len(order) else np.empty(0,bool)
    chosen=np.sort(order[unique])
    return ids[chosen],moved[ids[chosen]],q[chosen],n[chosen]


def solve_projective(source, target, normals, k, initial, *, iterations_per_scale=12, check=lambda:None):
    """Small-pose correction with bounded coarse-to-fine projective matches."""
    if type(iterations_per_scale) is not int or not 1<=iterations_per_scale<=40:
        raise ValueError('Invalid projective iteration budget')
    transform=np.asarray(initial,dtype=float).copy()
    if validate_pose(transform):raise ValueError('Invalid initial pose')
    rows=[]
    for stride,gate in ((4,.03),(2,.015),(1,.008)):
        check()
        scaled=np.asarray(k,dtype=float).copy();scaled[:2]/=stride
        points=np.asarray(target)[::stride,::stride];normal=np.asarray(normals)[::stride,::stride]
        # Coarse source sampling is deterministic and never changes source geometry.
        sample=np.asarray(source)[::stride]
        for iteration in range(iterations_per_scale):
            check()
            ids,p,q,n=correspondences(sample,points,normal,scaled,transform,gate_m=gate)
            if len(ids)<30:raise ValueError('projective_insufficient_correspondences')
            residual=np.einsum('ij,ij->i',p-q,n)
            normalized=residual/(gate*.5)
            weight=np.maximum(0,1-normalized**2)**2
            weight[np.abs(normalized)>=1]=0
            jacobian=np.c_[np.cross(p,n),n]
            root=np.sqrt(weight)
            delta,_,rank,singular=np.linalg.lstsq(jacobian*root[:,None],-residual*root,rcond=1e-6)
            if rank<6 or singular[-1]<=singular[0]*1e-6:
                raise ValueError('projective_degenerate_geometry')
            rotation=float(np.linalg.norm(delta[:3]));translation=float(np.linalg.norm(delta[3:]))
            if translation>.02 or rotation>np.deg2rad(3):raise ValueError('projective_step_exceeds_budget')
            correction=np.eye(4);correction[:3,:3]=cv2.Rodrigues(delta[:3])[0];correction[:3,3]=delta[3:]
            transform=correction@transform
            if validate_pose(transform):raise ValueError('Invalid projective transform')
            rows.append(dict(stride=stride,iteration=iteration,correspondences=len(ids),
                median_abs_plane_residual_m=float(np.median(np.abs(residual))),
                translation_step_m=translation,rotation_step_rad=rotation,condition=float(singular[0]/singular[-1])))
            if translation<1e-6 and rotation<1e-5:break
    return transform,dict(method=VERSION,iterations=rows)
