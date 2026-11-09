"""Organized RGB-D research primitives; independently implemented in NumPy.

Uses PCL's documented integral-gradient, normal-space and edge-label concepts,
not its source code. Inputs are undistorted camera RDF / metres. Invalid
observations stay unknown; no smoothing of points or inferred depth is returned.
"""
from __future__ import annotations

import cv2
import numpy as np

VERSION = 'organized-surface-experiment-v1'
MISSING_BOUNDARY, OCCLUDING, OCCLUDED, HIGH_CURVATURE = 1, 2, 4, 8


def point_map(depth_mm, k, minimum=.25, maximum=2.21):
    depth=np.asarray(depth_mm)
    k=np.asarray(k,dtype=float)
    if (depth.ndim!=2 or min(depth.shape)<1 or k.shape!=(3,3)
            or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0
            or not np.allclose(k[2],[0,0,1]) or k[0,1]!=0 or k[1,0]!=0
            or not np.isfinite([minimum,maximum]).all() or not 0<minimum<maximum):
        raise ValueError('Expected undistorted pinhole depth and valid metre range')
    valid=np.isfinite(depth)&(depth>=minimum*1000)&(depth<=maximum*1000)
    y,x=np.indices(depth.shape,dtype=float)
    z=np.where(valid,depth,0).astype(float)/1000
    points=np.stack(((x-k[0,2])*z/k[0,0],(y-k[1,2])*z/k[1,1],z),axis=-1)
    points[~valid]=np.nan
    return points


def _points(value):
    value=np.asarray(value,dtype=float)
    if value.ndim!=3 or value.shape[2]!=3 or min(value.shape[:2])<1:
        raise ValueError('Expected H x W x 3 organized points')
    valid=np.isfinite(value).all(axis=-1)&(value[...,2]>0)
    return value,valid


def _box_sum(values,radius):
    padded=np.pad(values,((radius,radius),(radius,radius)),mode='constant')
    integral=cv2.integral(padded,sdepth=cv2.CV_64F)
    side=2*radius+1
    return integral[side:,side:]-integral[:-side,side:]-integral[side:,:-side]+integral[:-side,:-side]


def integral_normals(points, *, radius=3, depth_gate_m=.02, minimum_support=.8):
    """Average 3-D gradients through integral images, orient towards camera.

    Only measured central differences are summed. A window touching missing
    depth or a neighbor jump beyond the gate remains unknown, preventing a normal
    averaged across an occlusion or a held-out validation block.
    """
    points,valid=_points(points)
    if (type(radius) is not int or not 1<=radius<=32 or not np.isfinite(depth_gate_m)
            or depth_gate_m<=0 or not 0<minimum_support<=1):
        raise ValueError('Invalid integral-normal experiment parameters')
    h,w=valid.shape
    normals=np.full_like(points,np.nan)
    if min(h,w)<2*radius+3:
        return normals,np.zeros((h,w),bool)
    gradient_valid=np.zeros_like(valid)
    z=points[...,2]
    center=z[1:-1,1:-1]
    local=valid[1:-1,1:-1].copy()
    for neighbor in (z[:-2,1:-1],z[2:,1:-1],z[1:-1,:-2],z[1:-1,2:]):
        local &= np.isfinite(neighbor)&(np.abs(neighbor-center)<=depth_gate_m)
    gradient_valid[1:-1,1:-1]=local
    count=_box_sum(gradient_valid.astype(float),radius)
    gradients=[]
    for axis in (1,0):
        delta=np.zeros_like(points)
        delta[1:-1,1:-1]=(points[1:-1,2:]-points[1:-1,:-2] if axis==1
                               else points[2:,1:-1]-points[:-2,1:-1])*.5
        delta[~gradient_valid]=0
        gradients.append(np.stack([_box_sum(delta[...,j],radius) for j in range(3)],axis=-1))
    normal=np.cross(gradients[0],gradients[1])
    length=np.linalg.norm(normal,axis=-1)
    side=2*radius+1
    kernel=np.ones((side,side),np.uint8)
    safe=valid.copy();safe[1:-1,1:-1]=local
    complete=cv2.erode(safe.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    good=complete&(count>=minimum_support*side*side)&(length>1e-15)
    normals[good]=normal[good]/length[good,None]
    away=np.einsum('ijk,ijk->ij',np.nan_to_num(normals),np.nan_to_num(points))>0
    normals[away]*=-1
    return normals,good


def edge_labels(points, normals=None, *, depth_gate_m=.03, curvature_deg=25.):
    """Label valid missing-boundary, foreground/background and normal-angle edges.

    Curvature here means an angular discontinuity proxy, not principal
    curvature. RGB texture is intentionally absent from geometry decisions.
    """
    points,valid=_points(points)
    if not np.isfinite([depth_gate_m,curvature_deg]).all() or depth_gate_m<=0 or not 0<curvature_deg<180:
        raise ValueError('Invalid edge experiment parameters')
    if normals is not None and np.shape(normals)!=points.shape:
        raise ValueError('Normal map shape mismatch')
    labels=np.zeros(valid.shape,np.uint8)
    for first,second in ((np.s_[:,:-1],np.s_[:,1:]),(np.s_[:-1,:],np.s_[1:,:])):
        a,b=valid[first],valid[second]
        left,right=labels[first],labels[second]
        left[a&~b] |= MISSING_BOUNDARY;right[b&~a] |= MISSING_BOUNDARY
        za,zb=points[first][...,2],points[second][...,2]
        jump=a&b&(np.abs(za-zb)>depth_gate_m)
        left[jump&(za<zb)] |= OCCLUDING;right[jump&(za<zb)] |= OCCLUDED
        left[jump&(za>zb)] |= OCCLUDED;right[jump&(za>zb)] |= OCCLUDING
        if normals is not None:
            na,nb=np.asarray(normals)[first],np.asarray(normals)[second]
            finite=np.isfinite(na).all(axis=-1)&np.isfinite(nb).all(axis=-1)
            denominator=np.linalg.norm(na,axis=-1)*np.linalg.norm(nb,axis=-1)
            cosine=np.einsum('ijk,ijk->ij',na,nb)/np.maximum(denominator,1e-30)
            bend=a&b&~jump&finite&(denominator>1e-15)&(cosine<np.cos(np.deg2rad(curvature_deg)))
            left[bend] |= HIGH_CURVATURE;right[bend] |= HIGH_CURVATURE
    return labels


def normal_space_indices(normals, count, *, bins=6):
    """Deterministic balanced sampling of occupied bins in normal direction.

    Returned indices refer to the original input, without moving any point.
    Within a bin, evenly spaced original indices retain spatial spread.
    """
    normals=np.asarray(normals,dtype=float)
    if normals.ndim!=2 or normals.shape[1]!=3 or type(count) is not int or count<1 or type(bins) is not int or not 2<=bins<=32:
        raise ValueError('Invalid normal-space sampling input')
    length=np.linalg.norm(normals,axis=1)
    valid=np.isfinite(normals).all(axis=1)&(length>1e-12)
    ids=np.flatnonzero(valid)
    if count>=len(ids):return ids
    unit=normals[ids]/length[ids,None]
    cells=np.clip(((unit+1)*.5*bins).astype(int),0,bins-1)
    code=cells@np.array([bins*bins,bins,1])
    order=np.argsort(code,kind='stable');sorted_code=code[order]
    groups=np.split(ids[order],np.flatnonzero(np.diff(sorted_code))+1)
    quotas=np.zeros(len(groups),int);remaining=count
    while remaining:
        available=np.flatnonzero(quotas<np.array([len(g) for g in groups]))
        allocation=min(remaining,len(available));quotas[available[:allocation]]+=1;remaining-=allocation
    selected=[group[np.linspace(0,len(group)-1,quota,dtype=int)] for group,quota in zip(groups,quotas) if quota]
    return np.sort(np.concatenate(selected))
