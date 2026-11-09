"""Bounded Float64/exact-sign retries for Float32 ray misses on dense meshes.

Only an actual intersection with an original triangle may recover support.
Nearest points propose triangles; their distance never closes a hole.
"""
from fractions import Fraction
import numpy as np
import open3d as o3d


def _exact_sign(a, b, r):
    a,b,r=[[Fraction(float(x)) for x in value] for value in (a,b,r)]
    value=(a[1]*b[2]-a[2]*b[1])*r[0]+(a[2]*b[0]-a[0]*b[2])*r[1]+(a[0]*b[1]-a[1]*b[0])*r[2]
    return (value>0)-(value<0)


def retry_missed_rays(mesh, scene, rays, approximate_points, check=lambda:None):
    rays=np.asarray(rays,float);result=np.full(len(rays),np.nan)
    if not len(rays):return result
    if len(rays)>100000:raise ValueError('原表面射线复核超过 10 万点预算')
    v=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles)
    check()
    nearest=scene.compute_closest_points(o3d.core.Tensor(np.asarray(approximate_points,np.float32)))['primitive_ids'].numpy().astype(np.int64)
    # Incident faces preserve holes: the retry searches original triangles only.
    flat=faces.ravel();order=np.argsort(flat,kind='stable');ordered=flat[order]
    recovered=[];queried=[]
    for i,face in enumerate(nearest):
        if i%512==0:check()
        if face>=len(faces):continue
        ids=faces[face]
        candidates=np.unique(np.concatenate([order[np.searchsorted(ordered,k,'left'):np.searchsorted(ordered,k,'right')]//3 for k in ids]))
        if len(candidates)>128:continue
        recovered.extend(candidates);queried.extend([i]*len(candidates))
    if not recovered:return result
    recovered=np.asarray(recovered);queried=np.asarray(queried);triangles=v[faces[recovered]];directions=rays[queried]
    signs=[]
    for a,b in ((0,1),(1,2),(2,0)):
        left,right=triangles[:,a],triangles[:,b]
        terms=np.cross(left,right)*directions
        value=terms.sum(axis=1)
        # Absolute product bound includes cancellation within cross products.
        cross_bound=np.abs(left[:,[1,2,0]]*right[:,[2,0,1]])+np.abs(left[:,[2,0,1]]*right[:,[1,2,0]])
        bound=(cross_bound*np.abs(directions)).sum(axis=1)*64*np.finfo(float).eps
        sign=np.sign(value)
        for index in np.flatnonzero(np.abs(value)<=bound):
            check();sign[index]=_exact_sign(left[index],right[index],directions[index])
        signs.append(sign)
    signs=np.column_stack(signs)
    inside=((signs>=0).all(axis=1)|(signs<=0).all(axis=1))&(triangles[:,:,2]>0).all(axis=1)
    normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    denominator=np.einsum('ij,ij->i',normal,directions)
    with np.errstate(invalid='ignore',divide='ignore'):distance=np.einsum('ij,ij->i',normal,triangles[:,0])/denominator
    good=inside&np.isfinite(distance)&(distance>0)&(denominator!=0)
    temporary=np.full(len(rays),np.inf);np.minimum.at(temporary,queried[good],distance[good])
    result[np.isfinite(temporary)]=temporary[np.isfinite(temporary)]
    check();return result
