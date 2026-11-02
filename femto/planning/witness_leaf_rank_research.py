"""Bounded proximity ranking of old proof leaves, never a coverage decision.

The caller must independently validate the original proof, target identity,
route and seeds. Scores only select which witness proposals to investigate.
"""
import numpy as np

from femto.geometry.exact_surface import _dyadic_coefficients
from femto.planning import coverage_certificate as coverage

VERSION='research-old-leaf-vertex-nearest-sample-rank-v1'
MAX_LEAVES=5000
MAX_SEEDS=200000
MAX_COMPARISONS=20_000_000
MAX_BATCH_POINTS=100000


def rank(target,certificate,pending_leaves,seeds,*,check=lambda:None):
    """Rank confirmed old leaf indices by largest nearest-sample vertex distance."""
    check();coverage._checked_assets(certificate,check)
    if not isinstance(pending_leaves,(list,tuple,np.ndarray)) or not isinstance(seeds,list) or not seeds:
        raise ValueError('见证叶排序输入无效')
    if (len(pending_leaves)>MAX_LEAVES or len(seeds)>MAX_SEEDS
            or 3*len(pending_leaves)*len(seeds)>MAX_COMPARISONS):
        raise ValueError('见证叶排序超过预算')
    indices=[]
    for item in pending_leaves:
        if isinstance(item,(bool,np.bool_)) or not isinstance(item,(int,np.integer)):
            raise ValueError('见证叶索引无效')
        indices.append(int(item))
    if len(set(indices))!=len(indices):raise ValueError('见证叶索引重复')
    values=certificate.arrays;arrays=getattr(target,'arrays',None)
    if not isinstance(arrays,dict):raise ValueError('见证叶目标数组缺失')
    if any(i<0 or i>=len(values['root']) or values['status'][i]!=1 for i in indices):
        raise ValueError('见证叶必须属于原已确认集合')
    vertices=np.asarray(arrays['vertices']);faces=np.asarray(arrays['faces'])
    source_faces=np.asarray(arrays['source_face']);bary=np.asarray(arrays['barycentric'])
    if (vertices.ndim!=2 or vertices.shape[1:]!=(3,) or not np.isfinite(vertices).all()
            or faces.ndim!=2 or faces.shape[1:]!=(3,) or faces.dtype.kind not in 'iu'
            or source_faces.ndim!=1 or source_faces.dtype.kind not in 'iu'
            or bary.shape!=(len(source_faces),3,3) or not np.isfinite(bary).all()
            or (faces.size and (faces.min()<0 or faces.max()>=len(vertices)))
            or (source_faces.size and (source_faces.min()<0 or source_faces.max()>=len(faces)))):
        raise ValueError('见证叶目标几何无效')
    try:xyz=np.asarray([item['xyz_m'] for item in seeds],dtype=float)
    except (TypeError,ValueError,KeyError) as error:raise ValueError('见证叶排序采样坐标无效') from error
    if xyz.shape!=(len(seeds),3) or not np.isfinite(xyz).all():raise ValueError('见证叶排序采样坐标无效')
    if not indices:return dict(version=VERSION,order=[],scores_m=[],semantics='ranking_only_not_coverage')
    roots=np.asarray(values['root'])[indices]
    if roots.min()<0 or roots.max()>=len(source_faces):raise ValueError('见证叶目标索引无效')
    original=vertices[faces[source_faces[roots]]]
    root_xyz=np.einsum('nij,njk->nik',bary[roots],original)
    coeff=np.asarray([_dyadic_coefficients(int(values['depth'][i]),int(values['code'][i])) for i in indices])
    leaf_xyz=np.einsum('nij,njk->nik',coeff,root_xyz)
    if not np.isfinite(leaf_xyz).all():raise ValueError('见证叶坐标无效')
    scores=[];batch=max(1,min(32,MAX_BATCH_POINTS//(3*len(xyz))))
    for begin in range(0,len(indices),batch):
        check();part=leaf_xyz[begin:begin+batch]
        squared=((part[:,:,None,:]-xyz[None,None,:,:])**2).sum(axis=3)
        if not np.isfinite(squared).all():raise ValueError('见证叶距离溢出')
        scores.extend(np.sqrt(squared.min(axis=2)).max(axis=1).tolist())
    order=sorted(range(len(indices)),key=lambda i:(-scores[i],indices[i]))
    coverage._checked_assets(certificate,check);check()
    return dict(version=VERSION,order=[indices[i] for i in order],scores_m=[scores[i] for i in order],
                semantics='ranking_only_not_coverage')
