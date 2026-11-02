"""Experimental geometry-only gate for a bounded spatial patch proposal.

The score estimates how a worst face can constrain a larger area. It does not
certify actual row density, coverage, continuity, or physical robot motion.
This component is not yet registered as a planning or archive version.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np
from femto.planning import spatial_patch_split

VERSION='adaptive-worst-face-area-split-proposal-v1'
MIN_AREA_WIDTHS_SQUARED=80.
MIN_TAIL_SPREAD=.12
MIN_PREDICTED_REDUCTION=.10
MIN_AXIS_FACTOR=.30
MAX_CANDIDATE_BINS=2


def _geometry(vertices,faces,part,axis,check):
    check()
    ids=part.get('face_ids') if isinstance(part,dict) else None
    if (not isinstance(ids,list) or not ids or len(ids)>spatial_patch_split.MAX_FACES
            or any(type(i) is not int or not 0<=i<len(faces) for i in ids)
            or len(set(ids))!=len(ids)):
        raise ValueError('自适应分区原面无效或超过预算')
    selected=faces[np.asarray(ids,dtype=int)]
    if np.any(selected<0) or np.any(selected>=len(vertices)):
        raise ValueError('自适应分区面引用无效')
    triangles=vertices[selected]
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);weights=np.linalg.norm(cross,axis=1)
    if not np.isfinite(weights).all():raise ValueError('自适应分区包含非有限几何')
    valid=weights>0
    if not valid.any():raise ValueError('自适应分区没有正面积面片')
    normal=cross[valid]/weights[valid,None];weights=weights[valid]
    factors=np.sqrt(np.maximum(0.,1-(normal@axis)**2))
    order=np.argsort(factors,kind='stable');cumulative=np.cumsum(weights[order])
    index=min(len(order)-1,int(np.searchsorted(cumulative,cumulative[-1]*.1,side='left')))
    check()
    return dict(area_m2=float(weights.sum()/2),minimum=float(factors.min()),
                percentile_10=float(factors[order[index]]))


def propose(vertices,faces,patch,along,across,width_mm,check=lambda:None):
    """Return an unchanged patch or a 2-bin connected proposal with evidence.

    Width only scales the minimum patch size; it is never modified. The caller
    still generates rows and verifies every original geometry/width predicate.
    """
    if (isinstance(width_mm,bool) or not isinstance(width_mm,(int,float))
            or not math.isfinite(width_mm) or width_mm<=0):raise ValueError('自适应分区作用宽度无效')
    vertices=np.asarray(vertices,dtype=float);faces=np.asarray(faces)
    along=np.asarray(along,dtype=float);across=np.asarray(across,dtype=float)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all()
            or faces.ndim!=2 or faces.shape[1]!=3 or not np.issubdtype(faces.dtype,np.integer)
            or along.shape!=(3,) or across.shape!=(3,) or not np.isfinite(along).all() or not np.isfinite(across).all()):
        raise ValueError('自适应分区网格或方向无效')
    a=np.linalg.norm(along);b=np.linalg.norm(across)
    if not math.isfinite(a) or not math.isfinite(b) or a<=1e-12 or b<=1e-12 or abs(float(along@across)/(a*b))>1e-6:
        raise ValueError('自适应分区方向不正交')
    across=across/b
    base=_geometry(vertices,faces,patch,across,check)
    record=dict(version=VERSION,adopted=False,reason='not_assessed',baseline=base,
                min_area_widths_squared=MIN_AREA_WIDTHS_SQUARED,min_tail_spread=MIN_TAIL_SPREAD,
                min_predicted_reduction=MIN_PREDICTED_REDUCTION,bins=MAX_CANDIDATE_BINS)
    def fallback(reason):
        record['reason']=reason
        return [deepcopy(patch)],record
    if base['area_m2']<MIN_AREA_WIDTHS_SQUARED*(width_mm/1000)**2:return fallback('small_patch')
    if base['minimum']<MIN_AXIS_FACTOR:return fallback('needs_other_direction')
    if base['percentile_10']-base['minimum']<MIN_TAIL_SPREAD:return fallback('no_local_metric_outlier')
    try:pieces=spatial_patch_split.split(vertices,faces,patch,along,MAX_CANDIDATE_BINS,check)
    except ValueError as error:
        if str(error)=='空间分区连通片超过预算':return fallback('component_budget')
        raise
    if len(pieces)<2:return fallback('one_component')
    try:measure=[_geometry(vertices,faces,piece,across,check) for piece in pieces]
    except ValueError as error:
        if str(error)=='自适应分区没有正面积面片':return fallback('degenerate_piece')
        raise
    before=base['area_m2']/base['minimum']
    after=sum(p['area_m2']/p['minimum'] for p in measure)
    reduction=1-after/before
    record.update(predicted_before=before,predicted_after=after,predicted_reduction=reduction,
                  pieces=[dict(seed_face=p['seed_face'],face_count=len(p['face_ids']),**q) for p,q in zip(pieces,measure)])
    if reduction<MIN_PREDICTED_REDUCTION:return fallback('insufficient_predicted_gain')
    record.update(adopted=True,reason='geometry_proposal')
    return pieces,record
