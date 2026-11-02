"""Bounded proposals for aligning neighboring local scan directions.

Only original shared mesh edges establish neighborhood. A candidate is used
only after rows are regenerated and existing geometry/coverage checks run.
This proposal never claims physical clearance or complete coverage.
"""
from __future__ import annotations
import math
import numpy as np

VERSION='research-shared-seam-direction-proposal-v1'
MAX_NEIGHBORS=4
ALIGN_ANGLE_DEG=10.
MINIMUM_TRANSVERSE_FACTOR=.85
EPS=1e-12


def propose(vertices,faces,patch,frame,previous,check=lambda:None):
    """Offer one compatible direction without increasing its row-count proxy."""
    if not isinstance(patch,dict) or not isinstance(frame,dict) or not isinstance(previous,list):
        raise ValueError('分区方向协调输入无效')
    vertices=np.asarray(vertices,dtype=float);faces=np.asarray(faces);ids=np.asarray(patch.get('face_ids'))
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3
            or ids.ndim!=1 or ids.dtype.kind not in 'iu' or not len(ids)
            or np.any(ids<0) or np.any(ids>=len(faces))):raise ValueError('分区方向协调几何无效')
    ids=ids.astype(int);selected_faces=faces[ids]
    if (selected_faces.dtype.kind not in 'iu' or np.any(selected_faces<0)
            or np.any(selected_faces>=len(vertices))):raise ValueError('分区方向协调面索引无效')
    tri=vertices[selected_faces]
    if not np.isfinite(tri).all():raise ValueError('分区方向协调包含非有限坐标')
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);magnitudes=np.linalg.norm(cross,axis=1);valid=magnitudes>0
    if not valid.any():raise ValueError('分区方向协调缺少有效三角面')
    normals=cross[valid]/magnitudes[valid,None]
    normal=np.asarray(frame.get('n'),float);axis=np.asarray(frame.get('e2'),float);origin=np.asarray(frame.get('origin'),float)
    if (normal.shape!=(3,) or axis.shape!=(3,) or origin.shape!=(3,) or not np.isfinite([normal,axis,origin]).all()
            or np.linalg.norm(normal)<=0 or np.linalg.norm(axis)<=0):raise ValueError('分区方向协调坐标系无效')
    normal=normal/np.linalg.norm(normal);axis=axis/np.linalg.norm(axis)
    points=vertices[np.unique(faces[ids])]-origin
    def proxy(candidate):
        factor=float(np.sqrt(np.maximum(0.,1.-(normals@candidate)**2)).min())
        return float(np.ptp(points@candidate))/max(factor,1e-12),factor
    original,original_factor=proxy(axis);seams={tuple(edge) for edge in patch.get('seam_edges',[])}
    neighbors=[item for item in previous if seams.intersection(map(tuple,item.get('seam_edges',[]))) and item.get('frame')]
    neighbors=sorted(neighbors,key=lambda item:int(item['seed_face']))[:MAX_NEIGHBORS]
    choices=[];limit=math.cos(math.radians(ALIGN_ANGLE_DEG))
    for neighbor in neighbors:
        check();ref=np.asarray(neighbor['frame'].get('e2'),float)
        if ref.shape!=(3,) or not np.isfinite(ref).all() or np.linalg.norm(ref)<=0:
            raise ValueError('相邻分区扫描方向无效')
        ref=ref/np.linalg.norm(ref);before=abs(float(axis@ref))
        if before>=limit:continue
        candidate=ref-normal*float(ref@normal);length=float(np.linalg.norm(candidate))
        if length<=1e-8:continue
        candidate=candidate/length
        if float(candidate@axis)<0:candidate=-candidate
        after=abs(float(candidate@ref));cost,factor=proxy(candidate)
        if (after+EPS<limit or cost>original+EPS or factor<=1e-8
                or factor+EPS<min(original_factor,MINIMUM_TRANSVERSE_FACTOR)):continue
        chosen=dict(frame,e1=np.cross(candidate,normal).tolist(),e2=candidate.tolist(),
                    direction_source='adjacent_seam_compatible',minimum_transverse_metric_factor=factor)
        decision=dict(version=VERSION,neighbor_seed_face=int(neighbor['seed_face']),before_alignment=before,
            after_alignment=after,original_span_proxy_m=original,candidate_span_proxy_m=cost,
            original_metric_factor=original_factor,candidate_metric_factor=factor,
            semantics='direction_proposal_not_geometry_or_coverage_proof')
        choices.append((cost,-after,int(neighbor['seed_face']),chosen,decision))
    if not choices:return None
    _,_,_,chosen,decision=min(choices,key=lambda item:item[:3])
    return chosen,decision
