"""Experimental deterministic spatial subdivision of one planning patch.

Subdivision changes only planning ownership, never the original mesh. It does
not establish suitable rows, route connectivity, treatment coverage or motion.
The caller must independently validate generated geometry and original target.
"""
from __future__ import annotations

from collections import defaultdict,deque
import math
import numpy as np

VERSION='connected-spatial-patch-split-v1'
MAX_FACES=100_000
MAX_PIECES=100
MAX_BINS=8


def split(vertices,faces,patch,axis,bins,check=lambda:None,*,cut_fraction=None):
    """Split along an axis and connected face ownership, retaining every face.

    Existing seam and outer edges keep their meaning. Only shared manifold
    edges between newly split pieces become artificial seams. No adjacency is
    invented through nonmanifold edges or coincident coordinates.
    """
    if type(bins) is not int or not 2<=bins<=MAX_BINS:
        raise ValueError('空间分区数量无效')
    if cut_fraction is not None and (bins!=2 or isinstance(cut_fraction,bool)
            or not isinstance(cut_fraction,(int,float)) or not math.isfinite(cut_fraction)
            or not 0<cut_fraction<1):
        raise ValueError('空间分区切分位置无效')
    vertices=np.asarray(vertices,dtype=float);faces=np.asarray(faces)
    axis=np.asarray(axis,dtype=float)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all()
            or faces.ndim!=2 or faces.shape[1]!=3 or not np.issubdtype(faces.dtype,np.integer)
            or axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<=1e-12):
        raise ValueError('空间分区网格或方向无效')
    if not isinstance(patch,dict) or not isinstance(patch.get('face_ids'),list) or not patch['face_ids']:
        raise ValueError('空间分区缺少原面')
    raw=patch['face_ids']
    if (len(raw)>MAX_FACES or any(type(i) is not int or not 0<=i<len(faces) for i in raw)
            or len(set(raw))!=len(raw) or type(patch.get('seed_face')) is not int or patch['seed_face'] not in raw):
        raise ValueError('空间分区原面无效或超过预算')
    ids=np.asarray(sorted(raw),dtype=np.int64);selected=faces[ids]
    if np.any(selected<0) or np.any(selected>=len(vertices)):
        raise ValueError('空间分区面引用无效')
    def declared(name):
        edges=patch.get(name)
        if (not isinstance(edges,list) or any(not isinstance(e,list) or len(e)!=2
                or any(type(v) is not int or not 0<=v<len(vertices) for v in e) or e[0]>=e[1] for e in edges)):
            raise ValueError('空间分区边界记录无效')
        values=set(map(tuple,edges))
        if len(values)!=len(edges):raise ValueError('空间分区边界重复')
        return values
    original_seams=declared('seam_edges');original_outer=declared('outer_edges')
    if original_seams & original_outer:raise ValueError('空间分区边界类型冲突')
    triangles=vertices[selected]
    levels=triangles.mean(axis=1)@axis;low,high=float(levels.min()),float(levels.max())
    if cut_fraction is None:
        labels=np.minimum(bins-1,np.floor((levels-low)/(high-low)*bins).astype(int)) if high>low else np.zeros(len(ids),int)
    else:
        labels=(levels>=low+(high-low)*cut_fraction).astype(int) if high>low else np.zeros(len(ids),int)
    edge_faces=defaultdict(list)
    for i,face in enumerate(selected):
        if i%256==0:check()
        for a,b in zip(face,np.roll(face,-1)):edge_faces[tuple(sorted((int(a),int(b))))].append(i)
    if not (original_seams|original_outer)<=set(edge_faces):raise ValueError('空间分区边界不属于原面')
    neighbors=[[] for _ in ids]
    for attached in edge_faces.values():
        if len(attached)==2:
            a,b=attached
            if labels[a]==labels[b]:neighbors[a].append(b);neighbors[b].append(a)
    owners=np.full(len(ids),-1,int);groups=[];visited=0
    for seed in range(len(ids)):
        if owners[seed]>=0:continue
        check();idx=len(groups)
        if idx>=MAX_PIECES:raise ValueError('空间分区连通片超过预算')
        owners[seed]=idx;queue=deque([seed]);group=[]
        while queue:
            current=queue.popleft();group.append(current);visited+=1
            if visited%256==0:check()
            for nxt in neighbors[current]:
                if owners[nxt]<0:owners[nxt]=idx;queue.append(nxt)
        groups.append(sorted(group))
    seams=[set() for _ in groups];outer=[set() for _ in groups]
    for index,(edge,attached) in enumerate(edge_faces.items()):
        if index%256==0:check()
        owner_set={int(owners[i]) for i in attached}
        if edge in original_seams:
            for owner in owner_set:seams[owner].add(edge)
        elif edge in original_outer:
            for owner in owner_set:outer[owner].add(edge)
        elif len(attached)==2 and len(owner_set)>1:
            for owner in owner_set:seams[owner].add(edge)
        elif len(attached)!=2:raise ValueError('空间分区缺少原始边界记录')
    sizes=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    result=[]
    for group,ss,oo in zip(groups,seams,outer):
        face_ids=sorted(int(ids[i]) for i in group)
        seed=patch['seed_face'] if patch['seed_face'] in face_ids else int(ids[min(group,key=lambda i:(-sizes[i],ids[i]))])
        result.append(dict(face_ids=face_ids,seed_face=seed,seam_edges=[list(e) for e in sorted(ss)],
                           outer_edges=[list(e) for e in sorted(oo)],area_m2=float(sizes[group].sum()/2)))
    check()
    return sorted(result,key=lambda p:p['face_ids'][0])
