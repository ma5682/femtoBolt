"""Winding-independent split of severe physical foldbacks within patches.

This is a geometric proposal, not a coverage or physical-safety certificate.
Faces and their source indices are retained. Nonmanifold and degenerate edges
never establish adjacency, and the caller must retain all downstream checks.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
import numpy as np

VERSION='connected-foldback-split-v1'
FOLDBACK_ANGLE_DEG=120.
MAX_FACES=100_000


def _continuous(vertices,faces,edge,left,right,cosine):
    tips=[]
    for fi in (left,right):
        corners=faces[fi];other=[int(v) for v in corners if int(v) not in edge]
        if len(other)!=1:return False
        tips.append(vertices[other[0]])
    start,end=vertices[list(edge)];tangent=end-start;length=float(np.linalg.norm(tangent))
    if not np.isfinite(length) or length<=0:return False
    tangent=tangent/length;inside=[]
    for point in tips:
        delta=point-start;delta=delta-tangent*float(delta@tangent);length=float(np.linalg.norm(delta))
        if not np.isfinite(length) or length<=0:return False
        inside.append(delta/length)
    return -float(inside[0]@inside[1])>=cosine


def split(vertices,faces,patches,check=lambda:None):
    """Split patches where a shared edge folds back beyond the fixed threshold."""
    vertices=np.asarray(vertices);faces=np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3
            or not np.isfinite(vertices).all() or faces.ndim != 2
            or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer)):
        raise ValueError('折返分区的网格无效')
    ids=[]
    for patch in patches:
        check()
        part=patch['face_ids']
        if not part or any(isinstance(fi, (bool, np.bool_)) or not isinstance(fi, (int, np.integer)) for fi in part):
            raise ValueError('折返分区的原面编号无效')
        if patch['seed_face'] not in part:
            raise ValueError('折返分区种子面无效')
        ids.extend(int(fi) for fi in part)
    if len(ids)>MAX_FACES:raise ValueError('折返分区面数超过预算，请缩小区域')
    if len(ids)!=len(set(ids)) or any(fi<0 or fi>=len(faces) for fi in ids):
        raise ValueError('折返分区的原面编号无效')
    selected=faces[ids]
    if np.any(selected < 0) or np.any(selected >= len(vertices)):
        raise ValueError('折返分区的顶点编号无效')
    edges=defaultdict(list)
    for fi in ids:
        check();face=faces[fi]
        for a,b in zip(face,np.roll(face,-1)):
            edges[tuple(sorted((int(a),int(b))))].append(fi)
    adjacency=defaultdict(list);blocked=defaultdict(list);cosine=math.cos(math.radians(FOLDBACK_ANGLE_DEG))
    for edge,incident in edges.items():
        check()
        if len(incident)==2:
            if _continuous(vertices,faces,edge,incident[0],incident[1],cosine):
                adjacency[incident[0]].append(incident[1]);adjacency[incident[1]].append(incident[0])
            else:
                blocked[incident[0]].append(incident[1])
    result=[]
    for original in patches:
        check();remaining=set(original['face_ids']);groups=[]
        for start in sorted(remaining):
            check()
            if start not in remaining:continue
            remaining.remove(start);stack=[start];group=[]
            while stack:
                check();fi=stack.pop();group.append(fi)
                for neighbor in sorted(adjacency[fi]):
                    if neighbor in remaining:remaining.remove(neighbor);stack.append(neighbor)
            groups.append(sorted(group))
        membership={fi:index for index,group in enumerate(groups) for fi in group}
        if any(b in membership and membership[a]==membership[b]
               for a in membership for b in blocked[a]):
            raise ValueError('折返边可从其他路径绕回，无法可靠分区')
        if len(groups)==1:
            result.append(deepcopy(original));continue
        for group in groups:
            check();owned=set(group);touched=set()
            for fi in group:
                face=faces[fi];touched.update(tuple(sorted((int(a),int(b)))) for a,b in zip(face,np.roll(face,-1)))
            seam=[];outer=[]
            for edge in sorted(touched):
                incident=edges[edge]
                if len(incident)==2 and any(fi not in owned for fi in incident):seam.append(list(edge))
                elif len(incident)!=2:outer.append(list(edge))
            triangle=vertices[faces[group]];sizes=np.linalg.norm(np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]),axis=1)
            seed=(original['seed_face'] if original['seed_face'] in owned else
                  group[min(range(len(group)),key=lambda i:(-sizes[i],group[i]))])
            result.append(dict(face_ids=group,seed_face=int(seed),seam_edges=seam,outer_edges=outer,
                               area_m2=float(sizes.sum()/2)))
    return sorted(result,key=lambda p:p['face_ids'][0])
