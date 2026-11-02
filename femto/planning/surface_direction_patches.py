"""Deterministic connected planning patches, separate from measured components.

Patches own every input face exactly once. Their seams are artificial planning
boundaries: never missing geometry, forbidden area, or a footprint inset rule.
Normal variation bounds limit accumulated curvature as well as sharp folds.
"""
from __future__ import annotations

from collections import defaultdict
import heapq
import math
import numpy as np

VERSION = 'connected-normal-cone-patches-v1'
SEED_ANGLE_DEG = 35.
NEIGHBOR_ANGLE_DEG = 25.
SMALL_PATCH_FACES = 12
MERGE_NORMAL_ANGLE_DEG = 45.
MERGE_MINIMUM_METRIC = .85


def partition(vertices, faces, face_ids, check=lambda: None):
    ids = np.asarray(face_ids, dtype=np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('表面规划分区的原面编号无效')
    if not len(ids):
        return []
    ids = np.sort(ids)
    triangles = np.asarray(vertices)[np.asarray(faces)[ids]]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    sizes = np.linalg.norm(cross, axis=1)
    if not np.isfinite(sizes).all():
        raise ValueError('表面规划分区包含非有限几何')
    normals = np.divide(cross, sizes[:, None], out=np.zeros_like(cross), where=sizes[:, None]>0)
    edge_faces = defaultdict(list)
    for local, face in enumerate(np.asarray(faces)[ids]):
        check()
        for a, b in zip(face, np.roll(face, -1)):
            edge_faces[tuple(sorted((int(a), int(b))))].append(local)
    neighbors = [[] for _ in ids]
    for edge in sorted(edge_faces):
        fs = edge_faces[edge]
        # Do not invent an adjacency through nonmanifold edges or coordinate
        # coincidence. Original mesh vertex IDs remain authoritative.
        if len(fs) == 2:
            a, b = fs
            neighbors[a].append(b);neighbors[b].append(a)
    seed_limit = math.cos(math.radians(SEED_ANGLE_DEG))
    neighbor_limit = math.cos(math.radians(NEIGHBOR_ANGLE_DEG))
    available = np.ones(len(ids), dtype=bool)
    patches = []
    for seed in sorted(range(len(ids)), key=lambda i: (-sizes[i], int(ids[i]))):
        check()
        if not available[seed]:
            continue
        queue = [seed];available[seed] = False;owned = []
        while queue:
            check();current = heapq.heappop(queue);owned.append(current)
            for nxt in sorted(neighbors[current]):
                if (available[nxt] and sizes[seed]>0 and sizes[nxt]>0
                        and abs(float(normals[nxt]@normals[seed])) >= seed_limit
                        and abs(float(normals[nxt]@normals[current])) >= neighbor_limit):
                    available[nxt] = False;heapq.heappush(queue,nxt)
        owned = sorted(owned)
        patch_ids = ids[owned]
        seam_edges, outer_edges = [], []
        own_set = set(owned)
        touched = set()
        for local in owned:
            face = np.asarray(faces)[ids[local]]
            touched.update(tuple(sorted((int(a),int(b)))) for a,b in zip(face,np.roll(face,-1)))
        for edge in sorted(touched):
            fs = edge_faces[edge]
            if len(fs) == 2 and any(i not in own_set for i in fs):
                seam_edges.append(list(edge))
            elif len(fs) != 2:
                outer_edges.append(list(edge))
        patches.append(dict(face_ids=patch_ids.tolist(), seed_face=int(ids[seed]),
                            seam_edges=seam_edges, outer_edges=outer_edges,
                            area_m2=float(sizes[owned].sum()/2)))
    return _merge_small(vertices,faces,patches,check)


def _merge_small(vertices,faces,patches,check):
    """Coalesce tiny neighbors only when the combined direction remains usable.

    No face disappears. Sharp/unsupported slivers which cannot meet the cone
    and metric bounds retain a separate patch for explicit downstream handling.
    """
    patches={i:p for i,p in enumerate(patches)}
    changed=True
    while changed:
        check();changed=False
        edge_owner=defaultdict(list)
        for i,p in patches.items():
            for edge in p['seam_edges']:edge_owner[tuple(edge)].append(i)
        for i in sorted(patches,key=lambda i:(len(patches[i]['face_ids']),patches[i]['face_ids'][0])):
            check();p=patches[i]
            if len(p['face_ids'])>=SMALL_PATCH_FACES:continue
            neighbors={j for edge in p['seam_edges'] for j in edge_owner[tuple(edge)] if j!=i}
            for j in sorted(neighbors,key=lambda j:(-len(patches[j]['face_ids']),patches[j]['face_ids'][0])):
                q=patches[j]
                edges_p=set(map(tuple,p['seam_edges']));edges_q=set(map(tuple,q['seam_edges']))
                merged=dict(face_ids=sorted(p['face_ids']+q['face_ids']),
                    seed_face=(p if p['area_m2']>q['area_m2'] else q)['seed_face'],
                    seam_edges=[list(e) for e in sorted(edges_p^edges_q)],
                    outer_edges=sorted(p['outer_edges']+q['outer_edges']),area_m2=p['area_m2']+q['area_m2'])
                try:local=frame(vertices,faces,merged,check)
                except ValueError:continue
                if local['minimum_transverse_metric_factor']<MERGE_MINIMUM_METRIC:continue
                triangles=np.asarray(vertices)[np.asarray(faces)[merged['face_ids']]]
                cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
                norms=np.linalg.norm(cross,axis=1);valid=norms>0
                if np.min(np.abs((cross[valid]/norms[valid,None])@np.asarray(local['n'])))<math.cos(math.radians(MERGE_NORMAL_ANGLE_DEG)):
                    continue
                patches[j]=merged;del patches[i];changed=True
                break
            if changed:break
    return sorted(patches.values(),key=lambda p:p['face_ids'][0])


def frame(vertices, faces, patch, check=lambda: None):
    """Local normal plane and bounded in-plane direction candidates.

    Minimize transverse extent divided by the worst transverse metric: a
    conservative row-count proxy. This is proposal geometry only.
    """
    check()
    ids = np.asarray(patch['face_ids'], dtype=int)
    triangles = np.asarray(vertices)[np.asarray(faces)[ids]]
    cross = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    weights = np.linalg.norm(cross,axis=1)
    valid = weights>0
    if not valid.any():
        raise ValueError('表面规划分区没有正面积三角面')
    normals = cross[valid]/weights[valid,None]
    seed = np.flatnonzero(ids==patch['seed_face'])[0]
    n0 = cross[seed]/weights[seed]
    signs = np.where(normals@n0<0,-1.,1.)
    n = np.sum(normals*(weights[valid]*signs)[:,None],axis=0);n /= np.linalg.norm(n)
    if n[2]>0:n=-n
    origin = np.average(triangles[valid].mean(axis=1),axis=0,weights=weights[valid])
    points = np.asarray(vertices)[np.unique(np.asarray(faces)[ids])]-origin
    projected = points-(points@n)[:,None]*n
    values, vectors = np.linalg.eigh(projected.T@projected)
    e1 = vectors[:,-1];e1 -= n*(e1@n);e1 /= np.linalg.norm(e1)
    # Stable signs; the frame itself is not a robot orientation.
    for component in e1:
        if abs(component)>1e-8:
            if component<0:e1=-e1
            break
    e2 = np.cross(n,e1)
    options = []
    for index in range(8):
        check();angle=index*math.pi/8
        along=e1*math.cos(angle)+e2*math.sin(angle)
        across=np.cross(n,along)
        factor=float(np.sqrt(np.maximum(0.,1-(normals@across)**2)).min())
        span=float(np.ptp(points@across))
        options.append(((span/max(factor,1e-12),-factor,index),along,across,factor))
    _, along, across, factor=min(options,key=lambda option:option[0])
    return dict(origin=origin.tolist(),e1=along.tolist(),e2=across.tolist(),n=n.tolist(),
                direction_source='local_normal_metric_eight_directions',normal_source='patch_area_weighted',
                minimum_transverse_metric_factor=factor)


def rows(vertices, faces, patch, width_mm, overlap_percent, reference, check=lambda: None, *, local_frame=None):
    """Propose local slices, keeping artificial seams eligible for scan centers.

    Only true fragment boundary sides receive a nominal-width inset. A side
    with a nearby seam gets no blanket inset; original whole-edge predicates
    must still trim any portion near a real excluded boundary. No masks change.
    """
    from femto.planning.fixed_width_layout import request,metric_positions
    spec=request(width_mm,overlap_percent)
    local=frame(vertices,faces,patch,check) if local_frame is None else local_frame
    origin=np.asarray(local['origin']);axis=np.asarray(local['e2'])
    points=np.asarray(vertices)[np.unique(np.asarray(faces)[patch['face_ids']])]
    levels=(points-origin)@axis;low,high=float(levels.min()),float(levels.max())
    seam_ids=sorted({v for edge in patch['seam_edges'] for v in edge})
    seam_levels=(np.asarray(vertices)[seam_ids]-origin)@axis if seam_ids else np.empty(0)
    radius=spec['effective_width_mm']/2000
    strip=radius+(reference.global_residual if reference is not None else 0.)
    seam_sides=[bool(len(seam_levels) and seam_levels.min()<=low+strip),
                bool(len(seam_levels) and seam_levels.max()>=high-strip)]
    insets=[]
    for side,mask in enumerate((levels<=low+strip,levels>=high-strip)):
        check();margin=0.
        if reference is not None and not seam_sides[side]:
            for point in points[mask]:
                check();_,value,_=reference.probe(point,0.);margin=max(margin,value)
        insets.append(0. if seam_sides[side] else radius+margin)
    first,last=low+insets[0],high-insets[1]
    positions=np.empty(0)
    if last>=first:
        triangle=np.asarray(vertices)[np.asarray(faces)[patch['face_ids']]]
        normals=np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0])
        norms=np.linalg.norm(normals,axis=1);valid=norms>0;normals=normals[valid]/norms[valid,None]
        triangle_levels=(triangle[valid]-origin)@axis
        intervals=np.column_stack((triangle_levels.min(axis=1),triangle_levels.max(axis=1)))
        factors=np.sqrt(np.maximum(0.,1-(normals@axis)**2))
        positions,_=metric_positions(first,last,intervals,factors,
                    spec['effective_width_mm']*(1-spec['overlap_percent']/100)/1000)
    return dict(frame=local,positions_m=positions.tolist(),insets_mm=[v*1000 for v in insets],
                artificial_seam_sides=seam_sides,status='proposed' if len(positions) else 'narrower_than_fixed_width')
