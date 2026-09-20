"""Fixed-reference configurable serpentine sweeps joined by surface-grid routes, demo only."""
from __future__ import annotations
from typing import Any, Callable

import heapq
import cv2
import numpy as np
from femto.planning.region_path import signature
from femto.planning.region_exclusions import allowed_region


def _finish_plan(projection,mesh,region,allowed,removed,chains,runs,breaks,faces,xx,yy,
                 width_mm,overlap_percent,boundary_margin_mm,pixel_radius,row_step,
                 turn_edges,total,modern,direction,start_corner,mask,safe,w,h,check):
    plan=dict(schema='femto-region-metric-path-v1',coordinate_frame='world_rdf',unit='metre',
        region_id=region['annotation_id'],region_signature=signature(region),segments=chains,
        spacing_mm=width_mm*(1-overlap_percent/100),row_step_px=row_step,
        spacing_semantics='reference_horizontal_rows_mm_converted_at_median_depth_not_uniform_surface_spacing',
        coverage_parameters=dict(width_mm=width_mm,overlap_percent=overlap_percent,boundary_margin_mm=boundary_margin_mm,pattern='serpentine_v1'),
        connection_policy='surface_grid_connectors_within_eroded_allowed_mask',
        execution_ready=False,physical_accuracy_verified=False,
        route_order=dict(mode='left_right_then_right_left',sweep_runs=len(runs),continuous_chains=len(chains),
            disconnected_breaks=len(chains)-1,connector_edges=turn_edges),
        boundary_check=dict(schema='femto-serpentine-boundary-v1',width_mm=width_mm,margin_mm=boundary_margin_mm,
            candidate_edges=total-len(chains),retained_edges=total-len(chains),rejected_edges=0,pixel_clearance=pixel_radius,
            method='conservative_projected_radius_mask_erosion_plus_edge_adjacent_depth_gated_grid',contact_footprint_verified=False),
        motion_demo=dict(speed_mm_s=20.,segment_gap_seconds=.5,transitions_planned=len(chains)==1,execution_ready=False))
    if modern:
        plan['coverage_parameters'].update(pattern='serpentine_v2',direction=direction,start_corner=start_corner)
        plan['spacing_semantics']='reference_rows_mm_converted_at_median_depth_not_uniform_surface_spacing'
        plan['route_order']['mode']=direction+'_'+start_corner
        plan['chain_breaks']=breaks
        bounds=np.asarray(region.get('screen_polygon') or np.c_[xx,yy],float)
        preferred=[float(bounds[:,0].max() if 'right' in start_corner else bounds[:,0].min()),
                   float(bounds[:,1].max() if 'bottom' in start_corner else bounds[:,1].min())]
        first=chains[0][0];adjusted=not np.allclose(preferred,first['pixel'],rtol=0,atol=1e-9)
        px,py=(int(round(v)) for v in preferred)
        if not (0<=px<w and 0<=py<h) or not projection.valid[py,px]:cause='偏好角点无可信可见几何支持'
        elif projection.ids[py,px] not in allowed['triangle_indices']:cause='偏好角点位于禁区剔除或选区外网格'
        elif not mask[py,px]:cause='偏好角点不在区域像素掩膜内'
        elif not safe[py,px]:cause='偏好角点不满足扫描半径和边界预留约束'
        else:cause='偏好角点不是当前扫掠行的可用端点'
        plan['start_selection']=dict(preference=start_corner,preferred_pixel=preferred,
            actual_pixel=first['pixel'],actual_xyz_m=first['xyz_m'],adjusted=bool(adjusted),
            rule='first_valid_sweep_endpoint_from_preferred_corner',
            adjustment_cause=cause if adjusted else None,
            reason=cause+'；按扫描顺序选取首条有效扫描端点，不保证最近点' if adjusted else '偏好位置满足扫描约束')
        plan['motion_demo']=dict(schema='femto-motion-demo-v2',speed_mm_s=20.,
            time_semantics='scan_and_connector_chords_only_constant_speed',cross_chain_time_s=None if breaks else 0.,
            gap_policy='stop_require_explicit_continue_no_interpolated_transition',execution_ready=False)
    # Cross-face chords are not face-local: seed only their endpoints for a
    # conservative estimate; same-face chords remain valid on their triangle.
    from femto.planning.region_coverage import evaluate_coverage
    coverage_segments=[]
    for chain in chains:
        for left,right in zip(chain,chain[1:]):
            if left['triangle_index']==right['triangle_index']:
                coverage_segments.append([left,right])
            else:
                coverage_segments.extend([[left,left],[right,right]])
    plan['coverage']=evaluate_coverage(mesh,allowed,dict(segments=coverage_segments),width_mm,check)
    if removed:plan['exclusions']=dict(polygons=region['exclusion_polygons'],removed_triangle_indices=removed,
        allowed_triangle_indices=allowed['triangle_indices'],rule='remove_whole_triangles_touching_dilated_reference_mask')
    if region.get('screen_polygon'):plan['image_support']=projection.region_support(region['screen_polygon'],check=check)
    return plan


def serpentine_plan(projection: Any,mesh: Any,region: Any,width_mm: float=10.,overlap_percent: float=20.,check: Callable[..., Any]=lambda:None,*,boundary_margin_mm: float=1.,direction: Any=None,start_corner: Any=None) -> Any:
    for value,lo,hi in ((width_mm,.2,100),(overlap_percent,0,90),(boundary_margin_mm,0,50)):
        if type(value) not in (int,float) or not np.isfinite(value) or not lo<=value<=hi:raise ValueError('Invalid serpentine parameters')
    modern=direction is not None or start_corner is not None
    direction=direction or 'horizontal';start_corner=start_corner or 'top_left'
    if direction not in ('horizontal','vertical') or start_corner not in ('top_left','top_right','bottom_left','bottom_right'):raise ValueError('Invalid sweep direction/start preference')
    allowed,removed=allowed_region(projection,mesh,region,check)
    mask=projection.valid & np.isin(projection.ids,allowed['triangle_indices'])
    h,w=mask.shape
    from femto.annotation.fixed_rgb_geometry import polygon_contains
    if region.get('screen_polygon'):
        for top in range(0,h,64):
            check();yy,xx=np.mgrid[top:min(top+64,h),:w]
            mask[top:top+len(yy)] &= polygon_contains(np.c_[xx.ravel(),yy.ravel()],region['screen_polygon']).reshape(xx.shape)
    yy,xx=np.nonzero(mask)
    if not len(yy):raise ValueError('No supported allowed surface')
    z=projection.z[mask];k=projection.k
    radius=(width_mm/2+boundary_margin_mm)/1000
    zmin=float(z.min())
    if zmin<=radius:raise ValueError('Scan radius exceeds working depth')
    slope=np.sqrt(((xx-k[0,2])/k[0,0])**2+((yy-k[1,2])/k[1,1])**2).max()
    pixel_radius=max(k[0,0],k[1,1])*radius/(zmin-radius)*(1+slope)+2
    distance=cv2.distanceTransform(np.pad(mask.astype(np.uint8),1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    safe=mask & (distance>pixel_radius)
    ys,xs=np.nonzero(safe)
    if not len(ys):raise ValueError('扫描宽度与边界余量约束后无可用路线，请调整模拟参数。')
    if len(ys)>800000:raise ValueError('Serpentine grid exceeds 800000 nodes; reduce selected region')
    # Neighbor connections require same or edge-adjacent source faces, and no depth jump.
    faces=np.asarray(mesh.triangles);n=len(faces)
    edges={};pairs=[]
    for tid in allowed['triangle_indices']:
        if tid%1024==0:check()
        for a,b in ((faces[tid,0],faces[tid,1]),(faces[tid,1],faces[tid,2]),(faces[tid,2],faces[tid,0])):
            edges.setdefault(tuple(sorted((int(a),int(b)))),[]).append(tid)
    for incident in edges.values():
        if len(incident)==2:pairs.append(min(incident)*n+max(incident))
    def connected(a,b,sa,sb,za,zb):
        return sa&sb&(np.abs(za-zb)<=.005)&((a==b)|np.isin(np.minimum(a,b).astype(np.int64)*n+np.maximum(a,b),pairs))
    ids=projection.ids;depth=projection.z
    right=connected(ids[:,:-1],ids[:,1:],safe[:,:-1],safe[:,1:],depth[:,:-1],depth[:,1:])
    down=connected(ids[:-1],ids[1:],safe[:-1],safe[1:],depth[:-1],depth[1:])
    vertical=direction=='vertical'
    row_step=max(1,int(round(width_mm*(1-overlap_percent/100)/1000*k[0 if vertical else 1,0 if vertical else 1]/float(np.median(z)))))
    grid=safe.T if vertical else safe
    links=down.T if vertical else right
    across=xs if vertical else ys
    reverse_rows=('right' in start_corner) if vertical else ('bottom' in start_corner)
    reverse_first=('bottom' in start_corner) if vertical else ('right' in start_corner)
    low,high=int(across.min()),int(across.max())
    rows=range(high,low-1,-row_step) if reverse_rows else range(low,high+1,row_step)
    runs=[]
    for ordinal,row_index in enumerate(rows):
        check();columns=np.flatnonzero(grid[row_index]);current=[];row=[]
        for column in columns:
            if current and (column!=current[-1]+1 or not links[row_index,current[-1]]):
                if len(current)>1:row.append(current)
                current=[]
            current.append(int(column))
        if len(current)>1:row.append(current)
        if bool(ordinal%2)^reverse_first:row=[list(reversed(run)) for run in reversed(row)]
        runs.extend([[(c,row_index) if vertical else (row_index,c) for c in run] for run in row])
    if not runs:raise ValueError('No usable sweep under selected direction and surface constraints')
    expanded=0
    def connect(start,goal):
        nonlocal expanded
        queue=[(0,0,start)];cost={start:0};previous={}
        while queue:
            _,g,node=heapq.heappop(queue)
            if g!=cost[node]:continue
            expanded+=1
            if expanded%2048==0:check()
            if expanded>4000000:raise ValueError('Serpentine connection search budget exceeded')
            if node==goal:
                route=[node]
                while route[-1]!=start:route.append(previous[route[-1]])
                return route[::-1]
            y,x=node
            neighbours=[]
            if x<w-1 and right[y,x]:neighbours.append((y,x+1))
            if x>0 and right[y,x-1]:neighbours.append((y,x-1))
            if y<h-1 and down[y,x]:neighbours.append((y+1,x))
            if y>0 and down[y-1,x]:neighbours.append((y-1,x))
            for nxt in neighbours:
                ng=g+1
                if ng<cost.get(nxt,float('inf')):
                    cost[nxt]=ng;previous[nxt]=node
                    heapq.heappush(queue,(ng+abs(nxt[0]-goal[0])+abs(nxt[1]-goal[1]),ng,nxt))
        return None
    chains=[];current=[];turn_edges=0;total=0;breaks=[]
    def sample(node,kind):
        y,x=node;hit=projection.pixel(x,y);tid=hit['triangle_index'];tri=np.asarray(mesh.vertices)[faces[tid]]
        normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=max(np.linalg.norm(normal),1e-15)
        return dict(xyz_m=hit['point_world_m'],pixel=[x,y],triangle_index=tid,normal=normal.tolist(),motion_kind=kind)
    for run in runs:
        check()
        if current:
            route=connect((current[-1]['pixel'][1],current[-1]['pixel'][0]),run[0])
            if route is None:
                breaks.append(dict(from_chain=len(chains),to_chain=len(chains)+1,
                    from_pixel=current[-1]['pixel'],to_pixel=[run[0][1],run[0][0]],
                    from_xyz_m=current[-1]['xyz_m'],to_xyz_m=projection.pixel(run[0][1],run[0][0])['point_world_m'],
                    reason='no_path_in_allowed_surface_graph',
                    explanation='有效表面图不连通：禁区、孔洞、遮挡/无支持、边界收缩或面邻接/深度门禁阻断；未规划跨链移动',
                    transition_planned=False,transition_time_s=None))
                chains.append(current);current=[]
            else:
                current.extend(sample(node,'connector') for node in route[1:]);turn_edges+=len(route)-1;total+=len(route)-1
        start=1 if current else 0
        current.extend(sample(node,'scan') for node in run[start:]);total+=len(run)-start
        if total>200000:raise ValueError('Serpentine path exceeds 200000 samples')
    if current:chains.append(current)
    return _finish_plan(projection,mesh,region,allowed,removed,chains,runs,breaks,faces,xx,yy,
        width_mm,overlap_percent,boundary_margin_mm,pixel_radius,row_step,turn_edges,total,
        modern,direction,start_corner,mask,safe,w,h,check)


from femto.planning.planning_state import versioned
serpentine_plan = versioned(serpentine_plan)
