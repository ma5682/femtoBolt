"""Versioned, evidence-based coverage inspection; unknown geometry has no area."""
from __future__ import annotations
from typing import Any, Callable

import copy
import numpy as np
import cv2
import open3d as o3d
from femto.planning.region_coverage import evaluate_coverage,point_segment_distance
from femto.annotation.fixed_rgb_geometry import polygon_contains
from femto.planning.planning_state import versioned

CAUSES={
 'forbidden':'禁区剔除', 'no_geometry':'无可信几何支持', 'selection_limit':'选区几何离散限制',
 'boundary_margin':'边界预留约束', 'boundary_clearance':'扫描包络边界约束', 'disconnected':'断连分量无路线',
 'route_spacing':'路线间距/布局', 'algorithm_limit':'算法未证实覆盖',
 'unknown':'未知', 'footprint_filter':'模拟覆盖范围约束断点'}
EVIDENCE={
 'selection_limit':'圈内像素有可信可见几何，但其三角面不在本区域保存的面集合中；选区离散限制，未纳入分母，不当作无几何。',
 'boundary_clearance':'未覆盖采样点若作为扫描中心，会违反模拟半径加预留的边界约束；不宣称该点必然不能被其他中心覆盖。',
 'boundary_margin':'采样点位于按边界预留距离计算的保守约束带；不是实际工具接触边界。',
 'disconnected':'允许网格的共边连通分量中没有任何路线种子；不根据链数推测。',
 'route_spacing':'到已有路线的欧氏距离下界大于模拟半径；只证明当前布局未到达，不保证减小行距即可解决。',
 'algorithm_limit':'距某路线节点的欧氏距离在模拟半径内，但网格距离上界超过半径；算法未证实覆盖，真实遗漏原因未知。',
 'unknown':'原因未知；现有检查没有足够证据确定原因。',
 'forbidden':'用户禁区及其保守整面剔除范围，排除于覆盖率分母；不是规划失败。',
 'no_geometry':'圈内参考像素无可信第一可见表面支持；只能定位图像，未知三维面积，不参与分母。'}


def pixel_runs(mask: Any) -> list[Any]:
    runs=[]
    for y,row in enumerate(mask):
        edges=np.flatnonzero(np.diff(np.r_[False,row,False].astype(np.int8)))
        runs.extend([[y,int(a),int(b)] for a,b in edges.reshape(-1,2)])
    return runs


def face_graph(mesh: Any,ids: Any) -> tuple[list[Any], dict[Any, Any]]:
    faces=np.asarray(mesh.triangles)[ids];edge_faces={};pairs=[]
    for i,face in enumerate(faces):
        for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
            edge_faces.setdefault(tuple(sorted((int(a),int(b)))),[]).append(i)
    for neighbours in edge_faces.values():
        if len(neighbours)==2:pairs.append(neighbours)
    graph=[[] for _ in ids]
    for a,b in pairs:graph[a].append(b);graph[b].append(a)
    return graph,edge_faces


def component_labels(graph: Any,nodes: Any=None) -> np.ndarray:
    nodes=list(range(len(graph))) if nodes is None else list(nodes)
    allowed=set(nodes);labels={}
    for node in nodes:
        if node in labels:continue
        label=len(labels);labels[node]=label;stack=[node]
        while stack:
            current=stack.pop()
            for other in graph[current]:
                if other in allowed and other not in labels:labels[other]=label;stack.append(other)
    return np.asarray([labels[node] for node in nodes],int)


def pixel_mask(projection: Any,polygon: Any,check: Any) -> np.ndarray:
    shape=(projection.height,projection.width)
    if not polygon:return np.ones(shape,bool)
    mask=np.zeros(shape,bool)
    for top in range(0,shape[0],64):
        check();yy,xx=np.mgrid[top:min(top+64,shape[0]),:shape[1]]
        mask[top:top+len(yy)]=polygon_contains(np.c_[xx.ravel(),yy.ravel()],polygon).reshape(xx.shape)
    return mask


def inspect_coverage(projection: Any,mesh: Any,region: Any,allowed: Any,plan: Any,check: Callable[..., Any]=lambda:None) -> tuple[Any, dict[str, Any]]:
    """Immutable report derived from final retained path; no repair or hole filling."""
    width=plan['coverage_parameters']['width_mm'];margin=plan['coverage_parameters']['boundary_margin_mm']/1000
    ids=np.asarray(allowed['triangle_indices'],int);faces=np.asarray(mesh.triangles);vertices=np.asarray(mesh.vertices)
    # Cross-face chords never act as face-local scan segments.
    seeds=[]
    for chain in plan['segments']:
        check()
        for a,b in zip(chain,chain[1:]):
            seeds.extend([[a,b]] if a['triangle_index']==b['triangle_index'] else [[a,a],[b,b]])
    if len(ids)*3>600000:raise ValueError('覆盖诊断采样超过60万点预算，请缩小区域')
    coverage=evaluate_coverage(mesh,allowed,dict(segments=seeds),width,check,details=True)
    xyz=np.asarray(coverage['sample_xyz_m']).reshape(-1,3);uv=projection.project(xyz)
    weights=np.repeat(np.asarray(coverage['face_area_m2'])/3,3)
    if len(xyz)>600000:raise ValueError('覆盖诊断采样超过60万点预算，请缩小区域')
    roi=pixel_mask(projection,region.get('screen_polygon'),check)
    supported=roi & projection.valid
    effective=supported & np.isin(projection.ids,ids)
    forbidden=np.zeros_like(roi)
    for polygon in region.get('exclusion_polygons',[]):forbidden |= pixel_mask(projection,polygon,check)
    removed=sorted(set(region['triangle_indices'])-set(ids.tolist()))
    forbidden=roi & (forbidden | (projection.valid & np.isin(projection.ids,removed)))
    missing_geometry=roi & ~projection.valid
    selection_limit=supported & ~effective & ~forbidden
    eligible=projection.visible(xyz,check=check)
    if region.get('screen_polygon'):eligible &= polygon_contains(uv,region['screen_polygon'])
    # Match the trusted raster layer, while visible() checks continuous first hit.
    pix=np.floor(uv+.5).astype(int);inside=(pix[:,0]>=0)&(pix[:,0]<projection.width)&(pix[:,1]>=0)&(pix[:,1]<projection.height)
    eligible &= inside
    positions=np.flatnonzero(inside)
    eligible[positions] &= effective[pix[positions,1],pix[positions,0]] & ~forbidden[pix[positions,1],pix[positions,0]]
    graph,edge_faces=face_graph(mesh,ids)
    components=component_labels(graph)
    seed_ids={p['triangle_index'] for chain in plan['segments'] for p in chain}
    seeded={int(components[i]) for i,tid in enumerate(ids) if int(tid) in seed_ids}
    connected=np.repeat(np.asarray([int(c) in seeded for c in components]),3)
    # Conservative Euclidean lower clearance agrees with the footprint guard.
    boundary=vertices[np.asarray([edge for edge,neighbours in edge_faces.items() if len(neighbours)==1],int)]
    if len(xyz)*len(boundary)>100_000_000:raise ValueError('覆盖边界采样检查超过预算，请缩小区域')
    clearance=np.full(len(xyz),np.inf)
    for i,(a,b) in enumerate(boundary):
        if i%128==0:check()
        clearance=np.minimum(clearance,point_segment_distance(xyz,a,b))
    reserved=eligible & (clearance<margin+1e-9) if margin else np.zeros(len(xyz),bool)
    center_limited=eligible & (clearance<(width/2000+margin))
    # Projected margin adds image ROI/unsupported boundaries without assigning an area to holes.
    image_distance=cv2.distanceTransform(np.pad((effective & ~forbidden).astype(np.uint8),1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    if margin:
        radius_px=max(projection.k[0,0],projection.k[1,1])*margin/np.maximum(xyz[:,2]-margin,1e-12)*(1+np.linalg.norm(xyz[:,:2],axis=1)/np.maximum(xyz[:,2],1e-12))
        indices=np.flatnonzero(eligible)
        reserved[indices] |= image_distance[pix[indices,1],pix[indices,0]]<=radius_px[indices]
    raw=np.asarray(coverage['sample_covered']).reshape(-1)
    covered=raw & eligible & ~reserved
    # Distance to vertices minus the longest edge is a lower bound to every chord.
    all_points=np.asarray([p['xyz_m'] for chain in plan['segments'] for p in chain])
    max_edge=max((np.linalg.norm(np.diff(np.asarray([p['xyz_m'] for p in chain]),axis=0),axis=1).max() for chain in plan['segments']),default=0.)
    lower=np.full(len(xyz),np.inf);nearest=np.full(len(xyz),np.inf)
    if len(all_points):
        search=o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(all_points));search.knn_index()
        for first in range(0,len(xyz),32768):
            check();_,squared=search.knn_search(o3d.core.Tensor(xyz[first:first+32768]),1)
            nearest[first:first+32768]=np.sqrt(squared.numpy().ravel())
            lower[first:first+32768]=np.maximum(0,nearest[first:first+32768]-max_edge)
    reasons=np.full(len(xyz),'unknown',object)
    reasons[eligible & ~covered & connected & (lower>width/2000+1e-12)]='route_spacing'
    # A finite graph estimate above radius is evidence of estimator rejection, not proof of true coverage.
    upper=np.array([float('inf') if d is None else d for row in coverage['sample_upper_distance_m'] for d in row])
    reasons[eligible & ~covered & connected & (nearest<=width/2000+1e-12) & np.isfinite(upper) & (upper>width/2000)]='algorithm_limit'
    reasons[eligible & ~covered & center_limited]='boundary_clearance'
    reasons[eligible & ~covered & ~connected]='disconnected'
    reasons[reserved]='boundary_margin'
    records=[]
    for index,point in enumerate(xyz):
        if index%2048==0:check()
        records.append(dict(triangle_index=int(ids[index//3]),sample_index=index%3,xyz_m=point.tolist(),pixel=uv[index].tolist(),
            represented_area_m2=float(weights[index]),eligible=bool(eligible[index]),covered=bool(covered[index]),
            reason=None if covered[index] or not eligible[index] else str(reasons[index]),
            evidence=None if covered[index] or not eligible[index] else dict(
                boundary_distance_lower_m=float(clearance[index]) if np.isfinite(clearance[index]) else None,
                route_distance_lower_m=float(lower[index]) if np.isfinite(lower[index]) else None,
                nearest_route_node_distance_m=float(nearest[index]) if np.isfinite(nearest[index]) else None,
                graph_distance_upper_m=float(upper[index]) if np.isfinite(upper[index]) else None,
                component_has_route_seed=bool(connected[index]),center_boundary_constraint=bool(center_limited[index]),
                margin_constraint=bool(reserved[index]))))
    omissions=[]
    def add(cause,indices=None,runs=None):
        indices=[] if indices is None else [int(v) for v in indices]
        pixels=np.asarray([records[i]['pixel'] for i in indices]) if indices else np.asarray([[x,y] for y,a,b in runs for x in (a,b-1)])
        bbox=[float(pixels[:,0].min()),float(pixels[:,1].min()),float(pixels[:,0].max()),float(pixels[:,1].max())]
        omissions.append(dict(id='omission_'+str(len(omissions)+1),reason=cause,label=CAUSES[cause],evidence=EVIDENCE[cause],
            in_denominator=bool(indices),sample_indices=indices,pixel_runs=runs or [],bbox_pixel=bbox,
            pixel_count=sum(b-a for _,a,b in runs) if runs else None,
            sampled_area_m2=float(weights[indices].sum()) if indices else None,
            focus_xyz_m=records[indices[0]]['xyz_m'] if indices else None))
    for cause in ('boundary_margin','boundary_clearance','disconnected','route_spacing','algorithm_limit','unknown'):
        sample_ids=np.flatnonzero(eligible & ~covered & (reasons==cause));face_ids=np.unique(sample_ids//3)
        if not len(face_ids):continue
        labels=component_labels(graph,face_ids)
        groups={}
        for index in sample_ids:groups.setdefault(int(labels[np.searchsorted(face_ids,index//3)]),[]).append(int(index))
        for group in sorted(groups):check();add(cause,indices=groups[group])
    for cause,mask in [('forbidden',forbidden),('no_geometry',missing_geometry),('selection_limit',selection_limit)]:
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=4)
        for index in range(1,n):
            check();x,y,w,h,_=stats[index]
            runs=[[int(yy+y),int(a+x),int(b+x)] for yy,a,b in pixel_runs(labels[y:y+h,x:x+w]==index)]
            add(cause,runs=runs)
    if len(omissions)>20000:raise ValueError('遗漏分量超过2万条预算，请缩小区域')
    denominator=float(weights[eligible].sum());area=float(weights[covered].sum())
    coverage.update(target_mesh_area_m2=denominator,sampled_covered_area_m2=area,
        sampled_coverage_percent=100*area/denominator if denominator else 0.,
        covered_samples_per_face=covered.reshape(-1,3).sum(axis=1).tolist(),
        eligible_samples_per_face=eligible.reshape(-1,3).sum(axis=1).tolist(),
        sample_covered=covered.reshape(-1,3).tolist(),denominator_method='sum_face_area_over_3_for_allowed_visible_in_roi_samples; excludes_forbidden_and_unsupported; includes_reserved_samples',
        sample_eligibility=eligible.reshape(-1,3).tolist(),method='three_barycentric_samples_per_face_area_weighted_mesh_graph_upper_distance_gated_v2')
    if denominator<=0:raise ValueError('覆盖报告无可用几何采样分母，未提交新方案')
    for key in ('sample_xyz_m','sample_upper_distance_m','sample_weights','face_area_m2','sample_covered','sample_eligibility'):
        coverage.pop(key,None)
    # Raster boundary margin layer has pixel units only; sample area is separate.
    boundary_mask=np.zeros_like(roi)
    if margin:
        yy,xx=np.nonzero(effective);z=projection.z[yy,xx]
        slope=np.sqrt(((xx-projection.k[0,2])/projection.k[0,0])**2+((yy-projection.k[1,2])/projection.k[1,1])**2)
        margin_px=max(projection.k[0,0],projection.k[1,1])*margin/np.maximum(z-margin,1e-12)*(1+slope)
        boundary_mask[yy,xx]=image_distance[yy,xx]<=margin_px
    layers={name:dict(pixel_runs=pixel_runs(mask),pixel_count=int(mask.sum())) for name,mask in
        [('selection',roi),('effective_geometry',effective & ~forbidden),('forbidden',forbidden),('boundary_margin',boundary_mask),('no_geometry',missing_geometry),('selection_limit',selection_limit)]}
    return coverage,dict(schema='femto-coverage-inspection-v1',method=coverage['method'],coordinate_frame='world_rdf',unit='metre',
        reference_image_size=[projection.width,projection.height],sampling=dict(samples_per_face=3,barycentric_weights=[[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]],
            density='fixed_per_face_not_uniform_per_square_metre',sample_count=len(records),eligible_sample_count=int(eligible.sum())),
        denominator=dict(method=coverage['denominator_method'],sampled_allowed_supported_area_m2=denominator,
            unknown_geometry_area_m2=None,forbidden_included=False,unsupported_included=False,boundary_margin_included=True),
        layers=layers,samples=records,omissions=omissions,chain_breaks=copy.deepcopy(plan['chain_breaks']),
        limitations=['离散面积加权采样，不保证完整曲面覆盖；采样密度随网格大小变化。',
            '图上界距离可能高估遗漏；跨面路线只以端点作为覆盖种子。',
            '禁区整面剔除和边界包络检查可能保守缩小可规划范围。',
            '像素区域无三维面积推算；分母是允许且可信的采样面积估计，并非精确裁剪曲面面积。',
            '边界预留图层是像素保守带，定量遗漏以三维采样点为准。',
            '只检查模拟半径包络，非真实接触、碰撞或临床精度验证。'],full_surface_coverage_verified=False)


def check_document_budget(plan: Any,check: Callable[..., Any]=lambda:None,limit: int=60_000_000) -> None:
    # Leave space for source metadata and indentation inside the 100 MB replay limit.
    import json
    total=0
    for ordinal,chunk in enumerate(json.JSONEncoder(ensure_ascii=False,indent=2,allow_nan=False).iterencode(plan)):
        if ordinal%4096==0:check()
        total+=len(chunk.encode('utf-8'))
        if total>limit:raise ValueError('覆盖方案明细超过可回读大小预算，请缩小区域；未提交新方案')


@versioned
def inspected_serpentine_plan(projection: Any,mesh: Any,region: Any,width_mm: float=10.,overlap_percent: float=20.,check: Callable[..., Any]=lambda:None,*,boundary_margin_mm: float=1.,direction: str='horizontal',start_corner: str='top_left') -> Any:
    from femto.planning.serpentine_path import serpentine_plan
    from femto.planning.region_exclusions import allowed_region
    from femto.planning.region_path_guard import constrain_path
    plan=serpentine_plan(projection,mesh,region,width_mm,overlap_percent,check,boundary_margin_mm=boundary_margin_mm,direction=direction,start_corner=start_corner)
    allowed,_=allowed_region(projection,mesh,region,check)
    original_breaks=plan['chain_breaks'];original_guard=plan['boundary_check']
    guard=constrain_path(projection,mesh,allowed,plan,width_mm,boundary_margin_mm,check)
    if not plan['segments']:raise ValueError('模拟扫描范围及边界/禁区约束后无可用路线；保留旧方案，不能自动放宽约束')
    plan['coverage_parameters']['pattern']='serpentine_v3'
    plan['boundary_check']=dict(guard,grid_precheck=original_guard,accepted_footprint_checked=True)
    breaks=[]
    for i,(left,right) in enumerate(zip(plan['segments'],plan['segments'][1:])):
        old=next((b for b in original_breaks if b['from_pixel']==left[-1]['pixel'] and b['to_pixel']==right[0]['pixel']),None)
        gap=copy.deepcopy(old) if old else dict(reason='footprint_filter',explanation='完整弦段的模拟包络、边界/禁区或端点/中点可见性门禁剔除了中间路线；未分辨唯一触发项，跨链运动未规划')
        gap.update(from_chain=i,to_chain=i+1,from_pixel=left[-1]['pixel'],to_pixel=right[0]['pixel'],from_xyz_m=left[-1]['xyz_m'],to_xyz_m=right[0]['xyz_m'],transition_planned=False,transition_time_s=None)
        breaks.append(gap)
    plan['chain_breaks']=breaks
    plan['route_order'].update(continuous_chains=len(plan['segments']),disconnected_breaks=len(breaks),
        connector_edges=sum(p.get('motion_kind')=='connector' for c in plan['segments'] for p in c[1:]))
    first=plan['segments'][0][0];start=plan['start_selection']
    if start['actual_pixel']!=first['pixel']:
        start.update(actual_pixel=first['pixel'],actual_xyz_m=first['xyz_m'],adjusted=True,adjustment_cause='模拟覆盖包络进一步约束首端点',reason='原首端点未通过整个模拟覆盖范围检查；按保留路线顺序调整')
    plan['motion_demo']['cross_chain_time_s']=None if breaks else 0.
    plan['coverage'],plan['coverage_inspection']=inspect_coverage(projection,mesh,region,allowed,plan,check)
    check_document_budget(plan,check)
    return plan
