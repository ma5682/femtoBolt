"""Conservative mesh-graph reach estimates at area-weighted surface samples."""
import heapq
import numpy as np
from femto.planning.metric_region_path import metric_path


def point_segment_distance(points,a,b):
    delta=b-a;den=float(delta@delta)
    t=np.clip((points-a)@delta/max(den,1e-30),0,1)
    return np.linalg.norm(points-(a+t[:,None]*delta),axis=1)


def evaluate_coverage(mesh,region,plan,width_mm,check=lambda:None,*,details=False):
    vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles)
    ids=np.asarray(region['triangle_indices'],int);triangles=vertices[faces[ids]]
    selected_ids=set(ids.tolist())
    weights=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    samples=np.einsum('ij,tjk->tik',weights,triangles)
    graph={int(v):{} for v in np.unique(faces[ids])}
    distances={v:float('inf') for v in graph};by_face={}
    for ordinal,face in enumerate(faces[ids]):
        if ordinal%1024==0:check()
        for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
            length=float(np.linalg.norm(vertices[a]-vertices[b]));graph[a][b]=length;graph[b][a]=length
    for segment in plan['segments']:
        check()
        for left,right in zip(segment,segment[1:]):
            # Metric generator emits face-local edges; the destination owns
            # the shared-boundary starting point after changing triangles.
            face_id=right['triangle_index']
            if face_id not in selected_ids:continue
            a=np.asarray(left['xyz_m']);b=np.asarray(right['xyz_m'])
            by_face.setdefault(face_id,[]).append((a,b))
            for v,d in zip(faces[face_id],point_segment_distance(vertices[faces[face_id]],a,b)):
                if v in distances:distances[v]=min(distances[v],float(d))
    queue=[(d,int(v)) for v,d in distances.items() if np.isfinite(d)];heapq.heapify(queue)
    count=0
    while queue:
        d,v=heapq.heappop(queue);count+=1
        if count%1024==0:check()
        if d!=distances[v]:continue
        for other,length in graph[v].items():
            if d+length<distances[other]:distances[other]=d+length;heapq.heappush(queue,(d+length,int(other)))
    reachable=[];estimates=[]
    for i,face_id in enumerate(ids):
        if i%256==0:check()
        ds=np.array([distances[v] for v in faces[face_id]])
        estimate=(np.linalg.norm(samples[i,:,None,:]-triangles[i,None,:,:],axis=2)+ds).min(axis=1)
        for a,b in by_face.get(int(face_id),[]):estimate=np.minimum(estimate,point_segment_distance(samples[i],a,b))
        reachable.append(estimate<=width_mm/2000+1e-12)
        if details:estimates.append(estimate.tolist())
    reachable=np.asarray(reachable)
    area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
    covered=float((area*reachable.mean(axis=1)).sum());total=float(area.sum())
    result=dict(method='three_barycentric_samples_per_face_area_weighted_mesh_graph_upper_distance',
        triangle_indices=ids.tolist(),covered_samples_per_face=reachable.sum(axis=1).tolist(),
        sampled_covered_area_m2=covered,target_mesh_area_m2=total,
        sampled_coverage_percent=100*covered/total if total else 0.,
        samples_per_face=3,full_surface_coverage_verified=False,
        limitations='Discrete sample estimate, graph routes may overestimate distance; excludes missing geometry and is not treatment coverage.')
    if details:
        result.update(sample_xyz_m=samples.tolist(),sample_covered=reachable.tolist(),
            sample_upper_distance_m=[[float(d) if np.isfinite(d) else None for d in row] for row in estimates],
            sample_weights=weights.tolist(),face_area_m2=area.tolist())
    return result


def coverage_plan(projection,mesh,region,width_mm=10.,overlap_percent=20.,check=lambda:None,*,boundary_margin_mm=None):
    from femto.planning.region_exclusions import allowed_region
    from femto.planning.region_path import signature
    original_region=region
    region,excluded_ids=allowed_region(projection,mesh,region,check)
    if region.get('exclusion_polygons') and boundary_margin_mm is None:boundary_margin_mm=0.
    if type(width_mm) not in (int,float) or not np.isfinite(width_mm) or not .2<=width_mm<=100:raise ValueError('Scan width must be 0.2–100 mm')
    if type(overlap_percent) not in (int,float) or not np.isfinite(overlap_percent) or not 0<=overlap_percent<=90:raise ValueError('Overlap must be 0–90%')
    if boundary_margin_mm is not None and (type(boundary_margin_mm) not in (int,float) or not np.isfinite(boundary_margin_mm) or not 0<=boundary_margin_mm<=50):raise ValueError('Boundary margin must be 0–50 mm')
    spacing=max(.1,width_mm*(1-overlap_percent/100))
    plan=metric_path(projection,mesh,dict(region,exclusion_polygons=[]),spacing,min(1.,width_mm/4),check,allow_empty=True)
    plan['region_signature']=signature(original_region)
    report=evaluate_coverage(mesh,region,plan,width_mm,check)
    plan['coverage_parameters']=dict(width_mm=width_mm,overlap_percent=overlap_percent)
    # Bounded supplementary face-local sweeps. Never bridge unsupported areas.
    vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles)
    missing=[tid for tid,n in zip(report['triangle_indices'],report['covered_samples_per_face']) if n<3]
    added=0
    for tid in missing[:500]:
        check();triangle=vertices[faces[tid]]
        normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);normal/=max(np.linalg.norm(normal),1e-15)
        a=(triangle[0]+triangle[1])/2;b=triangle[2]*.8+(triangle[0]+triangle[1])*.1
        length=np.linalg.norm(b-a)
        if length<1e-10:continue
        count=max(2,int(np.ceil(length/(min(1.,width_mm/4)/1000)))+1)
        if sum(map(len,plan['segments']))+count>200000:break
        xyz=np.linspace(a,b,count);uv=projection.project(xyz)
        plan['segments'].append([dict(xyz_m=p.tolist(),pixel=q.tolist(),normal=normal.tolist(),triangle_index=int(tid)) for p,q in zip(xyz,uv)])
        added+=1
    if boundary_margin_mm is not None:
        from femto.planning.region_path_guard import constrain_path
        plan['boundary_check']=constrain_path(projection,mesh,region,plan,width_mm,boundary_margin_mm,check)
        plan['coverage_parameters']['boundary_margin_mm']=boundary_margin_mm
        if not plan['segments']:raise ValueError('扫描宽度与边界余量约束后无可用路线；请检查选区或调整模拟宽度，不能自动跨越边界。')
    final=evaluate_coverage(mesh,region,plan,width_mm,check)
    if not plan['segments'] or final['target_mesh_area_m2']<=0:raise ValueError('Region has no usable surface path')
    final.update(initial_sampled_coverage_percent=report['sampled_coverage_percent'],supplementary_segments=added,
                 supplementary_limit=500)
    plan['coverage']=final
    if excluded_ids:
        plan['exclusions']=dict(polygons=original_region['exclusion_polygons'],removed_triangle_indices=excluded_ids,
            allowed_triangle_indices=region['triangle_indices'],rule='remove_whole_triangles_touching_dilated_reference_mask',
            coverage_denominator='allowed_mesh_only_excludes_forbidden_and_missing_surface')
    plan['route_order']=dict(mode='ordered_segments',start=plan['segments'][0][0]['xyz_m'],
        end=plan['segments'][-1][-1]['xyz_m'],transitions='unplanned_no_interpolation')
    if region.get('screen_polygon') and hasattr(projection,'region_support'):
        plan['image_support']=projection.region_support(region['screen_polygon'],check=check)
    plan['motion_demo']=dict(speed_mm_s=20.,segment_gap_seconds=.5,
                             transitions_planned=False,execution_ready=False)
    return plan


from femto.planning.planning_state import versioned
coverage_plan = versioned(coverage_plan)
