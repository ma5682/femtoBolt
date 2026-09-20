"""Mesh-edge distance isolines, in metres; approximate surface coverage only."""
import heapq
import numpy as np
from region_path import signature


def metric_path(projection, mesh, region, spacing_mm=5., sample_step_mm=1., check=lambda: None, allow_empty=False, *, include_levels=False):
    if region.get('exclusion_polygons'):raise ValueError('存在禁止区域，请使用区域覆盖规划。')
    for value in (spacing_mm, sample_step_mm):
        if type(value) not in (int, float) or not np.isfinite(value) or not .1 <= value <= 100:
            raise ValueError('Metric spacing must be 0.1–100 mm')
    vertices=np.asarray(mesh.vertices); faces=np.asarray(mesh.triangles)
    ids=np.asarray(region['triangle_indices'],dtype=int)
    if not len(ids):raise ValueError('Empty region')
    selected=faces[ids]; used=np.unique(selected)
    graph={int(v):{} for v in used}; edges={}
    for ordinal,face in enumerate(selected):
        if ordinal%1024==0:check()
        for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
            a,b=int(a),int(b);key=tuple(sorted((a,b)))
            edges[key]=edges.get(key,0)+1
            weight=float(np.linalg.norm(vertices[a]-vertices[b]))
            graph[a][b]=weight;graph[b][a]=weight
    if any(count>2 for count in edges.values()):raise ValueError('Non-manifold region; cannot form scan contours')
    boundary={v for edge,count in edges.items() if count==1 for v in edge}
    # A separate upper boundary seed set per connected component avoids
    # connecting disconnected islands or inventing travel across a hole.
    remaining=set(graph); seeds=[]
    while remaining:
        check();root=min(remaining);remaining.remove(root);component={root};stack=[root]
        while stack:
            v=stack.pop()
            if len(component)%1024==0:check()
            for neighbour in graph[v]:
                if neighbour in remaining:
                    remaining.remove(neighbour);component.add(neighbour);stack.append(neighbour)
        candidates=component & boundary
        if not candidates:raise ValueError('Region component has no boundary')
        top=min(vertices[v,1] for v in candidates)
        seeds.extend(v for v in candidates if vertices[v,1]<=top+1e-9)
    distances={v:float('inf') for v in graph};queue=[]
    for v in seeds:distances[v]=0.;heapq.heappush(queue,(0.,v))
    iterations=0
    while queue:
        distance,v=heapq.heappop(queue);iterations+=1
        if iterations%1024==0:check()
        if distance!=distances[v]:continue
        for neighbour,weight in graph[v].items():
            candidate=distance+weight
            if candidate<distances[neighbour]:
                distances[neighbour]=candidate;heapq.heappush(queue,(candidate,neighbour))
    field=np.full(len(vertices),np.inf)
    for v,distance in distances.items():field[v]=distance
    maximum=max(distances.values());spacing=spacing_mm/1000.;step=sample_step_mm/1000.
    if maximum/spacing*len(ids)>20_000_000:raise ValueError('Contour budget exceeded; increase spacing')
    levels=np.arange(spacing/2.,maximum,spacing)
    if not len(levels) and not allow_empty:raise ValueError('Region smaller than half the requested spacing')
    if len(levels)*len(ids)>20_000_000:raise ValueError('Contour budget exceeded; increase spacing')
    segments=[];total=0
    for level in levels:
        check();pieces=[];adjacency={}
        for ordinal,(face_id,face) in enumerate(zip(ids,selected)):
            if ordinal%1024==0:check()
            hits={}
            for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
                da,db=field[a],field[b]
                if (da<=level<db) or (db<=level<da):
                    ratio=(level-da)/(db-da)
                    key=('v',int(a)) if ratio==0 else ('v',int(b)) if ratio==1 else ('e',*sorted((int(a),int(b))))
                    hits[key]=vertices[a]+ratio*(vertices[b]-vertices[a])
            if len(hits)!=2:continue
            (ka,pa),(kb,pb)=hits.items()
            if np.linalg.norm(pb-pa)<1e-12:continue
            index=len(pieces);pieces.append((ka,kb,pa,pb,int(face_id)))
            adjacency.setdefault(ka,[]).append(index);adjacency.setdefault(kb,[]).append(index)
        unused=set(range(len(pieces)))
        while unused:
            first=min(unused);a,b,*_=pieces[first]
            start=next((key for key,links in adjacency.items() if len(links)!=2 and any(i in unused for i in links)),a)
            current=start;segment=[]
            while True:
                options=[i for i in adjacency[current] if i in unused]
                if not options:break
                index=min(options);unused.remove(index)
                a,b,pa,pb,face_id=pieces[index]
                if current==b:a,b,pa,pb=b,a,pb,pa
                triangle=vertices[faces[face_id]]
                normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])
                normal/=max(float(np.linalg.norm(normal)),1e-12)
                count=max(1,int(np.ceil(np.linalg.norm(pb-pa)/step)))
                if total+count+1>200000:raise ValueError('Path exceeds 200000 samples; increase spacing')
                for t in np.linspace(0.,1.,count+1)[1 if segment else 0:]:
                    point=pa+t*(pb-pa);pixel=projection.project([point])[0]
                    if not np.isfinite(pixel).all():raise ValueError('Path projects outside valid camera geometry')
                    segment.append(dict(xyz_m=point.tolist(),normal=normal.tolist(),pixel=pixel.tolist(),triangle_index=face_id))
                    if include_levels:segment[-1].update(surface_level_m=float(level),motion_kind='scan')
                    total+=1
                current=b
                if len(adjacency[current])!=2:break
            if len(segment)>1:segments.append(segment)
    if not segments and not allow_empty:raise ValueError('No metric contour generated')
    return dict(schema='femto-region-metric-path-v1',coordinate_frame='world_rdf',unit='metre',
        region_id=region['annotation_id'],region_signature=signature(region),spacing_mm=spacing_mm,
        sample_step_mm=sample_step_mm,segments=segments,
        spacing_semantics='mesh_edge_shortest_distance_isolines_approximate_surface_distance',
        seed_rule='minimum_W0_Y_boundary_vertices_per_component',seed_vertex_ids=sorted(seeds),
        first_level_mm=spacing_mm/2.,connection_policy='separate_segments_no_transition_moves',
        normal_semantics='mesh_winding_normal_not_tool_orientation',
        surface_constraint='piecewise_linear_on_selected_mesh_not_collision_checked',
        execution_ready=False,physical_accuracy_verified=False)


from planning_state import versioned
metric_path = versioned(metric_path)
