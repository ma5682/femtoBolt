"""Measured boundary-edge row turns; never invent a bridge across a mesh gap."""
from collections import Counter, defaultdict
from dataclasses import replace
import heapq
import math

import numpy as np

from femto.planning.surface_trajectory import Gap, ScanSegment, validate_trajectory


def _intersects(a, b, c, d):
    def cross(u, v):return float(u[0]*v[1]-u[1]*v[0])
    if np.any(np.maximum(np.minimum(a,b),np.minimum(c,d)) >
              np.minimum(np.maximum(a,b),np.maximum(c,d))+1e-9):return False
    return (cross(b-a,c-a)*cross(b-a,d-a)<=1e-12 and
            cross(d-c,a-c)*cross(d-c,b-c)<=1e-12)


class BoundaryTurns:
    def __init__(self, mesh, region, frame, check):
        from femto.planning.mesh_slice_planner import _forbidden_polygons
        self.vertices=np.asarray(mesh.vertices,float)
        self.check=check;self.frame=frame
        view=region['view_reference']
        self.k=np.asarray(view['undistorted_intrinsic']);self.size=np.asarray(view['image_size'])
        self.polygons=_forbidden_polygons(region)
        edges=defaultdict(list)
        for index,face in enumerate(mesh.faces):
            if index%1024==0:check()
            for a,b in zip(face,np.roll(face,-1)):
                edges[tuple(sorted((int(a),int(b))))].append(index)
        self.edges=[(a,b,ids[0]) for (a,b),ids in sorted(edges.items()) if len(ids)==1]
        degrees=Counter(v for a,b,_ in self.edges for v in (a,b))
        self.ambiguous={v for v,count in degrees.items() if count!=2}
        self.e2=np.asarray(frame['e2'])

    def allowed(self, a, b):
        from femto.planning.mesh_slice_planner import _project, _point_in_polygon
        uv=[_project(p,self.k,self.size) for p in (a,b)]
        if any(p is None for p in uv):return False
        u,v=map(np.asarray,uv)
        for polygon in self.polygons:
            if _point_in_polygon(u,polygon) or _point_in_polygon(v,polygon):return False
            if any(_intersects(u,v,c,d) for c,d in zip(polygon,np.roll(polygon,-1,axis=0))):return False
        return True

    def route(self, left, right, step):
        a=np.asarray(left['xyz_m']);b=np.asarray(right['xyz_m'])
        low,high=sorted((float(a@self.e2),float(b@self.e2)))
        vertices=self.vertices
        points={-2:a,-1:b}
        graph=defaultdict(list)
        def link(i,j,face):
            if i in self.ambiguous or j in self.ambiguous:return
            p=points[i] if i<0 else vertices[i];q=points[j] if j<0 else vertices[j]
            if not (low-1e-9<=p@self.e2<=high+1e-9 and low-1e-9<=q@self.e2<=high+1e-9):return
            if not self.allowed(p,q):return
            length=float(np.linalg.norm(q-p))
            graph[i].append((j,length,face));graph[j].append((i,length,face))
        for index,(i,j,face) in enumerate(self.edges):
            if index%256==0:self.check()
            u,v=vertices[i],vertices[j];delta=v-u;den=float(delta@delta)
            if den<=1e-24:continue
            link(i,j,face)
            on=[]
            for node,p in points.items():
                t=float((p-u)@delta)/den
                if -1e-9<=t<=1+1e-9 and np.linalg.norm(p-(u+np.clip(t,0,1)*delta))<=1e-8:
                    on.append(node);link(node,i,face);link(node,j,face)
            if len(on)==2:link(-2,-1,face)
        distances={-2:0.};previous={};queue=[(0.,-2)]
        while queue:
            self.check();distance,node=heapq.heappop(queue)
            if distance!=distances[node]:continue
            if node==-1:break
            for nxt,length,face in sorted(graph[node]):
                cost=distance+length
                if cost<distances.get(nxt,float('inf')):
                    distances[nxt]=cost;previous[nxt]=(node,face);heapq.heappush(queue,(cost,nxt))
        if -1 not in previous:return None
        path=[];node=-1
        while node!=-2:
            prev,face=previous[node];path.append((prev,node,face));node=prev
        from femto.planning.mesh_slice_planner import _project
        samples=[]
        # Keep every mesh-edge corner so consecutive samples share a real face.
        for i,j,face in reversed(path):
            self.check()
            p=points[i] if i<0 else vertices[i];q=points[j] if j<0 else vertices[j]
            count=max(1,math.ceil(float(np.linalg.norm(q-p))/step))
            if len(samples)+count>200000:raise ValueError('Row turn exceeds 200000 samples')
            for t in np.linspace(0,1,count+1)[1:]:
                xyz=p*(1-t)+q*t
                samples.append(dict(xyz_m=xyz.tolist(),pixel=_project(xyz,self.k,self.size),
                    triangle_index=int(face),geometry_support='measured_mesh',role='connector',
                    row_index=right['row_index']))
        return samples


def connect_row_turns(trajectory, mesh, region, eligible, check):
    router=BoundaryTurns(mesh,region,trajectory.planner['sweep_frame'],check)
    step=trajectory.planner['parameters']['sample_step_mm']/1000
    old=trajectory.ordered_items
    chains=[[dict(p) for p in old[0].points]];gaps=[];connected=0;length=0.;added=0
    for index in range(1,len(old),2):
        check();gap=old[index];following=old[index+1]
        route=None
        if gap.reason=='row_turn' and gap.gap_index in eligible:
            route=router.route(chains[-1][-1],following.points[0],step)
        if route:
            xyz=np.asarray([chains[-1][-1]['xyz_m']]+[p['xyz_m'] for p in route])
            length+=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum())
            chains[-1].extend(route);chains[-1].extend(dict(p) for p in following.points[1:])
            connected+=1;added+=len(route)-1
        else:
            gaps.append(gap);chains.append([dict(p) for p in following.points])
        if trajectory.statistics['point_count']+added>200000:
            raise ValueError('Trajectory including row turns exceeds 200000 samples')
    segments=[];route_index=0
    for index,points in enumerate(chains):
        for point in points:point['route_index']=route_index;route_index+=1
        segments.append(ScanSegment(index,points[0]['row_index'],points))
    items=[]
    for index,segment in enumerate(segments):
        items.append(segment)
        if index<len(gaps):
            before=segment.points[-1];after=segments[index+1].points[0]
            items.append(replace(gaps[index],gap_index=index,
                previous_point=dict(route_index=before['route_index'],xyz_m=before['xyz_m']),
                next_point=dict(route_index=after['route_index'],xyz_m=after['xyz_m'])))
    counts=dict(sorted(Counter(g.reason for g in gaps).items()))
    row_report=[dict(row,gap_count=sum(g.row_index_from==row['row_index'] for g in gaps))
                for row in trajectory.row_report]
    # Spacing is measured on original scan slices, not on the merged playback chains.
    spacing=dict(trajectory.spacing_report)
    spacing['scan_segment_rows']=[s.row_index for s in old if isinstance(s,ScanSegment)]
    planner=dict(trajectory.planner,algorithm_version='mesh-slice-serpentine-v2',
                 parameters=dict(trajectory.planner['parameters'],connect_rows=True))
    stats=dict(trajectory.statistics,segment_count=len(segments),gap_count=len(gaps),gap_counts=counts,
               point_count=route_index,connector_length_m=length,connected_row_turns=connected,
               total_length_m=trajectory.statistics['scan_length_m']+length)
    result=replace(trajectory,planner=planner,ordered_items=items,statistics=stats,
                   gap_report=dict(trajectory.gap_report,counts=counts),row_report=row_report,
                   spacing_report=spacing)
    validate_trajectory(result,mesh=mesh)
    return result
