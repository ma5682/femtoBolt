"""Measured boundary-edge row turns; never invent a bridge across a mesh gap."""
from collections import Counter, defaultdict
from dataclasses import replace
import heapq
import math

import numpy as np

from femto.planning.surface_trajectory import Gap, POINT_BUDGET, ScanSegment, validate_trajectory


def _intersects(a, b, c, d):
    def cross(u, v):return float(u[0]*v[1]-u[1]*v[0])
    if np.any(np.maximum(np.minimum(a,b),np.minimum(c,d)) >
              np.minimum(np.maximum(a,b),np.maximum(c,d))+1e-9):return False
    return (cross(b-a,c-a)*cross(b-a,d-a)<=1e-12 and
            cross(d-c,a-c)*cross(d-c,b-c)<=1e-12)


class BoundaryTurns:
    def __init__(self, mesh, region, frame, check, face_ids=None, snap_tolerance_m=1e-8):
        from femto.planning.mesh_slice_planner import _forbidden_polygons
        self.vertices=np.asarray(mesh.vertices,float)
        self.check=check;self.frame=frame
        # Numerical tolerance for recognising a row end that lies on a boundary edge
        # (slice ends at shared vertices differ by ~1e-8 m between triangles). Not a geometry gate.
        self.snap=float(snap_tolerance_m)
        view=region['view_reference']
        self.k=np.asarray(view['undistorted_intrinsic']);self.size=np.asarray(view['image_size'])
        self.polygons=_forbidden_polygons(region)
        faces=np.asarray(mesh.faces)
        # Restricting to one block keeps its boundary graph free of any other
        # surface, even one that only touches or overlaps it geometrically.
        selected=range(len(faces)) if face_ids is None else [int(v) for v in face_ids]
        edges=defaultdict(list)
        for ordinal,index in enumerate(selected):
            if ordinal%1024==0:check()
            face=faces[index]
            for a,b in zip(face,np.roll(face,-1)):
                edges[tuple(sorted((int(a),int(b))))].append(int(index))
        self.edges=[(a,b,ids[0]) for (a,b),ids in sorted(edges.items()) if len(ids)==1]
        degrees=Counter(v for a,b,_ in self.edges for v in (a,b))
        self.ambiguous={v for v,count in degrees.items() if count!=2}
        self.e2=None if frame is None else np.asarray(frame['e2'])

    def allowed(self, a, b):
        from femto.planning.mesh_slice_planner import _project, _point_in_polygon
        uv=[_project(p,self.k,self.size) for p in (a,b)]
        if any(p is None for p in uv):return False
        u,v=map(np.asarray,uv)
        for polygon in self.polygons:
            if _point_in_polygon(u,polygon) or _point_in_polygon(v,polygon):return False
            if any(_intersects(u,v,c,d) for c,d in zip(polygon,np.roll(polygon,-1,axis=0))):return False
        return True

    def route(self, left, right, step, e2=None):
        e2=self.e2 if e2 is None else np.asarray(e2,float)
        if e2 is None:raise ValueError('Row-turn routing needs a sweep frame')
        a=np.asarray(left['xyz_m']);b=np.asarray(right['xyz_m'])
        low,high=sorted((float(a@e2),float(b@e2)))
        vertices=self.vertices
        points={-2:a,-1:b}
        graph=defaultdict(list)
        def link(i,j,face):
            if i in self.ambiguous or j in self.ambiguous:return
            p=points[i] if i<0 else vertices[i];q=points[j] if j<0 else vertices[j]
            if not (low-1e-9<=p@e2<=high+1e-9 and low-1e-9<=q@e2<=high+1e-9):return
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
                if -1e-9<=t<=1+1e-9 and np.linalg.norm(p-(u+np.clip(t,0,1)*delta))<=self.snap:
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
            if len(samples)+count>POINT_BUDGET:raise ValueError(f'Row turn exceeds {POINT_BUDGET} samples')
            for t in np.linspace(0,1,count+1)[1:]:
                xyz=p*(1-t)+q*t
                samples.append(dict(xyz_m=xyz.tolist(),pixel=_project(xyz,self.k,self.size),
                    triangle_index=int(face),geometry_support='measured_mesh',role='connector',
                    row_index=right['row_index']))
        return samples


    def _distance_to_boundary(self, point):
        best=float('inf');p=np.asarray(point,float)
        for i,j,_ in self.edges:
            u,v=self.vertices[i],self.vertices[j];delta=v-u;den=float(delta@delta)
            if den<=1e-24:continue
            t=float(np.clip((p-u)@delta/den,0,1))
            best=min(best,float(np.linalg.norm(p-(u+t*delta))))
        return best

    def diagnose(self, left, right, step, e2=None):
        """Why no measured boundary turn exists between two row ends; diagnostic only, never a decision."""
        if max(self._distance_to_boundary(left['xyz_m']),self._distance_to_boundary(right['xyz_m']))>self.snap:
            return 'endpoint_not_on_boundary'
        polygons=self.polygons
        try:
            self.polygons=[]
            if self.route(left,right,step,e2):return 'forbidden_blocked'
        finally:self.polygons=polygons
        ambiguous=self.ambiguous
        try:
            self.ambiguous=set()
            if self.route(left,right,step,e2):return 'ambiguous_boundary'
        finally:self.ambiguous=ambiguous
        return 'no_boundary_path_in_band'


def connect_segments(segments, gaps, eligible, router, step, check, *, e2=None, used=0):
    """Merge eligible row-turn gaps along measured boundaries; every other gap is kept."""
    chains=[[dict(p) for p in segments[0].points]];kept=[];connected=0;length=0.;added=0
    for index,gap in enumerate(gaps):
        check();following=segments[index+1]
        route=None
        if gap.reason=='row_turn' and index in eligible:
            route=router.route(chains[-1][-1],following.points[0],step,e2)
        if route:
            xyz=np.asarray([chains[-1][-1]['xyz_m']]+[p['xyz_m'] for p in route])
            length+=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum())
            chains[-1].extend(route);chains[-1].extend(dict(p) for p in following.points[1:])
            connected+=1;added+=len(route)-1
        else:
            kept.append(gap);chains.append([dict(p) for p in following.points])
        if used+added>POINT_BUDGET:
            raise ValueError(f'Trajectory including row turns exceeds {POINT_BUDGET} samples')
    return chains,kept,connected,length,added


def chain_items(chains, kept, *, route_start=0, segment_start=0, gap_start=0, block_index=None):
    """Number merged chains globally and re-anchor kept gaps on the chain endpoints."""
    segments=[];route_index=route_start
    for offset,points in enumerate(chains):
        for point in points:point['route_index']=route_index;route_index+=1
        segments.append(ScanSegment(segment_start+offset,points[0]['row_index'],points,block_index))
    items=[]
    for offset,segment in enumerate(segments):
        items.append(segment)
        if offset<len(kept):
            before=segment.points[-1];after=segments[offset+1].points[0]
            items.append(replace(kept[offset],gap_index=gap_start+offset,
                previous_point=dict(route_index=before['route_index'],xyz_m=before['xyz_m']),
                next_point=dict(route_index=after['route_index'],xyz_m=after['xyz_m'])))
    return items,route_index


def connect_row_turns(trajectory, mesh, region, eligible, check):
    router=BoundaryTurns(mesh,region,trajectory.planner['sweep_frame'],check)
    step=trajectory.planner['parameters']['sample_step_mm']/1000
    old=trajectory.ordered_items
    segments=[s for s in old if isinstance(s,ScanSegment)];gaps=[g for g in old if isinstance(g,Gap)]
    chains,kept,connected,length,added=connect_segments(segments,gaps,eligible,router,step,check,
        used=trajectory.statistics['point_count'])
    items,route_index=chain_items(chains,kept)
    counts=dict(sorted(Counter(g.reason for g in kept).items()))
    row_report=[dict(row,gap_count=sum(g.row_index_from==row['row_index'] for g in kept))
                for row in trajectory.row_report]
    # Spacing is measured on original scan slices, not on the merged playback chains.
    spacing=dict(trajectory.spacing_report)
    spacing['scan_segment_rows']=[s.row_index for s in old if isinstance(s,ScanSegment)]
    planner=dict(trajectory.planner,algorithm_version='mesh-slice-serpentine-v2',
                 parameters=dict(trajectory.planner['parameters'],connect_rows=True))
    stats=dict(trajectory.statistics,segment_count=len(chains),gap_count=len(kept),gap_counts=counts,
               point_count=route_index,connector_length_m=length,connected_row_turns=connected,
               total_length_m=trajectory.statistics['scan_length_m']+length)
    result=replace(trajectory,planner=planner,ordered_items=items,statistics=stats,
                   gap_report=dict(trajectory.gap_report,counts=counts),row_report=row_report,
                   spacing_report=spacing)
    validate_trajectory(result,mesh=mesh)
    return result
