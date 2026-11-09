"""Measured boundary-edge row turns; never invent a bridge across a mesh gap."""
from collections import Counter, defaultdict
from dataclasses import replace
import heapq
import math

import numpy as np

from femto.planning.surface_trajectory import Gap, POINT_BUDGET, ScanSegment, validate_trajectory
from femto.planning.surface_polygon_index import PreparedPolygonIndex


def _intersects(a, b, c, d):
    def cross(u, v):return float(u[0]*v[1]-u[1]*v[0])
    if np.any(np.maximum(np.minimum(a,b),np.minimum(c,d)) >
              np.minimum(np.maximum(a,b),np.maximum(c,d))+1e-9):return False
    return (cross(b-a,c-a)*cross(b-a,d-a)<=1e-12 and
            cross(d-c,a-c)*cross(d-c,b-c)<=1e-12)


class BoundaryTurns:
    # Tolerance for recognising a row end on a boundary edge. Frozen with v2; changing it
    # changes which turns exist, so it needs an explicit decision, not a local tweak.
    SNAP_M=1e-8

    def __init__(self, mesh, region, frame, check, face_ids=None):
        from femto.planning.mesh_slice_planner import _forbidden_polygons
        self.vertices=np.asarray(mesh.vertices,float)
        self.check=check;self.frame=frame
        self.snap=self.SNAP_M
        view=region['view_reference']
        self.k=np.asarray(view['undistorted_intrinsic']);self.size=np.asarray(view['image_size'])
        self.polygons=_forbidden_polygons(region)
        self._polygon_source=self.polygons
        self._polygon_index=PreparedPolygonIndex(self.polygons)
        faces=np.asarray(mesh.faces)
        # Restricting to one block keeps its boundary graph free of any other
        # surface, even one that only touches or overlaps it geometrically.
        selected=range(len(faces)) if face_ids is None else [int(v) for v in face_ids]
        edges=defaultdict(list)
        for ordinal,index in enumerate(selected):
            if ordinal%1024==0:check()
            face=faces[index]
            a, b, c = face.tolist()
            for left, right in ((a, b), (b, c), (c, a)):
                edges[tuple(sorted((left, right)))].append(int(index))
        self.edges=[(a,b,ids[0]) for (a,b),ids in sorted(edges.items()) if len(ids)==1]
        degrees=Counter(v for a,b,_ in self.edges for v in (a,b))
        self.ambiguous={v for v,count in degrees.items() if count!=2}
        self.e2=None if frame is None else np.asarray(frame['e2'])

    def allowed(self, a, b):
        from femto.planning.mesh_slice_planner import _project
        uv=[_project(p,self.k,self.size) for p in (a,b)]
        if any(p is None for p in uv):return False
        # diagnose() temporarily replaces polygons; never reuse its old constraints.
        if self._polygon_source is not self.polygons:
            self._polygon_source=self.polygons
            self._polygon_index=PreparedPolygonIndex(self.polygons)
        return not self._polygon_index.blocks_segment(*uv)

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


def segment_pixels(a, b, limit=100000):
    """Exact set of reference pixels a projected straight segment passes through (supercover walk).

    Pixel (u, v) owns the cell [u-0.5, u+0.5) × [v-0.5, v+0.5), the same rounding every
    sample uses. A sub-segment's pixels are always a subset of the full segment's pixels, so
    planner, scan splitting and validator agree no matter how an edge is subdivided."""
    x0,y0=float(a[0]),float(a[1]);x1,y1=float(b[0]),float(b[1])
    cx,cy=int(math.floor(x0+.5)),int(math.floor(y0+.5));ex,ey=int(math.floor(x1+.5)),int(math.floor(y1+.5))
    pixels=[(cx,cy)]
    dx,dy=x1-x0,y1-y0
    stepx=1 if dx>0 else -1;stepy=1 if dy>0 else -1
    tx=((cx+.5*stepx)-x0)/dx if dx else math.inf
    ty=((cy+.5*stepy)-y0)/dy if dy else math.inf
    dtx=abs(1./dx) if dx else math.inf;dty=abs(1./dy) if dy else math.inf
    while (cx,cy)!=(ex,ey):
        if len(pixels)>limit:raise ValueError('Projected segment is too long for the reference image')
        if abs(tx-ty)<=1e-12:
            # Exact corner crossing: keep both side cells so a diagonal step never skips a pixel.
            pixels.append((cx+stepx,cy));pixels.append((cx,cy+stepy))
            cx+=stepx;cy+=stepy;tx+=dtx;ty+=dty
        elif tx<ty:cx+=stepx;tx+=dtx
        else:cy+=stepy;ty+=dty
        pixels.append((cx,cy))
    return pixels


def edge_pixels_allowed(p, q, k, size, allowed):
    """Every pixel the 3D edge p→q projects onto lies inside the image and on an allowed pixel."""
    from femto.planning.mesh_slice_planner import _project
    p=np.asarray(p,float);q=np.asarray(q,float)
    ends=[_project(v,k,size) for v in (p,q)]
    if any(v is None for v in ends):return False
    return projected_edge_pixels_allowed(ends[0],ends[1],allowed)


def projected_edge_pixels_allowed(start,end,allowed):
    """edge_pixels_allowed for endpoints already projected inside the image."""
    if allowed is None:return True
    height,width=allowed.shape
    for u,v in segment_pixels(start,end):
        if not (0<=u<width and 0<=v<height and allowed[v,u]):return False
    return True


class _LazyEdgeStates(dict):
    """One router's fixed edge predicates, evaluated only when search needs them."""
    def __init__(self, router, keys):
        from copy import copy, deepcopy
        frozen = copy(router)
        for name in ('vertices', 'k', 'size', 'allowed'):
            value = getattr(router, name)
            if value is not None:
                value = np.array(value, copy=True)
                value.setflags(write=False)
            setattr(frozen, name, value)
        frozen.polygons = deepcopy(router.polygons)
        frozen._polygon_source = frozen.polygons
        frozen._polygon_index = PreparedPolygonIndex(frozen.polygons)
        self.router, self.keys_allowed = frozen, frozenset(keys)
        # Supercover's near-corner tie can overrun an endpoint extremely close
        # to a pixel boundary. Retain eager checks there (including old errors).
        # The 1e-3 px guard is only a fallback threshold, never an admission rule.
        from femto.planning.mesh_slice_planner import _project
        safe = {}
        for ordinal, vertex in enumerate(sorted({v for edge in self.keys_allowed for v in edge})):
            if ordinal % 256 == 0:
                router.check()
            uv = _project(frozen.vertices[vertex], frozen.k, frozen.size)
            safe[vertex] = uv is None or all(abs((value+.5)-round(value+.5)) > 1e-3 for value in uv)
        self.eager_keys = {edge for edge in self.keys_allowed if not all(safe[v] for v in edge)}

    def __missing__(self, key):
        if key not in self.keys_allowed:
            raise KeyError(key)
        self.router.check()
        a, b = key
        state = self.router._state(self.router.vertices[a], self.router.vertices[b])
        self[key] = state
        return state


class SurfaceLinks:
    """v4 on-surface connectors inside one mesh fragment of a selection block.

    Adjacent rows first try the frozen v2/v3.1 boundary-band turn (identical output where it
    succeeds and stays on the block's selected pixels). Every other link, and every failed
    boundary turn, is routed as the shortest path along the fragment's own mesh edges: each
    connector edge lies on one selected triangle, no edge crosses a painted forbidden polygon,
    no edge leaves the reference image and every pixel it projects onto belongs to this block's
    selection component (`allowed`). No edge is ever invented across a mesh gap, so links
    between mesh fragments never exist here."""

    def __init__(self, mesh, region, frame, check, face_ids, allowed=None):
        from femto.planning.mesh_slice_planner import _forbidden_polygons
        self.vertices=np.asarray(mesh.vertices,float);self.faces=np.asarray(mesh.faces)
        self.check=check;self.frame=frame;self.allowed=allowed
        view=region['view_reference']
        self.k=np.asarray(view['undistorted_intrinsic']);self.size=np.asarray(view['image_size'])
        self.polygons=_forbidden_polygons(region)
        self._polygon_source=self.polygons
        self._polygon_index=PreparedPolygonIndex(self.polygons)
        self.face_ids=[int(v) for v in face_ids];self.member=set(self.face_ids)
        self.boundary=BoundaryTurns(mesh,region,frame,check,face_ids=self.face_ids)
        edges={}
        for ordinal,index in enumerate(self.face_ids):
            if ordinal%1024==0:check()
            face=self.faces[index]
            a, b, c = face.tolist()
            for left, right in ((a, b), (b, c), (c, a)):
                key=tuple(sorted((left, right)))
                if key not in edges or index<edges[key]:edges[key]=index
        self.edges=[(a,b,face) for (a,b),face in sorted(edges.items())]
        # Ordinary image extents bound every possible supercover walk below
        # its historical hard limit. Keep eager validation otherwise, so an
        # unused oversized edge cannot hide the old budget failure.
        lazy = (type(self)._state is SurfaceLinks._state
                and self.size.shape == (2,) and np.isfinite(self.size).all()
                and (self.size > 0).all() and (self.size <= 8192).all()
                and (self.allowed is None or (type(self.allowed) is np.ndarray
                     and self.allowed.ndim == 2 and self.allowed.dtype == np.dtype(bool)))
                and np.geterr() == dict(divide='warn', over='warn', under='ignore', invalid='warn'))
        self.edge_state = {}
        if lazy:
            try:
                self.edge_state = _LazyEdgeStates(self, ((a,b) for a,b,_ in self.edges))
            except MemoryError:
                lazy = False
        for index,(a,b,face) in enumerate(self.edges):
            if index%256==0:check()
            if not lazy or (a,b) in self.edge_state.eager_keys:
                self.edge_state[(a,b)] = self._state(self.vertices[a],self.vertices[b])
        self.adjacency=defaultdict(list)
        from femto.small_vectors import row_norms
        ends=np.asarray([(a,b) for a,b,_ in self.edges],dtype=np.int64).reshape(-1,2)
        lengths=row_norms(self.vertices[ends[:,1]]-self.vertices[ends[:,0]]).tolist()
        for (a,b,face),length in zip(self.edges,lengths):
            self.adjacency[a].append((b,length,face));self.adjacency[b].append((a,length,face))
        for node in self.adjacency:self.adjacency[node].sort()

    def _state(self, p, q):
        """0 allowed, 1 outside the reference image, 2 forbidden polygon, 3 explicitly unselected pixels."""
        from femto.planning.mesh_slice_planner import _project
        uv=[_project(v,self.k,self.size) for v in (p,q)]
        if any(v is None for v in uv):return 1
        if self._polygon_source is not self.polygons:
            self._polygon_source=self.polygons
            self._polygon_index=PreparedPolygonIndex(self.polygons)
        if self._polygon_index.blocks_segment(*uv):return 2
        # Reuse exactly the projections above; the supercover predicate is unchanged.
        if self.allowed is not None:
            height,width=self.allowed.shape
            for u,v in segment_pixels(*uv):
                if not (0<=u<width and 0<=v<height and self.allowed[v,u]):return 3
        return 0

    def _allowed(self, state, *, ignore_forbidden, ignore_image, ignore_selection=False):
        return (state==0 or (state==1 and ignore_image) or (state==2 and ignore_forbidden)
                or (state==3 and ignore_selection))

    def _endpoint_links(self, node, point, face, *, ignore_forbidden, ignore_image, ignore_selection=False):
        links=[]
        for vertex in self.faces[face]:
            vertex=int(vertex);q=self.vertices[vertex]
            state=self._state(point,q)
            if self._allowed(state,ignore_forbidden=ignore_forbidden,ignore_image=ignore_image,ignore_selection=ignore_selection):
                links.append((vertex,float(np.linalg.norm(q-point)),int(face)))
        return sorted(links)

    def _search(self, left, right, *, ignore_forbidden=False, ignore_image=False, ignore_selection=False):
        a=np.asarray(left['xyz_m'],float);b=np.asarray(right['xyz_m'],float)
        fa,fb=int(left['triangle_index']),int(right['triangle_index'])
        if fa not in self.member or fb not in self.member:return None
        flags=dict(ignore_forbidden=ignore_forbidden,ignore_image=ignore_image,ignore_selection=ignore_selection)
        if fa==fb:
            state=self._state(a,b)
            if self._allowed(state,**flags):
                return [(-2,-1,fa)]
        start=self._endpoint_links(-2,a,fa,**flags)
        finish={vertex:(length,face) for vertex,length,face in self._endpoint_links(-1,b,fb,**flags)}
        # A* with the straight-line distance to the target: admissible for Euclidean edge lengths,
        # so the result is the same shortest edge path Dijkstra would return, found much faster.
        def remaining(node):return 0. if node==-1 else float(np.linalg.norm(self.vertices[node]-b))
        distances={-2:0.};previous={};queue=[]
        for vertex,length,face in start:
            if length<distances.get(vertex,float('inf')):
                distances[vertex]=length;previous[vertex]=(-2,face);heapq.heappush(queue,(length+remaining(vertex),length,vertex))
        while queue:
            self.check();_,distance,node=heapq.heappop(queue)
            if distance!=distances.get(node):continue
            if node==-1:break
            candidates=[(other,length,face) for other,length,face in self.adjacency[node]
                        if self._allowed(self.edge_state[tuple(sorted((node,other)))],**flags)]
            if node in finish:candidates.append((-1,finish[node][0],finish[node][1]))
            for nxt,length,face in candidates:
                cost=distance+length
                if cost<distances.get(nxt,float('inf')):
                    distances[nxt]=cost;previous[nxt]=(node,face);heapq.heappush(queue,(cost+remaining(nxt),cost,nxt))
        if -1 not in previous:return None
        path=[];node=-1
        while node!=-2:
            prev,face=previous[node];path.append((prev,node,face));node=prev
        return list(reversed(path))

    def _samples(self, path, left, right, step):
        from femto.planning.mesh_slice_planner import _project
        a=np.asarray(left['xyz_m'],float);b=np.asarray(right['xyz_m'],float)
        points={-2:a,-1:b};samples=[]
        for i,j,face in path:
            self.check()
            p=points[i] if i<0 else self.vertices[i];q=points[j] if j<0 else self.vertices[j]
            length=float(np.linalg.norm(q-p))
            if length<=1e-12:continue
            count=max(1,math.ceil(length/step))
            if len(samples)+count>POINT_BUDGET:raise ValueError(f'Surface link exceeds {POINT_BUDGET} samples')
            for t in np.linspace(0,1,count+1)[1:]:
                xyz=p*(1-t)+q*t
                samples.append(dict(xyz_m=xyz.tolist(),pixel=_project(xyz,self.k,self.size),
                    triangle_index=int(face),geometry_support='measured_mesh',role='connector',
                    row_index=right['row_index']))
        if samples and np.linalg.norm(np.asarray(samples[-1]['xyz_m'])-b)>1e-12:
            return None
        return samples or None

    def _samples_on_selection(self, left, samples):
        """Boundary-band turns were routed without the selection mask: re-check every edge."""
        if self.allowed is None:return True
        previous=np.asarray(left['xyz_m'],float)
        for sample in samples:
            xyz=np.asarray(sample['xyz_m'],float)
            if not edge_pixels_allowed(previous,xyz,self.k,self.size,self.allowed):return False
            previous=xyz
        return True

    def route(self, left, right, step, e2, *, adjacent_rows):
        """(samples, method) or (None, None); boundary band first for adjacent rows."""
        if adjacent_rows:
            samples=self.boundary.route(left,right,step,e2)
            if samples and self._samples_on_selection(left,samples):return samples,'boundary_band'
        path=self._search(left,right)
        if path is None:return None,None
        samples=self._samples(path,left,right,step)
        return (samples,'surface_edges') if samples else (None,None)

    def diagnose(self, left, right):
        """Why no on-surface link exists; diagnostic only, never a decision."""
        if self._search(left,right,ignore_forbidden=True):return 'forbidden_blocked'
        if self._search(left,right,ignore_forbidden=True,ignore_image=True):return 'outside_reference_image'
        if self._search(left,right,ignore_forbidden=True,ignore_image=True,ignore_selection=True):return 'explicit_unselected_blocked'
        return 'no_surface_path'


def connect_links(segments, gaps, eligible, router, step, check, *, e2, used=0):
    """Try every gap inside one mesh fragment; kept gaps carry link (and row-turn) diagnostics."""
    chains=[[dict(p) for p in segments[0].points]];kept=[];length=0.;added=0
    connected=Counter();methods=Counter();failures=Counter()
    for index,gap in enumerate(gaps):
        check();following=segments[index+1]
        left=chains[-1][-1];right=following.points[0]
        adjacent=index in eligible
        route,method=router.route(left,right,step,e2,adjacent_rows=adjacent)
        if route:
            xyz=np.asarray([left['xyz_m']]+[p['xyz_m'] for p in route])
            length+=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum())
            chains[-1].extend(route);chains[-1].extend(dict(p) for p in following.points[1:])
            connected['row_turn' if left['row_index']!=right['row_index'] else 'detour']+=1
            methods[method]+=1;added+=len(route)-1
        else:
            detail=dict(gap.detail)
            if gap.reason=='row_turn':
                detail['row_turn_failure_reason']=('non_adjacent_slice' if not adjacent else
                    router.boundary.diagnose(left,right,step,e2))
            why=router.diagnose(left,right);detail['link_failure_reason']=why;failures[why]+=1
            kept.append(replace(gap,detail=detail));chains.append([dict(p) for p in following.points])
        if used+added>POINT_BUDGET:
            raise ValueError(f'Trajectory including surface links exceeds {POINT_BUDGET} samples '
                             f'({used+added}; scan/existing {used}, added links {added}); '
                             '总轨迹点数超限（不是搜索超限），未发布结果；请拆分工作区域后重新导出')
    return dict(chains=chains,kept=kept,connected=connected,methods=methods,failures=failures,
                length=length,added=added)
