"""Versioned continuous replanning inside the measured quality/width domain.

Scan samples preserve every triangle crossing. Rejected edges remain rejected;
new turns follow a finite shared-face graph whose emitted subedges pass the same
whole-chord, width, reference and boundary predicates as the final validator.
Legacy v4.4/v1/v2 generation and replay retain their original defaults.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import heapq
from itertools import combinations, islice
import math

import numpy as np
from femto.plain_copy import clone
from femto.operation_diagnostics import diagnosed

from femto.planning.surface_trajectory import ScanSegment, Gap, SurfaceTrajectory, ORDERED_SELECTION_ALGORITHM
from femto.planning import quality_constrained as quality

ALGORITHM = 'quality-constrained-surface-v3'
ROUTING = 'verified_shared_face_graph_v1'
LAZY_ROUTING = 'verified_shared_face_graph_lazy_v2'
DENSE_ROUTING = 'verified_shared_face_graph_quarter_portals_v1'
ORDERED_ROUTING = 'verified_shared_face_graph_visit_order_v1'
IMPROVED_ROUTING = 'verified_shared_face_graph_visit_order_two_opt_v2'
SUPPORT_ROUTING = 'verified_shared_face_graph_visit_order_support_v3'
BALANCED_ROUTING = 'verified_shared_face_graph_visit_order_balanced_v4'
SEAM_ROUTING = 'verified_shared_face_graph_visit_order_seam_v5'
SEAM_BALANCED_ROUTING = 'verified_shared_face_graph_visit_order_seam_balanced_v6'
PRESERVED_DETOUR_ROUTING = 'verified_shared_face_graph_visit_order_preserved_detour_v7'
SHORT_DETOUR_ROUTING = 'verified_shared_face_graph_visit_order_short_detour_v8'
JOINT_REORDER_ROUTING = 'verified_shared_face_graph_visit_order_joint_reorder_v9'
BRIDGE_REORDER_ROUTING = 'verified_shared_face_graph_visit_order_bridge_reorder_v10'
RELOCATION_ROUTING = 'verified_shared_face_graph_visit_order_relocation_v11'
DEEP_RELOCATION_ROUTING = 'verified_shared_face_graph_visit_order_relocation_v12'
FLOOR_DETOUR_ROUTING = 'verified_shared_face_graph_floor_detour_v13'
from femto.planning.algorithm_registry import UNIFIED_FLOOR_ROUTING, GLOBAL_FLOOR_ROUTING, REFINED_CONNECTOR_ROUTING
BUDGET = 200_000


@diagnosed('route.candidate')
def candidate(source, *, check=lambda: None, fixed_width_layout=None, layout_evidence=None, **values):
    from femto.planning.mesh_slice_planner import _load_input, _parameters
    from femto.planning.selection_blocks import plan_selection_blocks
    manifest_path, manifest, mesh = _load_input(source)
    reference = manifest['region']['view_reference']
    sampling = None
    if (reference.get('derivation') or {}).get('schema') in ('reduced-planning-reference-v1', 'surface-patch-planning-reference-v1'):
        from femto.planning.planning_resolution import route_sample_step_mm
        if reference['derivation'].get('sampling_policy') is not None:
            from femto.planning.route_sampling import decision
            check();sampling = decision(mesh,reference['derivation']['sampling_policy']);check()
        step = route_sample_step_mm(reference, decision=sampling)
        if values.get('sampling_decision', sampling) != sampling:
            raise ValueError('自动取点决策与选区不一致')
        if values.get('sample_step_mm', step) != step:
            raise ValueError('路线采样间隔与简化规划参考不一致')
        # Only this explicitly derived v3 input declares a coarser spacing.
        # Retain every other frozen rule, and keep legacy v1/v4 callers frozen.
        params = _parameters(dict(values, sample_step_mm=1.))
        params['sample_step_mm'] = step
        if sampling is not None:params['sampling_decision'] = sampling
    else:
        params = _parameters(values)
    params['preserve_face_crossings'] = True
    layout = None
    if fixed_width_layout is not None:
        from femto.planning.fixed_width_layout import Layout
        layout = Layout(fixed_width_layout.get('request', fixed_width_layout),layout_evidence,
                        row_shifts=fixed_width_layout.get('row_shifts'))
        if layout.is_patch:params['fixed_width_layout']={'request':deepcopy(layout.spec)}
    result = plan_selection_blocks(manifest_path, manifest, mesh, params, check,
                                   algorithm_version=ORDERED_SELECTION_ALGORITHM, row_layout=layout)
    if layout is not None:
        record = layout.record()
        if 'fragments' in fixed_width_layout and record != fixed_width_layout:
            raise ValueError('固定宽度自动排布与当前几何不一致')
        result = replace(result, planner=dict(result.planner, parameters=dict(result.planner['parameters'], fixed_width_layout=record)))
    return result


class VerifiedSurfaceGraph:
    """Finite face-interior routing; no XYZ welding or cross-gap interpolation."""
    def __init__(self, mesh, region, record, width, allowed, base, check):
        from femto.planning.selection_blocks import selection_face_assignment
        from femto.planning.surface_trajectory import _point_on_triangle
        self.mesh, self.record, self.width, self.allowed, self.check = mesh, record, width, allowed, check
        self.step = base.planner['parameters']['sample_step_mm']/1000
        self.k = np.asarray(record['view_reference']['undistorted_intrinsic'])
        self.size = np.asarray(record['image_size'])
        from femto.geometry.triangle_support import prepare_triangle_faces
        vertices, faces, prepared_faces = prepare_triangle_faces(mesh.vertices, mesh.faces)
        _, _, _, owners = selection_face_assignment(region, vertices, faces)
        self.nodes, self.support = [], []
        on_face, edge_faces = defaultdict(list), defaultdict(list)
        allowed_ids = np.flatnonzero(allowed)
        for fi in allowed_ids:
            check()
            a, b, c = faces[fi].tolist()
            for left, right in ((a, b), (b, c), (c, a)):
                edge_faces[tuple(sorted((left, right)))].append(int(fi))

        def node(xyz, fs):
            if len(self.nodes) >= BUDGET:raise ValueError('连续路线面内搜索节点超过预算，请缩小选区')
            index = len(self.nodes)
            self.nodes.append(np.asarray(xyz, float));self.support.append(int(fs[0]))
            for fi in fs:on_face[int(fi)].append(index)
            return index

        edges = iter(sorted(edge_faces.items()))
        while batch := list(islice(edges, 512)):
            check()
            shared = [(edge, fs) for edge, fs in batch if len(fs) == 2 and owners[fs[0]] == owners[fs[1]]]
            if not shared:
                continue
            pairs = vertices[[edge for edge, _ in shared]]
            if getattr(self, 'quarter_portals', False):
                positions = [pairs[:, 0]*(1-f) + pairs[:, 1]*f for f in (0.25, 0.5, 0.75)]
                for i, (_, fs) in enumerate(shared):
                    for xyz in positions:
                        node(xyz[i], fs)
            else:
                for xyz, (_, fs) in zip(pairs.mean(axis=1), shared):
                    node(xyz, fs)
        for start in range(0, len(allowed_ids), 512):
            check()
            ids = allowed_ids[start:start + 512]
            for xyz, fi in zip(vertices[faces[ids]].mean(axis=1), ids):
                node(xyz, [fi])

        @lru_cache(maxsize=256)
        def incident_faces(fi):
            a, b, c = faces[fi].tolist()
            candidates = dict.fromkeys(other for left, right in ((a, b), (b, c), (c, a))
                                       for other in edge_faces[tuple(sorted((left, right)))])
            return tuple(other for other in candidates if owners[other] == owners[fi])
        self.endpoints = []
        self.base_segments = [s for s in base.ordered_items if isinstance(s, ScanSegment)]
        for segment in self.base_segments:
            pair = []
            # Retained scan samples provide verified ways back into the
            # interior when a clipped endpoint lies close to the width limit.
            # These nodes are tied to actual incident faces, never XYZ-welded.
            for pi, p in enumerate(segment.points):
                fi = p['triangle_index'];fs = {fi}
                for other in incident_faces(fi):
                    if _point_on_triangle(p['xyz_m'], vertices, faces[other], _prepared=prepared_faces(other)):
                        fs.add(other)
                index = node(p['xyz_m'], sorted(fs))
                if pi == 0 or pi == len(segment.points)-1:pair.append(index)
            self.endpoints.append(pair)
        self.nodes = np.asarray(self.nodes)
        self.tested = 0
        self.margin_max = 0.
        self._edge_checks = quality._prepare_edge_checks(mesh, record, width, check, allowed)
        # Point preflight reduces graph work; only whole-edge tests admit a link.
        good = self._checks([(p,p,fi) for p,fi in zip(self.nodes,self.support)])
        self.adjacency = defaultdict(list)
        self.paths = {}
        pairs = []
        for fi, nodes in sorted(on_face.items()):
            check()
            ids = [n for n in nodes if good[n]]
            for a,b in combinations(ids,2):
                if len(pairs)>=BUDGET:raise ValueError('连续路线面内搜索边超过预算，请缩小选区')
                pairs.append((a,b,fi))
        self._admit_pairs(pairs)
        self.components = np.full(len(self.nodes), -1, int)
        component = 0
        for start in sorted(self.adjacency):
            if self.components[start] >= 0:continue
            stack = [start];self.components[start] = component
            while stack:
                check();current = stack.pop()
                for nxt, _, _ in self.adjacency[current]:
                    if self.components[nxt] < 0:
                        self.components[nxt] = component;stack.append(nxt)
            component += 1

    def _checks(self, edges):
        flags = []
        for start in range(0,len(edges),1024):
            self.check();batch = edges[start:start+1024]
            if self.tested+len(batch) > BUDGET:raise ValueError('连续路线整边搜索超过预算，请缩小选区')
            self.tested += len(batch)
            items = [ScanSegment(i,0,[dict(xyz_m=a,triangle_index=fi),dict(xyz_m=b,triangle_index=fi)],0)
                     for i,(a,b,fi) in enumerate(batch)]
            accepted, _, margins = self._edge_checks(items)
            self.margin_max = max(self.margin_max,float(margins.max(initial=0)))
            flags.extend(accepted.tolist())
        return flags

    def _admit_pairs(self, pairs):
        # Bound memory as well as total predicate work. Each graph edge is split
        # at the same locations later emitted into the saved connector.
        for first in range(0,len(pairs),256):
            self.check();group = pairs[first:first+256]
            tasks = []
            for pi,(a,b,fi) in enumerate(group):
                points = self._line(a,b)
                if self.tested+len(tasks)+len(points)-1>BUDGET:
                    raise ValueError('连续路线整边搜索超过预算，请缩小选区')
                tasks.extend((pi,float(i),float(i+1),p,q,0) for i,(p,q) in enumerate(zip(points,points[1:])))
            rejected=set();leaves=defaultdict(list)
            while tasks:
                active=[t for t in tasks if t[0] not in rejected]
                okay=self._checks([(t[3],t[4],group[t[0]][2]) for t in active])
                pending=[]
                for t,valid in zip(active,okay):
                    pi,lo,hi,a,b,depth=t
                    if valid:leaves[pi].append((lo,a,b))
                    elif depth>=12:rejected.add(pi)
                    else:pending.append(t)
                # A rejected midpoint cannot be bypassed by ever smaller edges.
                # Refinement changes only sampling, never the admission predicate.
                pending=[t for t in pending if t[0] not in rejected]
                mids=[(t[3]+t[4])/2 for t in pending]
                valid_mid=self._checks([(p,p,group[t[0]][2]) for p,t in zip(mids,pending)])
                tasks=[]
                for t,mid,valid in zip(pending,mids,valid_mid):
                    pi,lo,hi,a,b,depth=t
                    if not valid:rejected.add(pi);continue
                    half=(lo+hi)/2
                    tasks.extend([(pi,lo,half,a,mid,depth+1),(pi,half,hi,mid,b,depth+1)])
            from femto.small_vectors import row_norms
            ends=np.asarray([(a,b) for a,b,_ in group],dtype=np.int64).reshape(-1,2)
            distances=row_norms(self.nodes[ends[:,0]]-self.nodes[ends[:,1]]).tolist()
            for pi,(a,b,fi) in enumerate(group):
                if pi in rejected:continue
                points=np.asarray([self.nodes[a]]+[q for _,_,q in sorted(leaves[pi],key=lambda t:t[0])])
                self.paths[(a,b,fi)]=points
                self._add_verified_edge(a,b,fi,distances[pi])

    def _add_verified_edge(self,a,b,fi,distance):
        self.adjacency[a].append((b,distance,fi));self.adjacency[b].append((a,distance,fi))

    def _line(self,a,b):
        count = max(1,math.ceil(float(np.linalg.norm(self.nodes[a]-self.nodes[b]))/self.step))
        if count>BUDGET:raise ValueError('连续路线连接长度超过采样预算')
        return np.linspace(self.nodes[a],self.nodes[b],count+1)

    def route(self,left_index,right_index,*,left_reverse=False,right_reverse=False):
        start = self.endpoints[left_index][int(not left_reverse)];goal = self.endpoints[right_index][int(right_reverse)]
        if self.components[start] < 0 or self.components[start] != self.components[goal]:return None
        distances={start:0.};previous={};queue=[(0.,0.,start)]
        while queue:
            self.check();_,cost,node=heapq.heappop(queue)
            if cost != distances[node]:continue
            if node==goal:break
            for other,length,face in self.adjacency[node]:
                new=cost+length
                if new < distances.get(other,math.inf):
                    distances[other]=new;previous[other]=(node,face)
                    heapq.heappush(queue,(new+float(np.linalg.norm(self.nodes[other]-self.nodes[goal])),new,other))
        if goal not in previous:return None
        path=[];current=goal
        while current!=start:
            prior,face=previous[current];path.append((prior,current,face));current=prior
        return self._route_samples(reversed(path),right_index,right_reverse)

    def _route_samples(self,path,right_index,right_reverse=False):
        """Only previously certified subedges may become connector samples."""
        from femto.planning.mesh_slice_planner import _project
        row=self.base_segments[right_index].points[-1 if right_reverse else 0]['row_index']
        samples=[]
        for a,b,face in path:
            points=self.paths[(a,b,face)] if (a,b,face) in self.paths else self.paths[(b,a,face)][::-1]
            for p in points[1:]:
                samples.append(dict(xyz_m=p.tolist(),triangle_index=int(face),pixel=_project(p,self.k,self.size),
                    geometry_support='measured_mesh',role='connector',row_index=row))
        return samples


class LazyVerifiedSurfaceGraph(VerifiedSurfaceGraph):
    """Search the finite graph, verifying candidate links before emitting.

    Unchecked adjacency is optimistic and never motion evidence. Failed links
    are removed before searching again; predicates, sampling and budgets remain.
    """
    def _admit_pairs(self,pairs):
        self.candidate_pairs={}
        pairs=list(pairs)
        from femto.small_vectors import row_norms
        ends=np.asarray([(a,b) for a,b,_ in pairs],dtype=np.int64).reshape(-1,2)
        lengths=row_norms(self.nodes[ends[:,0]]-self.nodes[ends[:,1]]).tolist()
        for (a,b,fi),length in zip(pairs,lengths):
            self.check()
            self.candidate_pairs[(min(a,b),max(a,b),fi)]=(a,b,fi)
            self.adjacency[a].append((b,length,fi));self.adjacency[b].append((a,length,fi))

    def _add_verified_edge(self,a,b,fi,distance):
        # Optimistic adjacency already contains the link exactly once.
        pass

    def route(self,left_index,right_index,*,left_reverse=False,right_reverse=False):
        start=self.endpoints[left_index][int(not left_reverse)];goal=self.endpoints[right_index][int(right_reverse)]
        if self.components[start]<0 or self.components[start]!=self.components[goal]:return None
        while True:
            distances={start:0.};previous={};queue=[(0.,0.,start)]
            while queue:
                self.check();_,cost,node=heapq.heappop(queue)
                if cost!=distances[node]:continue
                if node==goal:break
                for other,length,face in self.adjacency[node]:
                    new=cost+length
                    if new<distances.get(other,math.inf):
                        distances[other]=new;previous[other]=(node,face)
                        heapq.heappush(queue,(new+float(np.linalg.norm(self.nodes[other]-self.nodes[goal])),new,other))
            if goal not in previous:return None
            path=[];current=goal
            while current!=start:
                prior,face=previous[current];path.append((prior,current,face));current=prior
            path.reverse()
            invalid=False
            for a,b,fi in path:
                self.check()
                if (a,b,fi) in self.paths or (b,a,fi) in self.paths:continue
                pair=self.candidate_pairs[(min(a,b),max(a,b),fi)]
                super()._admit_pairs([pair])
                if pair not in self.paths:
                    self.adjacency[a]=[v for v in self.adjacency[a] if not(v[0]==b and v[2]==fi)]
                    self.adjacency[b]=[v for v in self.adjacency[b] if not(v[0]==a and v[2]==fi)]
                    invalid=True
                    break
            if not invalid:return self._route_samples(path,right_index,right_reverse)


def _relocation_context(segments, visits, links, chains, block_items, classes, spacing, check):
    """Build a replayable v10 baseline without changing its face handoffs."""
    from femto.planning import verified_relocation_research as relocation
    from femto.planning.endpoint_preserving_detour import _signature
    from femto.planning.scan_visit_order import reconstruct
    if len(segments)>relocation.MAX_SEGMENTS:
        return 'segment_budget', None
    anchored,_,objective=reconstruct(segments,visits,links,check,preserve_endpoint_faces=True,budget=BUDGET)
    signature=_signature(anchored,check)
    if (signature!=_signature(chains,check)
            or signature!=_signature([(s.block_index,s.points) for s in segments],check)):
        return 'baseline_face_handoff', None
    if objective[0]:return 'internal_gap', None
    from femto.planning.selection_blocks import _mark_revisits
    marked=[(block,[dict(point) for point in points]) for block,points in chains]
    for _,points in marked:
        for point in points:point.pop('revisit',None)
    _mark_revisits([p for _,p in marked],spacing,check)
    stats,_=quality._summaries(quality._items(marked),block_items,classes)
    return None,dict(detours=stats['connected_detours'],connector_length_m=stats['connector_length_m'],
                     revisit_connector_m=stats['revisit_connector_length_m'])


@diagnosed('route.reconnect')
def replan(candidate, mesh, region, record, width, *, domain, check=lambda:None, routing=LAZY_ROUTING, search_budget=None):
    if routing==REFINED_CONNECTOR_ROUTING:
        from femto.planning.surface_connector_refinement import replan as refine_replan
        return refine_replan(candidate,mesh,region,record,width,domain=domain,check=check,search_budget=search_budget)
    if routing==GLOBAL_FLOOR_ROUTING:
        from femto.planning.global_search_budget_research import SearchBudget
        search=SearchBudget() if search_budget is None else search_budget
        if not isinstance(search,SearchBudget):raise ValueError('路线搜索预算对象无效')
    else:
        if search_budget is not None:raise ValueError('旧版路线不能使用全流程搜索预算')
        search=None
    if routing in (FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
        from femto.planning.bounded_floor_routing import check_layout
        check_layout(candidate.planner['parameters'],allow_adaptive=routing in (UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING))
    if routing not in (ROUTING,LAZY_ROUTING,DENSE_ROUTING,ORDERED_ROUTING,IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):raise ValueError('连续路线寻路版本不受支持')
    from femto.planning.effective_domain import validate as validate_domain
    allowed=validate_domain(domain,record,width,mesh,region,check)
    # constrain() would validate the identical domain again with the same inputs.
    base=quality.constrain(candidate,mesh,region,record,width,check,domain=domain,_allowed_faces=allowed.copy())
    segments=[s for s in base.ordered_items if isinstance(s,ScanSegment)]
    class QuarterPortalGraph(LazyVerifiedSurfaceGraph):
        quarter_portals = True
    graph_type=(QuarterPortalGraph if routing in (DENSE_ROUTING,ORDERED_ROUTING,IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING) else
                LazyVerifiedSurfaceGraph if routing==LAZY_ROUTING else VerifiedSurfaceGraph)
    graph=graph_type(mesh,region,record,width,allowed,base,check) if len(segments)>1 else None
    ordering=None
    chains=[(segments[0].block_index,deepcopy(segments[0].points))]
    connections=[];added=0
    for index,following in enumerate(segments[1:] if routing not in (ORDERED_ROUTING,IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING) else [],1):
        check();same=segments[index-1].block_index==following.block_index
        route=graph.route(index-1,index) if same else None
        if route:
            added+=len(route)
            if sum(len(p) for _,p in chains)+len(route)+len(following.points)>BUDGET:
                raise ValueError('连续路线采样点超过预算，请缩小选区')
            chains[-1][1].extend(route)
            chains[-1][1].extend(deepcopy(following.points[1:]))
        else:chains.append((following.block_index,deepcopy(following.points)))
        connections.append(dict(from_segment=index-1,to_segment=index,connected=bool(route),
            added_edges=len(route) if route else 0,
            samples=deepcopy(route) if route else [],
            reason='verified_surface_path' if route else 'no_verified_graph_path' if same else 'separate_work_blocks'))
    if routing in (ORDERED_ROUTING,IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
        from femto.planning.scan_visit_order import optimize
        chains,connections,added,ordering=optimize(graph,segments,check,BUDGET)
        if routing in (IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
            from femto.planning.verified_visit_improvement import improve,VERSION as IMPROVEMENT_VERSION
            prior_visits=deepcopy(ordering['visits']);prior_connections=deepcopy(connections)
            try:
                if routing in (SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
                    from femto.planning.seam_balanced_proposal import propose as combined_propose
                    chains,connections,added,improvement=combined_propose(graph,segments,prior_visits,prior_connections,
                        base.planner['parameters']['fixed_width_layout'],base.planner['parameters']['row_spacing_mm']/1000,check,BUDGET,
                        **({'search':search} if search is not None else {}))
                elif routing==SEAM_ROUTING:
                    from femto.planning.seam_visit_proposal import propose as seam_propose
                    chains,connections,added,improvement=seam_propose(graph,segments,prior_visits,prior_connections,
                        base.planner['parameters']['fixed_width_layout'],base.planner['parameters']['row_spacing_mm']/1000,check,BUDGET)
                elif routing==BALANCED_ROUTING:
                    from femto.planning.balanced_visit_improvement import propose as balanced_propose
                    chains,connections,added,improvement=balanced_propose(graph,segments,prior_visits,prior_connections,
                        base.planner['parameters']['row_spacing_mm']/1000,check,BUDGET)
                else:chains,connections,added,improvement=improve(graph,segments,prior_visits,prior_connections,check,BUDGET)
            except ValueError as error:
                if str(error)!='连续路线整边搜索超过预算，请缩小选区':raise
                if routing in (SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):raise
                if routing==SEAM_ROUTING:
                    from femto.planning.seam_visit_proposal import VERSION as SEAM_VERSION
                    from femto.planning.balanced_visit_improvement import _metrics
                    improvement=dict(version=SEAM_VERSION,adopted=False,attempts=[],reason='geometry_budget',matched_seams=0,
                                     baseline=_metrics(chains,base.planner['parameters']['row_spacing_mm']/1000,check))
                elif routing==BALANCED_ROUTING:
                    from femto.planning.balanced_visit_improvement import VERSION as BALANCED_VERSION,_metrics
                    improvement=dict(version=BALANCED_VERSION,adopted=False,attempts=[],skipped='geometry_budget',
                                     baseline=_metrics(chains,base.planner['parameters']['row_spacing_mm']/1000,check))
                else:improvement=dict(version=IMPROVEMENT_VERSION,adopted=False,rounds=[],skipped='geometry_budget')
            ordering['improvement']=dict(improvement,baseline_visits=prior_visits,baseline_connections=prior_connections)
            if routing in (SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
                from femto.planning.connector_endpoint_support import propose as support_propose
                from femto.planning.scan_visit_order import reconstruct
                visits=improvement['visits'] if improvement['adopted'] else prior_visits
                pre_support=deepcopy(connections)
                connections,support=support_propose(segments,visits,connections,mesh,record,width,allowed,check)
                chains,added,_=reconstruct(segments,visits,connections,check)
                ordering['endpoint_support']=dict(support,baseline_connections=pre_support)
                if routing in (PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
                    from femto.planning.endpoint_preserving_detour import propose as preserved_propose
                    before_detour=deepcopy(connections)
                    global_limits=None
                    if search is not None:
                        from femto.planning.unified_floor_routing import global_motion_limits
                        global_limits=global_motion_limits(segments,visits,before_detour,
                            base.planner['parameters']['row_spacing_mm']/1000,check,BUDGET)
                    chains,connections,added,detour=preserved_propose(graph,segments,visits,connections,
                        base.planner['parameters']['row_spacing_mm']/1000,check,BUDGET,short_candidates=routing in (SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING),
                        **({'search':search,'motion_limits':global_limits} if search is not None else {}))
                    ordering['preserved_detour']=dict(detour,baseline_connections=before_detour)
                    if routing in (JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
                        if routing in (BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
                            from femto.planning.bridge_joint_incremental import propose as joint_propose,VERSION as JOINT_VERSION
                        else:
                            from femto.planning.bounded_joint_reorder import propose as joint_propose,VERSION as JOINT_VERSION
                        from femto.planning.endpoint_preserving_detour import _signature
                        joint_visits=detour['visits'] if detour['adopted'] else visits
                        before_joint=deepcopy(connections)
                        anchored,_,_=reconstruct(segments,joint_visits,before_joint,check,preserve_endpoint_faces=True,budget=BUDGET)
                        if _signature(anchored,check)!=_signature(chains,check):
                            joint=dict(version=JOINT_VERSION,adopted=False,skipped='baseline_face_handoff')
                        else:
                            chains,connections,added,joint=joint_propose(graph,segments,joint_visits,before_joint,
                                base.planner['parameters']['row_spacing_mm']/1000,check,BUDGET,
                                **({'search':search,'motion_limits':global_limits} if search is not None else {}))
                        ordering['joint_reorder']=dict(joint,baseline_connections=before_joint)
                        if routing in (UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
                            from femto.planning.unified_floor_routing import propose as unified_propose
                            current=joint['visits'] if joint['adopted'] else joint_visits
                            classes,_=quality.checked_evidence(record,region)
                            spacing=base.planner['parameters']['row_spacing_mm']/1000
                            chains,connections,added,unified=unified_propose(graph,segments,current,connections,chains,added,
                                base.blocks['items'],classes,spacing,check,BUDGET,
                                **({'search':search,'global_limits':global_limits} if search is not None else {}))
                            ordering['relocation_unified']=unified
                            if search is not None:
                                from femto.planning.unified_floor_routing import check_global_motion
                                selected=unified['visits'] if unified.get('adopted') else current
                                check_global_motion(segments,selected,connections,spacing,global_limits,check,BUDGET)
                        if routing in (RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING):
                            from femto.planning import verified_relocation_research as relocation
                            current_visits=joint['visits'] if joint['adopted'] else joint_visits
                            prior_links=deepcopy(connections)
                            spacing=base.planner['parameters']['row_spacing_mm']/1000
                            classes,_=quality.checked_evidence(record,region)
                            skipped,limits=_relocation_context(segments,current_visits,prior_links,chains,
                                base.blocks['items'],classes,spacing,check)
                            if skipped:
                                decision=dict(version=relocation.VERSION,adopted=False,skipped=skipped)
                            else:
                                try:
                                    proposal,proposed_links,proposed_added,decision=relocation.propose(
                                        graph,segments,current_visits,prior_links,spacing,motion_limits=limits,
                                        attempts=32,rounds=2,check=check,budget=BUDGET)
                                except ValueError as error:
                                    if str(error)!='连续路线整边搜索超过预算，请缩小选区':raise
                                    decision=dict(version=relocation.VERSION,adopted=False,skipped='geometry_budget')
                                if decision['adopted']:
                                    chains,connections,added=proposal,proposed_links,proposed_added
                            ordering['relocation']=dict(decision,baseline_visits=[list(v) for v in current_visits],
                                                        baseline_connections=prior_links)
                            if routing in (DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING):
                                deep_visits=decision['visits'] if decision['adopted'] else current_visits
                                deep_links=deepcopy(connections)
                                skipped,deep_limits=_relocation_context(segments,deep_visits,deep_links,chains,
                                    base.blocks['items'],classes,spacing,check)
                                if skipped:
                                    deep_decision=dict(version=relocation.VERSION,adopted=False,skipped=skipped)
                                else:
                                    try:
                                        proposal,proposed_links,proposed_added,deep_decision=relocation.propose(
                                            graph,segments,deep_visits,deep_links,spacing,motion_limits=deep_limits,
                                            attempts=256,rounds=4,check=check,budget=BUDGET)
                                    except ValueError as error:
                                        if str(error)!='连续路线整边搜索超过预算，请缩小选区':raise
                                        deep_decision=dict(version=relocation.VERSION,adopted=False,skipped='geometry_budget')
                                    if deep_decision['adopted']:
                                        chains,connections,added=proposal,proposed_links,proposed_added
                                ordering['relocation_deep']=dict(deep_decision,baseline_visits=[list(v) for v in deep_visits],
                                                                 baseline_connections=deep_links)
                                if routing==FLOOR_DETOUR_ROUTING:
                                    from femto.planning.bounded_floor_routing import propose as floor_propose
                                    floor_visits=deep_decision['visits'] if deep_decision['adopted'] else deep_visits
                                    chains,connections,added,floor_record=floor_propose(graph,segments,floor_visits,connections,chains,added,
                                        base.blocks['items'],classes,spacing,check,BUDGET)
                                    ordering['relocation_floor']=floor_record
    from femto.planning.selection_blocks import _mark_revisits
    for _, points in chains:
        for point in points:point.pop('revisit',None)
    _mark_revisits([p for _,p in chains],base.planner['parameters']['row_spacing_mm']/1000,check)
    items=quality._items(chains)
    classes,_=quality.checked_evidence(record,region)
    stats,blocks=quality._summaries(items,base.blocks['items'],classes)
    gap_report=deepcopy(base.gap_report)
    gap_report['counts']=stats['gap_counts']
    gap_report['reconnection']=dict(schema='femto-verified-reconnection-v1',method=routing,
        base_items=[i.to_dict() for i in base.ordered_items],connections=connections,
        before_chains=len(segments),after_chains=len(chains),added_connector_edges=added,
        graph_checks=graph.tested if graph else 0,graph_margin_max_mm=graph.margin_max if graph else 0.,
        failed_search_is_proof_of_disconnection=False)
    if ordering is not None:
        if search is not None:ordering['search_budget']=search.report()
        gap_report['reconnection']['ordering']=ordering
    planner=dict(base.planner,algorithm_version=ALGORITHM,
                 parameters=dict(base.planner['parameters'],continuous_routing=routing))
    return replace(base,planner=planner,ordered_items=items,statistics=stats,blocks=blocks,gap_report=gap_report)


def validate(trajectory, mesh=None, region=None, check=lambda:None):
    """Recheck both the original exclusions and every newly emitted connector."""
    if trajectory.planner['parameters'].get('continuous_routing')==REFINED_CONNECTOR_ROUTING:
        from femto.planning.surface_connector_refinement import validate as validate_refined
        return validate_refined(trajectory,mesh,region,check)
    from femto.planning.surface_trajectory import validate_trajectory, CONNECTED_ALGORITHM
    from femto.planning.fixed_width_layout import ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION
    params=trajectory.planner['parameters']
    if params.get('continuous_routing') in (FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
        from femto.planning.bounded_floor_routing import check_layout
        check_layout(params,allow_adaptive=params.get('continuous_routing') in (UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING))
    layout=params.get('fixed_width_layout')
    if (layout is not None and layout['request']['version'] in ('fixed-width-local-direction-repair-v3','fixed-width-local-direction-budgeted-v4','fixed-width-local-direction-foldback-v5','research-fixed-width-local-direction-adaptive-half-v6','research-fixed-width-local-direction-minimal-v7','fixed-width-local-direction-spacing-floor-v9',ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION)
            and trajectory.planning_context is not None):
        from femto.planning.repair_workflow import validate as validate_repair
        validate_repair(trajectory,mesh,region,check)
    if layout is not None and mesh is not None and region is not None:
        from femto.planning.fixed_width_layout import validate as validate_layout
        validate_layout(layout,mesh,region,params,check)
        if layout['request']['effective_width_mm']!=params['effective_width_mm']:
            raise ValueError('自动排布与几何核验作用宽度不一致')
    reference = params['quality_evidence']['view_reference']
    if (reference.get('derivation') or {}).get('schema') in ('reduced-planning-reference-v1', 'surface-patch-planning-reference-v1'):
        from femto.planning.planning_resolution import route_sample_step_mm
        if params.get('sample_step_mm') != route_sample_step_mm(
                reference, decision=params.get('sampling_decision'), mesh=mesh):
            raise ValueError('路线采样间隔与绑定参考不一致')
    routing=params.get('continuous_routing')
    if params.get('preserve_face_crossings') is not True or routing not in (ROUTING,LAZY_ROUTING,DENSE_ROUTING,ORDERED_ROUTING,IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
        raise ValueError('连续路线的采样或寻路版本不一致')
    if trajectory.planning_context is not None:
        algorithms=trajectory.planning_context['revision']['inputs']['algorithms']
        if algorithms.get('route')==ALGORITHM and algorithms.get('routing',ROUTING)!=routing:
            raise ValueError('连续路线寻路版本与计算修订不一致')
        if layout is not None:
            if algorithms.get('layout')!=layout['request']['version']:
                raise ValueError('自动排布版本与计算修订不一致')
        elif algorithms.get('layout') is not None:
            raise ValueError('计算修订缺少自动排布决策')
    report=trajectory.gap_report.get('reconnection') or {}
    if (report.get('schema')!='femto-verified-reconnection-v1' or report.get('method')!=routing
            or report.get('failed_search_is_proof_of_disconnection') is not False
            or type(report.get('graph_checks')) is not int or not 0<=report['graph_checks']<=BUDGET
            or not np.isfinite(report.get('graph_margin_max_mm',np.nan))
            or report['graph_margin_max_mm']<0):
        raise ValueError('连续路线的寻路证据无效')
    base_items=[ScanSegment.from_dict(p) if p.get('kind')=='scan_segment' else Gap.from_dict(p)
                for p in report.get('base_items',[])]
    classes,residual=quality.checked_evidence(params['quality_evidence'],region)
    # One evidence check, effective domain and selection assignment per call;
    # every later consumer in this call receives a copy instead of a recheck.
    share=quality._ValidationShare(params['quality_evidence'],region,(classes.copy(),residual.copy()))

    def evidence():
        return share.evidence(params['quality_evidence'],region)

    def allowed_faces():
        found=share.allowed(params['effective_domain'],params['effective_width_mm'],mesh,region)
        if found is None:
            from femto.planning.effective_domain import validate as validate_domain
            shared=evidence()
            found=validate_domain(params['effective_domain'],params['quality_evidence'],params['effective_width_mm'],
                                  mesh,region,check,_classes=None if shared is None else shared[0])
            share.remember_allowed(params['effective_domain'],params['effective_width_mm'],mesh,region,found)
        return found

    def selection_labels():
        found=share.selection_labels(region,mesh)
        if found is None:
            from femto.planning.selection_blocks import selection_face_assignment
            found=selection_face_assignment(region,np.asarray(mesh.vertices),np.asarray(mesh.faces))[2]
            share.remember_selection(region,mesh,found.copy())
        return found

    base_stats,base_blocks=quality._summaries(base_items,trajectory.blocks['items'],classes)
    base_params={k:v for k,v in params.items() if k!='continuous_routing'}
    base_report={k:v for k,v in trajectory.gap_report.items() if k!='reconnection'}
    base_report['counts']=base_stats['gap_counts']
    base=replace(trajectory,planner=dict(trajectory.planner,algorithm_version=quality.DOMAIN_ALGORITHM,
        parameters=base_params),ordered_items=base_items,statistics=base_stats,blocks=base_blocks,gap_report=base_report)
    quality.validate(base,mesh,region,check,_share=share)
    segments=[s for s in base_items if isinstance(s,ScanSegment)]
    links=report.get('connections')
    if not isinstance(links,list) or len(links)!=len(segments)-1:
        raise ValueError('连续路线的连接记录缺失')
    chains=[(segments[0].block_index,clone(segments[0].points))];added=0
    for index,(link,following) in enumerate(zip(links,segments[1:] if routing not in (ORDERED_ROUTING,IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING) else []),1):
        check()
        same=segments[index-1].block_index==following.block_index
        samples=link.get('samples')
        if (link.get('from_segment')!=index-1 or link.get('to_segment')!=index
                or type(link.get('connected')) is not bool or not isinstance(samples,list)
                or link['connected']!=bool(samples) or link.get('added_edges')!=len(samples)):
            raise ValueError('连续路线的连接索引或点数不一致')
        if samples:
            if (not same or link.get('reason')!='verified_surface_path'
                    or samples[-1].get('xyz_m')!=following.points[0]['xyz_m']
                    or any(p.get('role')!='connector' or p.get('row_index')!=following.points[0]['row_index'] for p in samples)):
                raise ValueError('连续路线跨工作块或连接端点不一致')
            chains[-1][1].extend(clone(samples));chains[-1][1].extend(clone(following.points[1:]));added+=len(samples)
        else:
            if link.get('reason')!=('no_verified_graph_path' if same else 'separate_work_blocks'):
                raise ValueError('连续路线的未连接原因无效')
            chains.append((following.block_index,clone(following.points)))
    if routing in (ORDERED_ROUTING,IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
        from femto.planning.scan_visit_order import reconstruct,propose,VERSION as ORDER_VERSION
        ordering=report.get('ordering') or {}
        if routing==GLOBAL_FLOOR_ROUTING:
            from femto.planning.global_search_budget_research import validate_report
            counts=validate_report(ordering.get('search_budget'))
            recorded=ordering.get('relocation_unified') or {}
            tested=recorded.get('tested',0) if isinstance(recorded,dict) else None
            if type(tested) is not int or not 0<=tested<=counts['relocation']:
                raise ValueError('全流程搜索预算与移动候选数不一致')
        if ordering.get('version')!=ORDER_VERSION or type(ordering.get('adopted')) is not bool:
            raise ValueError('扫描段排序版本或采用状态无效')
        baseline=list(enumerate([False]*len(segments)))
        baseline_chains,_,baseline_objective=reconstruct(segments,baseline,ordering.get('baseline_connections'),check)
        visits=ordering.get('visits')
        expected_visits=propose(segments,check) if ordering['adopted'] else baseline
        if visits!=[list(v) for v in expected_visits]:raise ValueError('扫描段排序决策不能独立重算')
        improvement=ordering.get('improvement') if routing in (IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING) else None
        first_links=improvement.get('baseline_connections') if isinstance(improvement,dict) else links
        first_chains,_,objective=reconstruct(segments,visits,first_links,check)
        if (ordering.get('baseline_objective')!=list(baseline_objective)
                or ordering.get('selected_objective')!=list(objective)
                or (ordering['adopted'] and not (objective[0]<=baseline_objective[0] and objective[1]<baseline_objective[1]-1e-12))
                or (not ordering['adopted'] and first_links!=ordering['baseline_connections'])):
            raise ValueError('扫描段排序未保持连接数量与长度约束')
        if routing in (IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
            support=ordering.get('endpoint_support') if routing in (SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING) else None
            improved_links=support.get('baseline_connections') if isinstance(support,dict) else links
            if routing in (SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
                from femto.planning.seam_balanced_proposal import validate_record
                final_visits,middle_chains=validate_record(improvement,segments,visits,first_links,improved_links,params['row_spacing_mm']/1000,check,
                    **({'allow_search_budget':True} if routing==GLOBAL_FLOOR_ROUTING else {}))
                if mesh is not None:
                    middle_allowed=allowed_faces()
                    middle_segments=[s for s in quality._items(middle_chains) if isinstance(s,ScanSegment)]
                    flags,_,_=quality.edge_checks(middle_segments,mesh,params['quality_evidence'],params['effective_width_mm'],check,allowed_faces=middle_allowed,
                                                  _evidence=evidence())
                    if not flags.all():raise ValueError('接缝中间路线包含未核验连接')
            elif routing==SEAM_ROUTING:
                from femto.planning.seam_visit_proposal import validate_record
                final_visits=validate_record(improvement,segments,visits,first_links,improved_links,params['row_spacing_mm']/1000,check)
            elif routing==BALANCED_ROUTING:
                from femto.planning.balanced_visit_improvement import validate_record
                final_visits=validate_record(improvement,segments,visits,first_links,improved_links,params['row_spacing_mm']/1000,check)
            else:
                from femto.planning.verified_visit_improvement import validate_record
                final_visits=validate_record(improvement,segments,visits,first_links,improved_links,check)
            if routing in (SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
                from femto.planning.connector_endpoint_support import validate_record as validate_support
                allowed_support=allowed_faces() if mesh is not None else None
                support_decision={k:v for k,v in support.items() if k!='baseline_connections'} if isinstance(support,dict) else None
                detour=ordering.get('preserved_detour') if routing in (PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING) else None
                supported_links=detour.get('baseline_connections') if isinstance(detour,dict) else links
                validate_support(support_decision,segments,final_visits,improved_links,supported_links,mesh,params['quality_evidence'],params['effective_width_mm'],allowed_support,check)
                global_limits=None
                if routing==GLOBAL_FLOOR_ROUTING:
                    from femto.planning.unified_floor_routing import global_motion_limits
                    global_limits=global_motion_limits(segments,final_visits,supported_links,params['row_spacing_mm']/1000,check,BUDGET)
            if routing in (PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
                if not isinstance(detour,dict):raise ValueError('保留端点绕行记录缺失')
                from femto.planning.endpoint_preserving_detour import validate_record as validate_detour
                supported_chains,_,_=reconstruct(segments,final_visits,supported_links,check)
                if mesh is not None:
                    supported_segments=[s for s in quality._items(supported_chains) if isinstance(s,ScanSegment)]
                    flags,_,_=quality.edge_checks(supported_segments,mesh,params['quality_evidence'],params['effective_width_mm'],check,allowed_faces=allowed_support,
                                                  _evidence=evidence())
                    if not flags.all():raise ValueError('端点保留绕行基准包含未核验连接')
                detour_decision={k:v for k,v in detour.items() if k!='baseline_connections'}
                joint=ordering.get('joint_reorder') if routing in (JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING) else None
                if routing in (JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING) and not isinstance(joint,dict):raise ValueError('联合换序记录缺失')
                detour_links=joint.get('baseline_connections') if joint is not None else links
                chains,added=validate_detour(detour_decision,segments,final_visits,supported_links,detour_links,params['row_spacing_mm']/1000,check,BUDGET,short_candidates=routing in (SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING))
                if routing in (JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
                    if routing in (BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
                        from femto.planning.bridge_joint_incremental import validate_record as validate_joint,VERSION as JOINT_VERSION
                    else:
                        from femto.planning.bounded_joint_reorder import validate_record as validate_joint,VERSION as JOINT_VERSION
                    from femto.planning.endpoint_preserving_detour import _signature
                    joint_visits=detour_decision['visits'] if detour_decision['adopted'] else final_visits
                    joint_decision={k:v for k,v in joint.items() if k!='baseline_connections'}
                    relocation_record=ordering.get('relocation') if routing in (RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING) else None
                    if routing in (RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING) and not isinstance(relocation_record,dict):
                        raise ValueError('扫描段移动记录缺失')
                    deep_record=ordering.get('relocation_deep') if routing in (DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING) else None
                    if routing in (DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING) and not isinstance(deep_record,dict):
                        raise ValueError('额外扫描段移动记录缺失')
                    floor_record=ordering.get('relocation_floor') if routing==FLOOR_DETOUR_ROUTING else None
                    if routing==FLOOR_DETOUR_ROUTING and not isinstance(floor_record,dict):
                        raise ValueError('疏密绕行优化记录缺失')
                    unified_record=ordering.get('relocation_unified') if routing in (UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING) else None
                    if routing in (UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING) and not isinstance(unified_record,dict):
                        raise ValueError('统一预算换序记录缺失')
                    joint_output_links=(relocation_record.get('baseline_connections') if relocation_record is not None else
                                        unified_record.get('baseline_connections') if unified_record is not None else links)
                    joint_baseline,_,_=reconstruct(segments,joint_visits,detour_links,check,preserve_endpoint_faces=True,budget=BUDGET)
                    if mesh is not None and joint_decision.get('skipped')!='baseline_face_handoff':
                        joint_segments=[s for s in quality._items(joint_baseline) if isinstance(s,ScanSegment)]
                        flags,_,_=quality.edge_checks(joint_segments,mesh,params['quality_evidence'],params['effective_width_mm'],check,allowed_faces=allowed_support,
                                                      _evidence=evidence())
                        if not flags.all():raise ValueError('联合换序基准包含未核验连接')
                    if joint_decision.get('skipped')=='baseline_face_handoff':
                        if (joint_decision!={'version':JOINT_VERSION,'adopted':False,'skipped':'baseline_face_handoff'}
                                or joint_output_links!=detour_links or _signature(joint_baseline,check)==_signature(chains,check)):
                            raise ValueError('联合换序来源面回退记录无效')
                    else:
                        if _signature(joint_baseline,check)!=_signature(chains,check):raise ValueError('联合换序基准改变扫描来源面')
                        chains,added=validate_joint(joint_decision,segments,joint_visits,detour_links,joint_output_links,params['row_spacing_mm']/1000,check,BUDGET)
                    if routing in (RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING):
                        from femto.planning import verified_relocation_research as relocation
                        current_visits=joint_decision['visits'] if joint_decision['adopted'] else joint_visits
                        if relocation_record.get('baseline_visits')!=[list(v) for v in current_visits]:
                            raise ValueError('扫描段移动基准顺序不一致')
                        if mesh is not None:
                            prior_segments=[s for s in quality._items(chains) if isinstance(s,ScanSegment)]
                            flags,_,_=quality.edge_checks(prior_segments,mesh,params['quality_evidence'],params['effective_width_mm'],check,allowed_faces=allowed_support,
                                                          _evidence=evidence())
                            if not flags.all():raise ValueError('扫描段移动基准包含未核验连接')
                            if region is not None:
                                from femto.planning.selection_blocks import check_projection_inside_selection
                                check_projection_inside_selection(prior_segments,trajectory.blocks['items'],selection_labels(),
                                    np.asarray(region['view_reference']['undistorted_intrinsic']),np.asarray(region['view_reference']['image_size']))
                        spacing=params['row_spacing_mm']/1000
                        skipped,limits=_relocation_context(segments,current_visits,joint_output_links,chains,
                            trajectory.blocks['items'],classes,spacing,check)
                        decision={k:v for k,v in relocation_record.items() if k not in ('baseline_connections','baseline_visits')}
                        relocation_output_links=deep_record.get('baseline_connections') if deep_record is not None else links
                        if skipped or decision.get('skipped')=='geometry_budget':
                            reason=skipped or 'geometry_budget'
                            if decision!=dict(version=relocation.VERSION,adopted=False,skipped=reason) or relocation_output_links!=joint_output_links:
                                raise ValueError('扫描段移动回退记录无效')
                        else:
                            proposal,proposed_added=relocation.validate_record(decision,segments,current_visits,joint_output_links,
                                relocation_output_links,spacing,motion_limits=limits,attempts=32,rounds=2,check=check,budget=BUDGET)
                            if decision['adopted']:chains,added=proposal,proposed_added
                        if deep_record is not None:
                            deep_visits=decision['visits'] if decision['adopted'] else current_visits
                            if deep_record.get('baseline_visits')!=[list(v) for v in deep_visits]:
                                raise ValueError('额外扫描段移动基准顺序不一致')
                            if mesh is not None:
                                prior_segments=[s for s in quality._items(chains) if isinstance(s,ScanSegment)]
                                flags,_,_=quality.edge_checks(prior_segments,mesh,params['quality_evidence'],params['effective_width_mm'],check,allowed_faces=allowed_support,
                                                              _evidence=evidence())
                                if not flags.all():raise ValueError('额外扫描段移动基准包含未核验连接')
                                if region is not None:
                                    from femto.planning.selection_blocks import check_projection_inside_selection
                                    check_projection_inside_selection(prior_segments,trajectory.blocks['items'],selection_labels(),
                                        np.asarray(region['view_reference']['undistorted_intrinsic']),np.asarray(region['view_reference']['image_size']))
                            skipped,deep_limits=_relocation_context(segments,deep_visits,relocation_output_links,chains,
                                trajectory.blocks['items'],classes,spacing,check)
                            deep_decision={k:v for k,v in deep_record.items() if k not in ('baseline_connections','baseline_visits')}
                            deep_output_links=floor_record.get('baseline_connections') if floor_record is not None else links
                            if skipped or deep_decision.get('skipped')=='geometry_budget':
                                reason=skipped or 'geometry_budget'
                                if deep_decision!=dict(version=relocation.VERSION,adopted=False,skipped=reason) or deep_output_links!=relocation_output_links:
                                    raise ValueError('额外扫描段移动回退记录无效')
                            else:
                                proposal,proposed_added=relocation.validate_record(deep_decision,segments,deep_visits,relocation_output_links,
                                    deep_output_links,spacing,motion_limits=deep_limits,attempts=256,rounds=4,check=check,budget=BUDGET)
                                if deep_decision['adopted']:chains,added=proposal,proposed_added
                            if floor_record is not None:
                                from femto.planning.bounded_floor_routing import validate as validate_floor
                                floor_visits=deep_decision['visits'] if deep_decision['adopted'] else deep_visits
                                chains,added=validate_floor(floor_record,segments,floor_visits,links,chains,added,
                                    trajectory.blocks['items'],classes,params,mesh,region,allowed_support,check,BUDGET)
                    if unified_record is not None:
                        from femto.planning.unified_floor_routing import validate as validate_unified
                        current=joint_decision['visits'] if joint_decision['adopted'] else joint_visits
                        chains,added=validate_unified(unified_record,segments,current,links,chains,added,
                            trajectory.blocks['items'],classes,params,mesh,region,allowed_support,check,BUDGET,
                            **({'global_limits':global_limits} if routing==GLOBAL_FLOOR_ROUTING else {}))
                        if routing==GLOBAL_FLOOR_ROUTING:
                            from femto.planning.unified_floor_routing import check_global_motion
                            selected=unified_record['visits'] if unified_record.get('adopted') else current
                            check_global_motion(segments,selected,links,params['row_spacing_mm']/1000,global_limits,check,BUDGET)
            else:chains,added,_=reconstruct(segments,final_visits,links,check)
        else:chains,added=first_chains,reconstruct(segments,visits,links,check)[1]
        if mesh is not None:
            # Baseline cost is measured geometry evidence too, not a trusted
            # scalar that could be inflated to justify a longer proposal.
            baseline_segments=[s for s in quality._items(baseline_chains) if isinstance(s,ScanSegment)]
            order_allowed=allowed_faces()
            flags,_,_=quality.edge_checks(baseline_segments,mesh,params['quality_evidence'],params['effective_width_mm'],check,allowed_faces=order_allowed,
                                          _evidence=evidence())
            if not flags.all():raise ValueError('扫描段排序基准包含未核验连接')
            if routing in (IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING):
                first_segments=[s for s in quality._items(first_chains) if isinstance(s,ScanSegment)]
                flags,_,_=quality.edge_checks(first_segments,mesh,params['quality_evidence'],params['effective_width_mm'],check,allowed_faces=order_allowed,
                                              _evidence=evidence())
                if not flags.all():raise ValueError('优化前扫描段排序包含未核验连接')
    from femto.planning.selection_blocks import _mark_revisits
    for _,points in chains:
        for point in points:point.pop('revisit',None)
    _mark_revisits([p for _,p in chains],params['row_spacing_mm']/1000,check)
    expected=quality._items(chains)
    if ([i.to_dict() for i in expected]!=[i.to_dict() for i in trajectory.ordered_items]
            or report.get('before_chains')!=len(segments) or report.get('after_chains')!=len(chains)
            or report.get('added_connector_edges')!=added):
        raise ValueError('连续路线改变了保留扫描段或连接记录')
    stats,blocks=quality._summaries(expected,trajectory.blocks['items'],classes)
    if stats!=trajectory.statistics or blocks!=trajectory.blocks or trajectory.gap_report['counts']!=stats['gap_counts']:
        raise ValueError('连续路线统计不一致')
    # Reuse the stable basic validator for all final sample fields, support,
    # step limits and index order. Retained and added edges are then rechecked.
    shadow=[]
    for item in expected:
        if isinstance(item,ScanSegment):
            points=[dict({k:v for k,v in p.items() if k!='revisit'},role='scan') for p in item.points]
            shadow.append(replace(item,points=points,row_index=min(p['row_index'] for p in points),block_index=None))
        else:
            shadow.append(replace(item,reason='unsupported_geometry',detail=dict(inferred_cause='surface_gap',
                method='sweep_plane_outer_loop_containment',confidence='heuristic',decision_use='none')))
    validate_trajectory(replace(trajectory,planner=dict(trajectory.planner,algorithm_version=CONNECTED_ALGORITHM,
        parameters=dict(params,selection_blocks=False,block_planning=False,connect_rows=True)),
        ordered_items=shadow,blocks=None),mesh=mesh,check=check)
    if mesh is not None:
        allowed=allowed_faces()
        final=[s for s in expected if isinstance(s,ScanSegment)]
        if region is not None:
            from femto.planning.selection_blocks import check_projection_inside_selection
            check_projection_inside_selection(final,blocks['items'],selection_labels(),
                np.asarray(region['view_reference']['undistorted_intrinsic']),np.asarray(region['view_reference']['image_size']))
        okay,_,_=quality.edge_checks(final,mesh,params['quality_evidence'],params['effective_width_mm'],check,allowed_faces=allowed,
                                     _evidence=evidence())
        if not okay.all():raise ValueError('连续路线存在未通过几何或宽度检查的连接')
