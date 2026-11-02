"""Bounded same-corridor connector refinement after the retained v15 route.

Only existing manifold portals can move. Original scans, visits, endpoints,
surface predicates and selection stay bound to the input. Discrete angles and
nominal proximity are planning diagnostics, never robot or treatment evidence.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import math
import numpy as np

from femto.planning import scan_visit_order as order, balanced_visit_improvement as balanced
from femto.planning import quality_constrained as quality
from femto.planning.algorithm_registry import GLOBAL_FLOOR_ROUTING, REFINED_CONNECTOR_ROUTING
from femto.planning.endpoint_preserving_detour import _signature
from femto.planning.surface_trajectory import ScanSegment

SCHEMA = 'measured-corridor-connector-refinement-v1'
MAX_LINKS = 12
MAX_CHECKS = 1024
SWEEPS = 2
POINT_BUDGET = 200_000
MIN_GAIN_M = 1e-6
EPS = 1e-12
POLICY = dict(max_links=MAX_LINKS, max_edge_checks=MAX_CHECKS, portal_sweeps=SWEEPS,
              min_gain_m=MIN_GAIN_M, long_threshold_spacing_factor=2.,
              sharp_turn_deg=120., reversal_deg=150., scope='per_replan_at_most_three_replans')


def _endpoint(segments, visit, tail):
    return segments[visit[0]].points[-1 if bool(tail)!=visit[1] else 0]


def fold_counts(chains, check=lambda: None):
    """Same duplicate handling as polyline_quality; count connector joins only."""
    sharp = reversals = 0
    for _, points in chains:
        check()
        xyz = np.asarray([p['xyz_m'] for p in points], dtype=float)
        keep = np.r_[np.flatnonzero(np.linalg.norm(np.diff(xyz, axis=0), axis=1)>0), len(points)-1]
        vectors = np.diff(xyz[keep], axis=0)
        if len(vectors)<2:
            continue
        directions = vectors/np.linalg.norm(vectors, axis=1)[:, None]
        angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(directions[:-1],directions[1:]),axis=1),
                                      np.einsum('ij,ij->i',directions[:-1],directions[1:])))
        connector = np.asarray([points[int(a)+1].get('role')=='connector' or
                                points[int(b)+1].get('role')=='connector'
                                for a,b in zip(keep[:-2],keep[1:-1])])
        sharp += int(np.count_nonzero((angles>=120)&connector))
        reversals += int(np.count_nonzero((angles>=150)&connector))
    check()
    return dict(sharp_turns_120=sharp, reversals_150=reversals)


def measurements(segments, visits, links, reconstructed, spacing, check=lambda: None):
    lengths=[]; excess=[]
    for (a,b),link in zip(zip(visits,visits[1:]),links):
        check()
        if not link['connected']:
            continue
        tail,head=_endpoint(segments,a,True),_endpoint(segments,b,False)
        length=order._connector_length(tail,link['samples'])
        direct=float(np.linalg.norm(np.asarray(tail['xyz_m'])-head['xyz_m']))
        lengths.append(length);excess.append(max(0.,length-direct))
    return dict(balanced._metrics(reconstructed[0],spacing,check), **fold_counts(reconstructed[0],check),
                internal_gaps=reconstructed[2][0], connector_length_m=reconstructed[2][1],
                max_connector_m=max(lengths,default=0.),
                long_excess_m=sum(max(0.,x-2*spacing) for x in lengths),
                connector_backtrack_m=sum(excess))


class _EdgeBudget(Exception):
    pass


class _PointBudget(Exception):
    pass


def _candidate_indices(segments, visits, links):
    lengths=[order._connector_length(_endpoint(segments,a,True),link['samples']) if link['connected'] else 0.
             for a,link in zip(visits,links)]
    return sorted((i for i,link in enumerate(links) if link['connected']),key=lambda i:(-lengths[i],i))[:MAX_LINKS]


def refine(mesh, segments, visits, links, spacing, sample_step, intrinsic, image_size,
           edge_checker, check=lambda: None):
    """Derive independent links and a deterministic record; no input mutation."""
    from femto.geometry.triangle_support import prepare_triangle_faces
    from femto.planning.surface_trajectory import _point_on_triangle
    from femto.planning.surface_link_optimization import optimize_surface_path
    from femto.planning.mesh_slice_planner import _project
    from femto.planning.planning_revision import digest
    if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or v<=0
           for v in (spacing,sample_step)):
        raise ValueError('连接优化的行距或采样步长无效')
    visits=order.checked_visits(segments,visits)
    initial=order.reconstruct(segments,visits,links,check,preserve_endpoint_faces=True,budget=POINT_BUDGET)
    signature=_signature(initial[0],check)
    if signature!=_signature([(s.block_index,s.points) for s in segments],check):
        raise ValueError('连接优化基准改变了原扫描边')
    before=measurements(segments,visits,links,initial,spacing,check)
    vertices,faces,prepared=prepare_triangle_faces(mesh.vertices,mesh.faces)
    counts=Counter()
    for i,face in enumerate(faces):
        if i%256==0:check()
        counts.update(tuple(sorted((int(a),int(b)))) for a,b in zip(face,np.roll(face,-1)))
    current=deepcopy(links);selected=initial;values=before;attempts=[];used=0
    candidates=_candidate_indices(segments,visits,links)

    def test(edges):
        nonlocal used
        check()
        if used+len(edges)>MAX_CHECKS:
            raise _EdgeBudget()
        used+=len(edges)
        accepted=list(edge_checker(edges))
        if len(accepted)!=len(edges):raise ValueError('连接优化几何检查结果不完整')
        check()
        return accepted

    for index in candidates:
        check();link=current[index];tail=_endpoint(segments,visits[index],True);head=_endpoint(segments,visits[index+1],False)
        xyz=np.asarray([tail['xyz_m']]+[p['xyz_m'] for p in link['samples']])
        support=[p['triangle_index'] for p in link['samples']]
        if any(type(fi) is not int or not 0<=fi<len(faces) for fi in support):
            raise ValueError('连接优化的来源面无效')
        corridor_faces=sorted(set(support))

        def allowed(a,b):
            for fi in corridor_faces:
                check()
                triangle=prepared(fi)
                if (_point_on_triangle(a,vertices,faces[fi],_prepared=triangle)
                        and _point_on_triangle(b,vertices,faces[fi],_prepared=triangle)
                        and test([(a,b,fi)])[0]):
                    return True
            return False

        status='retained'
        try:
            points,new_faces=optimize_surface_path(vertices,faces,xyz,support,allowed,check,
                                                   edge_counts=counts,sweeps=SWEEPS)
            retained_points=(sum(len(points) for _,points in selected[0])-len(link['samples'])
                             -int(support[-1]!=head['triangle_index']))
            available=POINT_BUDGET-retained_points-int(new_faces[-1]!=head['triangle_index'])
            samples=[]
            for a,b,fi in zip(points,points[1:],new_faces):
                check();count=max(1,math.ceil(float(np.linalg.norm(b-a))/sample_step))
                if len(samples)+count>available:raise _PointBudget()
                for p in np.linspace(a,b,count+1)[1:]:
                    samples.append(dict(xyz_m=p.tolist(),triangle_index=int(fi),pixel=_project(p,intrinsic,image_size),
                                        geometry_support='measured_mesh',role='connector',row_index=head['row_index']))
            tasks=[(np.asarray(tail['xyz_m'] if i==0 else samples[i-1]['xyz_m']),np.asarray(p['xyz_m']),p['triangle_index'])
                   for i,p in enumerate(samples)]
            if not all(test(tasks)):
                status='geometry_rejected'
            else:
                candidate_links=list(current)
                candidate_links[index]=dict(link,samples=samples,added_edges=len(samples))
                candidate=order.reconstruct(segments,visits,candidate_links,check,preserve_endpoint_faces=True,budget=POINT_BUDGET)
                measured=measurements(segments,visits,candidate_links,candidate,spacing,check)
                if (all(measured[k]<=values[k]+EPS for k in values)
                        and measured['connector_length_m']<values['connector_length_m']-MIN_GAIN_M):
                    if _signature(candidate[0],check)!=signature:raise ValueError('连接优化改变了原扫描边')
                    current,selected,values=candidate_links,candidate,measured;status='adopted'
        except _EdgeBudget:
            attempts.append(dict(link=index,status='budget_exhausted'))
            break
        except _PointBudget:
            attempts.append(dict(link=index,status='sample_budget_exhausted'))
            break
        attempts.append(dict(link=index,status=status))
    check()
    record=dict(schema=SCHEMA,policy=dict(POLICY),visits=[list(v) for v in visits],attempts=attempts,
                edge_checks=used,baseline=before,selected=values,adopted=any(a['status']=='adopted' for a in attempts),
                final_links_sha256=digest(current))
    return selected,deepcopy(current),record


def _context(trajectory):
    report=trajectory.gap_report['reconnection'];stage=(report.get('ordering') or {}).get('relocation_unified')
    segments=[ScanSegment.from_dict(x) for x in report['base_items'] if x['kind']=='scan_segment']
    if not isinstance(stage,dict):return segments,None
    visits=stage['visits'] if stage.get('adopted') else stage.get('baseline_visits')
    return segments,order.checked_visits(segments,visits)


def _checker(mesh,region,params,check):
    from femto.planning.effective_domain import validate
    allowed=validate(params['effective_domain'],params['quality_evidence'],params['effective_width_mm'],mesh,region,check)
    prepared=quality._prepare_edge_checks(mesh,params['quality_evidence'],params['effective_width_mm'],check,allowed)
    def evaluate(edges):
        items=[ScanSegment(i,0,[dict(xyz_m=a,triangle_index=fi),dict(xyz_m=b,triangle_index=fi)],0)
               for i,(a,b,fi) in enumerate(edges)]
        flags,_,_=prepared(items)
        return flags.tolist()
    return evaluate


def _report_copy(report,*,read_only=False):
    """Temporary validators only replace the two copied outer mappings.

    Shared input evidence stays inside synchronous validation. Publication may
    use this view only after _rebuild has made an independent full evidence copy.
    """
    if read_only:return dict(report,reconnection=dict(report['reconnection']))
    return deepcopy(report)


def _rebuild(trajectory,segments,visits,links,check,*,read_only=False):
    from femto.planning.selection_blocks import _mark_revisits
    rebuilt=order.reconstruct(segments,visits,links,check,preserve_endpoint_faces=True,budget=POINT_BUDGET)
    _mark_revisits([points for _,points in rebuilt[0]],trajectory.planner['parameters']['row_spacing_mm']/1000,check)
    items=quality._items(rebuilt[0]);classes,_=quality.checked_evidence(trajectory.planner['parameters']['quality_evidence'],None)
    stats,blocks=quality._summaries(items,trajectory.blocks['items'],classes)
    report=_report_copy(trajectory.gap_report,read_only=read_only);rec=report['reconnection']
    rec.update(connections=deepcopy(links),after_chains=len(rebuilt[0]),added_connector_edges=rebuilt[1])
    report['counts']=stats['gap_counts']
    return replace(trajectory,ordered_items=items,statistics=stats,blocks=blocks,gap_report=report)


def replan(candidate,mesh,region,evidence,width,*,domain,check,search_budget):
    from femto.planning.continuous_quality import replan as baseline_replan
    base=baseline_replan(candidate,mesh,region,evidence,width,domain=domain,check=check,
                         routing=GLOBAL_FLOOR_ROUTING,search_budget=search_budget)
    segments,visits=_context(base);params=base.planner['parameters'];prior=deepcopy(base.gap_report['reconnection']['connections'])
    if visits is None:
        record=dict(schema=SCHEMA,skipped='no_relocation_stage')
        result=base
    else:
        reference=params['quality_evidence']['view_reference']
        _,links,record=refine(mesh,segments,visits,prior,params['row_spacing_mm']/1000,params['sample_step_mm']/1000,
                              np.asarray(reference['undistorted_intrinsic']),np.asarray(params['quality_evidence']['image_size']),
                              _checker(mesh,region,params,check),check)
        result=_rebuild(base,segments,visits,links,check)
    # _rebuild already owns an independent evidence copy in this branch.
    report=_report_copy(result.gap_report,read_only=visits is not None)
    report['reconnection'].update(method=REFINED_CONNECTOR_ROUTING,
                                  connector_refinement=dict(record,baseline_connections=prior))
    return replace(result,planner=dict(base.planner,parameters=dict(params,continuous_routing=REFINED_CONNECTOR_ROUTING)),gap_report=report)


def validate(trajectory,mesh=None,region=None,check=lambda:None):
    """Validate v15 inputs, then independently regenerate the refinement."""
    from femto.planning.continuous_quality import validate as validate_baseline
    from femto.planning.planning_revision import digest
    from femto.planning.fixed_width_layout import COORDINATED_FLOOR_PATCH_VERSION
    params=trajectory.planner['parameters'];report=trajectory.gap_report.get('reconnection') or {}
    record=report.get('connector_refinement')
    if (report.get('method')!=REFINED_CONNECTOR_ROUTING or not isinstance(record,dict)
            or record.get('schema')!=SCHEMA or not isinstance(record.get('baseline_connections'),list)):
        raise ValueError('连接优化记录缺失或版本无效')
    if (params.get('fixed_width_layout') or {}).get('request',{}).get('version')!=COORDINATED_FLOOR_PATCH_VERSION:
        raise ValueError('连接优化记录缺少接缝协调排布')
    if trajectory.planning_context is not None:
        algorithms=trajectory.planning_context['revision']['inputs']['algorithms']
        if algorithms.get('routing')!=REFINED_CONNECTOR_ROUTING:
            raise ValueError('连接优化与计算修订不一致')
        # The retained v15 baseline below deliberately has no v16 context.
        # Validate the actual v16 retry evidence before detaching that context.
        from femto.planning.repair_workflow import validate as validate_repair
        validate_repair(trajectory,mesh,region,check)
    segments,visits=_context(trajectory);prior=record['baseline_connections'];links=report.get('connections')
    if visits is None:
        if record!=dict(schema=SCHEMA,skipped='no_relocation_stage',baseline_connections=prior) or links!=prior:
            raise ValueError('连接优化回退记录无效')
        baseline=trajectory
    else:
        baseline=_rebuild(trajectory,segments,visits,prior,check,read_only=True)
    baseline_report=_report_copy(baseline.gap_report,read_only=True)
    baseline_report['reconnection'].pop('connector_refinement')
    baseline_report['reconnection']['method']=GLOBAL_FLOOR_ROUTING
    baseline=replace(baseline,planner=dict(trajectory.planner,parameters=dict(params,continuous_routing=GLOBAL_FLOOR_ROUTING)),
                     gap_report=baseline_report,planning_context=None)
    validate_baseline(baseline,mesh,region,check)
    if visits is None:return
    expected_keys={'schema','policy','visits','attempts','edge_checks','baseline','selected','adopted','final_links_sha256','baseline_connections'}
    if (set(record)!=expected_keys or digest(record['policy'])!=digest(POLICY) or digest(record['visits'])!=digest([list(v) for v in visits])
            or type(record['edge_checks']) is not int or not 0<=record['edge_checks']<=MAX_CHECKS
            or type(record['adopted']) is not bool or not isinstance(record['attempts'],list) or len(record['attempts'])>MAX_LINKS):
        raise ValueError('连接优化预算或步骤无效')
    for attempt in record['attempts']:
        if (not isinstance(attempt,dict) or set(attempt)!={'link','status'} or type(attempt['link']) is not int
                or not 0<=attempt['link']<len(prior)
                or attempt['status'] not in ('retained','adopted','geometry_rejected','budget_exhausted','sample_budget_exhausted')):
            raise ValueError('连接优化尝试记录无效')
    candidates=_candidate_indices(segments,visits,prior)
    attempts=record['attempts'];exhausted=('budget_exhausted','sample_budget_exhausted')
    if ([a['link'] for a in attempts]!=candidates[:len(attempts)]
            or any(a['status'] in exhausted for a in attempts[:-1])
            or (len(attempts)!=len(candidates) and (not attempts or attempts[-1]['status'] not in exhausted))):
        raise ValueError('连接优化尝试顺序或结束原因无效')
    initial=order.reconstruct(segments,visits,prior,check,preserve_endpoint_faces=True,budget=POINT_BUDGET)
    final=order.reconstruct(segments,visits,links,check,preserve_endpoint_faces=True,budget=POINT_BUDGET)
    before=measurements(segments,visits,prior,initial,params['row_spacing_mm']/1000,check)
    after=measurements(segments,visits,links,final,params['row_spacing_mm']/1000,check)
    changed=[i for i,(a,b) in enumerate(zip(prior,links)) if a!=b]
    adopted=[a['link'] for a in record['attempts'] if a['status']=='adopted']
    if (digest(before)!=digest(record['baseline']) or digest(after)!=digest(record['selected']) or any(after[k]>before[k]+EPS for k in before)
            or len(adopted)!=len(set(adopted)) or sorted(adopted)!=changed or record['adopted']!=bool(changed)
            or digest(links)!=record['final_links_sha256'] or _signature(initial[0],check)!=_signature(final[0],check)):
        raise ValueError('连接优化结果、扫描边或运动统计不一致')
    rebuilt=_rebuild(trajectory,segments,visits,links,check,read_only=True)
    if (rebuilt.ordered_items!=trajectory.ordered_items or rebuilt.statistics!=trajectory.statistics
            or rebuilt.blocks!=trajectory.blocks or rebuilt.gap_report!=trajectory.gap_report):
        raise ValueError('连接优化的路线与保存结果不一致')
    if mesh is not None:
        reference=params['quality_evidence']['view_reference']
        _,replayed,decision=refine(mesh,segments,visits,prior,params['row_spacing_mm']/1000,params['sample_step_mm']/1000,
                                  np.asarray(reference['undistorted_intrinsic']),np.asarray(params['quality_evidence']['image_size']),
                                  _checker(mesh,region,params,check),check)
        if replayed!=links or dict(decision,baseline_connections=prior)!=record:
            raise ValueError('连接优化不能由已绑定输入独立重现')
        from femto.planning.effective_domain import validate as validate_domain
        from femto.planning.selection_blocks import selection_face_assignment,check_projection_inside_selection
        allowed=validate_domain(params['effective_domain'],params['quality_evidence'],params['effective_width_mm'],mesh,region,check)
        scan=[item for item in trajectory.ordered_items if isinstance(item,ScanSegment)]
        accepted,_,_=quality.edge_checks(scan,mesh,params['quality_evidence'],params['effective_width_mm'],check,allowed_faces=allowed)
        if not accepted.all():raise ValueError('连接优化未通过完整表面和作用宽度检查')
        if region is not None:
            _,_,labels,_=selection_face_assignment(region,np.asarray(mesh.vertices),np.asarray(mesh.faces))
            check_projection_inside_selection(scan,trajectory.blocks['items'],labels,
                np.asarray(region['view_reference']['undistorted_intrinsic']),np.asarray(region['view_reference']['image_size']))
