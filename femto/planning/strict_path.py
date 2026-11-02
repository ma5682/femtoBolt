"""P4 candidate reconstruction and the shared strict support/quality adapter."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from fractions import Fraction as Q
import math

import numpy as np

from femto.geometry.surface_support import MeshSupport, NumericalLimit
from femto.geometry.exact_surface import rational, subtract, dot
from femto.planning import quality_constrained as quality, planning_revision as revisions
from femto.planning.surface_trajectory import ScanSegment, SurfaceTrajectory

ALGORITHM = 'constructed-surface-route-v1'
REFINED_ALGORITHM = 'constructed-coverage-refined-surface-v1'
SCHEMA = 'femto-strict-route-reconstruction-v1'
ANCHOR = 'surface_anchor'


@dataclass(frozen=True)
class SupportBudget:
    max_face_checks: int = 200_000
    max_filter_comparisons: int = 20_000_000
    max_edge_checks: int = 200_000
    max_output_points: int = 200_000
    max_point_displacement_m: float = .000001
    max_added_length_m: float = .01

    def checked(self):
        for key, cap in asdict(SupportBudget()).items():
            value = getattr(self, key)
            if isinstance(cap, int):
                if type(value) is not int or not 0 <= value <= cap:raise ValueError('严格构造预算无效：'+key)
            elif type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= cap:
                raise ValueError('严格构造预算无效：'+key)
        return self


def algorithms_for(base, budget):
    from femto.planning.refinement_validation import semantic_payload
    return dict(route=ALGORITHM, target=base.planning_context['revision']['inputs']['algorithms']['target'], support=dict(
        schema=SCHEMA, base_computation_sha256=base.planning_context['revision']['sha256'],
        base_route_sha256=revisions.digest(semantic_payload(base)), budget=asdict(budget)))


def checked_algorithms(algorithms):
    if (not isinstance(algorithms, dict) or set(algorithms) != {'route', 'target', 'support'}
            or algorithms['route'] != ALGORITHM or algorithms['target'] not in revisions.TARGET_ALGORITHMS):return False
    policy = algorithms['support']
    if not isinstance(policy, dict) or set(policy) != {'schema', 'base_computation_sha256', 'base_route_sha256', 'budget'}:
        return False
    if policy['schema'] != SCHEMA:return False
    for key in ('base_computation_sha256', 'base_route_sha256'):
        value = policy[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):return False
    try:SupportBudget(**policy['budget']).checked()
    except (TypeError, ValueError):return False
    return True


def is_strict(trajectory):
    return trajectory.planner['algorithm_version'] in (ALGORITHM, REFINED_ALGORITHM, 'nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1')


class Checks:
    def __init__(self, engine):
        self.engine = engine
        self.decisions = []

    def begin(self, pairs):
        self.decisions = [dict(status='not_checked', reason='earlier_quality_constraint') for _ in pairs]
        errors = []
        for i, (left, right) in enumerate(pairs):
            if i % 128 == 0:self.engine.check()
            bounds = []
            for point in (left, right):
                anchor = point.get(ANCHOR)
                if not anchor or anchor['face'] != point['triangle_index']:
                    raise ValueError('严格路径缺少相同支持面的构造来源')
                display, error = self.engine.display(anchor)
                if display != point['xyz_m']:raise ValueError('严格路径 XYZ 与构造来源不一致')
                bounds.append(error)
            errors.append(max(bounds))
        return np.asarray(errors)

    def supported(self, index, left, right):
        decision = self.engine.chord(left[ANCHOR], right[ANCHOR])
        self.decisions[index] = decision.to_dict()
        return decision.status == 'passed'

    def reason(self, index):
        record = self.decisions[index]
        return 'strict_support_'+record['status']+':'+record['reason']


def engine_for(mesh, allowed, budget=None, check=lambda: None, *, region=None):
    budget = (budget or SupportBudget()).checked()
    groups = None
    if region is not None:
        from femto.planning.selection_blocks import selection_face_assignment
        _, _, _, groups = selection_face_assignment(region, np.asarray(mesh.vertices), np.asarray(mesh.faces))
    return MeshSupport(mesh.vertices, mesh.faces, allowed=allowed, groups=groups, max_face_checks=budget.max_face_checks,
        max_filter_comparisons=budget.max_filter_comparisons, check=check)


def point_from_anchor(engine, anchor, template, params):
    from femto.planning.mesh_slice_planner import _project
    point = deepcopy(template)
    point.pop('route_index', None);point.pop('revisit', None)
    display, _ = engine.display(anchor)
    point.update(xyz_m=display, triangle_index=anchor['face'], surface_anchor=anchor,
        pixel=_project(np.asarray(display), np.asarray(params['quality_evidence']['view_reference']['undistorted_intrinsic']),
                       np.asarray(params['quality_evidence']['image_size'])))
    return point


def construct_point(engine, point, params, budget):
    if ANCHOR in point:
        anchor = point[ANCHOR]
        if anchor['face'] != point['triangle_index'] or engine.display(anchor)[0] != point['xyz_m']:
            raise ValueError('已有严格路径构造已改变')
        return deepcopy(point)
    try:anchor = engine.closest(point['xyz_m'], point['triangle_index'])
    except ValueError as error:
        if str(error) != 'degenerate_support_face':raise
        raise NumericalLimit('degenerate_support_face') from error
    delta = subtract(engine.resolve(anchor), rational(point['xyz_m']))
    if dot(delta, delta) > Q(budget.max_point_displacement_m)**2:
        raise NumericalLimit('candidate_displacement_limit')
    return point_from_anchor(engine, anchor, point, params)


def _edge_parameter(a, b, u, v):
    """Exact closest pair of two closed segments; stable tie-breaking."""
    d, e, w = subtract(b, a), subtract(v, u), subtract(a, u)
    dd, ee, de = dot(d, d), dot(e, e), dot(d, e)
    if not ee:raise ValueError('degenerate_shared_edge')
    clamp = lambda value: min(Q(1), max(Q(0), value))
    pairs = [(Q(0), clamp(dot(w, e)/ee)), (Q(1), clamp(dot(subtract(b,u),e)/ee))]
    if dd:
        pairs.extend([(clamp(-dot(w,d)/dd), Q(0)), (clamp(dot(subtract(v,a),d)/dd), Q(1))])
        determinant = dd*ee-de*de
        if determinant:
            t = (dot(w,e)*dd-dot(w,d)*de)/determinant
            s = (t*de-dot(w,d))/dd
            if 0 <= s <= 1 and 0 <= t <= 1:pairs.append((s,t))
    values = []
    for s,t in pairs:
        delta = tuple(x+s*y-t*z for x,y,z in zip(w,d,e))
        values.append((dot(delta,delta),t,s))
    distance2,t,_ = min(values)
    return t,distance2


def seam(engine, left, right, params, budget):
    first, last = left['triangle_index'], right['triangle_index']
    shared = sorted(set(engine.faces[first]) & set(engine.faces[last]))
    if len(shared) != 2:return None
    u,v = (rational(engine.vertices[int(index)]) for index in shared)
    t,distance2 = _edge_parameter(engine.resolve(left[ANCHOR]), engine.resolve(right[ANCHOR]), u, v)
    if distance2 > Q(budget.max_point_displacement_m)**2:return None
    weights = [Q(0)]*3
    weights[list(engine.faces[first]).index(shared[0])] = 1-t
    weights[list(engine.faces[first]).index(shared[1])] = t
    anchor = engine.point(first, weights)
    template = dict(right, role=right.get('role','scan'))
    return point_from_anchor(engine, anchor, template, params)


def _sample_supported(engine, left, right, decision, params, budget):
    a,b = engine.resolve(left[ANCHOR]), engine.resolve(right[ANCHOR])
    cuts = sorted({Q(0), Q(1)} | {t for lo,hi,_ in decision.intervals for t in (lo,hi) if 0 < t < 1})
    result = [deepcopy(left)]
    step = Q(params['sample_step_mm']/1000)
    if step <= 0:raise ValueError('严格采样步长必须大于零')
    for lo,hi in zip(cuts,cuts[1:]):
        middle = (lo+hi)/2
        face = min(face for start,end,face in decision.intervals if start <= middle <= end)
        delta = tuple((hi-lo)*(y-x) for x,y in zip(a,b));length2 = dot(delta,delta)
        available=budget.max_output_points-len(result)
        if available<1 or length2>(available*step)**2:raise NumericalLimit('candidate_point_budget')
        # Float only proposes a count. The exact square comparison determines
        # whether the construction satisfies the existing sampling step.
        count = max(1, math.ceil(math.sqrt(float(length2))/float(step)))
        while (count*step)**2 < length2:count += 1
        if len(result)+count > budget.max_output_points:raise NumericalLimit('candidate_point_budget')
        for index in range(1,count+1):
            if index % 128 == 0:engine.check()
            t = lo+(hi-lo)*Q(index,count)
            if t == 1:point = deepcopy(right)
            else:
                anchor = engine.on_face(tuple(x+t*(y-x) for x,y in zip(a,b)), face)
                if anchor is None:raise ValueError('严格区间与面内采样不一致')
                point = point_from_anchor(engine, anchor, right, params)
            result.append(point)
    return result


def construct_edge(engine, left, right, params, budget):
    decision = engine.chord(left[ANCHOR], right[ANCHOR])
    original = decision.to_dict()
    if decision.status == 'passed':
        return _sample_supported(engine,left,right,decision,params,budget), dict(method='constructed_chord', initial_support=original)
    middle = seam(engine,left,right,params,budget) if decision.status == 'rejected' else None
    if middle is not None:
        before,after = engine.chord(left[ANCHOR],middle[ANCHOR]),engine.chord(middle[ANCHOR],right[ANCHOR])
        if before.status == after.status == 'passed':
            points = _sample_supported(engine,left,middle,before,params,budget)
            points.extend(_sample_supported(engine,middle,right,after,params,budget)[1:])
            return points, dict(method='shared_edge_split', crossing=middle[ANCHOR], initial_support=original)
    return None, dict(method='unconstructed', initial_support=original)


def construct_polyline(engine, points, params, budget):
    """A new graph candidate is usable only if every reconstructed edge exists."""
    if len(points) < 2:raise ValueError('严格折线至少需要两个点')
    prepared = [construct_point(engine,p,params,budget) for p in points]
    output, recipes = [prepared[0]], []
    for left,right in zip(prepared,prepared[1:]):
        engine.check()
        fresh,recipe = construct_edge(engine,left,right,params,budget);recipes.append(recipe)
        if fresh is None:return None,recipes
        output.extend(fresh[1:])
        if len(output) > budget.max_output_points:raise NumericalLimit('candidate_point_budget')
    return output,recipes


def context_for(base, budget):
    inputs = deepcopy(base.planning_context['revision']['inputs']);inputs['algorithms'] = algorithms_for(base,budget)
    revision = revisions.create(inputs)
    target = deepcopy(base.planning_context['target']);target['computation_sha256'] = revision['sha256']
    if target.get('construction_pages') is not None:
        from femto.planning.target_paging import rebind_header_records
        target['construction_pages']=rebind_header_records(target,revision)
    target['sha256'] = revisions.digest({k:v for k,v in target.items() if k != 'sha256'})
    return dict(schema=revisions.CONTEXT_SCHEMA, revision=revision, target=target)


def items_from_records(base, records, check=lambda: None):
    from femto.planning.refinement_candidates import retained_chains
    chains=[];last=None
    for record in records:
        check()
        if record['status'] != 'checked':last=None;continue
        points=record['points'];decisions=record['edge_checks'];support=record['support_checks']
        if len(points)<2 or len(decisions)!=len(points)-1 or len(support)!=len(decisions):
            raise ValueError('严格重建缺少对应的逐边记录')
        if any(type(d['accepted']) is not bool or d['accepted']!=(d['reason']=='accepted')
               or d['accepted'] and s.get('status')!='passed' for d,s in zip(decisions,support)):
            raise ValueError('严格重建存在未经证明的接纳边')
        flags=[d['accepted'] for d in decisions]
        for block,run in retained_chains(ScanSegment(0,points[0]['row_index'],points,record['block_index']),flags):
            key=(record['source_segment'],record['source_edge'])
            same=(last is not None and key==(last[0],last[1]+1) and chains[-1][0]==block
                  and run[0]==points[0] and chains[-1][1][-1][ANCHOR]==run[0][ANCHOR])
            if same:chains[-1][1].extend(deepcopy(run[1:]))
            else:chains.append((block,deepcopy(run)))
            last=key if run[-1]==points[-1] else None
        if not any(flags):last=None
    if not chains:return None
    from femto.planning.selection_blocks import _mark_revisits
    _mark_revisits([points for _,points in chains],base.planner['parameters']['row_spacing_mm']/1000,check)
    items=quality._items(chains)
    classes,_=quality.checked_evidence(base.planner['parameters']['quality_evidence'])
    stats,blocks=quality._summaries(items,base.blocks['items'],classes)
    return items,stats,blocks


def route_from_report(base, report, check=lambda: None):
    budget=SupportBudget(**report['budget']).checked()
    result=items_from_records(base,report['records'],check)
    if result is None:raise quality.PlanningConstraintBlocked('严格构造后没有通过全部约束的路线；原路线保留。',report)
    items,stats,blocks=result
    if stats['total_length_m'] > base.statistics['total_length_m']+budget.max_added_length_m:
        raise quality.PlanningConstraintBlocked('严格构造候选超过新增长度预算；原路线保留。',report)
    context=context_for(base,budget)
    planner=dict(base.planner,kind='constructed_surface',algorithm_version=ALGORITHM,
        parameters=dict(base.planner['parameters'],surface_support=context['revision']['inputs']['algorithms']['support']))
    return replace(base,planner=planner,planning_context=context,ordered_items=items,statistics=stats,blocks=blocks,
        gap_report=dict(base.gap_report,counts=stats['gap_counts'],surface_support=report))


def outcome_counts(records):
    outcomes = Counter()
    for record in records:
        for decision in record['edge_checks']:
            reason = decision['reason']
            kind = ('accepted' if decision['accepted'] else 'support_unconfirmed' if reason.startswith('strict_support_unconfirmed:')
                    else 'support_rejected' if reason.startswith('strict_support_rejected:') else 'quality_rejected')
            outcomes[kind] += 1
    return dict(sorted(outcomes.items()))


def reconstruct(base, mesh, region, *, budget=None, check=lambda: None, progress=lambda text: None):
    """Reconstruct a new candidate from an unchanged v3 route, then check all edges."""
    from femto.planning.effective_domain import validate as validate_domain
    budget = (budget or SupportBudget()).checked()
    if base.planner['algorithm_version'] != revisions.ALGORITHMS['route'] or base.planning_context is None:
        raise ValueError('严格构造需要带原目标的常规 v3 路线；请先恢复补充前路线')
    if region is None:region=base.planning_context['revision']['inputs']['selection']
    params = base.planner['parameters']
    allowed = validate_domain(params['effective_domain'],params['quality_evidence'],params['effective_width_mm'],mesh,region,check)
    engine = engine_for(mesh,allowed,budget,check,region=region)
    progress('按支持面和共享边重建候选，保留原路线作为对照')
    records,candidates = [],[];edge_count = 0;point_count = 0;stopped = None
    for segment in (s for s in base.ordered_items if isinstance(s,ScanSegment)):
        prepared, failures = [], []
        for point in segment.points:
            check()
            try:
                prepared.append(construct_point(engine,point,params,budget));failures.append(None)
            except NumericalLimit as error:
                prepared.append(None);failures.append(str(error))
        for index,(left,right) in enumerate(zip(prepared,prepared[1:])):
            check()
            record = dict(source_segment=segment.segment_index,source_edge=index,block_index=segment.block_index,
                          points=[],edge_checks=[],support_checks=[])
            records.append(record)
            if left is None or right is None:
                record.update(status='unconfirmed',reason=failures[index] or failures[index+1]);continue
            component = base.blocks['items'][segment.block_index]['component_id']
            if any(engine.groups[p['triangle_index']] != component for p in (left,right)):
                record.update(status='rejected',reason='outside_work_block');continue
            try:points,recipe = construct_edge(engine,left,right,params,budget)
            except NumericalLimit as error:
                record.update(status='unconfirmed',reason=str(error));stopped=str(error);break
            record['construction'] = recipe
            if points is None:
                record.update(status=recipe['initial_support']['status'],reason=recipe['initial_support']['reason'])
                if record['reason'].endswith('_budget'):stopped=record['reason'];break
                continue
            if edge_count+len(points)-1 > budget.max_edge_checks or point_count+len(points) > budget.max_output_points:
                record.update(status='unconfirmed',reason='output_or_edge_budget');stopped=record['reason'];break
            edge_count += len(points)-1;point_count += len(points)
            record.update(status='constructed',reason='pending_quality_check',points=points)
            candidates.append(ScanSegment(len(candidates),points[0]['row_index'],points,segment.block_index))
        if stopped:break
    progress('对新候选重新检查严格支持、作用宽度、参考深度与边界')
    checker = Checks(engine)
    flags,reasons,margins = quality.edge_checks(candidates,mesh,params['quality_evidence'],params['effective_width_mm'],
        check,allowed_faces=allowed,support_checker=checker) if candidates else ([],[],[])
    offset = 0
    for record in records:
        if record['status'] != 'constructed':continue
        points = record['points'];count = len(points)-1
        values = flags[offset:offset+count]
        record['edge_checks'] = [dict(accepted=bool(a),reason=b,residual_margin_mm=float(c))
            for a,b,c in zip(values,reasons[offset:offset+count],margins[offset:offset+count])]
        record['support_checks'] = checker.decisions[offset:offset+count];offset += count
        record.update(status='checked',reason='all_constraints_rechecked')
    if stopped is None:
        stopped=next((d['reason'] for d in checker.decisions if d.get('status')=='unconfirmed'
                      and d['reason'].endswith('_budget')),None)
    report = dict(schema=SCHEMA,budget=asdict(budget),base_trajectory=base.to_dict(),records=records,
        counts=dict(sorted(Counter(r['status'] for r in records).items())),
        edge_outcomes=outcome_counts(records),
        face_checks=engine.face_checks,filter_comparisons=engine.filter_comparisons,construction_fast_paths=engine.fast_paths,
        support_queries=engine.queries,interval_queries=engine.interval_queries,aabb_rejected_faces=engine.aabb_rejected_faces,
        edge_checks=edge_count,stop_reason=stopped or 'completed',
        original_edge_count=sum(len(s.points)-1 for s in base.ordered_items if isinstance(s,ScanSegment)),
        processed_original_edges=len(records),projection_creates_new_candidate=True,
        support_scope='exact_constructed_numeric_surface_only',physical_accuracy_verified=False,execution_ready=False,
        registration_uncertainty_mm=None)
    from femto.planning.surface_trajectory import canonical_json_bytes
    if len(canonical_json_bytes(report)) > 64*1024*1024:
        raise quality.PlanningConstraintBlocked('严格构造报告超过 64 MiB；原路线保留。',
            {k:v for k,v in report.items() if k not in ('records','base_trajectory')})
    result=route_from_report(base,report,check)
    if len(canonical_json_bytes(result.to_dict()))>64*1024*1024:
        raise quality.PlanningConstraintBlocked('严格路线超过 64 MiB 保存预算；原路线保留。',
            {k:v for k,v in report.items() if k not in ('records','base_trajectory')})
    return result


def summary_text(trajectory):
    # Presentation reads an already installed record. Parsing here would
    # recursively validate the complete history on every Qt refresh.
    gaps=trajectory.get('gap_report',{}) if isinstance(trajectory,dict) else trajectory.gap_report
    report=gaps.get('surface_support')
    if not report:return ''
    unresolved=sum(value for key,value in report['counts'].items() if key != 'checked')
    outcomes=report['edge_outcomes']
    refused=sum(value for key,value in outcomes.items() if key != 'accepted')
    return (f'严格表面路径：已重建并重新验收；处理 {report["processed_original_edges"]} / '
            f'{report["original_edge_count"]} 条原边，{unresolved} 条未构造或未确认。'
            f'新子边接纳 {outcomes.get("accepted",0)} 条，拒绝或未确认 {refused} 条。'
            '支持结论仅对应当前数值网格；覆盖需要另行核验。')
