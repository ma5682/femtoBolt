"""Bounded single-scan relocation using verified surface links.

This is a research proposal, not a product route, coverage proof or robot
trajectory. The caller must independently validate the source and final edges
and reprove any coverage evidence before adopting a proposal.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

from femto.planning import balanced_visit_improvement, scan_visit_order
from femto.planning.endpoint_preserving_detour import _signature

VERSION = 'research-verified-scan-relocation-v1'
LIMITED_DETOUR_VERSION = 'research-limited-detour-priority-relocation-v2'
MAX_SEGMENTS = 256
MAX_TRIALS = 4096
MAX_ROUNDS = 4
EXTENDED_MAX_ROUNDS = 8
MAX_CACHE_LINKS = 4096
MAX_CACHE_POINTS = 100_000
MAX_DISTANCE_CACHE = 200_000
MAX_PREVIEW_POINTS = 100_000
EPS = 1e-12


def _detour_priority_improves(values,length,prior,prior_length):
    if values['detours']!=prior['detours']:return values['detours']<prior['detours']
    if values['revisit_connector_m']<prior['revisit_connector_m']-EPS:return True
    return (abs(values['revisit_connector_m']-prior['revisit_connector_m'])<=EPS
            and length<prior_length-EPS)


class _ConnectionLengths:
    """Per-search arithmetic reuse; every use snapshots the full coordinates.

    This is not a geometry or route acceptance cache. Link roles, endpoints,
    blocks and point budgets are still checked by anchored_objective. Every
    possible best candidate is independently rebuilt and measured afterwards.
    """
    def __init__(self):
        self.values={};self.points=0

    def measure(self,tail,samples):
        # Restrict keys to the exact native float representation used by graph
        # samples. Other dtypes retain the original NumPy arithmetic path.
        key=[]
        for point in (tail,*samples):
            xyz=point.get('xyz_m')
            if (type(xyz) not in (list,tuple) or len(xyz)!=3
                    or any(type(x) is not float or not math.isfinite(x) for x in xyz)):
                return scan_visit_order._connector_length(tail,samples)
            key.append(tuple(xyz))
        key=tuple(key)
        if key in self.values:return self.values[key]
        value=scan_visit_order._connector_length(tail,samples)
        if (math.isfinite(value) and len(self.values)<MAX_CACHE_LINKS
                and self.points+len(key)<=MAX_CACHE_POINTS):
            self.values[key]=value;self.points+=len(key)
        return value


class _LocalGraph:
    """Per-call cache of validated baseline links and newly verified paths."""
    def __init__(self, graph, visits, links, check):
        self.graph, self.check = graph, check
        self.cache, self.points = {}, 0
        for (a, ar), (b, br), link in zip(visits, visits[1:], links):
            if link['connected']:
                self._remember((a, b, ar, br), link['samples'])

    def _remember(self, key, value):
        count = len(value) if value else 0
        if len(self.cache) < MAX_CACHE_LINKS and self.points + count <= MAX_CACHE_POINTS:
            self.cache[key] = value
            self.points += count

    def route(self, left, right, *, left_reverse=False, right_reverse=False):
        self.check()
        key = (left, right, left_reverse, right_reverse)
        if key not in self.cache:
            value = self.graph.route(left, right, left_reverse=left_reverse, right_reverse=right_reverse)
            self._remember(key, value)
            return value
        return self.cache[key]


def _endpoint(segments, visit, tail):
    segment = segments[visit[0]]
    return np.asarray(segment.points[-1 if bool(tail) != visit[1] else 0]['xyz_m'], float)


def _estimate(segments, visits):
    return sum(float(np.linalg.norm(_endpoint(segments, a, True) - _endpoint(segments, b, False)))
               for a, b in zip(visits, visits[1:])
               if segments[a[0]].block_index == segments[b[0]].block_index)


def _candidates(segments, current, check, cache=None):
    """Rank relocations from the six changed boundary distances only."""
    endpoints={(i,reverse,tail):_endpoint(segments,(i,reverse),tail)
               for i in range(len(segments)) for reverse in (False,True) for tail in (False,True)}
    # The same segment endpoints are used by each round. Cache only these
    # distance calculations within this proposal, and invalidate the entire
    # cache if any endpoint changes. This cache never certifies a graph edge.
    signature=tuple(tuple(float(x) for x in value) for value in endpoints.values())
    if cache is None or MAX_DISTANCE_CACHE<=0:distances={}
    else:
        if cache.get('signature')!=signature:cache.clear();cache['signature']=signature
        distances=cache.setdefault('distances',{})
    def cost(a,b):
        if a is None or b is None or segments[a[0]].block_index!=segments[b[0]].block_index:return 0.
        key=(a,b)
        if key not in distances:
            value=float(np.linalg.norm(endpoints[(a[0],a[1],True)]-endpoints[(b[0],b[1],False)]))
            if len(distances)<MAX_DISTANCE_CACHE:distances[key]=value
            return value
        return distances[key]
    original=sum(cost(a,b) for a,b in zip(current,current[1:]))
    candidates=[];count=0
    for i,value in enumerate(current):
        check();block=segments[value[0]].block_index
        if i==0 or segments[current[i-1][0]].block_index!=block:continue
        left=current[i-1];right=current[i+1] if i+1<len(current) else None
        removal=cost(left,right)-cost(left,value)-cost(value,right)
        remaining=current[:i]+current[i+1:]
        for j in range(1,len(remaining)+1):
            if segments[remaining[j-1][0]].block_index!=block:continue
            before=remaining[j-1];after=remaining[j] if j<len(remaining) else None
            for reverse in (False,True):
                count+=1
                if count%128==0:check()
                if j==i and reverse==value[1]:continue
                moved=(value[0],reverse)
                score=original+removal+cost(before,moved)+cost(moved,after)-cost(before,after)
                candidates.append((score,i,j,reverse))
    return sorted(candidates)


class _Preview:
    """Read-only candidate chains for scoring; never returned or installed."""
    def __init__(self, segments, check):
        self.segments, self.check = segments, check
        self.cache, self.points = {}, 0

    def _oriented(self, index, reverse):
        key=(index,reverse)
        if key in self.cache:return self.cache[key]
        source=self.segments[index].points
        points=[dict(point) for point in (reversed(source) if reverse else source)]
        if reverse:
            for i in range(1,len(points)):points[i]['role']=source[-i].get('role','scan')
            points[0]['role']='scan'
        for point in points:point.pop('revisit',None)
        if self.points+len(points)<=MAX_PREVIEW_POINTS:
            self.cache[key]=points;self.points+=len(points)
        return points

    def chains(self, visits, links):
        """Use only after anchored_objective has validated links and budget."""
        first=visits[0];chains=[(self.segments[first[0]].block_index,list(self._oriented(*first)))]
        for (_, _), (right,reverse), link in zip(visits,visits[1:],links):
            self.check();following=self._oriented(right,reverse);samples=link['samples']
            if samples:
                keep=samples[-1]['triangle_index']!=following[0]['triangle_index']
                chains[-1][1].extend(samples)
                chains[-1][1].extend(following if keep else following[1:])
            else:chains.append((self.segments[right].block_index,list(following)))
        return chains


def propose(graph, segments, visits, links, spacing_m, *, motion_limits, check=lambda: None,
            budget=200_000, attempts=256, rounds=2, objective_policy='pareto'):
    """Move one scan at a time, keeping every block start and scan edge.

    Adopt only a shorter or less redundant checked route with no increase in
    gaps, connector length, detours or nominal revisit, including the caller's
    independently validated published-route limits. The geometric estimate
    ranks work only. The source links must already have been validated by the
    caller; no cached link or proposal establishes coverage.

    Explicit detour_first_with_limits uses a separate record version. It can
    spend the caller's connector/revisit allowance to reduce detours, then
    revisit, then connector length. All three absolute caps still apply;
    callers comparing two layouts must separately check the published result.
    """
    if (objective_policy not in ('pareto','detour_first_with_limits')
            or not isinstance(motion_limits,dict) or set(motion_limits)!={'detours','connector_length_m','revisit_connector_m'}
            or type(motion_limits.get('detours')) is not int or motion_limits['detours']<0
            or any(isinstance(motion_limits.get(key),bool) or not isinstance(motion_limits.get(key),(float,int))
                   or not math.isfinite(motion_limits[key]) or motion_limits[key]<0
                   for key in ('connector_length_m','revisit_connector_m'))
            or not isinstance(segments, list) or not segments or len(segments) > MAX_SEGMENTS
            or isinstance(spacing_m, bool) or not isinstance(spacing_m, (float, int))
            or not math.isfinite(spacing_m) or spacing_m <= 0
            or type(budget) is not int or not 1 <= budget <= 200_000
            or type(attempts) is not int or not 1 <= attempts <= MAX_TRIALS
            or type(rounds) is not int or not 1 <= rounds <= MAX_ROUNDS):
        raise ValueError('扫描段移动研究参数无效或超过预算')
    check()
    current = scan_visit_order.checked_visits(segments, visits)
    initial = scan_visit_order.reconstruct(segments, current, links, check,
                                           preserve_endpoint_faces=True, budget=budget)
    if initial[2][0]:raise ValueError('扫描段移动需要工作块内已连接')
    initial_metrics = balanced_visit_improvement._metrics(initial[0], spacing_m, check)
    signature = _signature(initial[0], check)
    if signature != _signature([(segment.block_index,segment.points) for segment in segments],check):
        raise ValueError('扫描段移动基准改变了原扫描边')
    local = _LocalGraph(graph, current, links, check)
    preview = _Preview(segments, check)
    lengths = _ConnectionLengths()
    current_links, selected = deepcopy(links), initial
    metrics, steps, tested = initial_metrics, [], 0
    rank_cache={}
    for _ in range(rounds):
        check();candidates=_candidates(segments,current,check,rank_cache)
        best=None
        for _,i,j,reverse in candidates[:attempts]:
            check();tested+=1
            value=current[i];remaining=current[:i]+current[i+1:]
            proposal=remaining[:j]+[(value[0],reverse)]+remaining[j:]
            scan_visit_order.checked_visits(segments,proposal)
            candidate_links=scan_visit_order.connect(local,segments,proposal,check,budget)
            objective=scan_visit_order.anchored_objective(segments,proposal,candidate_links,check,budget=budget,_lengths=lengths)
            if (objective[0]>selected[2][0] or objective[1]>motion_limits['connector_length_m']+EPS
                    or (objective_policy=='pareto' and objective[1]>selected[2][1]+EPS)):continue
            values=balanced_visit_improvement._metrics(preview.chains(proposal,candidate_links),spacing_m,check)
            if (values['detours']>motion_limits['detours'] or
                    values['revisit_connector_m']>motion_limits['revisit_connector_m']+EPS):continue
            if objective_policy=='pareto':
                if (values['detours']>metrics['detours'] or values['revisit_connector_m']>metrics['revisit_connector_m']+EPS):continue
                if not (objective[1]<selected[2][1]-EPS or values['detours']<metrics['detours'] or
                        values['revisit_connector_m']<metrics['revisit_connector_m']-EPS):continue
            elif not _detour_priority_improves(values,objective[1],metrics,selected[2][1]):continue
            rank=(values['detours'],values['revisit_connector_m'],objective[1],i,j,reverse)
            if best is None or rank<best[0]:
                candidate=scan_visit_order.reconstruct(segments,proposal,candidate_links,check,
                                                       preserve_endpoint_faces=True,budget=budget)
                verified=balanced_visit_improvement._metrics(candidate[0],spacing_m,check)
                if (verified!=values or candidate[2][0]!=objective[0] or abs(candidate[2][1]-objective[1])>EPS):
                    raise ValueError('扫描段移动预筛与完整重建不一致')
                if _signature(candidate[0],check)!=signature:raise ValueError('扫描段移动改变了原扫描边')
                best=(rank,proposal,candidate_links,candidate,values)
        if best is None:break
        rank,current,current_links,selected,metrics=best
        steps.append(dict(source=rank[3],destination=rank[4],reverse=rank[5]))
    check()
    report=dict(version=VERSION if objective_policy=='pareto' else LIMITED_DETOUR_VERSION,adopted=bool(steps),visits=[list(v) for v in current],steps=steps,
        tested=tested,baseline=initial_metrics,selected=metrics,baseline_objective=list(initial[2]),
        selected_objective=list(selected[2]),motion_limits=deepcopy(motion_limits),
        semantics='candidate_requires_geometry_and_coverage_replay')
    return selected[0],deepcopy(current_links),selected[1],report


def _validate_record(record,segments,baseline_visits,baseline_links,final_links,spacing_m,*,motion_limits,
                     attempts=256,rounds=2,check=lambda:None,budget=200_000,objective_policy='pareto',maximum_rounds=MAX_ROUNDS):
    """Rebuild a relocation record and verify its motion and scan invariants.

    This verifies the recorded result, not that the bounded search found an
    optimum. The caller must independently check every emitted geometric edge.
    """
    expected_version=VERSION if objective_policy=='pareto' else LIMITED_DETOUR_VERSION
    if (objective_policy not in ('pareto','detour_first_with_limits')
            or not isinstance(segments,list) or not segments or len(segments)>MAX_SEGMENTS
            or not isinstance(record,dict) or set(record)!={'version','adopted','visits','steps','tested','baseline','selected',
            'baseline_objective','selected_objective','motion_limits','semantics'}
            or record['version']!=expected_version or type(record['adopted']) is not bool
            or record['semantics']!='candidate_requires_geometry_and_coverage_replay'
            or record['motion_limits']!=motion_limits or not isinstance(record['steps'],list)
            or type(attempts) is not int or not 1<=attempts<=MAX_TRIALS
            or type(rounds) is not int or not 1<=rounds<=maximum_rounds
            or len(record['steps'])>rounds or type(record['tested']) is not int
            or not len(record['steps'])<=record['tested']<=attempts*rounds
            or type(budget) is not int or not 1<=budget<=200_000
            or isinstance(spacing_m,bool) or not isinstance(spacing_m,(float,int))
            or not math.isfinite(spacing_m) or spacing_m<=0):
        raise ValueError('扫描段移动记录或预算无效')
    if (not isinstance(motion_limits,dict) or set(motion_limits)!={'detours','connector_length_m','revisit_connector_m'}
            or type(motion_limits.get('detours')) is not int or motion_limits['detours']<0
            or any(isinstance(motion_limits.get(key),bool) or not isinstance(motion_limits.get(key),(float,int))
                   or not math.isfinite(motion_limits[key]) or motion_limits[key]<0
                   for key in ('connector_length_m','revisit_connector_m'))):
        raise ValueError('扫描段移动记录的原路线限制无效')
    for values in (record['baseline'],record['selected']):
        if (not isinstance(values,dict) or set(values)!={'detours','revisit_connector_m'}
                or type(values.get('detours')) is not int or values['detours']<0
                or isinstance(values.get('revisit_connector_m'),bool)
                or not isinstance(values.get('revisit_connector_m'),(float,int))
                or not math.isfinite(values['revisit_connector_m']) or values['revisit_connector_m']<0):
            raise ValueError('扫描段移动记录的重访统计无效')
    for objective in (record['baseline_objective'],record['selected_objective']):
        if (not isinstance(objective,list) or len(objective)!=2 or type(objective[0]) is not int or objective[0]<0
                or isinstance(objective[1],bool) or not isinstance(objective[1],(float,int))
                or not math.isfinite(objective[1]) or objective[1]<0):
            raise ValueError('扫描段移动记录的连接统计无效')
    check();current=scan_visit_order.checked_visits(segments,baseline_visits)
    baseline=scan_visit_order.reconstruct(segments,current,baseline_links,check,preserve_endpoint_faces=True,budget=budget)
    if baseline[2][0]:raise ValueError('扫描段移动基准存在工作块内部断点')
    before=balanced_visit_improvement._metrics(baseline[0],spacing_m,check)
    signature=_signature(baseline[0],check)
    if signature!=_signature([(s.block_index,s.points) for s in segments],check):raise ValueError('扫描段移动基准改变了原扫描边')
    if record['baseline']!=before or record['baseline_objective']!=list(baseline[2]):
        raise ValueError('扫描段移动基准统计不一致')
    for step in record['steps']:
        check()
        if (not isinstance(step,dict) or set(step)!={'source','destination','reverse'}
                or type(step['source']) is not int or type(step['destination']) is not int
                or type(step['reverse']) is not bool or not 0<=step['source']<len(current)
                or not 1<=step['destination']<len(current)):
            raise ValueError('扫描段移动步骤无效')
        index=step['source'];value=current[index]
        if index==0 or segments[current[index-1][0]].block_index!=segments[value[0]].block_index:
            raise ValueError('扫描段移动不能改变工作块起点')
        remaining=current[:index]+current[index+1:]
        candidate=remaining[:step['destination']]+[(value[0],step['reverse'])]+remaining[step['destination']:]
        if candidate==current:raise ValueError('扫描段移动步骤没有改变路线')
        current=scan_visit_order.checked_visits(segments,candidate)
    if record['visits']!=[list(v) for v in current] or record['adopted']!=bool(record['steps']):
        raise ValueError('扫描段移动步骤与最终顺序不一致')
    selected=scan_visit_order.reconstruct(segments,current,final_links,check,preserve_endpoint_faces=True,budget=budget)
    values=balanced_visit_improvement._metrics(selected[0],spacing_m,check)
    if (_signature(selected[0],check)!=signature or record['selected']!=values
            or record['selected_objective']!=list(selected[2])):
        raise ValueError('扫描段移动最终路线或统计不一致')
    if record['adopted']:
        if (selected[2][0]>baseline[2][0] or selected[2][1]>motion_limits['connector_length_m']+EPS
                or values['detours']>motion_limits['detours']
                or values['revisit_connector_m']>motion_limits['revisit_connector_m']+EPS):
            raise ValueError('扫描段移动未满足原路线约束')
        if objective_policy=='pareto':
            if (selected[2][1]>baseline[2][1]+EPS or values['detours']>before['detours']
                    or values['revisit_connector_m']>before['revisit_connector_m']+EPS
                    or not (selected[2][1]<baseline[2][1]-EPS or values['detours']<before['detours']
                            or values['revisit_connector_m']<before['revisit_connector_m']-EPS)):
                raise ValueError('扫描段移动未满足原路线约束')
        elif not _detour_priority_improves(values,selected[2][1],before,baseline[2][1]):
            raise ValueError('有界绕行优先移动没有改善排序目标')
    elif current!=scan_visit_order.checked_visits(segments,baseline_visits) or final_links!=baseline_links:
        raise ValueError('未采用的扫描段移动改变了原路线')
    check();return selected[0],selected[1]


def validate_record(record,segments,baseline_visits,baseline_links,final_links,spacing_m,*,motion_limits,
                    attempts=256,rounds=2,check=lambda:None,budget=200_000,objective_policy='pareto'):
    """Retain the original four-round replay contract for existing versions."""
    return _validate_record(record,segments,baseline_visits,baseline_links,final_links,spacing_m,
        motion_limits=motion_limits,attempts=attempts,rounds=rounds,check=check,budget=budget,
        objective_policy=objective_policy,maximum_rounds=MAX_ROUNDS)


def validate_extended_record(record,segments,baseline_visits,baseline_links,final_links,spacing_m,*,motion_limits,
                             attempts,rounds,check=lambda:None,budget=200_000):
    """Replay the separately versioned bounded policy with at most eight rounds."""
    return _validate_record(record,segments,baseline_visits,baseline_links,final_links,spacing_m,
        motion_limits=motion_limits,attempts=attempts,rounds=rounds,check=check,budget=budget,
        objective_policy='detour_first_with_limits',maximum_rounds=EXTENDED_MAX_ROUNDS)
