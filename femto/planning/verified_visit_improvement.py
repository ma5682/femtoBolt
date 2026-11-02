"""Bounded surface-link 2-opt proposal for versioned continuous planning.

Candidate distance is a lower bound only. A candidate can improve the current
order only after both replacement links have been returned by the supplied
verified surface graph. The complete final order is reconnected and compared
with the original before it can be returned.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from femto.planning.scan_visit_order import checked_visits, connect, reconstruct

VERSION = 'verified-surface-link-two-opt-proposal-v1'
MAX_SEGMENTS = 512
MAX_ROUNDS = 8
MAX_CANDIDATES_PER_ROUND = 60
# Per-proposal endpoint arithmetic only; bounded and never reused as geometry evidence.
MAX_ENDPOINT_DISTANCES = 200_000


def _length(start, samples):
    xyz = np.asarray([start] + [point['xyz_m'] for point in samples], dtype=float)
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def improve(graph, segments, visits, links, check, budget, *, search=None):
    """Propose a non-worse connected order without modifying its inputs.

    This is research infrastructure, not a route or coverage certificate.
    Callers must apply all existing validation to the complete emitted route.
    """
    if not segments:raise ValueError('扫描段不能为空')
    current = checked_visits(segments, visits)
    original_links = deepcopy(links)
    original = reconstruct(segments, current, original_links, check)
    if not current or len(current) > MAX_SEGMENTS:
        return original[0], original_links, original[1], dict(version=VERSION, adopted=False, rounds=[])
    endpoints = np.asarray([[s.points[0]['xyz_m'], s.points[-1]['xyz_m']] for s in segments], dtype=float)

    def entry(visit):return endpoints[visit[0], int(visit[1])]
    def exit(visit):return endpoints[visit[0], 1-int(visit[1])]
    distances = {}
    def endpoint_distance(left, left_side, right, right_side):
        a=(left[0], left_side(left));b=(right[0], right_side(right))
        key=(a,b) if a<=b else (b,a)
        if key in distances:return distances[key]
        value=float(np.linalg.norm(endpoints[a]-endpoints[b]))
        if len(distances)<MAX_ENDPOINT_DISTANCES:distances[key]=value
        return value
    def entry_side(visit):return int(visit[1])
    def exit_side(visit):return 1-int(visit[1])

    lengths = [_length(exit(current[i]).tolist(), link['samples']) if link['connected'] else 0.
               for i, link in enumerate(original_links)]
    connected = [link['connected'] for link in original_links]
    disconnected_prefix = [0]
    for value in connected:disconnected_prefix.append(disconnected_prefix[-1]+int(not value))
    rounds = []
    for _ in range(MAX_ROUNDS):
        if search is not None and not search.can_try('two_opt'):break
        check();candidates = []
        for first in range(1, len(current)):
            check()
            for last in range(first, len(current)-1):
                if segments[current[first-1][0]].block_index != segments[current[last+1][0]].block_index:
                    continue
                if (not connected[first-1] or not connected[last]
                        or disconnected_prefix[last] != disconnected_prefix[first]):
                    continue
                lower = (endpoint_distance(current[first-1],exit_side,current[last],exit_side)
                         + endpoint_distance(current[first],entry_side,current[last+1],entry_side))
                potential = lengths[first-1]+lengths[last]-float(lower)
                if potential > 1e-9:candidates.append((potential, first, last))
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        best = None
        for potential, first, last in candidates[:MAX_CANDIDATES_PER_ROUND]:
            if search is not None and not search.take('two_opt'):break
            check();left, old_first, old_last, right = current[first-1], current[first], current[last], current[last+1]
            new_first = (old_last[0], not old_last[1]);new_last = (old_first[0], not old_first[1])
            a = graph.route(left[0], new_first[0], left_reverse=left[1], right_reverse=new_first[1])
            b = graph.route(new_last[0], right[0], left_reverse=new_last[1], right_reverse=right[1])
            if not a or not b:continue
            a_length = _length(exit(left).tolist(), a);b_length = _length(exit(new_last).tolist(), b)
            saving = lengths[first-1]+lengths[last]-a_length-b_length
            choice = (saving, -first, -last, first, last, a_length, b_length, potential)
            if saving > 1e-9 and (best is None or choice[:3] > best[:3]):best = choice
        if best is None:break
        saving, _, _, first, last, a_length, b_length, potential = best
        current[first:last+1] = [(index, not reverse) for index, reverse in reversed(current[first:last+1])]
        lengths[first-1:last+1] = [a_length, *reversed(lengths[first:last]), b_length]
        rounds.append(dict(first=first, last=last, saving_m=saving, lower_bound_saving_m=potential))
    if not rounds:
        return original[0], original_links, original[1], dict(version=VERSION, adopted=False, rounds=[])
    # Re-search all links; reverse-direction sampling or graph state must not
    # turn the bound-based estimate into a worse final route.
    final_links = connect(graph, segments, current, check, budget)
    final = reconstruct(segments, current, final_links, check)
    accepted = final[2][0] <= original[2][0] and final[2][1] < original[2][1]-1e-12
    if not accepted:
        return original[0], original_links, original[1], dict(version=VERSION, adopted=False, rounds=rounds)
    return final[0], final_links, final[1], dict(version=VERSION, adopted=True, rounds=rounds,
        visits=[list(v) for v in current], baseline_objective=list(original[2]), selected_objective=list(final[2]))


def validate_record(record, segments, visits, initial_links, final_links, check):
    """Validate the saved decisions; full reproduction reruns the graph search."""
    if (not isinstance(record,dict) or record.get('version')!=VERSION
            or type(record.get('adopted')) is not bool or not isinstance(record.get('rounds'),list)
            or len(record['rounds'])>MAX_ROUNDS or record.get('baseline_visits')!=visits
            or record.get('baseline_connections')!=initial_links):
        raise ValueError('表面连接优化记录无效')
    skipped=record.get('skipped')
    if skipped not in (None,'geometry_budget') or (skipped and (record['adopted'] or record['rounds'])):
        raise ValueError('表面连接优化预算记录无效')
    current=checked_visits(segments,visits)
    if len(current)>MAX_SEGMENTS and record['rounds']:raise ValueError('表面连接优化超出扫描段预算')
    for step in record['rounds']:
        check()
        if (not isinstance(step,dict) or set(step)!={'first','last','saving_m','lower_bound_saving_m'}
                or type(step['first']) is not int or type(step['last']) is not int
                or not 1<=step['first']<=step['last']<len(current)-1
                or any(isinstance(step[k],bool) or not isinstance(step[k],(float,int)) or not np.isfinite(step[k]) or step[k]<=0
                       for k in ('saving_m','lower_bound_saving_m'))):
            raise ValueError('表面连接优化步骤无效')
        first,last=step['first'],step['last']
        if segments[current[first-1][0]].block_index != segments[current[last+1][0]].block_index:
            raise ValueError('表面连接优化跨越工作块')
        current[first:last+1]=[(index,not reverse) for index,reverse in reversed(current[first:last+1])]
        checked_visits(segments,current)
    original=reconstruct(segments,visits,initial_links,check)
    if not record['adopted']:
        if final_links!=initial_links:raise ValueError('未采用的表面连接优化改变了路线')
        return visits
    if not record['rounds'] or record.get('visits')!=[list(v) for v in current]:
        raise ValueError('表面连接优化的扫描顺序与步骤不一致')
    final=reconstruct(segments,current,final_links,check)
    if (record.get('baseline_objective')!=list(original[2]) or record.get('selected_objective')!=list(final[2])
            or final[2][0]>original[2][0] or final[2][1]>=original[2][1]-1e-12):
        raise ValueError('表面连接优化未保持断点和长度约束')
    return current
