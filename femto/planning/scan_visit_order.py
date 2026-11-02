"""Bounded endpoint proposals; distances here never certify a surface link.

Keep each block's first scan and its direction. Every remaining scan is visited
once. The caller must compare verified links and retain the original if worse.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import numpy as np

VERSION = 'endpoint_order_orientation_proposal_v1'
MAX_SEGMENTS = 512
MAX_PASSES = 8
_ATOMIC_POINT_VALUES = (str, int, float, bool, type(None), bytes)


def _clone_points(points):
    """Copy common JSON point records while preserving shared references.

    Unknown or nested values use the original deepcopy path. The returned
    points never share mutable lists or dictionaries with their source.
    """
    if type(points) is not list:return deepcopy(points)
    result=[];memo={id(points):result}
    for point in points:
        if type(point) is not dict:return deepcopy(points)
        if id(point) in memo:
            result.append(memo[id(point)]);continue
        copied=point.copy();memo[id(point)]=copied
        for key,value in point.items():
            if type(key) is not str:return deepcopy(points)
            kind=type(value)
            if kind is list:
                for item in value:
                    if type(item) not in _ATOMIC_POINT_VALUES:return deepcopy(points)
                identity=id(value)
                if identity not in memo:memo[identity]=value.copy()
                copied[key]=memo[identity]
            elif kind is tuple:
                for item in value:
                    if type(item) not in _ATOMIC_POINT_VALUES:return deepcopy(points)
            elif kind not in _ATOMIC_POINT_VALUES:return deepcopy(points)
        result.append(copied)
    return result


def oriented(segment, reverse):
    """Reverse points while preserving incoming-edge roles, not point roles."""
    points = _clone_points(segment.points)
    if reverse:
        points.reverse()
        for index in range(1, len(points)):
            points[index]['role'] = segment.points[-index].get('role', 'scan')
        points[0]['role'] = 'scan'
    for point in points:
        point.pop('revisit', None)
    return replace(segment, points=points, row_index=points[0]['row_index'])


def endpoint_cost(endpoints, visits):
    return sum(float(np.linalg.norm(endpoints[a, 1-int(ar)]-endpoints[b, int(br)]))
               for (a, ar), (b, br) in zip(visits, visits[1:]))


def _directions(endpoints, order, check):
    """Exact two-state dynamic program for a fixed order and fixed start."""
    costs = np.array([0., np.inf])
    parents = []
    for left, right in zip(order, order[1:]):
        check()
        distances = np.linalg.norm(endpoints[left, ::-1, None, :]-endpoints[right, None, :, :], axis=2)
        options = costs[:, None]+distances
        parent = np.argmin(options, axis=0)
        parents.append(parent)
        costs = options[parent, np.arange(2)]
    direction = int(np.argmin(costs))
    visits = [(order[-1], bool(direction))]
    for index in range(len(parents)-1, -1, -1):
        direction = int(parents[index][direction])
        visits.append((order[index], bool(direction)))
    return list(reversed(visits))


def _improve(endpoints, visits, check):
    # Reversing a subchain and every scan inside it leaves all internal link
    # lengths unchanged. Only two boundary distances need comparison.
    result = list(visits)
    for _ in range(MAX_PASSES):
        check()
        best = None
        indices=np.asarray([i for i,_ in result],dtype=int)
        directions=np.asarray([int(reverse) for _,reverse in result],dtype=int)
        heads=endpoints[indices,directions];tails=endpoints[indices,1-directions]
        for first in range(1, len(result)):
            check()
            deltas=np.linalg.norm(tails[first:]-tails[first-1],axis=1)-np.linalg.norm(tails[first-1]-heads[first])
            deltas[:-1]+=np.linalg.norm(heads[first+1:]-heads[first],axis=1)-np.linalg.norm(tails[first:-1]-heads[first+1:],axis=1)
            last=int(np.argmin(deltas))+first
            choice=(float(deltas[last-first]),first,last)
            if choice[0] < -1e-12 and (best is None or choice < best):best=choice
        if best is None:
            break
        _, first, last = best
        result[first:last+1] = [(i, not reverse) for i, reverse in reversed(result[first:last+1])]
    return _directions(endpoints, [i for i, _ in result], check)


def propose(segments, check=lambda: None):
    """Return block-preserving visits, with no worse Euclidean connector cost.

    A proposal is NOT accepted motion. Above the explicit search size budget,
    retain order and optimize orientations only (linear time).
    """
    if not segments:
        return []
    endpoints = np.asarray([[s.points[0]['xyz_m'], s.points[-1]['xyz_m']] for s in segments], float)
    result = []
    blocks = list(dict.fromkeys(s.block_index for s in segments))
    for block in blocks:
        check()
        indices = [i for i, s in enumerate(segments) if s.block_index == block]
        baseline = [(i, False) for i in indices]
        candidates = [baseline, _directions(endpoints, indices, check)]
        if len(indices) <= MAX_SEGMENTS:
            greedy = [baseline[0]]
            remaining = set(indices[1:])
            while remaining:
                check()
                i, reverse = greedy[-1]
                current = endpoints[i, 1-int(reverse)]
                # Stable tie breaking makes persisted reproduction independent
                # of hash/set iteration order.
                _, nxt, direction = min((float(np.linalg.norm(current-endpoints[j, direction])), j, direction)
                                         for j in sorted(remaining) for direction in (0, 1))
                greedy.append((nxt, bool(direction)))
                remaining.remove(nxt)
            candidates.extend((_improve(endpoints, candidates[1], check), _improve(endpoints, greedy, check)))
        chosen = min(enumerate(candidates), key=lambda item: (endpoint_cost(endpoints, item[1]), item[0]))[1]
        result.extend(chosen)
    return result


def checked_visits(segments, visits):
    if (not isinstance(visits, list) or len(visits) != len(segments)
            or any(not isinstance(v, (list, tuple)) or len(v) != 2
                   or type(v[0]) is not int or type(v[1]) is not bool for v in visits)
            or sorted(v[0] for v in visits) != list(range(len(segments)))):
        raise ValueError('扫描段访问顺序必须保留每段且仅访问一次')
    blocks = list(dict.fromkeys(s.block_index for s in segments))
    expected = [block for block in blocks for s in segments if s.block_index == block]
    if [segments[i].block_index for i, _ in visits] != expected:
        raise ValueError('扫描段访问顺序不得跨工作块混排')
    for block in blocks:
        first = next(i for i, s in enumerate(segments) if s.block_index == block)
        if tuple(next(v for v in visits if segments[v[0]].block_index == block)) != (first, False):
            raise ValueError('扫描段访问顺序改变了工作块起点')
    return [(i, reverse) for i, reverse in visits]


def connect(graph, segments, visits, check, budget):
    visits = checked_visits(segments, visits)
    links = []
    total = sum(len(s.points) for s in segments)
    for (left, lr), (right, rr) in zip(visits, visits[1:]):
        check()
        same = segments[left].block_index == segments[right].block_index
        route = graph.route(left, right, left_reverse=lr, right_reverse=rr) if same else None
        total += len(route) if route else 0
        if total > budget:
            raise ValueError('连续路线采样点超过预算，请缩小选区')
        links.append(dict(from_segment=left, to_segment=right, connected=bool(route),
            added_edges=len(route) if route else 0, samples=route or [],
            reason='verified_surface_path' if route else 'no_verified_graph_path' if same else 'separate_work_blocks'))
    return links


def _connector_length(tail, samples):
    xyz=np.asarray([tail['xyz_m']]+[p['xyz_m'] for p in samples])
    return float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum())


def anchored_objective(segments, visits, links, check=lambda: None, *, budget=None, _lengths=None):
    """Validate anchored links and measure their cost without copying chains.

    This is a prefilter, not geometric acceptance. Qualifying routes must still
    be reconstructed and independently checked by the caller. Its checks and
    point count match ``reconstruct(..., preserve_endpoint_faces=True)``.
    """
    visits=checked_visits(segments,visits)
    if not isinstance(links,list) or len(links)!=len(visits)-1:
        raise ValueError('访问顺序的连接记录缺失')
    count=len(segments[visits[0][0]].points);gaps=0;length=0.
    if budget is not None and count>budget:raise ValueError('连续路线采样点超过预算，请缩小选区')
    for (left,lr),(right,rr),link in zip(visits,visits[1:],links):
        check();first=segments[left];following=segments[right]
        tail=first.points[0 if lr else -1];head=following.points[-1 if rr else 0]
        same=first.block_index==following.block_index;samples=link.get('samples')
        if (link.get('from_segment')!=left or link.get('to_segment')!=right
                or type(link.get('connected')) is not bool or not isinstance(samples,list)
                or link['connected']!=bool(samples) or link.get('added_edges')!=len(samples)):
            raise ValueError('访问顺序的连接索引或点数不一致')
        if samples:
            if (not same or link.get('reason')!='verified_surface_path'
                    or samples[-1].get('xyz_m')!=head['xyz_m']
                    or any(p.get('role')!='connector' or p.get('row_index')!=head['row_index'] for p in samples)):
                raise ValueError('访问顺序的连接端点或工作块不一致')
            length+=_connector_length(tail,samples) if _lengths is None else _lengths.measure(tail,samples)
            keep=samples[-1]['triangle_index']!=head['triangle_index']
            count+=len(samples)+len(following.points)-(0 if keep else 1)
        else:
            if link.get('reason')!=('no_verified_graph_path' if same else 'separate_work_blocks'):
                raise ValueError('访问顺序的未连接原因无效')
            gaps+=int(same);count+=len(following.points)
        if budget is not None and count>budget:raise ValueError('连续路线采样点超过预算，请缩小选区')
    return gaps,length


def reconstruct(segments, visits, links, check=lambda: None, *, preserve_endpoint_faces=False, budget=None):
    visits = checked_visits(segments, visits)
    if not isinstance(links, list) or len(links) != len(visits)-1:
        raise ValueError('访问顺序的连接记录缺失')
    first = oriented(segments[visits[0][0]], visits[0][1])
    chains = [(first.block_index, first.points)]
    count = len(first.points)
    if budget is not None and count > budget:
        raise ValueError('连续路线采样点超过预算，请缩小选区')
    added, gaps, length = 0, 0, 0.
    for (left, lr), (right, rr), link in zip(visits, visits[1:], links):
        check()
        following = oriented(segments[right], rr)
        same = segments[left].block_index == following.block_index
        samples = link.get('samples')
        if (link.get('from_segment') != left or link.get('to_segment') != right
                or type(link.get('connected')) is not bool or not isinstance(samples, list)
                or link['connected'] != bool(samples) or link.get('added_edges') != len(samples)):
            raise ValueError('访问顺序的连接索引或点数不一致')
        if samples:
            if (not same or link.get('reason') != 'verified_surface_path'
                    or samples[-1].get('xyz_m') != following.points[0]['xyz_m']
                    or any(p.get('role') != 'connector' or p.get('row_index') != following.points[0]['row_index'] for p in samples)):
                raise ValueError('访问顺序的连接端点或工作块不一致')
            xyz = np.asarray([chains[-1][1][-1]['xyz_m']]+[p['xyz_m'] for p in samples])
            length += float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
            keep = preserve_endpoint_faces and samples[-1]['triangle_index'] != following.points[0]['triangle_index']
            rest = following.points if keep else following.points[1:]
            count += len(samples)+len(rest)
            if budget is not None and count > budget:
                raise ValueError('连续路线采样点超过预算，请缩小选区')
            chains[-1][1].extend(_clone_points(samples))
            chains[-1][1].extend(rest)
            added += len(samples)
        else:
            if link.get('reason') != ('no_verified_graph_path' if same else 'separate_work_blocks'):
                raise ValueError('访问顺序的未连接原因无效')
            gaps += int(same)
            count += len(following.points)
            if budget is not None and count > budget:
                raise ValueError('连续路线采样点超过预算，请缩小选区')
            chains.append((following.block_index, following.points))
    return chains, added, (gaps, length)


def optimize(graph, segments, check, budget):
    baseline = list(enumerate([False]*len(segments)))
    baseline_links = connect(graph, segments, baseline, check, budget)
    baseline_result = reconstruct(segments, baseline, baseline_links, check)
    proposal = propose(segments, check)
    proposal_links = connect(graph, segments, proposal, check, budget) if proposal != baseline else baseline_links
    proposal_result = reconstruct(segments, proposal, proposal_links, check)
    # Fewer gaps must not purchase a longer route; both requirements hold.
    accepted = (proposal_result[2][0] <= baseline_result[2][0]
                and proposal_result[2][1] < baseline_result[2][1]-1e-12)
    visits, links, result = (proposal, proposal_links, proposal_result) if accepted else (baseline, baseline_links, baseline_result)
    record = dict(version=VERSION, visits=[list(v) for v in visits], adopted=accepted,
                  baseline_connections=baseline_links,
                  baseline_objective=list(baseline_result[2]), selected_objective=list(result[2]))
    return result[0], links, result[1], record
