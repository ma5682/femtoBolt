"""Bounded seam-aware ordering proposal using existing verified connections.

The result is a proposal only; it does not certify coverage or execute motion.
Near-coincident points are grouped only when the surface graph verifies their
connection, and every final transition is checked again by that graph.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

from femto.planning.scan_visit_order import checked_visits,connect,propose as order_propose,reconstruct
from femto.planning.surface_trajectory import ScanSegment
from femto.planning.balanced_visit_improvement import _metrics

VERSION='verified-seam-visit-proposal-v1'
MAX_SEGMENTS=512
MAX_PAIRS=8192
MAX_SWAP_TRIALS=4
SEAM_SEARCH_RADIUS_M=1e-8
MAX_VERIFIED_SEAM_LENGTH_M=1e-12


def _endpoint(segment, reverse, *, end):
    """Read just the fields used by this proposal, without copying a scan."""
    point=segment.points[0 if reverse == end else -1]
    return dict(xyz_m=point['xyz_m'],row_index=point['row_index'])


def _directions(proxies,order,check,*,_distance_cache=None):
    """Optimize each work block separately, preserving every block's start."""
    result=[];total=(0,0.);blocks=list(dict.fromkeys(proxies[i].block_index for i in order))
    if [proxies[i].block_index for i in order]!=[b for b in blocks for i in order if proxies[i].block_index==b]:
        raise ValueError('接缝排序不得跨工作块混排')
    for block in blocks:
        indices=[i for i in order if proxies[i].block_index==block]
        cost=[(0,0.),(10**9,float('inf'))];parents=[]
        for left,right in zip(indices,indices[1:]):
            check();row=[];new=[]
            for r in (0,1):
                head=proxies[right].points[r];options=[]
                for l in (0,1):
                    tail=proxies[left].points[1-l]
                    key=(left,right,l,r)
                    if _distance_cache is not None and key in _distance_cache:distance=_distance_cache[key]
                    else:
                        distance=float(np.linalg.norm(np.asarray(tail['xyz_m'])-head['xyz_m']))
                        if _distance_cache is not None:_distance_cache[key]=distance
                    options.append((cost[l][0]+int(tail['row_index']==head['row_index']),
                        cost[l][1]+distance,l))
                best=min(options);row.append(best[2]);new.append(best[:2])
            cost=new;parents.append(row)
        direction=min((cost[i],i) for i in (0,1))[1];chosen=[direction]
        for parent in reversed(parents):direction=parent[direction];chosen.append(direction)
        result.extend(zip(indices,map(bool,reversed(chosen))))
        best=min(cost);total=(total[0]+best[0],total[1]+best[1])
    return result,total


def _patch_rows(layout,check):
    mapping={}
    for fi,fragment in enumerate(layout.get('fragments',[])):
        for pi,part in enumerate(fragment.get('patches',[])):
            first,count=part['first_row_index'],part['retained_row_count']
            if type(first) is not int or type(count) is not int or first<0 or count<0:
                raise ValueError('接缝排序的局部扫描行范围无效')
            for row in range(first,first+count):
                check();key=(fragment['block_index'],row)
                if key in mapping:raise ValueError('接缝排序的局部扫描行不唯一')
                mapping[key]=(fi,pi)
    return mapping


def propose(graph,segments,visits,links,layout,spacing_m,check,budget,*,search=None):
    """Return a verified non-worse candidate, or the original connections."""
    if not segments:raise ValueError('接缝排序缺少扫描段')
    if isinstance(spacing_m,bool) or not isinstance(spacing_m,(float,int)) or not math.isfinite(spacing_m) or spacing_m<=0:
        raise ValueError('接缝排序的重访判定行距无效')
    visits=checked_visits(segments,visits);original=reconstruct(segments,visits,links,check)
    baseline=_metrics(original[0],spacing_m,check)
    def fallback(reason,matched=0):
        return original[0],deepcopy(links),original[1],dict(version=VERSION,adopted=False,baseline=baseline,
            reason=reason,matched_seams=matched,attempts=[])
    if search is not None and not search.can_try('seam_pair'):return fallback('search_budget')
    if not segments or len(segments)>MAX_SEGMENTS:return fallback('segment_budget')
    mapping=_patch_rows(layout,check);xyz=[];meta=[];position={v[0]:i for i,v in enumerate(visits)}
    for i,segment in enumerate(segments):
        for side,point in ((0,segment.points[0]),(-1,segment.points[-1])):
            xyz.append(point['xyz_m']);meta.append((i,side,segment.block_index,mapping.get((segment.block_index,point['row_index']))))
    if not xyz:return fallback('no_seams')
    import open3d as o3d
    points=np.asarray(xyz,dtype=float);cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(points)
    tree=o3d.geometry.KDTreeFlann(cloud);pairs=set()
    for a,point in enumerate(points):
        check();_,ids,_=tree.search_radius_vector_3d(point,SEAM_SEARCH_RADIUS_M)
        for b in ids:
            if (a<b and meta[a][0]!=meta[b][0] and meta[a][2]==meta[b][2]
                    and meta[a][3]!=meta[b][3] and None not in (meta[a][3],meta[b][3])):
                pairs.add((a,b))
                if len(pairs)>MAX_PAIRS:return fallback('pair_budget')
    first_by_block={block:next(i for i,s in enumerate(segments) if s.block_index==block)
                    for block in dict.fromkeys(s.block_index for s in segments)}
    used=set();groups=[];matched=0
    for a,b in sorted(pairs,key=lambda pair:(min(position[meta[pair[0]][0]],position[meta[pair[1]][0]]),pair)):
        check();ai,aside,block,_=meta[a];bi,bside,_,_=meta[b]
        if ai in used or bi in used:continue
        if bi==first_by_block[block] or (ai!=first_by_block[block] and position[bi]<position[ai]):
            ai,bi,aside,bside=bi,ai,bside,aside
        if ai==first_by_block[block] and aside!=-1:continue
        av,bv=(ai,aside==0),(bi,bside==-1)
        if search is not None and not search.take('seam_pair'):break
        route=graph.route(ai,bi,left_reverse=av[1],right_reverse=bv[1])
        if not route:continue
        coordinates=np.asarray([_endpoint(segments[ai],av[1],end=True)['xyz_m']]+[p['xyz_m'] for p in route])
        if float(np.linalg.norm(np.diff(coordinates,axis=0),axis=1).sum())>MAX_VERIFIED_SEAM_LENGTH_M:continue
        groups.append([av,bv]);used.update((ai,bi));matched+=1
    if not matched:return fallback('no_verified_seams')
    for i in range(len(segments)):
        if i not in used:groups.append([(i,False)])
    groups.sort(key=lambda group:min(position[i] for i,_ in group));proxies=[]
    for index,group in enumerate(groups):
        check();a=_endpoint(segments[group[0][0]],group[0][1],end=False)
        b=_endpoint(segments[group[-1][0]],group[-1][1],end=True)
        proxies.append(ScanSegment(index,a['row_index'],[a,b],segments[group[0][0]].block_index))
    initial=order_propose(proxies,check);order=[index for index,_ in initial]
    # Proxies are private snapshots for this invocation. Distances are pure
    # endpoint arithmetic, not geometry/acceptance verdicts, and are discarded
    # when this proposal returns. Graph links still receive all original checks.
    distance_cache={}
    initial,cost=_directions(proxies,order,check,_distance_cache=distance_cache)
    candidates=[('base',initial)]
    swaps=[]
    for k in range(1,len(order)-1):
        check();block=proxies[order[k]].block_index
        if block!=proxies[order[k+1]].block_index:continue
        if any(i==first_by_block[block] for i,_ in groups[order[k]]):continue
        modified=list(order);modified[k],modified[k+1]=modified[k+1],modified[k]
        choices,estimate=_directions(proxies,modified,check,_distance_cache=distance_cache)
        if estimate[0]<cost[0]:swaps.append((estimate,k,choices))
    swaps.sort(key=lambda value:(value[0],value[1]))
    candidates.extend((f'swap:{index}',choices) for _,index,choices in swaps[:MAX_SWAP_TRIALS])
    attempts=[];best=None
    for name,choices in candidates:
        if search is not None and not search.take('seam_order'):break
        check();actual=[]
        for group_index,reverse in choices:
            group=groups[group_index]
            actual.extend([(i,not r) for i,r in reversed(group)] if reverse else group)
        checked_visits(segments,actual)
        selected_links=connect(graph,segments,actual,check,budget)
        selected=reconstruct(segments,actual,selected_links,check)
        metrics=_metrics(selected[0],spacing_m,check)
        accepted=(selected[2][0]<=original[2][0] and selected[2][1]<original[2][1]-1e-12
                  and metrics['detours']<=baseline['detours']
                  and metrics['revisit_connector_m']<=baseline['revisit_connector_m']+1e-12)
        attempts.append(dict(name=name,metrics=metrics,objective=list(selected[2]),accepted=accepted))
        if accepted:
            rank=(metrics['detours'],metrics['revisit_connector_m'],selected[2][1],len(attempts))
            if best is None or rank<best[0]:best=(rank,actual,selected_links,selected,metrics)
            # Swaps are ordered by optimistic detour count. First verified
            # improvement suffices; avoid repeating the full graph traversal.
            if name!='base':break
    if best is None:
        result=fallback('no_nonworse_candidate',matched)
        result[3]['attempts']=attempts
        return result
    _,actual,selected_links,selected,metrics=best
    return selected[0],selected_links,selected[1],dict(version=VERSION,adopted=True,baseline=baseline,
        selected=metrics,baseline_objective=list(original[2]),selected_objective=list(selected[2]),
        visits=[list(v) for v in actual],matched_seams=matched,attempts=attempts)


def validate_record(record,segments,visits,initial_links,final_links,spacing_m,check,*,allow_search_budget=False):
    """Validate saved candidate outcomes; full route replay reruns the search."""
    if (not isinstance(record,dict) or record.get('version')!=VERSION or type(record.get('adopted')) is not bool
            or record.get('baseline_visits')!=visits or record.get('baseline_connections')!=initial_links
            or type(record.get('matched_seams')) is not int or not 0<=record['matched_seams']<=len(segments)//2
            or not isinstance(record.get('attempts'),list) or len(record['attempts'])>MAX_SWAP_TRIALS+1):
        raise ValueError('接缝排序记录无效')
    original=reconstruct(segments,visits,initial_links,check);baseline=_metrics(original[0],spacing_m,check)
    if record.get('baseline')!=baseline:raise ValueError('接缝排序基准不能重算')
    attempts=record['attempts']
    for index,attempt in enumerate(attempts):
        if (not isinstance(attempt,dict) or set(attempt)!={'name','metrics','objective','accepted'}
                or type(attempt['accepted']) is not bool or not isinstance(attempt['name'],str)
                or (index==0 and attempt['name']!='base')
                or (index>0 and not (attempt['name'].startswith('swap:') and attempt['name'][5:].isdigit()))):
            raise ValueError('接缝排序尝试记录无效')
        metrics,objective=attempt['metrics'],attempt['objective']
        if (not isinstance(metrics,dict) or set(metrics)!={'detours','revisit_connector_m'}
                or type(metrics['detours']) is not int or metrics['detours']<0
                or isinstance(metrics['revisit_connector_m'],bool) or not isinstance(metrics['revisit_connector_m'],(int,float))
                or not math.isfinite(metrics['revisit_connector_m']) or metrics['revisit_connector_m']<0
                or not isinstance(objective,list) or len(objective)!=2 or type(objective[0]) is not int or objective[0]<0
                or isinstance(objective[1],bool) or not isinstance(objective[1],(int,float))
                or not math.isfinite(objective[1]) or objective[1]<0):
            raise ValueError('接缝排序尝试指标无效')
        accepted=(objective[0]<=original[2][0] and objective[1]<original[2][1]-1e-12
                  and metrics['detours']<=baseline['detours']
                  and metrics['revisit_connector_m']<=baseline['revisit_connector_m']+1e-12)
        if accepted!=attempt['accepted']:raise ValueError('接缝排序尝试判断无效')
    if not record['adopted']:
        if (record.get('reason') not in (('segment_budget','pair_budget','no_seams','no_verified_seams',
                                         'no_nonworse_candidate','geometry_budget')+ (('search_budget',) if allow_search_budget else ()))
                or any(a['accepted'] for a in attempts) or final_links!=initial_links):
            raise ValueError('未采用的接缝排序改变了路线')
        return visits
    actual=checked_visits(segments,record.get('visits'));selected=reconstruct(segments,actual,final_links,check)
    metrics=_metrics(selected[0],spacing_m,check)
    if (record.get('baseline_objective')!=list(original[2]) or record.get('selected_objective')!=list(selected[2])
            or record.get('selected')!=metrics or selected[2][0]>original[2][0] or selected[2][1]>=original[2][1]-1e-12
            or metrics['detours']>baseline['detours'] or metrics['revisit_connector_m']>baseline['revisit_connector_m']+1e-12
            or not any(a['accepted'] and a['metrics']==metrics and a['objective']==list(selected[2]) for a in attempts)):
        raise ValueError('接缝排序未保持连接与重访约束')
    return actual
