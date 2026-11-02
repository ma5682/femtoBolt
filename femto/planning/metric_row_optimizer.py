"""Bounded row-placement proposal with a global prior-level choice.

This chooses levels only. It does not alter a route, relax width/geometry
predicates, establish target coverage, or supply a robot motion command.
"""
from __future__ import annotations

import math
import numpy as np

from femto.planning.local_metric_alignment import schedule

VERSION='minimum-metric-rows-then-prior-levels-v1'
SLACK_VERSION='bounded-extra-metric-rows-prior-levels-v1'
FLOOR_VERSION='bounded-extra-metric-rows-spacing-floor-v2'
MAX_ROWS=2000
MAX_PAIR_CHECKS=250_000
MAX_TRANSITIONS=500_000
MIN_FRACTION=.5


def positions(first,last,intervals,factors,maximum,existing,check=lambda:None):
    """Minimize row count, then maximize retained old levels, on a metric axis.

    New intermediate rows are uniformly inserted in integrated metric space.
    If the bounded search is too large, reject rather than silently removing
    prior levels or relaxing the caller's spacing bound.
    """
    if (isinstance(maximum,bool) or not isinstance(maximum,(int,float))
            or not math.isfinite(maximum) or maximum<=0):
        raise ValueError('局部优化行距上限无效')
    old=np.asarray(existing,dtype=float)
    if old.ndim!=1 or not np.isfinite(old).all():raise ValueError('局部优化旧行位置无效')
    levels,cumulative=schedule(first,last,intervals,factors,check)
    if first==last:
        return np.asarray([first]),dict(version=VERSION,row_count=1,retained_prior_levels=int(first in old),pair_checks=0)
    prior=sorted(set(float(v) for v in old if first<v<last))
    by_distance={float(d):value for d,value in zip(np.interp(prior,levels,cumulative),prior)}
    end=float(cumulative[-1]);nodes=[0.,*sorted(x for x in by_distance if 0<x<end),end]
    if len(nodes)>MAX_ROWS+1:raise ValueError('局部优化候选行超过预算')
    if len(nodes)*(len(nodes)-1)//2>MAX_PAIR_CHECKS:raise ValueError('局部优化候选比较超过预算')
    # Cost is steps, negative retained levels, then deviation from nominal step.
    best=[None]*len(nodes);best[0]=(0,0,0.);parents=[None]*len(nodes);pair_checks=0
    for j in range(1,len(nodes)):
        for i in range(j):
            if pair_checks%128==0:check()
            pair_checks+=1
            if best[i] is None:continue
            distance=nodes[j]-nodes[i]
            steps=max(1,math.ceil((distance-1e-12)/maximum))
            while distance/steps>maximum+1e-12:steps+=1
            spacing=distance/steps
            if spacing<maximum*MIN_FRACTION-1e-12 and not (i==0 and j==len(nodes)-1):continue
            old_score=0 if j==len(nodes)-1 else 1
            proposed=(best[i][0]+steps,best[i][1]-old_score,
                      best[i][2]+steps*(maximum-spacing)**2)
            if best[j] is None or proposed<best[j]:best[j]=proposed;parents[j]=(i,steps)
    if best[-1] is None or best[-1][0]+1>MAX_ROWS:
        raise ValueError('局部优化无法在预算内保持尺度行距')
    edges=[];j=len(nodes)-1
    while j:
        i,steps=parents[j];edges.append((i,j,steps));j=i
    distances=[0.]
    for i,j,steps in reversed(edges):
        distances.extend(float(nodes[i]+(nodes[j]-nodes[i])*k/steps) for k in range(1,steps))
        distances.append(nodes[j])
    distances[-1]=end
    result=np.asarray([by_distance.get(x,float(np.interp(x,cumulative,levels))) for x in distances])
    # Keep caller boundary floats and retained prior floats bit-exact.
    result[0]=first;result[-1]=last
    actual=np.diff(np.interp(result,levels,cumulative))
    if np.any(actual>maximum+2e-12) or np.any(actual<=0):
        raise ValueError('局部优化结果未保持尺度行距')
    return result,dict(version=VERSION,row_count=len(result),retained_prior_levels=sum(float(v) in prior for v in result),
                       metric_step_min_m=float(actual.min()) if len(actual) else None,
                       metric_step_max_m=float(actual.max()) if len(actual) else None,pair_checks=pair_checks)


def positions_with_slack(first,last,intervals,factors,maximum,existing,check=lambda:None,*,extra_fraction=.1,minimum_fraction=None):
    """Prefer prior levels within a bounded increase over the minimum rows."""
    if (isinstance(maximum,bool) or not isinstance(maximum,(int,float)) or not math.isfinite(maximum) or maximum<=0
            or isinstance(extra_fraction,bool) or not isinstance(extra_fraction,(int,float))
            or not math.isfinite(extra_fraction) or not 0<=extra_fraction<=.5
            or (minimum_fraction is not None and (isinstance(minimum_fraction,bool)
                or not isinstance(minimum_fraction,(int,float)) or not math.isfinite(minimum_fraction)
                or not .5<=minimum_fraction<=1.))):
        raise ValueError('局部优化行距或额外行预算无效')
    old=np.asarray(existing,dtype=float)
    if old.ndim!=1 or not np.isfinite(old).all():raise ValueError('局部优化旧行位置无效')
    levels,cumulative=schedule(first,last,intervals,factors,check)
    if first==last:
        report=dict(version=SLACK_VERSION,row_count=1,retained_prior_levels=int(first in old),
                    extra_fraction=extra_fraction,pair_checks=0,transitions=0)
        if minimum_fraction is not None:
            report.update(version=FLOOR_VERSION,minimum_fraction=float(minimum_fraction),effective_minimum_metric_step_m=None)
        return np.asarray([first]),report
    prior=sorted(set(float(v) for v in old if first<v<last))
    by_distance={float(d):value for d,value in zip(np.interp(prior,levels,cumulative),prior)}
    end=float(cumulative[-1]);nodes=[0.,*sorted(x for x in by_distance if 0<x<end),end]
    if len(nodes)>MAX_ROWS+1:raise ValueError('局部优化候选行超过预算')
    if len(nodes)*(len(nodes)-1)//2>MAX_PAIR_CHECKS:raise ValueError('局部优化候选比较超过预算')
    minimum=max(1,math.ceil((end-1e-12)/maximum))
    floor=maximum*MIN_FRACTION if minimum_fraction is None else min(maximum*minimum_fraction,end/minimum)
    limit=minimum+math.ceil(minimum*extra_fraction)
    if limit+1>MAX_ROWS:raise ValueError('局部优化行数超过预算')
    # Each state records retained count, deviation and its predecessor.
    states=[{} for _ in nodes];states[0][0]=(0,0.,None)
    pair_checks=transitions=0
    for j in range(1,len(nodes)):
        suffix=max(0,math.ceil((end-nodes[j]-1e-12)/maximum))
        for i in range(j):
            if pair_checks%128==0:check()
            pair_checks+=1
            if not states[i]:continue
            distance=nodes[j]-nodes[i];steps=max(1,math.ceil((distance-1e-12)/maximum))
            while distance/steps>maximum+1e-12:steps+=1
            spacing=distance/steps
            if spacing<floor-1e-12 and not (minimum_fraction is None and i==0 and j==len(nodes)-1):continue
            added=0 if j==len(nodes)-1 else 1
            deviation=steps*(maximum-spacing)**2
            for used,(retained,cost,_) in states[i].items():
                transitions+=1
                if transitions%128==0:check()
                if transitions>MAX_TRANSITIONS:raise ValueError('局部优化状态转移超过预算')
                total=used+steps
                if total+suffix>limit:continue
                candidate=(retained+added,cost+deviation,(i,used,steps))
                current=states[j].get(total)
                if current is None or (-candidate[0],candidate[1])<(-current[0],current[1]):
                    states[j][total]=candidate
    if not states[-1]:raise ValueError('局部优化无法在预算内保持尺度行距')
    total,(_,_,_)=min(states[-1].items(),key=lambda item:(-item[1][0],item[0],item[1][1]))
    edges=[];j=len(nodes)-1;used=total
    while j:
        i,previous,steps=states[j][used][2];edges.append((i,j,steps));j,used=i,previous
    distances=[0.]
    for i,j,steps in reversed(edges):
        distances.extend(float(nodes[i]+(nodes[j]-nodes[i])*k/steps) for k in range(1,steps))
        distances.append(nodes[j])
    result=np.asarray([by_distance.get(x,float(np.interp(x,cumulative,levels))) for x in distances])
    result[0]=first;result[-1]=last
    actual=np.diff(np.interp(result,levels,cumulative))
    if np.any(actual>maximum+2e-12) or np.any(actual<=0):raise ValueError('局部优化结果未保持尺度行距')
    if minimum_fraction is not None and np.any(actual<floor-2e-12):raise ValueError('局部优化结果未保持尺度行距下限')
    report=dict(version=SLACK_VERSION,row_count=len(result),retained_prior_levels=sum(float(v) in prior for v in result),
        metric_step_min_m=float(actual.min()) if len(actual) else None,metric_step_max_m=float(actual.max()) if len(actual) else None,
        extra_fraction=float(extra_fraction),minimum_steps=minimum,maximum_steps=limit,pair_checks=pair_checks,transitions=transitions)
    if minimum_fraction is not None:
        report.update(version=FLOOR_VERSION,minimum_fraction=float(minimum_fraction),effective_minimum_metric_step_m=float(floor))
    return result,report
