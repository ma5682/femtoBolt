"""Bounded row proposals preserving eligible prior slice levels.

The minimum transverse metric is integrated only over each face's own
projected interval. Old levels are preferred when they meet the current bound;
the choice itself proves neither selected-area coverage nor robot safety.
"""
from __future__ import annotations

import heapq
import numpy as np

VERSION='local-metric-prior-level-alignment-v1'
MIN_USEFUL_FRACTION=.5
MAX_ROWS=2000
SHIFT_VERSION='local-metric-bounded-phase-shift-v1'
SHIFT_WIDTH_FRACTION=.05


def schedule(first,last,intervals,factors,check=lambda:None):
    intervals=np.asarray(intervals,dtype=float);factors=np.asarray(factors,dtype=float)
    if (not np.isfinite([first,last]).all() or last<first or intervals.shape!=(len(factors),2)
            or not np.isfinite(intervals).all() or not np.isfinite(factors).all()
            or np.any(factors<0) or np.any(factors>1) or np.any(intervals[:,1]<intervals[:,0])):
        raise ValueError('局部表面尺度或区间无效')
    events={}
    for i,((lo,hi),factor) in enumerate(zip(intervals,factors)):
        if i%128==0:check()
        lo,hi=max(first,float(lo)),min(last,float(hi))
        if hi<=lo:continue
        if factor<=1e-8:raise ValueError('局部表面需要另一扫描方向')
        events.setdefault(lo,[]).append((True,i));events.setdefault(hi,[]).append((False,i))
    levels=sorted({first,last,*events});active=set();queue=[];cumulative=[0.]
    for left,right in zip(levels,levels[1:]):
        check()
        for entering,index in events.get(left,()):
            if entering:active.add(index);heapq.heappush(queue,(factors[index],index))
            else:active.discard(index)
        while queue and queue[0][1] not in active:heapq.heappop(queue)
        if not queue:raise ValueError('局部表面横向范围存在无表面区间')
        cumulative.append(cumulative[-1]+(right-left)/queue[0][0])
    return np.asarray(levels),np.asarray(cumulative)


def positions(first,last,intervals,factors,maximum,existing,check=lambda:None):
    """Greedily choose prior levels under the integrated step bound.

    Original float values are copied back exactly so compatible scan anchors
    may be recognized. New boundary levels are explicit proposals and undergo
    all normal quality, width, reference and connectivity checks downstream.
    """
    if not np.isfinite(maximum) or maximum<=0:raise ValueError('局部表面行距上限无效')
    levels,cumulative=schedule(first,last,intervals,factors,check)
    if last==first:return np.asarray([first])
    old=np.asarray(existing,dtype=float)
    if old.ndim!=1 or not np.isfinite(old).all():raise ValueError('旧扫描位置无效')
    prior=sorted(set(float(v) for v in old if first<v<last))
    distances=np.interp(prior,levels,cumulative)
    candidates=sorted(set([0.,*distances,float(cumulative[-1])]))
    choices=[0.];end=float(cumulative[-1])
    while end-choices[-1]>1e-12:
        check()
        if len(choices)>=MAX_ROWS:raise ValueError('局部表面排布超过扫描行预算')
        feasible=[x for x in candidates if choices[-1]+1e-12<x<=choices[-1]+maximum+1e-12]
        next_value=max(feasible) if feasible else 0.
        if next_value-choices[-1]<maximum*MIN_USEFUL_FRACTION and end-choices[-1]>maximum:
            next_value=choices[-1]+maximum
        if next_value<=choices[-1]+1e-12:next_value=min(end,choices[-1]+maximum)
        choices.append(next_value)
    if len(choices)>2 and choices[-1]-choices[-3]<=maximum+1e-12:choices.pop(-2)
    originals=dict(zip(distances,prior))
    return np.asarray([originals.get(value,float(np.interp(value,cumulative,levels))) for value in choices])


def shifted(positions_m,index,shift_m,intervals,factors,maximum,check=lambda:None):
    """Move one interior row and rebalance neighbors without changing bounds.

    This generates a proposal only. The caller must re-slice the measured mesh
    and independently validate selection, width, connectivity and coverage.
    Endpoints stay fixed. If no valid rebalance exists the proposal is rejected.
    """
    values=np.asarray(positions_m,dtype=float)
    if (values.ndim!=1 or len(values)<3 or not np.isfinite(values).all()
            or np.any(np.diff(values)<=0) or type(index) is not int or not 0<index<len(values)-1
            or not np.isfinite(shift_m) or shift_m==0 or not np.isfinite(maximum) or maximum<=0):
        raise ValueError('局部行位置调整输入无效')
    target=values[index]+shift_m
    if not values[index-1]<target<values[index+1]:raise ValueError('局部行位置调整越过相邻行')
    def distance(first,last):
        return float(schedule(float(first),float(last),intervals,factors,check)[1][-1])
    if any(distance(a,b)>maximum+1e-12 for a,b in zip(values,values[1:])):
        raise ValueError('原排布已超过局部尺度行距上限')
    output=values.copy();output[index]=target
    if shift_m>0:
        for i in range(index-1,0,-1):
            check()
            if distance(output[i],output[i+1])<=maximum+1e-12:break
            left,right=output[i],output[i+1]
            for _ in range(40):
                middle=(left+right)/2
                if distance(middle,output[i+1])>maximum:left=middle
                else:right=middle
            output[i]=right
    else:
        for i in range(index+1,len(output)-1):
            check()
            if distance(output[i-1],output[i])<=maximum+1e-12:break
            left,right=output[i-1],output[i]
            for _ in range(40):
                middle=(left+right)/2
                if distance(output[i-1],middle)>maximum:right=middle
                else:left=middle
            output[i]=left
    if np.any(np.diff(output)<=0) or any(distance(a,b)>maximum+1e-12 for a,b in zip(output,output[1:])):
        raise ValueError('局部行位置调整无法保持边界和尺度行距')
    return output
