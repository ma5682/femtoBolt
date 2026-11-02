"""Bounded arithmetic-only relocation queues; no geometry acceptance cache.

The legacy queue reproduces scalar distances, built-in sums and
tuple ordering. Only selected moves receive Python edge-set objects. A caller
must fall back to its retained scalar implementation for unsupported inputs.
"""
from __future__ import annotations

import math
import numpy as np

MAX_SEGMENTS=256
MAX_COORDINATE=1e100


def supported(segments,visits):
    if not 1<=len(segments)<=MAX_SEGMENTS or len(visits)!=len(segments):return False
    if any(type(s.block_index) is not int or not -(2**63)<=s.block_index<2**63 for s in segments):return False
    if any(type(i) is not int or not 0<=i<len(segments) or type(r) is not bool for i,r in visits):return False
    if len({i for i,_ in visits})!=len(segments):return False
    for segment in segments:
        if not segment.points:return False
        for point in (segment.points[0],segment.points[-1]):
            xyz=point.get('xyz_m')
            if (type(xyz) not in (list,tuple) or len(xyz)!=3
                    or any(type(v) is not float or not math.isfinite(v) or abs(v)>MAX_COORDINATE for v in xyz)):
                return False
    return True


def _distances(segments,check,cache):
    endpoints=np.asarray([[s.points[0]['xyz_m'],s.points[-1]['xyz_m']] for s in segments],dtype=np.float64)
    blocks=tuple(s.block_index for s in segments)
    signature=(blocks,endpoints.tobytes())
    if cache.get('batch_signature')==signature:return cache['batch_distances']
    n=len(segments);distance=np.zeros((2*n+1,2*n+1),dtype=np.float64)
    # Each entry uses exactly the retained scalar norm. Keeping the two
    # orientations explicit also preserves zero-length and same-block rules.
    for a in range(2*n):
        check();tail=endpoints[a//2,1-a%2]
        for b in range(2*n):
            if blocks[a//2]==blocks[b//2]:
                distance[a,b]=float(np.linalg.norm(tail-endpoints[b//2,b%2]))
    check();cache['batch_signature']=signature;cache['batch_distances']=distance
    return distance


def table(segments,visits,original,check,cache):
    """Return one bounded array per key, preserving the scalar legacy keys."""
    if not supported(segments,visits):raise ValueError('不支持批量移动候选')
    n=len(visits)
    if n<2:return None
    check();distance=_distances(segments,check,cache)
    encoded=np.asarray([2*i+int(r) for i,r in visits]+[2*n],dtype=np.int64)
    blocks=np.asarray([s.block_index for s in segments],dtype=np.int64)
    position_blocks=blocks[encoded[:-1]//2]
    source,destination,reverse=np.meshgrid(np.arange(1,n),np.arange(1,n),np.arange(2),indexing='ij')
    source=source.reshape(-1);destination=destination.reshape(-1);reverse=reverse.reshape(-1)
    before_pos=np.where(destination<=source,destination-1,destination)
    after_pos=np.where(destination<source,destination,destination+1)
    allowed=(position_blocks[source]==position_blocks[source-1]) & (position_blocks[before_pos]==position_blocks[source])
    allowed &= ~((destination==source)&(reverse==(encoded[source]%2)))
    source=source[allowed];destination=destination[allowed];reverse=reverse[allowed]
    before_pos=before_pos[allowed];after_pos=after_pos[allowed]
    left,value,right=encoded[source-1],encoded[source],encoded[source+1]
    before,after=encoded[before_pos],encoded[after_pos]
    moved=(value//2)*2+reverse
    baseline=sum(float(distance[a,b]) for a,b in zip(encoded[:-2],encoded[1:-1]))
    removal=(distance[left,right]-distance[left,value])-distance[value,right]
    estimate=(((baseline+removal)+distance[before,moved])+distance[moved,after])-distance[before,after]
    # Original removed edges are sorted by their left visit, just like the
    # retained sorted(set(...)) implementation. The sentinel contributes 0.
    removed_third=np.where(destination==source,n-1,before_pos)
    removed=np.stack((source-1,source,removed_third),axis=1)
    edge_left=np.r_[encoded[:-2],2*n]
    edge_order=np.argsort(edge_left[removed],axis=1,kind='stable')
    removed=np.take_along_axis(removed,edge_order,axis=1)
    metrics=np.asarray([original[(a,b)] for a,b in zip(visits,visits[1:])]+[(0,0.,0.)],dtype=np.float64)
    # CPython's float sum is not ordinary left-to-right array addition. Use
    # the same built-in on native floats to preserve ties on every runtime.
    triples=metrics[removed].transpose(0,2,1).reshape(-1,3).tolist()
    bad=np.asarray([sum(row) for row in triples],dtype=np.float64).reshape(-1,3)
    check()
    return dict(source=source,destination=destination,reverse=reverse,
                detours=-bad[:,0],revisit=-bad[:,1],length=-bad[:,2],estimate=estimate,
                left=left,right=right,before=before,after=after,moved=moved,removed=removed,
                distance=distance,encoded=encoded)


def legacy_queue(segments,visits,original,count,check,cache):
    """Exactly the old 7/8 distance + 1/8 bad-link queue, for supported inputs."""
    from femto.planning.budgeted_relocation_research import _changes
    data=table(segments,visits,original,check,cache)
    if data is None:return []
    i,j,r=data['source'],data['destination'],data['reverse']
    by_distance=np.lexsort((r,j,i,data['estimate']))
    by_bad=np.lexsort((r,j,i,data['estimate'],data['length'],data['revisit'],data['detours']))
    selected=[];seen=set();quota=(count*7)//8
    for index in np.concatenate((by_distance[:quota],by_bad,by_distance[quota:])):
        key=(int(i[index]),int(j[index]),bool(r[index]))
        if key in seen:continue
        seen.add(key);selected.append(int(index))
        if len(selected)>=count:break
    result=[]
    for index in selected:
        check();move=(int(i[index]),int(j[index]),bool(r[index]))
        removed,added=_changes(visits,*move)
        key=(int(data['detours'][index]),float(data['revisit'][index]),float(data['length'][index]),float(data['estimate'][index]),*move)
        result.append((key,removed,added))
    return result
