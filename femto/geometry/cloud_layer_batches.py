"""Exact dense-cloud diagnostics using cell bounds and observed witnesses.

Cells can prove absence of a height witness; sparse *original* points can prove
its presence. Unresolved queries use bounded complete native batches, with the
original one-point query at numerical boundaries or unsupported conditions.
"""
import itertools
import numpy as np
import open3d as o3d
from femto.performance_detail import observed, timed, count


_DEFAULT_ERROR = dict(divide='warn', over='warn', under='ignore', invalid='warn')


@observed('cloud.negative_envelope')
def negative_bounds(points,normals,*,oriented=False,check=lambda:None):
    """Refine only unresolved cell envelopes; resource failure retains old bounds."""
    if np.geterr()!=_DEFAULT_ERROR:
        return _original_negative_bounds(points,normals,oriented=oriented,check=check)
    try:
        return _subbox_negative_bounds(points,normals,oriented=oriented,check=check)
    except MemoryError:
        return _original_negative_bounds(points,normals,oriented=oriented,check=check)


def _subbox_negative_bounds(points,normals,*,oriented=False,check=lambda:None):
    from femto.geometry.cloud_layer_consistency import _CANDIDATE_RADIUS,_HEIGHT,_ROUNDING_FACTOR
    # The caller bounds finite coordinates by 2**20 m. This cell width exceeds
    # the widened radius by ~1e-6 m, well above binning rounding at that bound;
    # every possible neighbor remains in the same or an adjacent cell.
    # Custom floating-error policies retain the previous arithmetic.
    cell=.010001
    cells=np.floor(points/cell).astype(np.int64)
    start=cells.min(axis=0)-1;span=cells.max(axis=0)-start+2
    if np.prod(span.astype(object))>2**61:return np.zeros(len(points),bool)
    stride=np.array([int(span[1])*int(span[2]),int(span[2]),1],np.int64)
    keys=(cells-start)@stride;unique,inverse=np.unique(keys,return_inverse=True)
    low=np.full((len(unique),3),np.inf);high=np.full((len(unique),3),-np.inf)
    np.minimum.at(low,inverse,points);np.maximum.at(high,inverse,points)
    # Membership partitions actual observed points; empty subboxes stay empty.
    check()
    midpoint=low/2+high/2
    bits=(points>=midpoint[inverse]).astype(np.int64)
    sub=inverse*8+bits@np.array([4,2,1])
    sub_low=np.full((len(unique)*8,3),np.inf);sub_high=np.full_like(sub_low,-np.inf)
    np.minimum.at(sub_low,sub,points);np.maximum.at(sub_high,sub,points)
    sub_low=sub_low.reshape(-1,8,3);sub_high=sub_high.reshape(-1,8,3)
    check()
    envelope=None
    if oriented:
        from femto.geometry.cloud_oriented_bounds import OrientedBounds
        with timed('cloud.oriented_prepare'):
            envelope=OrientedBounds(points,normals,inverse,low,high,check)
    result=np.ones(len(points),bool)
    from femto.geometry.cloud_radius_kernel import negative_boxes
    offsets=np.array(list(itertools.product((-1,0,1),repeat=3)),np.int64)@stride
    for begin in range(0,len(points),4096):
        check()
        end=min(begin+4096,len(points));p=points[begin:end];n=normals[begin:end]
        native=None if oriented else negative_boxes(p,n,keys[begin:end],unique,offsets,low,high,sub_low,sub_high)
        if native is not None:
            result[begin:end]=native
            continue
        window_low=np.nextafter(p-_CANDIDATE_RADIUS,-np.inf)
        window_high=np.nextafter(p+_CANDIDATE_RADIUS,np.inf)
        for offset in itertools.product((-1,0,1),repeat=3):
            target=keys[begin:end]+np.asarray(offset)@stride
            ids=np.searchsorted(unique,target);safe=np.minimum(ids,len(unique)-1)
            present=(ids<len(unique))&(unique[safe]==target)
            a=np.maximum(low[safe],window_low);b=np.minimum(high[safe],window_high)
            present &= (a<=b).all(axis=1)
            with np.errstate(over='ignore',invalid='ignore'):
                left=(a-p)*n;right=(b-p)*n
                maximum=np.maximum(np.abs(np.minimum(left,right).sum(axis=1)),np.abs(np.maximum(left,right).sum(axis=1)))
                guard=_ROUNDING_FACTOR*np.maximum(((np.maximum(np.abs(a),np.abs(b))+np.abs(p))*np.abs(n)).sum(axis=1),_HEIGHT)
            possible=~np.isfinite(maximum)|~np.isfinite(guard)|(maximum+guard>_HEIGHT)
            candidates=np.flatnonzero(present&possible&result[begin:end])
            if len(candidates):
                pp=p[candidates,None,:];nn=n[candidates,None,:]
                aa=np.maximum(sub_low[safe[candidates]],window_low[candidates,None,:])
                bb=np.minimum(sub_high[safe[candidates]],window_high[candidates,None,:])
                occupied=(aa<=bb).all(axis=2)
                with np.errstate(over='ignore',invalid='ignore'):
                    ll=(aa-pp)*nn;rr=(bb-pp)*nn
                    mm=np.maximum(np.abs(np.minimum(ll,rr).sum(axis=2)),np.abs(np.maximum(ll,rr).sum(axis=2)))
                    gg=_ROUNDING_FACTOR*np.maximum(((np.maximum(np.abs(aa),np.abs(bb))+np.abs(pp))*np.abs(nn)).sum(axis=2),_HEIGHT)
                uncertain=~np.isfinite(mm)|~np.isfinite(gg)|(mm+gg>_HEIGHT)
                possible[candidates]=(occupied&uncertain).any(axis=1)
            if envelope is not None:
                candidates=np.flatnonzero(present&possible&result[begin:end])
                if len(candidates):
                    certified=envelope.negative(p[candidates],n[candidates],safe[candidates],_HEIGHT)
                    possible[candidates[certified]]=False
            result[begin:end] &= ~(present&possible)
    return result


def _complete_radius_counter(points,normals,legacy,check):
    from femto.geometry.cloud_radius_batches import NativeRadiusCounter
    try:return NativeRadiusCounter(points,normals,legacy,check=check)
    except MemoryError:return legacy


class _SecondaryWitness:
    """A capped query can only prove a positive using an original observed point."""

    @observed('cloud.secondary_witness_prepare')
    def __init__(self,points,normals,*,check=lambda:None):
        self.points,self.normals=points,normals
        self.search=None;self.observed=None
        self.enabled=np.geterr()==_DEFAULT_ERROR
        self.failed=False
        if not self.enabled:return
        check()
        try:
            _,ids=np.unique(np.floor(points/.0035).astype(np.int64),axis=0,return_index=True)
            self.observed=points[ids]
            self.search=o3d.core.nns.NearestNeighborSearch(
                o3d.core.Tensor(self.observed,dtype=o3d.core.Dtype.Float64),
                index_dtype=o3d.core.Dtype.Int64)
            self.enabled=bool(self.search.hybrid_index())
        except (RuntimeError,TypeError,AttributeError,MemoryError):
            self.enabled=False
        if self.enabled:
            count('cloud.secondary_witness_work',representatives=len(self.observed))
        else:
            self.search=None;self.observed=None

    @observed('cloud.secondary_witness_queries')
    def query(self,indices):
        from femto.geometry.cloud_layer_consistency import _CANDIDATE_RADIUS,_RADIUS_SQUARED,_RADIUS_GUARD,_ROUNDING_FACTOR,_HEIGHT
        indices=np.asarray(indices,dtype=np.int64)
        unknown=np.zeros(len(indices),bool)
        if not self.enabled or self.failed or not len(indices) or np.geterr()!=_DEFAULT_ERROR:return unknown
        try:
            ids,_,counts=self.search.hybrid_search(
                o3d.core.Tensor(self.points[indices],dtype=o3d.core.Dtype.Float64),
                radius=_CANDIDATE_RADIUS,max_knn=64)
            ids,counts=ids.numpy(),counts.numpy()
            valid=np.arange(64)[None,:]<counts[:,None]
            delta=self.observed[np.maximum(ids,0)]-self.points[indices,None,:]
            squared=(delta[...,0]*delta[...,0]+delta[...,1]*delta[...,1])+delta[...,2]*delta[...,2]
            with np.errstate(over='ignore',invalid='ignore'):
                products=delta*self.normals[indices,None,:]
                heights=np.abs((products[...,0]+products[...,1])+products[...,2])
                guard=_ROUNDING_FACTOR*np.maximum(np.abs(products).sum(axis=2),_HEIGHT)
            certified=(valid&(squared<_RADIUS_SQUARED-_RADIUS_GUARD)&np.isfinite(heights)&np.isfinite(guard)
                       &(heights>_HEIGHT+guard)).any(axis=1)
        except (RuntimeError,TypeError,AttributeError,MemoryError):
            # Failure or truncation never certifies absence. The caller keeps
            # every unresolved query on the complete-radius/legacy path.
            self.failed=True;self.search=None;self.observed=None
            return unknown
        count('cloud.secondary_witness_work',queries=len(indices),proven_positive=int(certified.sum()))
        return certified


def count_dense(points,normals,legacy,*,check=lambda:None):
    from femto.geometry.cloud_layer_consistency import _CANDIDATE_RADIUS,_RADIUS_SQUARED,_RADIUS_GUARD,_ROUNDING_FACTOR,_HEIGHT
    negative=negative_bounds(points,normals,check=check)
    todo=np.flatnonzero(~negative)
    count('cloud.classifications',input_points=len(points),proven_negative=int(negative.sum()))
    if not len(todo):return 0
    with timed('cloud.witness_representatives'):
        _,representatives=np.unique(np.floor(points/.002).astype(np.int64),axis=0,return_index=True)
    observed=points[representatives]
    try:
        tensor=o3d.core.Tensor(observed,dtype=o3d.core.Dtype.Float64)
        search=o3d.core.nns.NearestNeighborSearch(tensor,index_dtype=o3d.core.Dtype.Int64)
        if not search.hybrid_index():return legacy(todo)
    except (RuntimeError,TypeError,AttributeError):return legacy(todo)
    total=0;complete=None;secondary=None
    for start in range(0,len(todo),512):
        check()
        group=todo[start:start+512]
        try:
            with timed('cloud.witness_queries'):
                ids,_,counts=search.hybrid_search(o3d.core.Tensor(points[group],dtype=o3d.core.Dtype.Float64),
                    radius=_CANDIDATE_RADIUS,max_knn=64)
        except (RuntimeError,TypeError):return total+legacy(todo[start:])
        ids,counts=ids.numpy(),counts.numpy()
        valid=np.arange(64)[None,:]<counts[:,None]
        delta=observed[np.maximum(ids,0)]-points[group,None,:]
        squared=(delta[...,0]*delta[...,0]+delta[...,1]*delta[...,1])+delta[...,2]*delta[...,2]
        with np.errstate(over='ignore',invalid='ignore'):
            products=delta*normals[group,None,:]
            heights=np.abs((products[...,0]+products[...,1])+products[...,2])
            guard=_ROUNDING_FACTOR*np.maximum(np.abs(products).sum(axis=2),_HEIGHT)
        witness=(valid&(squared<_RADIUS_SQUARED-_RADIUS_GUARD)&np.isfinite(heights)&np.isfinite(guard)
                 &(heights>_HEIGHT+guard)).any(axis=1)
        if not witness.all():
            if secondary is None:secondary=_SecondaryWitness(points,normals,check=check)
            unresolved=np.flatnonzero(~witness)
            witness[unresolved] |= secondary.query(group[unresolved])
        total+=int(witness.sum())
        count('cloud.classifications',proven_positive=int(witness.sum()))
        if not witness.all():
            if complete is None:complete=_complete_radius_counter(points,normals,legacy,check)
            total+=complete(group[~witness])
    return total


@observed('cloud.negative_envelope')
def _original_negative_bounds(points,normals,*,oriented=False,check=lambda:None):
    from femto.geometry.cloud_layer_consistency import _CANDIDATE_RADIUS,_HEIGHT,_ROUNDING_FACTOR
    # The caller bounds finite coordinates by 2**20 m. This cell width exceeds
    # the widened radius by ~1e-6 m, well above binning rounding at that bound;
    # every possible neighbor remains in the same or an adjacent cell.
    # Custom floating-error policies retain the previous arithmetic.
    cell=.010001 if np.geterr()==_DEFAULT_ERROR else .02
    cells=np.floor(points/cell).astype(np.int64)
    start=cells.min(axis=0)-1;span=cells.max(axis=0)-start+2
    if np.prod(span.astype(object))>2**61:return np.zeros(len(points),bool)
    stride=np.array([int(span[1])*int(span[2]),int(span[2]),1],np.int64)
    keys=(cells-start)@stride;unique,inverse=np.unique(keys,return_inverse=True)
    low=np.full((len(unique),3),np.inf);high=np.full((len(unique),3),-np.inf)
    np.minimum.at(low,inverse,points);np.maximum.at(high,inverse,points)
    envelope=None
    if oriented:
        from femto.geometry.cloud_oriented_bounds import OrientedBounds
        with timed('cloud.oriented_prepare'):
            envelope=OrientedBounds(points,normals,inverse,low,high,check)
    result=np.ones(len(points),bool)
    for begin in range(0,len(points),4096):
        check()
        end=min(begin+4096,len(points));p=points[begin:end];n=normals[begin:end]
        window_low=np.nextafter(p-_CANDIDATE_RADIUS,-np.inf)
        window_high=np.nextafter(p+_CANDIDATE_RADIUS,np.inf)
        for offset in itertools.product((-1,0,1),repeat=3):
            target=keys[begin:end]+np.asarray(offset)@stride
            ids=np.searchsorted(unique,target);safe=np.minimum(ids,len(unique)-1)
            present=(ids<len(unique))&(unique[safe]==target)
            a=np.maximum(low[safe],window_low);b=np.minimum(high[safe],window_high)
            present &= (a<=b).all(axis=1)
            with np.errstate(over='ignore',invalid='ignore'):
                left=(a-p)*n;right=(b-p)*n
                maximum=np.maximum(np.abs(np.minimum(left,right).sum(axis=1)),np.abs(np.maximum(left,right).sum(axis=1)))
                guard=_ROUNDING_FACTOR*np.maximum(((np.maximum(np.abs(a),np.abs(b))+np.abs(p))*np.abs(n)).sum(axis=1),_HEIGHT)
            possible=~np.isfinite(maximum)|~np.isfinite(guard)|(maximum+guard>_HEIGHT)
            if envelope is not None:
                candidates=np.flatnonzero(present&possible&result[begin:end])
                if len(candidates):
                    certified=envelope.negative(p[candidates],n[candidates],safe[candidates],_HEIGHT)
                    possible[candidates[certified]]=False
            result[begin:end] &= ~(present&possible)
    return result
