"""Opt-in complete cell-neighborhood experiment, never a truncated kNN test.

The production entry point does not import this module. Candidate windows are
rounded outwards before the same monotone floating division used for binning.
Every occupied bin in that window is visited; ambiguous original-radius or
height decisions go to the supplied original query. Pair buffers are bounded.
"""
import itertools
import numpy as np
from femto.performance_detail import observed, count


class CellRadiusCounter:
    @observed('cloud.csr_prepare')
    def __init__(self, points, normals, legacy, *, check=lambda:None, max_pairs=65536):
        self.points,self.normals,self.legacy=points,normals,legacy
        self.check,self.max_pairs=check,max(1,int(max_pairs))
        self.enabled=False
        if (points.shape!=normals.shape or points.ndim!=2 or points.shape[1]!=3
                or not len(points) or not np.isfinite(points).all()
                or not np.isfinite(normals).all() or np.abs(points).max()>2**20):return
        self.check()
        cells=np.floor(points/.01).astype(np.int64)
        self.origin=cells.min(axis=0)-4
        span=cells.max(axis=0)-self.origin+5
        if np.prod(span.astype(object))>2**61:return
        self.stride=np.array([int(span[1])*int(span[2]),int(span[2]),1],np.int64)
        keys=(cells-self.origin)@self.stride
        self.order=np.argsort(keys,kind='stable')
        self.keys,self.starts,self.sizes=np.unique(keys[self.order],return_index=True,return_counts=True)
        self.enabled=True

    def _pairs(self, query_ids, first, sizes, positive, ambiguous):
        from femto.geometry.cloud_layer_consistency import _RADIUS_SQUARED,_RADIUS_GUARD,_HEIGHT,_ROUNDING_FACTOR
        prefix=np.r_[0,np.cumsum(sizes)]
        repeated=np.repeat(query_ids,sizes)
        source=first.repeat(sizes)+np.arange(prefix[-1])-np.repeat(prefix[:-1],sizes)
        delta=self.points[self.order[source]]-self.points[self._indices[repeated]]
        squared=(delta[:,0]*delta[:,0]+delta[:,1]*delta[:,1])+delta[:,2]*delta[:,2]
        with np.errstate(over='ignore',invalid='ignore'):
            products=delta*self.normals[self._indices[repeated]]
            heights=np.abs((products[:,0]+products[:,1])+products[:,2])
            guard=_ROUNDING_FACTOR*np.maximum(np.abs(products).sum(axis=1),_HEIGHT)
        radius_edge=np.abs(squared-_RADIUS_SQUARED)<=_RADIUS_GUARD
        inside=squared<_RADIUS_SQUARED-_RADIUS_GUARD
        finite=np.isfinite(heights)&np.isfinite(guard)
        good=inside&finite&(heights>_HEIGHT+guard)
        unclear=radius_edge|(inside&(~finite|(np.abs(heights-_HEIGHT)<=guard)))
        positive[query_ids] |= np.logical_or.reduceat(good,prefix[:-1])
        ambiguous[query_ids] |= np.logical_or.reduceat(unclear,prefix[:-1])
        count('cloud.csr_work',candidate_pairs=len(repeated),pair_batches=1)

    @observed('cloud.csr_queries')
    def __call__(self, indices):
        from femto.geometry.cloud_layer_consistency import _CANDIDATE_RADIUS
        if not self.enabled:return self.legacy(indices)
        self._indices=np.asarray(indices,dtype=np.int64)
        if not len(self._indices):return 0
        p=self.points[self._indices]
        low=np.floor(np.nextafter(p-_CANDIDATE_RADIUS,-np.inf)/.01).astype(np.int64)
        high=np.floor(np.nextafter(p+_CANDIDATE_RADIUS,np.inf)/.01).astype(np.int64)
        shape=high-low+1
        positive=np.zeros(len(indices),bool);ambiguous=positive.copy()
        base=(low-self.origin)@self.stride
        for offset in itertools.product(*(range(int(v)) for v in shape.max(axis=0))):
            self.check()
            eligible=np.flatnonzero(~positive&(shape>offset).all(axis=1))
            target=base[eligible]+np.asarray(offset)@self.stride
            found=np.searchsorted(self.keys,target)
            safe=np.minimum(found,len(self.keys)-1)
            present=(found<len(self.keys))&(self.keys[safe]==target)
            query_ids=eligible[present];bins=safe[present]
            first=self.starts[bins];sizes=self.sizes[bins]
            begin=0
            prefix=np.r_[0,np.cumsum(sizes)]
            while begin<len(query_ids):
                self.check()
                end=int(np.searchsorted(prefix,prefix[begin]+self.max_pairs,side='right'))-1
                if end>begin:
                    self._pairs(query_ids[begin:end],first[begin:end],sizes[begin:end],positive,ambiguous)
                    begin=end
                else:
                    # A single occupied bin can exceed the whole buffer budget.
                    for position in range(0,int(sizes[begin]),self.max_pairs):
                        self.check()
                        take=min(self.max_pairs,int(sizes[begin])-position)
                        self._pairs(query_ids[begin:begin+1],first[begin:begin+1]+position,np.array([take]),positive,ambiguous)
                        if positive[query_ids[begin]]:break
                    begin+=1
        fallback=self._indices[ambiguous&~positive]
        count('cloud.csr_work',queries=len(indices),positive=int(positive.sum()),ambiguous=len(fallback))
        return int(positive.sum())+(self.legacy(fallback) if len(fallback) else 0)
