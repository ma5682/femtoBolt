"""Opt-in native complete-radius queries with preflight memory bounds.

The cell index counts a conservative superset before native allocation. Batches
cannot contain more than max_pairs candidates; a single query exceeding that
limit uses the original one-query path. Nothing truncates the native neighbors.
"""
import itertools
import numpy as np
import open3d as o3d
from femto.geometry.cloud_csr_research import CellRadiusCounter
from femto.performance_detail import observed,count


class NativeRadiusCounter(CellRadiusCounter):
    @observed('cloud.native_complete_prepare')
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.failed=False
        if not self.enabled:return
        del self.order,self.starts
        try:
            self.search=o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(self.points,dtype=o3d.core.Dtype.Float64),index_dtype=o3d.core.Dtype.Int64)
            self.enabled=bool(self.search.fixed_radius_index())
        except (RuntimeError,TypeError,AttributeError):self.enabled=False

    @observed('cloud.native_complete_preflight')
    def _upper_counts(self,indices):
        from femto.geometry.cloud_layer_consistency import _CANDIDATE_RADIUS,_ROUNDING_FACTOR
        p=self.points[indices]
        radius=np.nextafter(_CANDIDATE_RADIUS*(1+_ROUNDING_FACTOR),np.inf)
        low=np.floor(np.nextafter(p-radius,-np.inf)/.01).astype(np.int64)
        high=np.floor(np.nextafter(p+radius,np.inf)/.01).astype(np.int64)
        shape=high-low+1;base=(low-self.origin)@self.stride
        sizes=np.zeros(len(indices),np.int64)
        for offset in itertools.product(*(range(int(v)) for v in shape.max(axis=0))):
            self.check()
            eligible=np.flatnonzero((shape>offset).all(axis=1))
            target=base[eligible]+np.asarray(offset)@self.stride
            found=np.searchsorted(self.keys,target);safe=np.minimum(found,len(self.keys)-1)
            present=(found<len(self.keys))&(self.keys[safe]==target)
            sizes[eligible[present]]+=self.sizes[safe[present]]
        return sizes

    @observed('cloud.native_complete_queries')
    def __call__(self,indices):
        from femto.geometry.cloud_layer_consistency import _CANDIDATE_RADIUS,_RADIUS_SQUARED,_RADIUS_GUARD,_HEIGHT,_ROUNDING_FACTOR
        if not self.enabled or self.failed:return self.legacy(indices)
        indices=np.asarray(indices,np.int64)
        if not len(indices):return 0
        counts=self._upper_counts(indices);prefix=np.r_[0,np.cumsum(counts)]
        begin=0;total=0
        while begin<len(indices):
            self.check()
            end=int(np.searchsorted(prefix,prefix[begin]+self.max_pairs,side='right'))-1
            if end<=begin:
                total+=self.legacy(indices[begin:begin+1]);begin+=1
                count('cloud.native_complete_work',oversized_queries=1)
                continue
            group=indices[begin:end]
            try:
                neighbor_ids,_,splits=self.search.fixed_radius_search(o3d.core.Tensor(self.points[group],dtype=o3d.core.Dtype.Float64),radius=_CANDIDATE_RADIUS,sort=False)
            except (RuntimeError,TypeError):
                self.failed=True
                return total+self.legacy(indices[begin:])
            ids=neighbor_ids.numpy();splits=splits.numpy();sizes=np.diff(splits)
            if len(ids)>self.max_pairs:
                raise ValueError('Complete radius candidate bound violated')
            repeated=np.repeat(np.arange(len(group)),sizes)
            delta=self.points[ids]-self.points[group[repeated]]
            squared=(delta[:,0]*delta[:,0]+delta[:,1]*delta[:,1])+delta[:,2]*delta[:,2]
            with np.errstate(over='ignore',invalid='ignore'):
                products=delta*self.normals[group[repeated]]
                heights=np.abs((products[:,0]+products[:,1])+products[:,2])
                guard=_ROUNDING_FACTOR*np.maximum(np.abs(products).sum(axis=1),_HEIGHT)
            inside=squared<_RADIUS_SQUARED-_RADIUS_GUARD
            finite=np.isfinite(heights)&np.isfinite(guard)
            good=inside&finite&(heights>_HEIGHT+guard)
            unclear=(np.abs(squared-_RADIUS_SQUARED)<=_RADIUS_GUARD)|(inside&(~finite|(np.abs(heights-_HEIGHT)<=guard)))
            positive=np.zeros(len(group),bool);ambiguous=sizes==0;nonempty=sizes>0
            if len(ids):
                positive[nonempty]=np.logical_or.reduceat(good,splits[:-1][nonempty])
                ambiguous[nonempty]=np.logical_or.reduceat(unclear,splits[:-1][nonempty])
            fallback=group[ambiguous&~positive]
            total+=int(positive.sum())+(self.legacy(fallback) if len(fallback) else 0)
            count('cloud.native_complete_work',query_batches=1,queries=len(group),neighbors=len(ids),ambiguous=len(fallback))
            begin=end
        return total
