"""Native complete-radius queries with preflight memory bounds.

The cell index counts a conservative superset before native allocation. Batches
cannot contain more than max_pairs candidates; a single query exceeding that
limit uses the original one-query path. Nothing truncates the native neighbors.
"""
import itertools
from time import perf_counter
import numpy as np
import open3d as o3d
from femto.performance_detail import observed,count,current,_record


class NativeRadiusCounter:
    @observed('cloud.native_complete_prepare')
    def __init__(self,points,normals,legacy,*,check=lambda:None,max_pairs=65536):
        self.points,self.normals,self.legacy=points,normals,legacy
        self.check,self.max_pairs=check,max(1,int(max_pairs))
        self.failed=False;self.enabled=False
        # Custom NumPy error policies retain the original arithmetic/notifications.
        if np.geterr()!=dict(divide='warn',over='warn',under='ignore',invalid='warn'):return
        if (points.shape!=normals.shape or points.ndim!=2 or points.shape[1]!=3
                or not len(points) or not np.isfinite(points).all()
                or not np.isfinite(normals).all() or np.abs(points).max()>2**20):return
        check()
        cells=np.floor(points/.01).astype(np.int64)
        self.origin=cells.min(axis=0)-4
        span=cells.max(axis=0)-self.origin+5
        if np.prod(span.astype(object))>2**61:return
        self.stride=np.array([int(span[1])*int(span[2]),int(span[2]),1],np.int64)
        keys=(cells-self.origin)@self.stride
        self.keys,self.sizes=np.unique(keys,return_counts=True)
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

    @observed('cloud.native_exact_fallback')
    def _fallback(self, indices):
        count('cloud.native_fallback_work',queries=len(indices))
        return self.legacy(indices)

    @observed('cloud.native_complete_queries')
    def __call__(self,indices):
        from femto.geometry.cloud_layer_consistency import _CANDIDATE_RADIUS,_RADIUS_SQUARED,_RADIUS_GUARD,_HEIGHT,_ROUNDING_FACTOR
        if not self.enabled or self.failed:return self._fallback(indices)
        indices=np.asarray(indices,np.int64)
        if not len(indices):return 0
        counts=self._upper_counts(indices);prefix=np.r_[0,np.cumsum(counts)]
        detail=current()
        begin=0;total=0
        while begin<len(indices):
            self.check()
            end=int(np.searchsorted(prefix,prefix[begin]+self.max_pairs,side='right'))-1
            if end<=begin:
                total+=self._fallback(indices[begin:begin+1]);begin+=1
                count('cloud.native_complete_work',oversized_queries=1)
                continue
            group=indices[begin:end]
            tick=perf_counter() if detail is not None else None
            try:
                query=o3d.core.Tensor(self.points[group],dtype=o3d.core.Dtype.Float64)
                if detail is not None:
                    _record(detail,'cloud.native_input',seconds=perf_counter()-tick,calls=1,queries=len(group))
                    tick=perf_counter()
                neighbor_ids,_,splits=self.search.fixed_radius_search(query,radius=_CANDIDATE_RADIUS,sort=False)
            except (RuntimeError,TypeError):
                self.failed=True
                return total+self._fallback(indices[begin:])
            del query
            if detail is not None:
                _record(detail,'cloud.native_search',seconds=perf_counter()-tick,calls=1)
                tick=perf_counter()
            ids=neighbor_ids.numpy();splits=splits.numpy();sizes=np.diff(splits)
            if len(ids)>self.max_pairs:
                raise ValueError('Complete radius candidate bound violated')
            from femto.geometry.cloud_radius_kernel import classify
            decision=classify(self.points,self.normals,group,ids,splits)
            if decision is not None:
                positive,ambiguous=decision
                fallback=group[ambiguous]
                if detail is not None:
                    _record(detail,'cloud.native_fused_predicate',seconds=perf_counter()-tick,calls=1,neighbors=len(ids))
                total+=int(positive.sum())+(self._fallback(fallback) if len(fallback) else 0)
                count('cloud.native_complete_work',query_batches=1,queries=len(group),neighbors=len(ids),ambiguous=len(fallback))
                begin=end
                continue
            repeated=np.repeat(np.arange(len(group)),sizes)
            # Work component-wise on live column views. Keep the original
            # binary64 operation order, without Nx3 coordinate/normal/product
            # temporaries. No copied coordinates or decisions survive a batch.
            query_ids=group[repeated]
            px,py,pz=self.points.T
            dx=px[ids]-px[query_ids];dy=py[ids]-py[query_ids];dz=pz[ids]-pz[query_ids]
            if detail is not None:
                _record(detail,'cloud.native_expand_and_delta',seconds=perf_counter()-tick,calls=1,neighbors=len(ids))
                tick=perf_counter()
            squared=(dx*dx+dy*dy)+dz*dz
            if detail is not None:
                _record(detail,'cloud.native_squared',seconds=perf_counter()-tick,calls=1)
                tick=perf_counter()
            with np.errstate(over='ignore',invalid='ignore'):
                nx,ny,nz=self.normals.T
                ax=dx*nx[query_ids];ay=dy*ny[query_ids];az=dz*nz[query_ids]
                if detail is not None:
                    _record(detail,'cloud.native_normal_products',seconds=perf_counter()-tick,calls=1)
                    tick=perf_counter()
                heights=np.abs((ax+ay)+az)
                guard=_ROUNDING_FACTOR*np.maximum((np.abs(ax)+np.abs(ay))+np.abs(az),_HEIGHT)
            del dx,dy,dz,ax,ay,az
            inside=squared<_RADIUS_SQUARED-_RADIUS_GUARD
            finite=np.isfinite(heights)&np.isfinite(guard)
            good=inside&finite&(heights>_HEIGHT+guard)
            unclear=(np.abs(squared-_RADIUS_SQUARED)<=_RADIUS_GUARD)|(inside&(~finite|(np.abs(heights-_HEIGHT)<=guard)))
            if detail is not None:
                _record(detail,'cloud.native_arithmetic_and_guards',seconds=perf_counter()-tick,calls=1)
                tick=perf_counter()
            positive=np.zeros(len(group),bool);ambiguous=sizes==0;nonempty=sizes>0
            if len(ids):
                positive[nonempty]=np.logical_or.reduceat(good,splits[:-1][nonempty])
                ambiguous[nonempty]=np.logical_or.reduceat(unclear,splits[:-1][nonempty])
            fallback=group[ambiguous&~positive]
            if detail is not None:
                _record(detail,'cloud.native_reduce',seconds=perf_counter()-tick,calls=1)
            total+=int(positive.sum())+(self._fallback(fallback) if len(fallback) else 0)
            count('cloud.native_complete_work',query_batches=1,queries=len(group),neighbors=len(ids),ambiguous=len(fallback))
            begin=end
        return total
