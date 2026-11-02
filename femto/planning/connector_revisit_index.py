"""Research helper for reusable nominal connector proximity statistics.

This index is not a surface-path verifier, a coverage certificate, or robot
motion guidance. It is a helper rather than a route or archive algorithm. The
caller must retain every scan head in each reconstructed chain and separately
verify every proposed connector against the original geometry.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np

MAX_SCAN_POINTS = 1_000_000
MAX_LINK_POINTS = 100_000


def _point(point):
    if not isinstance(point, dict):
        raise ValueError('重访统计点无效')
    xyz=point.get('xyz_m');row=point.get('row_index')
    if (not isinstance(xyz,(list,tuple,np.ndarray)) or len(xyz)!=3
            or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) or not math.isfinite(x) for x in xyz)
            or type(row) is not int or not 0<=row<2**63):
        raise ValueError('重访统计点坐标或行号无效')
    return tuple(float(x) for x in xyz),row


class ConnectorRevisitIndex:
    """Snapshot scan seeds for scoring isolated, fully anchored links.

    Scan segments must contain only scan points and use one row number per
    segment. The caller supplies previously obtained connector samples; this
    class neither searches nor caches graph paths or geometric decisions.
    """
    def __init__(self,segments,spacing_m,check=lambda:None):
        if (isinstance(spacing_m,bool) or not isinstance(spacing_m,Real)
                or not math.isfinite(spacing_m) or spacing_m<=0):
            raise ValueError('重访统计行距无效')
        self._radius=float(spacing_m)/2;self._check=check
        history=set();edges=set();endpoints=set();total=0
        for segment in segments:
            check();points=segment.points
            if not points:raise ValueError('扫描段不能为空')
            parsed=[]
            for point in points:
                total+=1
                if total>MAX_SCAN_POINTS:raise ValueError('重访统计扫描点超过预算')
                if total%256==0:check()
                parsed.append(_point(point))
                if point.get('role') not in (None,'scan'):raise ValueError('重访统计扫描角色无效')
            if len({row for _,row in parsed})!=1:raise ValueError('扫描段行号不一致')
            endpoints.update((*xyz,row) for xyz,row in (parsed[0],parsed[-1]))
            history.update((*xyz,row) for xyz,row in parsed)
            for (a,_),(b,row) in zip(parsed,parsed[1:]):
                if a!=b:
                    edges.add((*a,row));edges.add((*b,row))
        check()
        if not history:raise ValueError('扫描点不能为空')
        self._endpoints=frozenset(endpoints)
        self._history_tree,self._history_rows=self._tree(history)
        self._edge_tree,self._edge_rows=self._tree(edges)

    @staticmethod
    def _tree(points):
        if not points:return None,None
        import open3d as o3d
        keys=sorted(points);cloud=o3d.geometry.PointCloud()
        cloud.points=o3d.utility.Vector3dVector(np.asarray([p[:3] for p in keys],dtype=np.float64))
        return o3d.geometry.KDTreeFlann(cloud),np.asarray([p[3] for p in keys],dtype=np.int64)

    def measure(self,tail,head,samples):
        """Return link length and both nominal revisit lengths without mutation.

        ``tail`` and ``head`` are scan endpoints; ``samples`` must be the
        complete connector ending at ``head``. This validates representation,
        not geometric support or traversability.
        """
        self._check();a,ar=_point(tail);b,br=_point(head)
        if ((*a,ar) not in self._endpoints or (*b,br) not in self._endpoints
                or tail.get('role') not in (None,'scan') or head.get('role') not in (None,'scan')):
            raise ValueError('连接端点不属于扫描快照')
        if not isinstance(samples,list) or not samples or len(samples)>MAX_LINK_POINTS:
            raise ValueError('连接采样点无效或超过预算')
        joined={ar,br};previous=np.asarray(a);length=history=edge=0.
        history_count=edge_count=0
        for i,sample in enumerate(samples):
            if i%256==0:self._check()
            xyz,row=_point(sample)
            if sample.get('role')!='connector' or row!=br:raise ValueError('连接采样角色或行号无效')
            current=np.asarray(xyz);distance=float(np.linalg.norm(current-previous));length+=distance;previous=current
            _,found,_=self._history_tree.search_radius_vector_3d(current,self._radius)
            if any(int(self._history_rows[k]) not in joined for k in found):
                history+=distance;history_count+=1
            if self._edge_tree is not None:
                _,found,_=self._edge_tree.search_radius_vector_3d(current,self._radius)
                if any(int(self._edge_rows[k]) not in joined for k in found):
                    edge+=distance;edge_count+=1
        if xyz!=b:raise ValueError('连接终点与扫描起点不一致')
        self._check()
        return dict(length_m=length,history_revisit_m=history,edge_revisit_m=edge,
                    history_revisit_samples=history_count,edge_revisit_samples=edge_count)
