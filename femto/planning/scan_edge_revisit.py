"""Experimental revisit statistic based on endpoints of active scan edges.

This is a nominal proximity heuristic, not physical contact or coverage.
It is used only by an explicitly versioned experimental proposal; legacy
planner versions and their persisted statistics retain their original rules.
"""
from __future__ import annotations

import math
import numpy as np

CRITERION = 'nonzero_scan_edge_endpoints_row_spacing_v1'


def mark(chains, spacing_m, check=lambda: None):
    """Mark connector points near another scan row in caller-owned chains.

    Roles describe incoming edges. Both endpoints of each nonzero scan edge
    belong to the edge's row; isolated points and zero-length handoffs do not
    add scan samples. Existing revisit marks are cleared before recomputation.
    """
    if isinstance(spacing_m, bool) or not isinstance(spacing_m, (int, float)) or not math.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError('重访统计行距无效')
    points_by_row = set()
    examined = 0
    for chain in chains:
        for point in chain:
            point.pop('revisit', None)
        for left, right in zip(chain, chain[1:]):
            examined += 1
            if examined % 256 == 0:check()
            role=right.get('role')
            if role not in (None, 'scan', 'connector'):
                raise ValueError('重访统计扫描边角色无效')
            if role == 'connector':continue
            a,b=tuple(left['xyz_m']),tuple(right['xyz_m'])
            row=right['row_index']
            if (len(a)!=3 or len(b)!=3 or type(row) is not int or row<0
                    or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in a+b)):
                raise ValueError('重访统计扫描端点无效')
            if a==b:continue
            points_by_row.add((*a,row));points_by_row.add((*b,row))
    check()
    result=dict(connector_sample_count=0,connector_length_m=0.,rows=[],criterion=CRITERION,
                radius_m=spacing_m/2,semantics='nominal_row_spacing_not_treatment_head_width',
                decision_use='none',scan_sample_count=len(points_by_row))
    if not points_by_row:return result
    import open3d as o3d
    keys=sorted(points_by_row)
    xyz=np.asarray([key[:3] for key in keys],dtype=np.float64)
    rows=np.asarray([key[3] for key in keys],dtype=np.int64)
    tree=o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz)))
    revisited=set();count=0;length=0.;queries=0
    for chain in chains:
        index=0;last_row=None
        while index<len(chain):
            point=chain[index]
            if point.get('role')!='connector':
                last_row=point['row_index'];index+=1;continue
            start=index
            while index<len(chain) and chain[index].get('role')=='connector':index+=1
            joined={last_row,chain[index]['row_index'] if index<len(chain) else chain[start]['row_index']}
            for offset in range(start,index):
                queries+=1
                if queries%256==0:check()
                sample=chain[offset]
                _,found,_=tree.search_radius_vector_3d(np.asarray(sample['xyz_m'],dtype=np.float64),spacing_m/2)
                others={int(rows[i]) for i in found if int(rows[i]) not in joined}
                if others:
                    sample['revisit']=True;count+=1;revisited.update(others)
                    if offset>0:
                        length+=float(np.linalg.norm(np.asarray(sample['xyz_m'])-np.asarray(chain[offset-1]['xyz_m'])))
    check()
    result.update(connector_sample_count=count,connector_length_m=length,rows=sorted(revisited))
    return result
