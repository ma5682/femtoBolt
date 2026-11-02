"""Build bounded extra scan segments from independently validated old edges.

These segments are proposals only. The caller must validate the old route,
recheck every produced edge against the current target and reconnect it using
the verified graph. This module does not decide coverage or physical safety.
"""
from __future__ import annotations

from copy import deepcopy
from femto.planning.surface_trajectory import ScanSegment

VERSION='research-existing-scan-edge-segments-v1'
MAX_EDGES=50_000
MAX_SEGMENTS=5000


def materialize(old_trajectory,edges,*,row_offset,check=lambda:None):
    if type(row_offset) is not int or row_offset<0 or not isinstance(edges,(list,tuple)) or len(edges)>MAX_EDGES:
        raise ValueError('补线索引或行偏移无效或超过预算')
    sources={}
    for item in old_trajectory.ordered_items:
        check()
        if not isinstance(item,ScanSegment):continue
        if type(item.segment_index) is not int or item.segment_index in sources:
            raise ValueError('原扫描段索引无效或重复')
        sources[item.segment_index]=item
    selected=set()
    for i,pair in enumerate(edges):
        if i%256==0:check()
        if (not isinstance(pair,(list,tuple)) or len(pair)!=2 or any(type(x) is not int or x<0 for x in pair)
                or tuple(pair) in selected):raise ValueError('补线原扫描边索引无效或重复')
        segment,index=pair
        if segment not in sources or index+1>=len(sources[segment].points):raise ValueError('补线原扫描边不存在')
        left,right=sources[segment].points[index:index+2]
        if (right.get('role','scan')!='scan' or left['xyz_m']==right['xyz_m']
                or type(right.get('row_index')) is not int or right['row_index']<0):
            raise ValueError('补线必须来源于非零原扫描边')
        selected.add((segment,index))
    result=[]
    for segment,index in sorted(selected):
        source=sources[segment];row=source.points[index+1]['row_index']
        if (segment,index-1) in selected and source.points[index]['row_index']==row:continue
        last=index
        while (segment,last+1) in selected and source.points[last+2]['row_index']==row:last+=1
        points=deepcopy(source.points[index:last+2])
        for point in points:
            point['role']='scan';point['row_index']=row+row_offset;point.pop('revisit',None)
        result.append(ScanSegment(len(result),row+row_offset,points,source.block_index))
        if len(result)>MAX_SEGMENTS:raise ValueError('补线扫描段超过预算')
        check()
    return result,dict(version=VERSION,selected_edges=len(selected),segments=len(result),
        semantics='original_scan_candidates_require_current_geometry_and_route_checks')
