"""Approximate connection change for local-row candidate ordering only.

This never validates a connector, treats Euclidean distance as a proxy and
does not establish coverage. Ambiguous topology returns no prediction.
"""
from __future__ import annotations
import numpy as np
from femto.planning import auto_supplement_route_research

VERSION='research-local-connector-endpoint-proxy-v1'
MAX_ROW_SEGMENTS=32


def estimate(state,trajectory,block,row,changed,check=lambda:None):
    """Compare fixed-link endpoint lengths after greedy row-segment matching."""
    sources=[(i,s) for i,s in enumerate(state['base']) if s.block_index==block and s.points[0]['row_index']==row]
    targets=[s for s in changed if s.block_index==block and s.points[0]['row_index']==row]
    if len(sources)!=len(targets) or not sources or len(sources)>MAX_ROW_SEGMENTS:return None
    pairs=[]
    for i,s in sources:
        check();a,b=np.asarray(s.points[0]['xyz_m'],float),np.asarray(s.points[-1]['xyz_m'],float)
        for j,t in enumerate(targets):
            c,d=np.asarray(t.points[0]['xyz_m'],float),np.asarray(t.points[-1]['xyz_m'],float)
            if not np.isfinite([a,b,c,d]).all():return None
            forward=float(np.linalg.norm(a-c)+np.linalg.norm(b-d));reverse=float(np.linalg.norm(a-d)+np.linalg.norm(b-c))
            pairs.append((min(forward,reverse),i,j,reverse<forward))
    mapping={};used=set()
    for _,i,j,reverse in sorted(pairs):
        if i in mapping or j in used:continue
        pts=targets[j].points
        mapping[i]=(np.asarray(pts[-1 if reverse else 0]['xyz_m'],float),np.asarray(pts[0 if reverse else -1]['xyz_m'],float));used.add(j)
    if len(mapping)!=len(sources):return None
    visits=auto_supplement_route_research._source_visits(trajectory,state['base'])
    links=trajectory.gap_report['reconnection']['connections']
    if len(links)!=len(visits)-1:return None
    delta=0.;affected=0
    for (left,lr),(right,rr),link in zip(visits,visits[1:],links):
        check()
        if not link['connected'] or (left not in mapping and right not in mapping):continue
        samples=link['samples']
        if not samples:return None
        a=np.asarray(state['base'][left].points[0 if lr else -1]['xyz_m'],float)
        b=np.asarray(state['base'][right].points[-1 if rr else 0]['xyz_m'],float)
        aa=mapping[left][0 if lr else 1] if left in mapping else a
        bb=mapping[right][1 if rr else 0] if right in mapping else b
        if len(samples)==1:
            delta+=float(np.linalg.norm(aa-bb)-np.linalg.norm(a-b))
        else:
            first=np.asarray(samples[0]['xyz_m'],float);previous=np.asarray(samples[-2]['xyz_m'],float)
            delta+=float(np.linalg.norm(aa-first)-np.linalg.norm(a-first)+np.linalg.norm(previous-bb)-np.linalg.norm(previous-b))
        affected+=1
    if not affected or not np.isfinite(delta):return None
    return dict(version=VERSION,predicted_connector_delta_m=delta,affected_connections=affected,
                matched_segments=len(sources),semantics='ordering_proxy_not_verified_connection_or_coverage')
