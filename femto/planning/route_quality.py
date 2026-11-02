"""Quantitative polyline diagnostics; exact retraversal, not physical swath overlap."""
from typing import Any

import numpy as np
from femto.planning.route_motion import statistics

def quality(plan,check=lambda:None):
    result=statistics(plan);seen=set();repeat=0.;repeated_edges=0;turns=[]
    for ci,chain in enumerate(plan['segments']):
        check();xyz=np.asarray([p['xyz_m'] for p in chain]);delta=np.diff(xyz,axis=0);length=np.linalg.norm(delta,axis=1)
        for ordinal,(a,b,d) in enumerate(zip(xyz[:-1],xyz[1:],length)):
            if ordinal%1024==0:check()
            edge=tuple(sorted((tuple(a),tuple(b))))
            if edge in seen:repeat+=float(d);repeated_edges+=1
            seen.add(edge)
        for i in range(1,len(chain)-1):
            if i%1024==0:check()
            if min(length[i-1],length[i])<=1e-12:continue
            angle=float(np.degrees(np.arccos(np.clip(delta[i-1]@delta[i]/(length[i-1]*length[i]),-1,1))))
            if angle>=60-1e-9:
                turns.append(dict(chain_index=ci,sample_index=i,pixel=chain[i]['pixel'],xyz_m=chain[i]['xyz_m'],deflection_degrees=angle))
    result.update(repeated_edge_length_m=repeat,repeated_edge_count=repeated_edges,
        repeat_semantics='extra_traversals_of_exact_undirected_xyz_edges_not_nearby_swath_overlap',
        disconnected_break_count=max(0,len(plan['segments'])-1),sharp_turn_threshold_degrees=60.,sharp_turn_count=len(turns),
        sampled_coverage_percent=plan.get('coverage',{}).get('sampled_coverage_percent'),
        coverage_denominator_m2=plan.get('coverage',{}).get('target_mesh_area_m2'))
    return result,turns

from femto.planning.report_catalog import quality_text
def show_quality(parent: Any, plan: Any, locate: Any, is_current: Any = lambda: True) -> Any:
    from femto.planning.qt_reports import quality_report
    return quality_report(parent, plan, locate, is_current)
