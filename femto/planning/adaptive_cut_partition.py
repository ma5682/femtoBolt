"""Bounded, research-only asymmetric split proposals based on local metric.

The geometric score is a proxy for worst-face influence, not an estimate of
coverage, route quality, treatment effect or physical robot safety. The
proposals are not registered for route generation or archive replay.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from femto.planning import adaptive_spatial_partition as base,spatial_patch_split

VERSION='adaptive-metric-cut-proposal-v1'
CANDIDATE_FRACTIONS=(.5,.4,.6)


def propose(vertices,faces,patch,along,across,width_mm,check=lambda:None,*,fractions=None):
    """Screen a patch, then compare at most three connected cut proposals.

    The established gate validates input and the original patch. All faces
    remain in exactly one piece; existing seams remain attached to their owner.
    The caller must still independently plan, validate and assess coverage.
    """
    if fractions is None:fractions=CANDIDATE_FRACTIONS
    if (not isinstance(fractions,tuple) or not 1<=len(fractions)<=3
            or any(isinstance(f,bool) or not isinstance(f,(int,float)) or not np.isfinite(f) or not 0<f<1 for f in fractions)
            or len(set(fractions))!=len(fractions)):
        raise ValueError('自适应分区切分候选无效')
    fallback,gate=base.propose(vertices,faces,patch,along,across,width_mm,check)
    record=dict(version=VERSION,adopted=False,reason=gate['reason'],gate=gate,candidates=[])
    if gate['reason'] not in ('geometry_proposal','insufficient_predicted_gain'):
        return fallback,record
    vertices=np.asarray(vertices,dtype=float);faces=np.asarray(faces)
    axis=np.asarray(across,dtype=float);axis=axis/np.linalg.norm(axis)
    baseline=gate['baseline'];before=baseline['area_m2']/baseline['minimum']
    best=None
    for fraction in fractions:
        check()
        try:
            pieces=spatial_patch_split.split(vertices,faces,patch,along,2,check,cut_fraction=fraction)
            if len(pieces)<2:
                record['candidates'].append(dict(fraction=fraction,status='one_component'));continue
            measures=[base._geometry(vertices,faces,p,axis,check) for p in pieces]
        except ValueError as error:
            if str(error) not in ('空间分区连通片超过预算','自适应分区没有正面积面片'):raise
            record['candidates'].append(dict(fraction=fraction,status='budget_or_degenerate'));continue
        reduction=1-sum(x['area_m2']/x['minimum'] for x in measures)/before
        record['candidates'].append(dict(fraction=fraction,status='evaluated',pieces=len(pieces),predicted_reduction=reduction))
        if reduction>=base.MIN_PREDICTED_REDUCTION and (best is None or reduction>best[0]+1e-12):
            best=(reduction,fraction,pieces)
    if best is None:
        record['reason']='no_qualifying_cut'
        return [deepcopy(patch)],record
    reduction,fraction,pieces=best
    record.update(adopted=True,reason='geometry_proposal',selected_fraction=fraction,predicted_reduction=reduction)
    return pieces,record
