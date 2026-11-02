"""Strict P4 candidate admission; legacy P3 search remains a candidate source.

Every proposed scan and connector is reconstructed on actual source faces,
then checked again. Retained endpoints keep their exact original anchors.
"""
from copy import deepcopy
from dataclasses import replace

import numpy as np

from femto.geometry.surface_support import NumericalLimit
from femto.planning import strict_path as strict, coverage_refinement as refinement
from femto.planning import quality_constrained as quality
from femto.planning.surface_trajectory import ScanSegment


def _checks(segment, mesh, params, allowed, meter, check, diagnostics=None):
    meter.reserve(len(segment.points)-1)
    checker = strict.Checks(meter.support)
    flags, reasons, margins = quality.edge_checks([segment], mesh, params['quality_evidence'],
        params['effective_width_mm'], check, allowed_faces=allowed, support_checker=checker, diagnostics=diagnostics)
    decisions = [dict(accepted=bool(a),reason=b,residual_margin_mm=float(c)) for a,b,c in zip(flags,reasons,margins)]
    return flags, decisions, checker.decisions


def _budget_exit(recipes, support=()):
    for decision in [r['initial_support'] for r in recipes]+list(support):
        if decision.get('status')=='unconfirmed' and decision['reason'].endswith('_budget'):
            raise refinement._Stop(decision['reason'])


def _connect(chains, addition, trajectory, mesh, region, allowed, meter, check):
    from femto.planning.refinement_candidates_v2 import connection_limit, connection_fits
    block, points = addition
    positions = [i for i,(b,_) in enumerate(chains) if b==block]
    position = positions[-1] if positions else next((i-1 for i,(b,_) in enumerate(chains) if b>block),len(chains)-1)
    link = dict(block_index=block,connected=False,samples=[],edge_checks=[],support_checks=[],
                reason='no_verified_graph_path' if positions else 'no_prior_chain_in_block',
                construction=[],attempt=None)
    samples = None
    params = trajectory.planner['parameters']
    limit = connection_limit(chains, points, trajectory, meter)
    if limit is not None:link['connection_budget'] = limit
    limited = positions and not connection_fits(limit, [chains[position][1][-1], points[0]], 'endpoint_lower_bound')
    if positions and not limited:
        previous = ScanSegment(0,chains[position][1][-2]['row_index'],chains[position][1][-2:],block)
        following = ScanSegment(1,points[0]['row_index'],points[:2],block)
        component = trajectory.blocks['items'][block]['component_id']
        graph = refinement._Graph.__new__(refinement._Graph);graph.meter=meter
        try:
            graph.__init__(mesh,region,params['quality_evidence'],params['effective_width_mm'],
                allowed & (meter.support.groups==component),replace(trajectory,ordered_items=[previous,following]),check)
            samples = graph.route(0,1)
        except ValueError as error:
            if '预算' not in str(error):raise
            raise refinement._Stop('graph_budget') from error
        if samples and not connection_fits(limit, [chains[position][1][-1]]+samples, 'graph'):
            samples = None;limited = True
    if samples:
        source = [deepcopy(chains[position][1][-1])]+deepcopy(samples)
        # The graph uses floating display coordinates and may choose an
        # incident face. Restore both authoritative endpoint constructions.
        source[-1] = dict(deepcopy(points[0]),role='connector',row_index=points[0]['row_index'])
        try:
            fresh,recipes = strict.construct_polyline(meter.support,source,params,meter.support_budget)
        except NumericalLimit as error:
            raise refinement._Stop(str(error)) from error
        link['construction']=recipes
        link['attempt']=dict(source_points=source,status='unconstructed',points=[],edge_checks=[],support_checks=[])
        _budget_exit(recipes)
        if fresh is not None:
            segment = ScanSegment(0,points[0]['row_index'],fresh,block)
            flags,decisions,support = _checks(segment,mesh,params,allowed,meter,check)
            link['attempt'].update(status='checked',points=fresh,edge_checks=decisions,support_checks=support)
            _budget_exit((),support)
            fits = connection_fits(limit, fresh, 'constructed')
            if not fits:limited = True
            if flags.all() and fits:
                link.update(connected=True,reason='verified_surface_path',samples=deepcopy(fresh[1:]),
                            edge_checks=decisions,support_checks=support)
                chains[position][1].extend(deepcopy(fresh[1:]));chains[position][1].extend(deepcopy(points[1:]))
    if limited:link['reason'] = 'connection_length_budget'
    if not link['connected']:chains.insert(position+1,(block,deepcopy(points)))
    return link


def propose(current, domain, certificate, mesh, region, allowed, attempted, meter, check):
    from femto.planning.refinement_candidates import retained_chains
    recipes = refinement.candidate_recipes(current,domain,certificate,mesh,region,allowed,attempted,meter,check)
    chains=refinement._chains(current);records=[];params=current.planner['parameters']
    row=max(p['row_index'] for _,points in chains for p in points)+1
    status='no_candidates'
    for recipe in recipes[:meter.budget.candidates_per_round]:
        check();attempted.add(recipe['local_face'])
        record=dict(recipe=recipe,status='too_short_or_unconstructed',edge_checks=[],connections=[],
                    construction=[],support_checks=[],source_points=[])
        records.append(record)
        candidate=refinement.construct_candidate(recipe,domain,mesh,row,params['sample_step_mm']/1000);row+=1
        if candidate is None:continue
        record['source_points']=deepcopy(candidate.points)
        try:
            points,construction=strict.construct_polyline(meter.support,candidate.points,params,meter.support_budget)
            record['construction']=construction
            _budget_exit(construction)
            if points is None:continue
            candidate=replace(candidate,points=points)
            diagnostics=[] if getattr(meter,'candidate_policy',None) is not None else None
            flags,decisions,support=_checks(candidate,mesh,params,allowed,meter,check,diagnostics)
            record.update(status='checked',points=points,edge_checks=decisions,support_checks=support)
            if diagnostics is not None:record['quality_diagnostics']=diagnostics
            _budget_exit((),support)
            for addition in retained_chains(candidate,flags):
                record['connections'].append(_connect(chains,addition,current,mesh,region,allowed,meter,check))
            if flags.any():status='proposed'
        except (refinement._Stop,NumericalLimit) as error:
            record['status']=str(error)
            return refinement._chains(current),records,str(error)
    return chains,records,status


def validate_candidate(candidate):
    """Cheap structural validation; full construction is reproduced with assets."""
    from femto.planning.refinement_candidates import retained_chains
    from femto.geometry.surface_support import fraction, SCHEMA
    points=candidate.get('points',[]);decisions=candidate['edge_checks'];support=candidate['support_checks']
    if len(support)!=len(decisions):raise ValueError('严格补充缺少逐边支持证明')
    for decision,proof in zip(decisions,support):
        if decision['accepted'] and proof.get('status')!='passed':raise ValueError('严格补充接纳了未确认边')
    additions=retained_chains(ScanSegment(0,points[0]['row_index'],points,candidate['recipe']['block_index']),
                               [d['accepted'] for d in decisions])
    if len(additions)!=len(candidate['connections']):raise ValueError('严格补充连接数量不一致')
    all_points=list(points)
    for (_,run),link in zip(additions,candidate['connections']):
        if len(link['support_checks'])!=len(link['edge_checks']) or any(p.get('status')!='passed' for p in link['support_checks']):
            raise ValueError('严格补充连接缺少支持证明')
        if link['samples'] and link['samples'][-1].get(strict.ANCHOR)!=run[0].get(strict.ANCHOR):
            raise ValueError('严格补充连接改变了原端点锚点')
        all_points.extend(link['samples'])
    for point in all_points:
        anchor=point.get(strict.ANCHOR) or {}
        weights=[fraction(value) for value in anchor.get('barycentric',[])]
        if (anchor.get('schema')!=SCHEMA or anchor.get('face')!=point['triangle_index']
                or len(weights)!=3 or min(weights)<0 or sum(weights)!=1):
            raise ValueError('严格补充点的构造来源无效')
