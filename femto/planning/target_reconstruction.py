"""Explicit exact-partition calculation; preserve source evidence and old results."""
from copy import deepcopy
from pathlib import Path

from femto.planning import (planning_revision as revisions, target_surface as target,
                            target_paging as paging, target_continuation as continuation,
                            surface_workspace, planning_state)


def available(plan):
    context=(plan or {}).get('planning_context')
    if context is None:return False
    algorithms=context['revision']['inputs']['algorithms']
    reasons=context['target']['summary']['unknown_by_reason']
    return (algorithms['route']=='quality-constrained-surface-v3'
            and algorithms['target']!=revisions.EXACT_TARGET
            and bool(reasons['numerically_unconfirmed'] or reasons['partly_located_pixel_cell']))


def calculate(model,region,plan,output,check=lambda:None,progress=lambda text:None,*,budget=None):
    if not available(plan):raise ValueError('当前目标无需精确分区核验，或需先恢复常规路线')
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('精确分区结果必须保存到新目录')
    old_revision=plan['planning_context']['revision']
    revisions.require_current(model,region,plan['treatment_head'],old_revision)
    planning_state.validate(plan,model.store.mesh,region,check)
    original=surface_workspace.target_for(model,plan,check,reproduce=False)
    inputs=deepcopy(old_revision['inputs'])
    inputs['algorithms']['target']=revisions.EXACT_TARGET
    revision=revisions.create(inputs)
    source={name:original.arrays[name] for name in paging.SOURCE_NAMES}
    target._check_inputs(source,revision)
    pages=[];cursor=0;budget=(budget or target.TargetBudget()).checked()
    while len(pages)<paging.MAX_PAGES:
        check();progress(f'精确核验像素分区：第 {len(pages)+1} 块；已处理 {cursor} 个像素组')
        arrays,metrics=target._construct(source,revision,budget,check,start_group=cursor)
        page=target._seal(arrays,revision,budget,metrics,original.metadata['quality_evidence_sha256'],
                          original.metadata['effective_domain_sha256'])
        pages.append(page)
        # Check cumulative assets after every page, before constructing another.
        # This uses the same duplicate-source accounting as ordinary continuation.
        bundle=paging._seal(tuple(pages),revision)
        if metrics['stop_reason']=='complete':break
        if (metrics['stop_reason'] not in ('unit_budget','pair_budget','time_budget')
                or metrics['processed_pixel_groups']<=cursor):
            raise ValueError('精确分区核验未能推进；旧台账和方案保留')
        cursor=metrics['processed_pixel_groups']
    else:raise ValueError('精确分区核验超过分块数量预算；旧方案保留')
    check()
    ledger=paging.materialize(bundle,revision)
    return continuation.store_result(model,region,plan,original,ledger,revision,output,check,progress,
                                     operation='exact_pixel_partition_v2')
