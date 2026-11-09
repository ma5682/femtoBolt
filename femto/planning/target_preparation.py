"""Materialize optional area geometry without regenerating the saved route."""
from pathlib import Path
from uuid import uuid4

from femto.planning import planning_revision, planning_state, surface_workspace, target_surface
from femto.geometry.region_reuse import operation as geometry_operation
from femto.planning.evidence_preparation import operation as evidence_operation


@geometry_operation()
@evidence_operation()
def prepare(model, region, plan, check=lambda: None, progress=lambda text: None,
            *, output=None, budget=None):
    """Return a newly bound plan, leaving the caller's original plan intact.

    Existing computed targets keep their exact inventory and budget frontier.
    Pending inputs are materialized only for an explicit analysis operation;
    neither ordinary export nor reopening calls this function.
    """
    context = (plan or {}).get('planning_context')
    if context is None or not target_surface.is_deferred(context.get('target')):
        return plan
    if plan['surface_trajectory']['planner']['algorithm_version'] != 'quality-constrained-surface-v3':
        raise ValueError('请先恢复常规路线，再计算目标面积')
    if plan.get('coverage_evidence') is not None:
        raise ValueError('尚未计算目标面积的路线不能携带覆盖结论')
    revision = context['revision']
    planning_revision.require_current(model, region, plan['treatment_head'], revision)
    planning_state.validate(plan, model.store.mesh, region, check)
    previous = surface_workspace.target_for(model, plan, check, reproduce=True)
    params = plan['surface_trajectory']['planner']['parameters']
    progress('按需计算目标面积；保留现有路线坐标与顺序')
    ledger = target_surface.build(model, region, revision,
        evidence=params['quality_evidence'], domain=params['effective_domain'],
        budget=budget, check=check)
    planning_revision.require_current(model, region, plan['treatment_head'], revision)
    check()
    from femto.planning.target_continuation import store_result
    destination = (Path(output) if output is not None else
                   Path(model.reconstruction)/'target_analysis'/uuid4().hex)
    result, _ = store_result(model, region, plan, previous, ledger, revision,
                            destination, check, progress, operation='on_demand_area')
    return result
