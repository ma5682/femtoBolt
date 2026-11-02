"""Explicit nominal fixed-ball model for independent P2/B1 experiments."""
from __future__ import annotations

import math

from femto.planning.planning_revision import digest

SCHEMA = 'femto-nominal-fixed-ball-v1'


def nominal(effective_width_mm: float) -> dict:
    if isinstance(effective_width_mm, bool) or not isinstance(effective_width_mm, (int, float)):
        raise ValueError('有效宽度必须是有限正数')
    width = float(effective_width_mm)
    if not math.isfinite(width) or not 0 < width <= 1000:
        raise ValueError('名义球模型宽度须在 (0, 1000] mm 范围内')
    radius = width/2000
    if radius <= 0:
        raise ValueError('名义球模型半径小于可表示的正数范围')
    record = dict(schema=SCHEMA, model='fixed_euclidean_ball', mode='nominal',
                  effective_width_mm=width, nominal_radius_m=radius,
                  center_offset_assumed_m=0.0, physical_center_offset_bound_m=None,
                  registration_uncertainty_m=None, calibration_uncertainty_m=None,
                  active_edge_roles=['scan'], connector_effect_assumed=False,
                  seed_rule='existing_accepted_scan_edge_endpoints_only',
                  anchor_rule='exact_surface_projection_with_displacement_deducted_inner_radius',
                  coverage_claim='nominal_model_surface_geometry_only',
                  outer_acceptance='existing_v3_rules_unchanged',
                  execution_ready=False, physical_accuracy_verified=False)
    record['sha256'] = digest(record)
    return record


def validate(record: dict) -> dict:
    if isinstance(record, dict) and record.get('schema') == 'femto-research-nominal-dyadic-ball-v1':
        from femto.planning.fraction_footprint import validate as validate_fraction
        return validate_fraction(record)
    if isinstance(record, dict) and record.get('schema') == 'femto-nominal-piecewise-ball-v1':
        from femto.planning.variable_footprint import validate as validate_variable
        return validate_variable(record)
    if not isinstance(record, dict) or record != nominal(record.get('effective_width_mm')):
        raise ValueError('名义作用模型内容、单位或摘要不一致')
    return record


def for_trajectory(trajectory) -> dict:
    if trajectory.planner['algorithm_version'] in ('nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
        return validate(trajectory.planner['parameters']['variable_footprint'])
    return nominal(trajectory.planner['parameters']['effective_width_mm'])


def search_radius(record: dict) -> float:
    if record['schema'] in (SCHEMA, 'femto-research-nominal-dyadic-ball-v1'):return record['nominal_radius_m']
    # A positive bin size also handles the all-zero-inner-radius model. The
    # search bound can be larger than the actual seed balls, never smaller.
    return max(.00025, max(row['upper_half_width_m'] for row in record['knots']))


def seed_radius(record: dict, seed: dict) -> float:
    if record['schema'] == SCHEMA:return record['nominal_radius_m']
    if record['schema'] == 'femto-research-nominal-dyadic-ball-v1':
        from femto.planning.fraction_footprint import seed_radius as fraction_radius
        return fraction_radius(record, seed)
    from femto.planning.variable_footprint import seed_radius as variable_radius
    return variable_radius(record, seed)
