"""Versioned research model for bounded positions on accepted scan edges.

The model is not registered for product coverage archives. A caller must
regenerate every seed from a validated trajectory; the model alone cannot
establish route provenance or physical accuracy.
"""
from __future__ import annotations

from femto.planning.planning_revision import digest

SCHEMA = 'femto-research-nominal-dyadic-ball-v1'
PROVENANCE = 'dyadic_edge_position_with_l1_error_deduction'


def nominal(effective_width_mm: float) -> dict:
    # Reuse the established unit, width and radius checks without changing the
    # old model's bytes or its endpoint-only meaning.
    from femto.planning import footprint_model
    result = footprint_model.nominal(effective_width_mm)
    result.pop('sha256')
    result.update(schema=SCHEMA,
                  seed_rule='accepted_scan_edge_endpoints_and_bound_dyadic_positions',
                  fractional_position_rule='dyadic_fraction_with_upward_l1_rounding_bound',
                  fractional_radius_rule='subtract_rounding_bound_then_round_down',
                  publication_scope='research_not_product_coverage_archive')
    result['sha256'] = digest(result)
    return result


def validate(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ValueError('研究分数位置作用模型内容、单位或摘要不一致')
    expected = nominal(record.get('effective_width_mm'))
    # Python considers False equal to 0; canonical JSON hashing distinguishes
    # those values and rejects type substitutions even with an old digest.
    if record != expected or digest(record) != digest(expected):
        raise ValueError('研究分数位置作用模型内容、单位或摘要不一致')
    return record


def seed_radius(model: dict, seed: dict) -> float:
    """Deduct a declared bound; seed provenance must be checked separately."""
    if not isinstance(seed, dict):
        raise ValueError('研究分数位置种子无效')
    radius = model['nominal_radius_m']
    endpoint = seed.get('endpoint')
    if type(endpoint) is not int:
        raise ValueError('研究分数位置种子缺少端点类型')
    if endpoint in (0, 1):
        if 'midpoint_error_bound_m' in seed or 'provenance' in seed:
            raise ValueError('扫描端点不能携带分数位置误差')
        return radius
    if endpoint != 3 or seed.get('provenance') != PROVENANCE:
        raise ValueError('研究分数位置来源无效')
    numerator, denominator = seed.get('fraction_numerator'), seed.get('fraction_denominator')
    if (type(numerator) is not int or type(denominator) is not int
            or denominator < 2 or denominator > 256 or denominator & (denominator - 1)
            or not 0 < numerator < denominator):
        raise ValueError('研究分数位置分数无效')
    from femto.planning.scan_midpoint_seeds import conservative_radius
    return conservative_radius(radius, seed)
