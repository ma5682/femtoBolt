"""Explicit non-product dyadic reproof policy for a materialized paged target."""
from femto.planning.fraction_reproof_policy import POLICY as FRACTION_POLICY
from femto.planning.planning_revision import digest

POLICY = dict(FRACTION_POLICY,
    paged_target='validated_materialized_target_all_pages_and_original_root_indices')


def checked_policy(value):
    if not isinstance(value, dict) or value != POLICY or digest(value) != digest(POLICY):
        raise ValueError('分块研究分数位置复算策略无效或版本不一致')
    return value
