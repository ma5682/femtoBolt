"""Separate, non-product policy for rechecking endpoint proofs with fractions."""
from femto.planning.coverage_indexed_reproof import POLICY as INDEXED_POLICY
from femto.planning.planning_revision import digest

POLICY = dict(INDEXED_POLICY,
              guide_footprint='endpoint_fixed_ball_to_matching_width_dyadic_research_model',
              publication_scope='research_only_original_proofs_rechecked_not_inherited')


def checked_policy(value):
    if not isinstance(value, dict) or value != POLICY or digest(value) != digest(POLICY):
        raise ValueError('研究分数位置复算策略无效或版本不一致')
    return value
