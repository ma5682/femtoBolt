"""Request-local budgets for experimental multi-stage route proposals.

Only optional optimization candidates consume this budget. Constructing the
initial route, checking the final route and validating saved evidence remain
mandatory and are not subject to this search limit.
"""

SCHEMA = 'research-global-route-search-budget-v1'
TOTAL = 512
LIMITS = {
    'seam_pair': 32,
    'seam_order': 5,
    'two_opt': 96,
    'balanced_prefix': 8,
    'detour': 24,
    'joint': 48,
    'relocation': TOTAL,
}


class SearchBudget:
    """Owned by one route-generation request, including row-repair trials."""

    def __init__(self):
        self.counts = dict.fromkeys(LIMITS, 0)

    def can_try(self, stage):
        if stage not in LIMITS:
            raise ValueError('未知路线搜索阶段')
        return self.counts[stage] < LIMITS[stage] and sum(self.counts.values()) < TOTAL

    def take(self, stage):
        if not self.can_try(stage):
            return False
        self.counts[stage] += 1
        return True

    def report(self):
        return dict(schema=SCHEMA, total_limit=TOTAL, limits=dict(LIMITS),
                    counts=dict(self.counts), total_used=sum(self.counts.values()),
                    scope='optional_ordering_candidates_across_repair_trials')


def validate_report(report):
    if (not isinstance(report, dict) or set(report) != {
            'schema', 'total_limit', 'limits', 'counts', 'total_used', 'scope'}
            or report['schema'] != SCHEMA or report['total_limit'] != TOTAL
            or report['limits'] != LIMITS
            or report['scope'] != 'optional_ordering_candidates_across_repair_trials'
            or not isinstance(report['counts'], dict) or set(report['counts']) != set(LIMITS)
            or any(type(value) is not int or not 0 <= value <= LIMITS[stage]
                   for stage, value in report['counts'].items())
            or type(report['total_used']) is not int
            or report['total_used'] != sum(report['counts'].values())
            or report['total_used'] > TOTAL):
        raise ValueError('全流程搜索预算记录无效')
    return report['counts']
