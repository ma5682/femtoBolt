"""Versioned composition of two independently verified ordering proposals.

The result does not prove physical or clinical coverage.
The caller must also verify all edges in the intermediate and final routes.
"""
from copy import deepcopy

from femto.planning import balanced_visit_improvement, seam_visit_proposal
from femto.planning.scan_visit_order import reconstruct

VERSION = 'verified-seam-then-balanced-proposal-v1'
GEOMETRY_BUDGET = '连续路线整边搜索超过预算，请缩小选区'


def propose(graph, segments, visits, links, layout, spacing_m, check, budget, *, search=None):
    """Reorder after seam joining, falling back at each stage independently."""
    initial_visits, initial_links = deepcopy(visits), deepcopy(links)
    try:
        chains, middle_links, added, seam = seam_visit_proposal.propose(
            graph, segments, initial_visits, initial_links, layout, spacing_m, check, budget,
            **({'search': search} if search is not None else {}))
    except ValueError as error:
        if str(error) != GEOMETRY_BUDGET:
            raise
        chains, added, _ = reconstruct(segments, initial_visits, initial_links, check)
        middle_links = deepcopy(initial_links)
        seam = dict(version=seam_visit_proposal.VERSION, adopted=False, attempts=[],
                    reason='geometry_budget', matched_seams=0,
                    baseline=balanced_visit_improvement._metrics(chains, spacing_m, check))
    seam = dict(seam, baseline_visits=deepcopy(initial_visits), baseline_connections=deepcopy(initial_links))
    middle_visits = [list(v) for v in (seam['visits'] if seam['adopted'] else initial_visits)]
    saved_middle_links = deepcopy(middle_links)
    try:
        final_chains, final_links, final_added, balanced = balanced_visit_improvement.propose(
            graph, segments, middle_visits, middle_links, spacing_m, check, budget,
            **({'search': search} if search is not None else {}))
    except ValueError as error:
        if str(error) != GEOMETRY_BUDGET:
            raise
        final_chains, final_links, final_added = chains, deepcopy(middle_links), added
        balanced = dict(version=balanced_visit_improvement.VERSION, adopted=False, attempts=[],
                        skipped='geometry_budget',
                        baseline=balanced_visit_improvement._metrics(chains, spacing_m, check))
    balanced = dict(balanced, baseline_visits=deepcopy(middle_visits),
                    baseline_connections=deepcopy(saved_middle_links))
    adopted = seam['adopted'] or balanced['adopted']
    final_visits = balanced['visits'] if balanced['adopted'] else middle_visits
    result = dict(version=VERSION, adopted=adopted, seam=seam,
                  middle_connections=saved_middle_links, balanced=balanced,
                  baseline_visits=deepcopy(initial_visits), baseline_connections=deepcopy(initial_links))
    if adopted:
        result['visits'] = deepcopy(final_visits)
    return final_chains, final_links, final_added, result


def validate_record(record, segments, visits, links, final_links, spacing_m, check, *, allow_search_budget=False):
    """Rebuild both decisions; return the intermediate route for edge checking."""
    if (not isinstance(record, dict) or record.get('version') != VERSION
            or type(record.get('adopted')) is not bool
            or set(record) != ({'version', 'adopted', 'seam', 'middle_connections', 'balanced',
                               'baseline_visits', 'baseline_connections'}
                              | ({'visits'} if record['adopted'] else set()))
            or record['baseline_visits'] != visits or record['baseline_connections'] != links):
        raise ValueError('接缝后续排序记录无效')
    middle_links = record['middle_connections']
    middle_visits = seam_visit_proposal.validate_record(
        record['seam'], segments, visits, links, middle_links, spacing_m, check,
        **({'allow_search_budget':True} if allow_search_budget else {}))
    middle_visits = [list(v) for v in middle_visits]
    final_visits = balanced_visit_improvement.validate_record(
        record['balanced'], segments, middle_visits, middle_links, final_links, spacing_m, check)
    if (record['adopted'] != (record['seam']['adopted'] or record['balanced']['adopted'])
            or (record['adopted'] and record['visits'] != [list(v) for v in final_visits])):
        raise ValueError('接缝后续排序结果不一致')
    middle_chains, _, _ = reconstruct(segments, middle_visits, middle_links, check)
    return final_visits, middle_chains
