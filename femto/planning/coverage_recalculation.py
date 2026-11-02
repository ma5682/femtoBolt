"""Same-input P2 reproof with one untrusted, non-recursive scheduling hint.

The hint records a partition and candidate proof paths, not retained coverage
claims. Every current ball, leaf and positive subedge is proved again by the
delivered v4 engine before any search of the remaining target. A subsequent
recalculation keeps only the previous output arrays, never its older hints.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

from femto.planning import coverage_certificate as coverage, footprint_model
from femto.planning.coverage_residual import POLICY as RESIDUAL_POLICY
from femto.planning.coverage_prepared import PreparedResidualEngine
from femto.planning.coverage_schedule import POLICY as LAYERED_POLICY
from femto.planning.planning_revision import digest
from femto.operation_diagnostics import diagnosed

POLICY = dict(RESIDUAL_POLICY,
    schedule='same_input_retained_reproof_then_current_exclusions_then_remaining_target',
    hint_storage='one_previous_output_partition_and_paths_without_recursive_hint_assets',
    hint_authority='untrusted_search_input_every_current_claim_recomputed',
    seed_binding='identical_ordered_input_seeds_and_nominal_fixed_footprint',
    retained_frontier='current_reproved_positive_subedges_indexed_before_remaining_search',
    frontier_accounting='each_indexed_current_edge_charged_to_original_adjacency_budget',
    admission='independent_common_partition_retention_check_before_install')
HINT_PREFIX = 'recalculation_hint_'


@dataclass(frozen=True)
class RecalculatedCertificate(coverage.Certificate):
    guide: coverage.Certificate | None = None


def checked_policy(value: dict) -> dict:
    if not isinstance(value, dict) or value != POLICY or digest(value) != digest(POLICY):
        raise ValueError('原集合优先复算策略无效或版本不一致')
    return value


def flat_hint(certificate: coverage.Certificate) -> coverage.Certificate:
    """Keep a previous output as untrusted input, without its older dependencies."""
    return coverage.Certificate(deepcopy(certificate.metadata), dict(certificate.arrays))


def require_guide(certificate: coverage.Certificate, check: Callable = lambda: None) -> coverage.Certificate:
    guide = getattr(certificate, 'guide', None)
    if not isinstance(guide, coverage.Certificate):
        raise ValueError('原集合优先复算缺少单层引导资产')
    coverage._checked_assets(guide, check)
    if guide.metadata['sha256'] != certificate.metadata['inputs'].get('guide_sha256'):
        raise ValueError('原集合优先复算引导资产与计算绑定不一致')
    # The pair shares the pre-existing 256 MiB allocation bound. Hints cannot
    # silently double an individual coverage certificate's allocation budget.
    if sum(a.nbytes for c in (certificate, guide) for a in c.arrays.values()) > coverage.MAX_ARRAY_BYTES:
        raise ValueError('覆盖证据与复算引导合计超过 256 MiB 预算')
    return guide


class RecalculationEngine(PreparedResidualEngine):
    """Use v4's independently checked geometry, with exactly the same seeds."""

    def __init__(self, *args, guide: coverage.Certificate):
        super().__init__(*args, guide=guide)
        if (self.model['schema'] != footprint_model.SCHEMA
                or guide.metadata['inputs']['seeds_sha256'] != digest(self.records)):
            raise ValueError('原集合优先复算只能使用同一扫描种子和固定名义模型')
        self.meter.counts.update(retained_frontier_leaves=0, retained_frontier_edges=0)

    @diagnosed('coverage.retained_frontier')
    def _index_reproved_edges(self):
        for number in sorted(self.by_node):
            self.meter.phase = 'retained_frontier'; self.meter.touch()
            node = self.nodes[number]
            if not node.active or node.witness < 0:continue
            for edge in self._edges(number):
                self.meter.touch('adjacencies')
                self.edges[edge[0]].append((edge, number))
                self.meter.counts['retained_frontier_edges'] += 1
            self.meter.counts['retained_frontier_leaves'] += 1

    def cells(self):
        if not self._prepare():return
        self._pixel_bounds()
        if not self._reprove():return
        self._outside_faces(); self._outside_pixels(); self._outside_leaves()
        self._index_reproved_edges()
        self.pending.clear()
        for number in range(len(self.nodes)):
            self.meter.phase = 'remaining_original_cells'; self.meter.touch()
            node = self.nodes[number]
            if not node.active or node.status in (1, 2, 3, 6):continue
            self.meter.counts['original_cells_visited'] += 1
            self._visit(number)
            if number % LAYERED_POLICY['connectivity_batch'] == 0:self._drain(LAYERED_POLICY['proof_batch'])
        self._drain(); self._subdivide()
