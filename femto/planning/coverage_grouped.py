"""Versioned identical-bound elimination for paged retained-first coverage.

Only byte-identical current prism bounds within the same complete candidate
list share an exclusion calculation. No previous certificate state is reused.
"""
import numpy as np

from femto.operation_diagnostics import diagnosed
from femto.planning.coverage_recalculation import RecalculationEngine, POLICY as LEGACY_POLICY
from femto.planning.coverage_residual import POLICY as RESIDUAL_POLICY, _outside_pairs
from femto.planning.planning_revision import digest

POLICY = dict(LEGACY_POLICY,
    pixel_bounds_reuse='byte_identical_current_boxes_within_identical_complete_candidate_group',
    pixel_bounds_order='first_occurrence_then_original_member_order',
    pixel_bounds_accounting='one_comparison_per_unique_box_ball_pair_and_checked_member_install')


def checked_policy(value):
    if not isinstance(value, dict) or value != POLICY or digest(value) != digest(POLICY):
        raise ValueError('分块覆盖同界归并策略无效或版本不一致')
    return value


def identical_boxes(group, bounds, check):
    """Bounded by this input group; exact bytes, never rounded coordinates."""
    by_bounds, members, representatives = {}, {}, []
    for root in group:
        check()
        key = (bounds[0][root].tobytes(), bounds[1][root].tobytes())
        representative = by_bounds.get(key)
        if representative is None:
            representative = root
            by_bounds[key] = root
            representatives.append(root)
            members[root] = []
        members[representative].append(root)
    return representatives, members


class GroupedRecalculationEngine(RecalculationEngine):
    def __init__(self, *args, guide):
        super().__init__(*args, guide=guide)
        self.meter.counts['pixel_bounds_duplicates_removed'] = 0

    @diagnosed('coverage.outside_pixel_recheck')
    def _outside_pixels(self):
        if self.domain.ledger is None:return
        self.meter.phase = 'outside_pixel_recheck'
        numbers = self.nodes.unknown_numbers(self.meter.touch)
        roots = np.unique(self.nodes.roots[numbers])
        outside = np.zeros(len(self.domain.areas), bool)
        try:
            for key, group in self._box_groups(roots).items():
                candidates = np.asarray(self._candidates_in_box(key), np.int64)
                radii = np.asarray([self.seeds[i]['inner'] for i in candidates])
                low = self.prism_bounds[0][group].min(axis=0)[None,:]
                high = self.prism_bounds[1][group].max(axis=0)[None,:]
                rejected = np.zeros(len(candidates),bool)
                for offset in range(0,len(candidates),RESIDUAL_POLICY['outside_pair_batch']):
                    current = candidates[offset:offset+RESIDUAL_POLICY['outside_pair_batch']]
                    self.meter.touch_many(len(current),'comparisons')
                    self.meter.counts['root_group_ball_comparisons'] += len(current)
                    rejected[offset:offset+len(current)] = _outside_pairs(low,high,
                        self.centers_low[current],self.centers_high[current],radii[offset:offset+len(current)])[0]
                self.meter.counts['root_group_candidate_pairs_pruned'] += len(group)*int(rejected.sum())
                candidates,radii = candidates[~rejected],radii[~rejected]
                if not len(candidates):
                    outside[group] = True; continue
                representatives, members = identical_boxes(group, self.prism_bounds, self.meter.touch)
                self.meter.counts['pixel_bounds_duplicates_removed'] += len(group)-len(representatives)
                size = min(RESIDUAL_POLICY['outside_pixel_batch'], max(1, RESIDUAL_POLICY['outside_pair_batch']//max(1,len(candidates))))
                for start in range(0, len(representatives), size):
                    self.meter.touch()
                    batch = representatives[start:start+size]
                    low, high = self.prism_bounds[0][batch], self.prism_bounds[1][batch]
                    survivors = [[] for _ in batch]
                    stride = max(1, RESIDUAL_POLICY['outside_pair_batch']//len(batch))
                    for offset in range(0, len(candidates), stride):
                        current = candidates[offset:offset+stride]
                        self.meter.touch_many(len(batch)*len(current), 'comparisons')
                        self.meter.counts['pixel_ball_comparisons'] += len(batch)*len(current)
                        rejected = _outside_pairs(low, high, self.centers_low[current], self.centers_high[current],
                                                  radii[offset:offset+stride])
                        for row, mask in zip(survivors, rejected):row.extend(current[~mask].tolist())
                    # Only after all current candidates are checked can this
                    # root's survivors be used by the descendants below.
                    for root, remaining in zip(batch, survivors):
                        for member in members[root]:
                            self.meter.touch()
                            if remaining:self._root_candidates[member] = tuple(remaining)
                            else:outside[member] = True
        finally:
            self.nodes.mark_outside(outside)
            self.meter.counts['outside_pixel_roots_rechecked'] = int(outside.sum())
