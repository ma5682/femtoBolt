"""P3 v5: bounded interleaving of current exclusions and new-ball search.

All geometric predicates come from the delivered v4 implementation. Only work
ordering changes: reprove the whole retained set first, distribute the anchor
prefix across new seeds, then alternate small complete exclusion/search batches.
No clock-based per-stage allowance or saved classification grants a proof.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from femto.geometry import exact_surface as exact
from femto.operation_diagnostics import diagnosed
from femto.planning.coverage_residual import POLICY as V4_POLICY, _outside_pairs
from femto.planning.coverage_prepared import PreparedResidualEngine
from femto.planning.coverage_schedule import LayeredEngine, POLICY as LAYERED_POLICY
from femto.planning.planning_revision import digest

POLICY = dict(V4_POLICY,
    schedule='reprove_then_balanced_anchor_prefix_then_current_roots_then_interleaved_exclusions_and_new_scan',
    new_scan_order='nearest_new_seed_groups_round_robin_then_anchor_face_root_distance_leaf_id',
    new_seed_group_chunk=8, search_leaf_batch=32, residual_batches_per_cycle=8,
    interleave='one_new_scan_batch_then_eight_complete_residual_batches_until_both_exhausted',
    outside_group_order='descending_current_unconfirmed_nominal_area_then_original_group_order',
    residual_order='descending_current_unconfirmed_root_area_then_root_index_then_candidate_group_then_leaf_area',
    residual_live_filter='recheck_current_unconfirmed_active_leaves_before_each_batch',
    remaining_schedule='all_current_seeds_after_interleaved_search_and_exclusions')


def checked_policy(value: dict) -> dict:
    if not isinstance(value, dict) or value != POLICY or digest(value) != digest(POLICY):
        raise ValueError('交替覆盖调度策略无效或版本不一致')
    return value


class InterleavedEngine(PreparedResidualEngine):
    def __init__(self, *args, guide):
        super().__init__(*args, guide=guide)
        self.meter.counts.update(interleave_cycles=0, interleaved_search_batches=0,
            interleaved_residual_batches=0, interleaved_already_confirmed_skips=0,
            new_seed_groups=0)

    def _unknown_areas(self, numbers):
        # These weights order work only. Status still comes exclusively from
        # current complete geometric checks, never from the hint's old flags.
        self.meter.touch()
        weights = np.ldexp(np.ones(len(numbers)), -2*self.nodes.depths[numbers].astype(np.int64))
        areas = np.bincount(self.nodes.roots[numbers], weights=weights,
                            minlength=len(self.domain.areas))*self.domain.areas
        self.meter.touch()
        return areas

    def _box_groups(self, roots):
        groups = super()._box_groups(roots)
        areas = self._unknown_areas(self.nodes.unknown_numbers(self.meter.touch))
        ordered = []
        for index, (key, group) in enumerate(groups.items()):
            self.meter.touch()
            ordered.append((-float(areas[group].sum()), index, key, group))
        ordered.sort(key=lambda item: item[:2])
        return {key: group for _, _, key, group in ordered}

    @diagnosed('coverage.new_scan_order')
    def _new_order(self, *, anchor_only=False):
        if not self.new_seeds:return np.empty(0, np.int64)
        self.meter.phase = 'new_scan_order'; self.meter.touch()
        anchor_faces = np.zeros(len(self.domain.faces), bool)
        anchor_faces[[self.seeds[i]['face'] for i in self.new_seeds]] = True
        relevant = anchor_faces.copy()
        faces = np.empty(0, np.int64) if anchor_only else np.unique(self.domain.source_faces)
        for start in range(0, len(faces), 128):
            self.meter.touch()
            batch = faces[start:start+128]
            triangles = self.domain.vertices[self.domain.faces[batch]]
            low = np.nextafter(triangles.min(axis=1)-self.radius, -np.inf)
            high = np.nextafter(triangles.max(axis=1)+self.radius, np.inf)
            selected = np.zeros(len(batch), bool)
            for seed in self.new_seeds:
                self.meter.touch_many(len(batch), 'comparisons')
                selected |= (np.all(self.seeds[seed]['point'] >= low, axis=1)
                             & np.all(self.seeds[seed]['point'] <= high, axis=1))
            relevant[batch[selected]] = True
        root_indices = np.r_[self.nodes.roots, np.asarray([n.root for n in self.nodes.extra], np.int64)]
        faces_by_leaf = self.domain.source_faces[root_indices]
        eligible = relevant[faces_by_leaf] & ~np.isin(self.nodes.root_status[root_indices], (2, 3))
        if self._new_visited:eligible[list(self._new_visited)] = False
        numbers = np.flatnonzero(eligible); roots = root_indices[numbers]
        if self.prism_bounds is None:
            centers = self.domain.vertices[self.domain.faces[self.domain.source_faces[roots]]].mean(axis=1)
        else:centers = self.prism_bounds[0][roots]/2+self.prism_bounds[1][roots]/2
        distances = np.full(len(numbers), np.inf); nearest = np.zeros(len(numbers), np.int64)
        for position, seed in enumerate(self.new_seeds):
            self.meter.touch()
            with np.errstate(over='ignore', invalid='ignore'):
                delta = centers-self.seeds[seed]['point']
                values = np.einsum('ij,ij->i', delta, delta)
            better = values < distances
            nearest[better] = position; distances[better] = values[better]
        groups = []
        for position in np.unique(nearest):
            self.meter.touch()
            selected = np.flatnonzero(nearest == position)
            seed = self.new_seeds[int(position)]
            order = np.lexsort((numbers[selected], distances[selected],
                                faces_by_leaf[numbers[selected]] != self.seeds[seed]['face']))
            groups.append(numbers[selected[order]])
        self.meter.counts['new_seed_groups'] += len(groups)
        chunks = []
        step = POLICY['new_seed_group_chunk']
        for start in range(0, max((len(g) for g in groups), default=0), step):
            self.meter.touch()
            chunks.extend(group[start:start+step] for group in groups if start < len(group))
        return np.concatenate(chunks) if chunks else np.empty(0, np.int64)

    @diagnosed('coverage.new_scan_batch')
    def _search_batch(self, numbers, *, anchor_only=False):
        self.limited_seeds = set(self.new_seeds)
        try:
            for value in numbers:
                self.meter.phase = 'new_anchor_prefix' if anchor_only else 'new_scan_neighborhoods'
                self.meter.touch()
                number = int(value); node = self.nodes[number]
                if not node.active or node.status in (2, 3, 6):continue
                self.meter.counts['new_scan_leaves_visited'] += 1
                if anchor_only:self.meter.counts['new_anchor_leaves_visited'] += 1
                self._visit(number); self._new_visited.add(number)
            self._drain(LAYERED_POLICY['proof_batch'])
            self.meter.counts['interleaved_search_batches'] += 1
        finally:self.limited_seeds = None

    def _search_batches(self):
        if not self.new_seeds:return
        numbers = self._new_order()
        for start in range(0, len(numbers), POLICY['search_leaf_batch']):
            self._search_batch(numbers[start:start+POLICY['search_leaf_batch']])
            yield
        self.limited_seeds = set(self.new_seeds)
        try:self._drain(); self._subdivide()
        finally:self.limited_seeds = None

    def _outside_batches(self):
        # Same outward bounds, current all-ball survivors and batch ceilings as
        # v4. Search may confirm a queued leaf between batches, so filter again.
        self.meter.phase = 'outside_leaf_recheck'
        numbers = self.nodes.unknown_numbers(self.meter.touch)
        numbers = numbers[np.argsort(self.nodes.roots[numbers], kind='stable')]
        roots, starts, counts = np.unique(self.nodes.roots[numbers], return_index=True, return_counts=True)
        areas = self._unknown_areas(numbers)
        root_order = np.lexsort((roots, -areas[roots]))
        for start in range(0, len(roots), POLICY['residual_root_batch']):
            groups = defaultdict(list); lows, highs = [], []
            for index in root_order[start:start+POLICY['residual_root_batch']]:
                self.meter.phase = 'outside_leaf_recheck'; self.meter.touch()
                root = int(roots[index]); face = int(self.domain.source_faces[root])
                integer_root = self._integer_root_cache(root)
                if integer_root is None:
                    self.meter.counts['residual_unconstructed_roots'] += 1; continue
                if root not in self._reconstructed:
                    self._reconstructed.add(root); self.meter.counts['roots_reconstructed'] += 1
                bounds = self._root_vertices(root)
                self.meter.counts['residual_roots_reconstructed'] += 1
                candidates = self._root_candidates.get(root)
                if candidates is None:candidates = LayeredEngine.candidates_for(self, face)
                else:self.meter.counts['residual_current_root_candidates'] += len(candidates)
                rows = numbers[starts[index]:starts[index]+counts[index]]
                groups[tuple(candidates)].append((len(lows), rows))
                lows.append(bounds[0]); highs.append(bounds[1])
            lows, highs = np.asarray(lows), np.asarray(highs)
            for candidates, pieces in groups.items():
                rows = np.concatenate([part for _, part in pieces])
                positions = np.repeat([position for position, _ in pieces], [len(part) for _, part in pieces])
                leaf_areas = np.ldexp(self.domain.areas[self.nodes.roots[rows]],
                                     -2*self.nodes.depths[rows].astype(np.int64))
                order = np.lexsort((rows, -leaf_areas))
                rows, positions = rows[order], positions[order]
                candidates = np.asarray(candidates, np.int64)
                radii = np.asarray([self.seeds[i]['inner'] for i in candidates])
                for offset in range(0, len(rows), POLICY['residual_leaf_batch']):
                    self.meter.phase = 'outside_leaf_recheck'; self.meter.touch()
                    batch = rows[offset:offset+POLICY['residual_leaf_batch']]
                    root_positions = positions[offset:offset+POLICY['residual_leaf_batch']]
                    active = np.asarray([number not in self.nodes.touched or
                        (self.nodes.touched[int(number)].active and self.nodes.touched[int(number)].status == 7)
                        for number in batch], bool)
                    self.meter.counts['interleaved_already_confirmed_skips'] += int((~active).sum())
                    batch, root_positions = batch[active], root_positions[active]
                    if not len(batch):continue
                    low, high = exact.descendant_boxes(lows[root_positions], highs[root_positions],
                        self.nodes.depths[batch], self.nodes.codes[batch])
                    excluded = np.ones(len(batch), bool)
                    for index, current in enumerate(candidates):
                        selected = np.flatnonzero(excluded)
                        if not len(selected):break
                        self.meter.touch_many(len(selected), 'comparisons')
                        self.meter.counts['residual_ball_comparisons'] += len(selected)
                        rejected = _outside_pairs(low[selected], high[selected], self.centers_low[current:current+1],
                            self.centers_high[current:current+1], radii[index:index+1])[:, 0]
                        excluded[selected] = rejected
                    self.nodes.mark_leaves_outside(batch[excluded])
                    self.meter.counts['residual_leaves_rechecked'] += len(batch)
                    self.meter.counts['residual_outside_leaves'] += int(excluded.sum())
                    self.meter.counts['residual_short_circuited_leaves'] += int((~excluded).sum())
                    self.meter.counts['residual_batches_completed'] += 1
                    self.meter.counts['interleaved_residual_batches'] += 1
                    yield

    @diagnosed('coverage.interleaved_search')
    def _interleave(self):
        searches, exclusions = iter(self._search_batches()), iter(self._outside_batches())
        active_search = active_exclusion = True
        while active_search or active_exclusion:
            self.meter.phase = 'interleaved_search'; self.meter.touch()
            self.meter.counts['interleave_cycles'] += 1
            if active_search:
                try:next(searches)
                except StopIteration:active_search = False
            if active_exclusion:
                for _ in range(POLICY['residual_batches_per_cycle']):
                    try:next(exclusions)
                    except StopIteration:
                        active_exclusion = False; break

    def cells(self):
        if not self._prepare():return
        self._pixel_bounds()
        if not self._reprove():return
        prefix = self._new_order(anchor_only=True)[:POLICY['new_anchor_leaf_budget']]
        for start in range(0, len(prefix), POLICY['search_leaf_batch']):
            self._search_batch(prefix[start:start+POLICY['search_leaf_batch']], anchor_only=True)
        self._outside_faces(); self._outside_pixels()
        self._interleave()
        self.pending.clear()
        for number in range(len(self.nodes)):
            self.meter.phase = 'remaining_original_cells'; self.meter.touch()
            node = self.nodes[number]
            if not node.active or node.status in (1, 2, 3, 6):continue
            self.meter.counts['original_cells_visited'] += 1
            self._visit(number)
            if number % LAYERED_POLICY['connectivity_batch'] == 0:self._drain(LAYERED_POLICY['proof_batch'])
        self._drain(); self._subdivide()
