"""Equivalent P2/P3 leaf checks with request-local, immutable search inputs.

The delivered v4 evaluator remains the reference. Preparation stores candidate
IDs/coordinates, never a containment, exclusion or connectivity decision. Visit
order, every meter boundary and all directed/exact predicates remain the same.
"""
from __future__ import annotations

import numpy as np

from femto.geometry import exact_surface as exact
from femto.planning.coverage_domain import construction_cache
from femto.planning.coverage_residual import ResidualEngine, _outside_pairs
from femto.planning.coverage_schedule import LayeredEngine, POLICY as LAYERED_POLICY
from femto.resource_policy import preparation_resources


class PreparedResidualEngine(ResidualEngine):
    def __init__(self, *args, guide):
        super().__init__(*args, guide=guide)
        self.preparation_resources = preparation_resources(len(self.domain.source_faces) + len(self.records))
        self._candidate_inputs = construction_cache(self._prepare_candidates,
            maxsize=self.preparation_resources.candidate_cache_entries)

    def seed(self):
        super().seed()
        self._radii = np.asarray([seed['inner'] for seed in self.seeds], float)
        self._radii.setflags(write=False)
        self._candidate_inputs.cache_clear()

    def _prepare_candidates(self, face, selection):
        candidates = LayeredEngine.candidates_for(self, face)
        if selection is not None:
            candidates = tuple(seed for index, seed in enumerate(candidates) if (selection >> index) & 1)
        return self._candidate_arrays(candidates)

    def _candidate_arrays(self, candidates):
        numbers = np.asarray(candidates, np.int64)
        points = self.points[numbers]
        numbers.setflags(write=False); points.setflags(write=False)
        return numbers, points

    def _visit(self, number):
        node = self.nodes[number]
        if node.status == 3:return
        self.meter.touch()
        if self._checked_integer_root(node) is None:return
        previous = node.status, node.good
        try:
            self.meter.counts['new_scan_integer_leaves'] += 1
            face = int(self.domain.source_faces[node.root])
            triangle, bounds = self._geometry(number)
            minimum, maximum = bounds[0].min(axis=0), bounds[1].max(axis=0)
            low = np.nextafter(minimum-self.radius, -np.inf)
            high = np.nextafter(maximum+self.radius, np.inf)
            source_candidates = LayeredEngine.candidates_for(self, face)
            # Keys use at most 128 bits, independent of the total seed count.
            # Capacity is frozen per request, <=16384 entries; dense faces bypass.
            if len(source_candidates) <= 128:
                selection = (None if self.limited_seeds is None else sum(1 << index
                    for index, seed in enumerate(source_candidates) if seed in self.limited_seeds))
                candidates, points = self._candidate_inputs(face, selection)
            else:
                selected = (source_candidates if self.limited_seeds is None else
                    tuple(seed for seed in source_candidates if seed in self.limited_seeds))
                candidates, points = self._candidate_arrays(selected)
            self.meter.counts['spatial_candidates'] += len(candidates)
            candidates = candidates[(points >= low).all(axis=1) & (points <= high).all(axis=1)]
            self.meter.counts['aabb_ball_comparisons'] += len(candidates)
            rejected = _outside_pairs(minimum[None, :], maximum[None, :],
                self.centers_low[candidates], self.centers_high[candidates], self._radii[candidates])[0]
            candidates = candidates[~rejected]
            if not len(candidates):
                node.status = 3; return
            centroid = (bounds[0]/2+bounds[1]/2).mean(axis=0)
            delta = self.points[candidates]-centroid
            distance = np.sqrt((delta*delta).sum(axis=1))
            order = np.lexsort((candidates, distance)); cap = LAYERED_POLICY['max_cell_ball_candidates']
            self.meter.counts['cell_candidates_deferred'] += max(0, len(candidates)-cap)
            candidates = candidates[order[:cap]]
            for _ in candidates:self.meter.touch('comparisons')
            contained, fallbacks = exact.contains_balls(triangle, tuple(self.seeds[i]['center'] for i in candidates),
                self._radii[candidates], triangle_bounds=bounds,
                center_bounds=(self.centers_low[candidates], self.centers_high[candidates]))
            self.meter.counts['exact_fallbacks'] += fallbacks
            node.good = tuple(int(i) for i in candidates[contained])
            if node.good:
                node.status = 4; self._link(number)
            elif node.depth >= self.budget.max_depth:
                node.status = 5; self.meter.flags.add('depth_limit')
            else:self.pending.append(number)
        finally:
            if previous[0] == 1:
                if self.pending and self.pending[-1] == number:self.pending.pop()
                node.status = 1; node.good = tuple(sorted(set(node.good) | set(previous[1])))
            elif self.limited_seeds is not None and node.status not in (1, 6):node.status = 7
