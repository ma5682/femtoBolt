"""B2 schedule: visit original cells before refinement; connect incrementally.

The B1 evaluator is retained verbatim for old certificates. Candidate ranking
only chooses which sufficient proofs to attempt; it never establishes coverage.
Untried balls or an unfinished frontier cannot turn an unknown cell into a proof.
"""
from collections import defaultdict, deque

import numpy as np

from femto.geometry import exact_surface as exact
from femto.planning.coverage_certificate import _Engine, _BudgetStop, _Node
from femto.planning.coverage_domain import construction_cache
from femto.operation_diagnostics import diagnosed

POLICY = dict(schedule='original_cells_then_breadth_first_refinement',
              connectivity='incremental_positive_subedges_same_ball',
              max_cell_ball_candidates=8, connectivity_batch=128,
              proof_batch=1024, candidate_order='distance_to_cell_centroid_then_seed_id',
              pruning='outward_seed_ball_to_cell_aabb', untested_candidates='unconfirmed')


class LayeredEngine(_Engine):
    def __init__(self, *args):
        super().__init__(*args)
        self.edges = defaultdict(list)
        self.by_node = defaultdict(dict)
        self.proof_queue = deque()
        self.pending = deque()
        # Cache only small immutable ID arrays; no geometric decisions or
        # cross-request state are retained (at most 4096 * 128 int64 values).
        self._candidate_ids = construction_cache(self._make_candidate_ids, maxsize=4096)
        self.meter.counts.update(original_cells_visited=0, cell_candidates_deferred=0, aabb_ball_comparisons=0)

    def _make_candidate_ids(self, face):
        values = np.asarray(self.candidates_for(face), np.int64)
        values.setflags(write=False)
        return values

    def _remember_proof(self, number, seed_id, parent):
        known = self.by_node[number]
        if seed_id in known:return
        witness = self._proof(number, seed_id, parent)
        known[seed_id] = witness
        self.proof_queue.append((number, seed_id, witness))

    def _link(self, number):
        node = self.nodes[number]
        face = int(self.domain.source_faces[node.root])
        ids = tuple(int(i) for i in self.domain.faces[face])
        for a, b in zip(node.bary, node.bary[1:]+node.bary[:1]):
            self.meter.touch()
            edge = exact.edge_interval(face, ids, a, b)
            if edge is None:continue
            key, low, high = edge
            for other_low, other_high, other in self.edges[key]:
                self.meter.touch('adjacencies')
                if other != number and min(high, other_high) > max(low, other_low):
                    self.neighbors[number].add(other)
                    self.neighbors[other].add(number)
            self.edges[key].append((low, high, number))
        for seed_id in node.good:
            seed = self.seeds[seed_id]
            if face == seed['face'] and exact.contains_point(node.bary, seed['bary']):
                self._remember_proof(number, seed_id, -1)
        # An older proof may already have been drained before this cell existed.
        for other in sorted(self.neighbors[number]):
            for seed_id in node.good:
                parent = self.by_node[other].get(seed_id)
                if parent is not None:self._remember_proof(number, seed_id, parent)

    def _drain(self, count=None):
        self.meter.phase = 'connectivity'
        processed = 0
        while self.proof_queue and (count is None or processed < count):
            self.meter.touch()
            number, seed_id, parent = self.proof_queue.popleft()
            for other in sorted(self.neighbors[number]):
                if seed_id in self.nodes[other].good:
                    self._remember_proof(other, seed_id, parent)
            processed += 1

    def _visit(self, number):
        self.meter.touch()
        node = self.nodes[number]
        face = int(self.domain.source_faces[node.root])
        tri, bounds = self.domain.cell(face, node.bary)
        minimum, maximum = bounds[0].min(axis=0), bounds[1].max(axis=0)
        low = np.nextafter(minimum-self.radius, -np.inf)
        high = np.nextafter(maximum+self.radius, np.inf)
        ids = self.candidates_for(face)
        candidates = (self._candidate_ids(face) if len(ids) <= 128
                      else np.asarray(ids, np.int64))
        self.meter.counts['spatial_candidates'] += len(candidates)
        candidates = candidates[np.all(self.points[candidates] >= low, axis=1)
                                & np.all(self.points[candidates] <= high, axis=1)]
        # With no ball inside the conservative search box, the later AABB
        # filter is necessarily empty too. Keep its counter unchanged and
        # avoid allocating several empty intermediate arrays for this cell.
        if not len(candidates):
            node.status = 3; node.bary = None; return
        # A lower bound on distance to the entire cell AABB. Round every
        # subtraction, square and addition outwards, including underflow.
        self.meter.counts['aabb_ball_comparisons'] += len(candidates)
        delta = np.maximum(0, np.maximum(
            np.nextafter(minimum-self.centers_high[candidates], -np.inf),
            np.nextafter(self.centers_low[candidates]-maximum, -np.inf)))
        squared = np.maximum(0, np.nextafter(delta*delta, -np.inf))
        lower = np.nextafter(np.nextafter(squared[:, 0]+squared[:, 1], -np.inf)+squared[:, 2], -np.inf)
        radii = np.asarray([self.seeds[i]['inner'] for i in candidates])
        candidates = candidates[lower <= np.nextafter(radii*radii, np.inf)]
        if not len(candidates):
            node.status = 3; node.bary = None; return
        # Ranking is deliberately approximate. Every selected ball still passes
        # the exact/interval containment test and a same-ball connectivity proof.
        centroid = (bounds[0]/2+bounds[1]/2).mean(axis=0)
        distance = np.linalg.norm(self.points[candidates]-centroid, axis=1)
        order = np.lexsort((candidates, distance))
        cap = POLICY['max_cell_ball_candidates']
        self.meter.counts['cell_candidates_deferred'] += max(0, len(candidates)-cap)
        candidates = candidates[order[:cap]]
        for _ in candidates:self.meter.touch('comparisons')
        contained, fallbacks = exact.contains_balls(tri, tuple(self.seeds[i]['center'] for i in candidates),
            [self.seeds[i]['inner'] for i in candidates], triangle_bounds=bounds,
            center_bounds=(self.centers_low[candidates], self.centers_high[candidates]))
        self.meter.counts['exact_fallbacks'] += fallbacks
        node.good = tuple(int(i) for i in candidates[contained])
        if node.good:
            node.status = 4
            self._link(number)
        elif node.depth >= self.budget.max_depth:
            node.status = 5; node.bary = None; self.meter.flags.add('depth_limit')
        else:
            self.pending.append(number)

    @diagnosed('coverage.layered_cells')
    def cells(self):
        for root in range(len(self.domain.areas)):
            self.meter.phase = 'original_cells'
            self.meter.touch()
            self.meter.counts['original_cells_visited'] += 1
            if self.domain.quality[root]:continue
            face = int(self.domain.source_faces[root])
            if not self.candidates_for(face):self.nodes[root].status = 3; continue
            bary = self.domain.root(root)
            if bary is None:self.nodes[root].status = 6; continue
            self.meter.counts['roots_reconstructed'] += 1
            self.nodes[root].bary = bary
            self._visit(root)
            if root % POLICY['connectivity_batch'] == 0:self._drain(POLICY['proof_batch'])
        self._drain()
        while self.pending:
            self.meter.phase = 'refinement'
            self.meter.touch()
            number = self.pending.popleft()
            node = self.nodes[number]
            if self.meter.counts['added_cells']+3 > self.budget.max_added_cells:
                self.meter.flags.add('cell_budget')
                # All pending nodes are still active leaves with original weights.
                break
            first = len(self.nodes)
            self.nodes.extend(_Node(node.root, node.depth+1, 4*node.code+child, bary=part)
                              for child, part in enumerate(exact.subdivide(node.bary)))
            node.active = False; node.bary = None
            self.meter.counts['added_cells'] += 3
            for child in range(first, first+4):self._visit(child)
            self._drain(POLICY['proof_batch'])
        self._drain()

    def run(self):
        self.meter.check()
        try:
            self.seed(); self.cells()
        except _BudgetStop:
            pass
        self.meter.check()
