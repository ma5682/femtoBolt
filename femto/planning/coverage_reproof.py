"""Recompute a guided partition from current geometry and current scan balls.

The previous certificate supplies only a partition and candidate proof paths.
Every used ball, containment and positive subedge is checked anew. No confirmed
status, old proof or old array is installed without these checks. The complete
original target is retained even if the budget stops during guide preparation.
"""
from collections import defaultdict
import heapq
import math

import numpy as np

from femto.geometry import exact_surface as exact
from femto.planning.coverage_certificate import _BudgetStop, _Node, REPROOF_ALGORITHM as ALGORITHM
from femto.planning.coverage_domain import construction_cache
from femto.planning.coverage_schedule import LayeredEngine, POLICY as LAYERED_POLICY
from femto.operation_diagnostics import diagnosed

POLICY = dict(schedule='recompute_guided_partition_then_anchor_prefix_then_exclusions_and_new_scan_neighborhoods',
              witnesses='greedy_unclaimed_leaf_seed_groups_then_parent_closure',
              proof_batch=128, geometric_cache_entries=8192,
              guide_semantics='partition_and_candidate_paths_only_all_claims_rechecked',
              outside='current_all_ball_face_then_pixel_prism_aabb_lower_bound',
              outside_pixel_batch=128, outside_pair_batch=4096,
              outside_index='current_seed_grid_grouped_by_outward_pixel_box', outside_max_grid_bins=4096,
              batch_accounting='every_pair_charged_checks_at_most_32_entries_apart',
              containment_filter='current_pixel_prism_then_exact_cell_intervals',
              pixel_bounds='outward_source_plane_intersected_ray_prism_or_original_box_on_ambiguity',
              shared_edge_filter='two_identical_current_vertices_in_same_source_face_then_exact_intervals',
              new_scan_order='anchor_faces_then_root_box_distance_then_leaf_index',
              new_anchor_leaf_budget=128,
              interrupted_visit='keep_current_rechecked_witness_and_leave_partial_new_search_unconfirmed',
              unfinished='retain_all_original_weights_as_unconfirmed',
              remaining_schedule='layered-anchor-ball-subedge-coverage-v2')


def checked_policy(value):
    from femto.planning.planning_revision import digest
    if not isinstance(value, dict) or value != POLICY or digest(value) != digest(POLICY):
        raise ValueError('覆盖复算调度策略无效或版本不一致')
    return value


class _Partition:
    """Whole immutable input partition; allocate mutable nodes only when used.

    Only root/depth/code are taken from the guide. Every initial status comes
    from the current quality mask, and every witness starts absent.
    """
    def __init__(self, roots, depths, codes, quality):
        self.roots, self.depths, self.codes, self.quality = roots, depths, codes, quality
        self.count = len(roots)
        self.root_status = np.where(quality, 2, 7).astype('u1')
        self.touched = {}
        self.extra = []

    def __len__(self):return self.count+len(self.extra)

    def root(self, number):
        return int(self.roots[number]) if number < self.count else self.extra[number-self.count].root

    def __getitem__(self, number):
        if number < 0:raise IndexError(number)
        if number >= self.count:return self.extra[number-self.count]
        node = self.touched.get(number)
        if node is None:
            root = int(self.roots[number])
            node = _Node(root, int(self.depths[number]), int(self.codes[number]), status=int(self.root_status[root]))
            self.touched[number] = node
        return node

    def extend(self, nodes):self.extra.extend(nodes)

    def mark_outside(self, roots):
        self.root_status[roots] = 3
        for node in self.touched.values():
            if roots[node.root] and node.status == 7:node.status = 3

    def leaf_arrays(self):
        size = len(self)
        values = dict(root=np.empty(size, '<i8'), depth=np.empty(size, 'u1'), code=np.empty(size, '<u4'),
                      status=np.full(size, 7, 'u1'), witness=np.full(size, -1, '<i8'))
        count = len(self.roots)
        values['root'][:count], values['depth'][:count], values['code'][:count] = self.roots, self.depths, self.codes
        values['status'][:count] = self.root_status[self.roots]
        active = np.ones(size, bool)
        for number, node in self.touched.items():
            active[number] = node.active
            values['status'][number], values['witness'][number] = node.status, node.witness
        for number, node in enumerate(self.extra, count):
            active[number] = node.active
            for key in values:values[key][number] = getattr(node, key)
        order = np.flatnonzero(active)
        order = order[np.lexsort((values['depth'][order],
            values['code'][order]/np.power(4., values['depth'][order]), values['root'][order]))]
        remap = np.full(size, -1, '<i8'); remap[order] = np.arange(len(order))
        return {key:value[order] for key,value in values.items()}, remap


class ReproofEngine(LayeredEngine):
    def _initial_partition(self):
        count = len(self.domain.areas)
        return _Partition(np.arange(count, dtype='<i8'), np.zeros(count, 'u1'),
                          np.zeros(count, '<u4'), self.domain.quality)

    def leaf_arrays(self):return self.nodes.leaf_arrays()

    def __init__(self, *args, guide):
        super().__init__(*args)
        self.guide = guide
        self.limited_seeds = None
        self._root_cache = construction_cache(self.domain.root, maxsize=4096)
        self._integer_root_cache = construction_cache(self._make_integer_root, maxsize=4096)
        cap = POLICY['geometric_cache_entries']
        self._geometry = construction_cache(self._make_geometry, maxsize=cap)
        self._integer_bary = construction_cache(self._make_integer_bary, maxsize=cap)
        self._edges = construction_cache(self._make_edges, maxsize=cap)
        self._shared = construction_cache(self._shared_subedge, maxsize=cap)
        self._reconstructed = set()
        self._retained_flags = None
        self.prism_bounds = None
        self._new_visited = set()
        self.meter.counts.update(guided_leaf_count=0, hint_proofs_requested=0,
            hint_proofs_rechecked=0, retained_leaves_rechecked=0,
            unbound_hint_witnesses=0, hint_containment_failures=0, hint_connectivity_failures=0,
            outside_roots_rechecked=0, whole_face_ball_comparisons=0,
            outside_pixel_roots_rechecked=0, pixel_ball_comparisons=0, prism_contained_pairs=0,
            outside_index_bins_visited=0, new_scan_leaves_visited=0, new_anchor_leaves_visited=0,
            shared_endpoint_pairs=0)

    @diagnosed('coverage.guide_prepare')
    def _prepare(self):
        from femto.planning.coverage_partition import intervals
        a = self.guide.arrays
        self.meter.phase = 'guided_partition'
        intervals(self.guide, len(self.domain.areas), self.meter.touch)
        if np.any(a['depth'] > self.budget.max_depth):
            raise ValueError('引导分区深度超过本次覆盖预算')
        extra = len(a['root'])-len(self.domain.areas)
        if extra > self.budget.max_added_cells:
            self.meter.flags.add('cell_budget')
            return False
        count = len(a['proof_leaf'])
        parents, leaves, seeds = a['proof_parent'], a['proof_leaf'], a['proof_seed']
        parent_mask = parents >= 0
        if (np.any(leaves < 0) or np.any(leaves >= len(a['root']))
                or np.any(seeds < 0) or np.any(seeds >= len(a['seed_face']))
                or np.any(parents < -1) or np.any(parents >= np.arange(count))
                or np.any(seeds[parent_mask] != seeds[parents[parent_mask]])):
            raise ValueError('引导证明的索引、同球依赖或父节点顺序无效')
        confirmed = a['status'] == 1
        witness = a['witness'][confirmed]
        if (np.any(a['witness'] < -1) or np.any((a['witness'] >= 0) != confirmed) or np.any(witness >= count)
                or not np.array_equal(leaves[witness], np.flatnonzero(confirmed))):
            raise ValueError('引导确认单元缺少对应的候选证明')
        for _ in range(0, len(a['root']), 128):self.meter.touch()
        nodes = _Partition(a['root'], a['depth'], a['code'], self.domain.quality)
        # Install the whole partition together; a partial construction cannot
        # drop the remaining original roots on cancellation or budget exit.
        self.nodes = nodes
        self.meter.counts['added_cells'] = extra
        self.meter.counts['guided_leaf_count'] = len(nodes)
        self.old_confirmed = np.flatnonzero(confirmed)
        self._retained_flags = confirmed
        lookup = {(s['face'], *s['point']): i for i, s in enumerate(self.seeds)}
        self.seed_remap = [lookup.get((int(face), *xyz), -1)
                           for face, xyz in zip(a['seed_face'], a['seed_xyz'])]
        represented = {i for i in self.seed_remap if i >= 0}
        self.new_seeds = tuple(i for i in range(len(self.seeds)) if i not in represented)
        self.selected_proofs = self._choose_paths(confirmed)
        self.meter.counts['hint_proofs_requested'] = len(self.selected_proofs)
        return True

    @diagnosed('coverage.pixel_bounds')
    def _pixel_bounds(self):
        if self.domain.ledger is None:return
        self.meter.phase = 'current_pixel_bounds'; self.meter.touch()
        width = self.domain.ledger.arrays['mask'].shape[1]
        pixel = self.domain.ledger.arrays['pixel_index']
        triangles = self.domain.vertices[self.domain.faces[self.domain.source_faces]]
        intrinsic = np.asarray(self.domain.revision['inputs']['calibration']['intrinsic'], float)
        self.prism_bounds = exact.pixel_prism_bounds(triangles, intrinsic,
            np.column_stack((pixel%width, pixel//width)), plane_tightening=True)
        self.meter.touch()

    def _choose_paths(self, confirmed):
        """Choose shorter candidate forests; this choice does not prove a cell."""
        a = self.guide.arrays
        groups = defaultdict(list)
        for i, seed in enumerate(a['proof_seed']):
            if i % 128 == 0:self.meter.touch()
            groups[int(seed)].append(i)
        groups = {key: np.asarray(value, np.int64) for key, value in groups.items()}
        remaining = confirmed.copy()
        wanted = np.zeros(len(a['proof_leaf']), bool)
        heap = [(-int(remaining[a['proof_leaf'][rows]].sum()), key) for key, rows in groups.items()]
        heapq.heapify(heap)
        while heap:
            self.meter.touch()
            estimate, seed = heapq.heappop(heap)
            rows = groups[seed]
            requested = rows[remaining[a['proof_leaf'][rows]]]
            if not len(requested):continue
            if -estimate != len(requested):
                heapq.heappush(heap, (-len(requested), seed)); continue
            for proof in requested:
                while proof >= 0 and not wanted[proof]:
                    self.meter.touch()
                    wanted[proof] = True
                    remaining[a['proof_leaf'][proof]] = False
                    proof = int(a['proof_parent'][proof])
        if remaining.any():raise ValueError('引导确认集合缺少完整候选路径')
        return np.flatnonzero(wanted)

    def _bary(self, number):
        node = self.nodes[number]
        if node.bary is None:
            root = self._root_cache(node.root)
            if root is None:
                node.status = 6
                return None
            if node.root not in self._reconstructed:
                self._reconstructed.add(node.root)
                self.meter.counts['roots_reconstructed'] += 1
            node.bary = exact.descend(root, node.depth, node.code)
        return node.bary

    def _make_geometry(self, number):
        bary = self._integer_bary(number)
        if bary is None:return None
        return self.domain.integer_cell(int(self.domain.source_faces[self.nodes[number].root]), bary)

    def _make_integer_root(self, root):
        triangle = self.domain.integer_root(root)
        return None if triangle is None else exact.IntegerTriangle(triangle)

    def _make_integer_bary(self, number):
        node = self.nodes[number]
        root = self._checked_integer_root(node)
        return None if root is None else root.descend(node.depth,node.code)

    def _checked_integer_root(self, node):
        root = self._integer_root_cache(node.root)
        if root is None:
            node.status = 6
            return None
        if node.root not in self._reconstructed:
            self._reconstructed.add(node.root)
            self.meter.counts['roots_reconstructed'] += 1
        return root

    def _make_edges(self, number):
        bary = self._integer_bary(number)
        if bary is None:return ()
        face = int(self.domain.source_faces[self.nodes[number].root])
        ids = tuple(int(v) for v in self.domain.faces[face])
        return exact.homogeneous_edge_intervals(face, ids, bary)

    def _shared_subedge(self, left, right):
        if left == right:return False
        a, b = self.nodes[left], self.nodes[right]
        if a.root == b.root:
            return exact.descendants_share_subedge(a.depth, a.code, b.depth, b.code)
        if self.domain.source_faces[a.root] == self.domain.source_faces[b.root]:
            aa, bb = self._integer_bary(left), self._integer_bary(right)
            # Two distinct identical vertices give a shared positive-length
            # triangle edge. These are current, exactly reconstructed vertices,
            # and source-face identity prevents welding coincident surfaces.
            if aa is not None and bb is not None and len(set(aa).intersection(bb)) >= 2:
                self.meter.counts['shared_endpoint_pairs'] += 1
                return True
        return any(exact.integer_subedges_overlap(a, b) for a in self._edges(left) for b in self._edges(right))

    def _proof(self, number, seed_id, parent):
        first = self.nodes[number].witness < 0
        witness = super()._proof(number, seed_id, parent)
        if first and self._retained_flags is not None and number < len(self._retained_flags) and self._retained_flags[number]:
            self.meter.counts['retained_leaves_rechecked'] += 1
        return witness

    @diagnosed('coverage.retained_reproof')
    def _reprove(self):
        self.meter.phase = 'retained_proofs'
        a = self.guide.arrays
        remap = np.full(len(a['proof_leaf']), -1, np.int64)
        batch = POLICY['proof_batch']
        for offset in range(0, len(self.selected_proofs), batch):
            rows, seed_ids, roots = [], [], []
            for proof in self.selected_proofs[offset:offset+batch]:
                self.meter.touch('comparisons')
                number = int(a['proof_leaf'][proof])
                seed_id = self.seed_remap[a['proof_seed'][proof]]
                if seed_id < 0:
                    self.meter.counts['unbound_hint_witnesses'] += 1; continue
                if self.domain.quality[self.nodes[number].root]:continue
                # A proved enclosing root prism also encloses every dyadic
                # child. Reconstruct its exact root now; construct the child
                # only if containment, anchoring or an edge test needs it.
                if self._checked_integer_root(self.nodes[number]) is None:continue
                rows.append((int(proof), number, seed_id))
                seed_ids.append(seed_id); roots.append(self.nodes[number].root)
            inside = np.zeros(len(rows), bool)
            if rows and self.prism_bounds is not None:
                inside = exact.boxes_in_balls(self.prism_bounds[0][roots], self.prism_bounds[1][roots],
                    [self.seeds[i]['inner'] for i in seed_ids],
                    center_bounds=(self.centers_low[seed_ids], self.centers_high[seed_ids]))
                self.meter.counts['prism_contained_pairs'] += int(inside.sum())
            pending = np.flatnonzero(~inside)
            triangles, lows, highs, remaining_seeds = [], [], [], []
            for index in pending:
                _, number, seed_id = rows[index]
                tri, bounds = self._geometry(number)
                triangles.append(tri); lows.append(bounds[0]); highs.append(bounds[1]); remaining_seeds.append(seed_id)
            contained, fallbacks = exact.contains_ball_pairs(triangles,
                tuple(self.seeds[i]['center'] for i in remaining_seeds), [self.seeds[i]['inner'] for i in remaining_seeds],
                triangle_bounds=(np.asarray(lows), np.asarray(highs)),
                center_bounds=(self.centers_low[remaining_seeds], self.centers_high[remaining_seeds]))
            inside[pending] = contained
            self.meter.counts['exact_fallbacks'] += fallbacks
            for (proof, number, seed_id), contained in zip(rows, inside):
                self.meter.touch()
                self.meter.counts['hint_proofs_rechecked'] += 1
                if not contained:
                    self.meter.counts['hint_containment_failures'] += 1; continue
                node, seed = self.nodes[number], self.seeds[seed_id]
                parent = int(a['proof_parent'][proof])
                self.meter.touch('adjacencies')
                connected = (int(self.domain.source_faces[node.root]) == seed['face']
                    and exact.integer_contains_point(self._integer_bary(number), seed['bary'])) if parent == -1 else (
                    remap[parent] >= 0 and self._shared(*sorted((number, int(a['proof_leaf'][parent])))))
                if not connected:
                    self.meter.counts['hint_connectivity_failures'] += 1; continue
                if seed_id not in node.good:node.good += (seed_id,)
                known = self.by_node[number].get(seed_id)
                if known is None:
                    known = self._proof(number, seed_id, -1 if parent == -1 else int(remap[parent]))
                    self.by_node[number][seed_id] = known
                remap[proof] = known
        checked = sum(self.nodes[int(i)].witness >= 0 for i in self.old_confirmed)
        self.meter.counts['retained_leaves_rechecked'] = checked
        if checked != len(self.old_confirmed):
            self.meter.flags.add('retained_proofs_unconfirmed')
            return False
        return True

    def candidates_for(self, face):
        candidates = super().candidates_for(face)
        if self.limited_seeds is None:return candidates
        return tuple(i for i in candidates if i in self.limited_seeds)

    def _visit(self, number):
        node = self.nodes[number]
        if self._bary(number) is None:return
        previous = node.status, node.good, node.bary
        try:
            super()._visit(number)
        finally:
            if previous[0] == 1:
                # A current recomputed proof also survives a budget/cancel exit
                # while another ball is linked. This cell is never split.
                if self.pending and self.pending[-1] == number:self.pending.pop()
                node.status, node.bary = 1, previous[2]
                node.good = tuple(sorted(set(node.good) | set(previous[1])))
            elif self.limited_seeds is not None and node.status not in (1, 6):
                # Only the new balls have been searched so far. Do not label
                # this as outside all balls, including on a partial visit.
                node.status = 7

    def _subdivide(self):
        while self.pending:
            self.meter.phase = 'refinement'
            self.meter.touch()
            number = self.pending.popleft(); node = self.nodes[number]
            if node.witness >= 0:continue
            if self.meter.counts['added_cells']+3 > self.budget.max_added_cells:
                self.meter.flags.add('cell_budget'); break
            first = len(self.nodes)
            self.nodes.extend(_Node(node.root, node.depth+1, 4*node.code+child, bary=part)
                              for child, part in enumerate(exact.subdivide(node.bary)))
            node.active = False; node.bary = None
            self.meter.counts['added_cells'] += 3
            for child in range(first, first+4):self._visit(child)
            self._drain(LAYERED_POLICY['proof_batch'])
        self._drain()

    @diagnosed('coverage.outside_recheck')
    def _outside_faces(self):
        """Re-establish outside status using all current balls, never old flags."""
        self.meter.phase = 'outside_recheck'
        outside = np.zeros(len(self.domain.faces), bool)
        try:
            for face in np.unique(self.domain.source_faces):
                self.meter.touch()
                face = int(face)
                candidates = np.asarray(super().candidates_for(face), np.int64)
                self.meter.touch_many(len(candidates), 'comparisons')
                self.meter.counts['whole_face_ball_comparisons'] += len(candidates)
                if not len(candidates):
                    outside[face] = True; continue
                triangle = self.domain.vertices[self.domain.faces[face]]
                delta = np.maximum(0, np.maximum(
                    np.nextafter(triangle.min(axis=0)-self.centers_high[candidates], -np.inf),
                    np.nextafter(self.centers_low[candidates]-triangle.max(axis=0), -np.inf)))
                squared = np.maximum(0, np.nextafter(delta*delta, -np.inf))
                lower = np.nextafter(np.nextafter(squared[:,0]+squared[:,1], -np.inf)+squared[:,2], -np.inf)
                radii = np.asarray([self.seeds[i]['inner'] for i in candidates])
                outside[face] = bool(np.all(lower > np.nextafter(radii*radii, np.inf)))
        finally:
            # Even a budget exit installs only fully completed exclusion tests.
            roots = outside[self.domain.source_faces] & ~self.domain.quality.astype(bool)
            self.nodes.mark_outside(roots)
            self.meter.counts['outside_roots_rechecked'] = int(roots.sum())

    @diagnosed('coverage.outside_pixel_recheck')
    def _outside_pixels(self):
        if self.domain.ledger is None:return
        self.meter.phase = 'outside_pixel_recheck'
        has_confirmation = np.zeros(len(self.domain.areas), bool)
        has_confirmation[self.guide.arrays['root'][self.old_confirmed]] = True
        roots = np.flatnonzero((self.nodes.root_status == 7) & ~has_confirmation)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            low_bins = np.floor(np.nextafter(self.prism_bounds[0][roots]-self.radius, -np.inf)/self.radius)
            high_bins = np.floor(np.nextafter(self.prism_bounds[1][roots]+self.radius, np.inf)/self.radius)
        valid = (np.isfinite(low_bins).all(axis=1) & np.isfinite(high_bins).all(axis=1)
                 & (np.abs(low_bins) < 2.**60).all(axis=1) & (np.abs(high_bins) < 2.**60).all(axis=1))
        groups = defaultdict(list)
        for index, root in enumerate(roots):
            if index % 128 == 0:self.meter.touch()
            key = (None, int(self.domain.source_faces[root]))
            if valid[index]:
                low, high = tuple(map(int, low_bins[index])), tuple(map(int, high_bins[index]))
                if math.prod(b-a+1 for a,b in zip(low,high)) <= POLICY['outside_max_grid_bins']:
                    key = (*low, *high)
            groups[key].append(int(root))
        outside = np.zeros(len(self.domain.areas), bool)
        try:
            for key, group in groups.items():
                if key[0] is None:
                    found = super().candidates_for(key[1])
                else:
                    found = []
                    for x in range(key[0],key[3]+1):
                        for y in range(key[1],key[4]+1):
                            for z in range(key[2],key[5]+1):
                                self.meter.touch(); self.meter.counts['outside_index_bins_visited'] += 1
                                found.extend(self.grid.get((x,y,z), ()))
                    found.sort()
                candidates = np.asarray(found, np.int64)
                radii = np.asarray([self.seeds[i]['inner'] for i in candidates])
                size = min(POLICY['outside_pixel_batch'], max(1, POLICY['outside_pair_batch']//max(1, len(candidates))))
                for start in range(0, len(group), size):
                    self.meter.touch()
                    batch = group[start:start+size]
                    low, high = self.prism_bounds[0][batch], self.prism_bounds[1][batch]
                    excluded = np.ones(len(batch), bool)
                    stride = max(1, POLICY['outside_pair_batch']//len(batch))
                    for offset in range(0, len(candidates), stride):
                        current = candidates[offset:offset+stride]
                        self.meter.touch_many(len(batch)*len(current), 'comparisons')
                        self.meter.counts['pixel_ball_comparisons'] += len(batch)*len(current)
                        delta = np.maximum(0, np.maximum(
                            np.nextafter(low[:,None,:]-self.centers_high[current], -np.inf),
                            np.nextafter(self.centers_low[current]-high[:,None,:], -np.inf)))
                        squared = np.maximum(0, np.nextafter(delta*delta, -np.inf))
                        lower = np.nextafter(np.nextafter(squared[:,:,0]+squared[:,:,1], -np.inf)+squared[:,:,2], -np.inf)
                        radius = radii[offset:offset+stride]
                        excluded &= np.all(lower > np.nextafter(radius*radius, np.inf), axis=1)
                    outside[batch] = excluded
        finally:
            self.nodes.mark_outside(outside)
            self.meter.counts['outside_pixel_roots_rechecked'] = int(outside.sum())

    @diagnosed('coverage.new_scan_search')
    def _new_neighborhoods(self, *, anchor_only=False):
        if not self.new_seeds:return
        self.limited_seeds = set(self.new_seeds)
        self.meter.phase = 'new_anchor_prefix' if anchor_only else 'new_scan_neighborhoods'
        anchor_faces = np.zeros(len(self.domain.faces), bool)
        anchor_faces[[self.seeds[i]['face'] for i in self.new_seeds]] = True
        relevant = set(np.flatnonzero(anchor_faces))
        # A whole-face outer AABB only ranks/selects work. Exact containment
        # and source-edge topology still decide every confirmation below.
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
                selected |= np.all(self.seeds[seed]['point'] >= low, axis=1) & np.all(self.seeds[seed]['point'] <= high, axis=1)
            relevant.update(int(face) for face in batch[selected])
        # Start near current new anchors. Global root order may otherwise spend
        # the entire remaining budget building cells before any anchor arrives.
        # This ranks work only; every ball/edge/anchor predicate is still run.
        root_indices = np.r_[self.nodes.roots, np.asarray([n.root for n in self.nodes.extra], np.int64)]
        faces_by_leaf = self.domain.source_faces[root_indices]
        flags = np.zeros(len(self.domain.faces), bool); flags[list(relevant)] = True
        eligible = flags[faces_by_leaf] & ~np.isin(self.nodes.root_status[root_indices], (2,3))
        if self._new_visited:eligible[list(self._new_visited)] = False
        numbers = np.flatnonzero(eligible); roots = root_indices[numbers]
        if self.prism_bounds is None:
            centres = self.domain.vertices[self.domain.faces[self.domain.source_faces[roots]]].mean(axis=1)
        else:centres = self.prism_bounds[0][roots]/2+self.prism_bounds[1][roots]/2
        distances = np.full(len(numbers), np.inf)
        for seed in self.new_seeds:
            self.meter.touch()
            delta = centres-self.seeds[seed]['point']
            distances = np.minimum(distances, np.einsum('ij,ij->i',delta,delta))
        numbers = numbers[np.lexsort((numbers, distances, ~anchor_faces[faces_by_leaf[numbers]]))]
        if anchor_only:numbers = numbers[:POLICY['new_anchor_leaf_budget']]
        for position, value in enumerate(numbers):
            if position % 128 == 0:self.meter.touch()
            number = int(value)
            self.meter.counts['new_scan_leaves_visited'] += 1
            if anchor_only:self.meter.counts['new_anchor_leaves_visited'] += 1
            self._visit(number)
            self._new_visited.add(number)
            if position % LAYERED_POLICY['connectivity_batch'] == 0:self._drain(LAYERED_POLICY['proof_batch'])
        self._drain()
        if not anchor_only:self._subdivide()
        self.limited_seeds = None

    def cells(self):
        if not self._prepare():return
        self._pixel_bounds()
        if not self._reprove():return
        self._new_neighborhoods(anchor_only=True)
        self._outside_faces()
        self._outside_pixels()
        self._new_neighborhoods()
        self.pending.clear()
        count = len(self.nodes)
        for number in range(count):
            self.meter.phase = 'remaining_original_cells'
            self.meter.touch()
            node = self.nodes[number]
            if not node.active or node.status in (1, 2, 3, 6):continue
            self.meter.counts['original_cells_visited'] += 1
            self._visit(number)
            if number % LAYERED_POLICY['connectivity_batch'] == 0:self._drain(LAYERED_POLICY['proof_batch'])
        self._drain(); self._subdivide()

    def run(self):
        self.meter.check()
        try:
            self.seed(); self.cells()
        except _BudgetStop:
            pass
        self.meter.check()
