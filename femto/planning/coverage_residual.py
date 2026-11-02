"""V4: recheck residual leaves against current balls within the same budget.

V3's delivered scheduler remains available verbatim. A current root exclusion
may remove a ball from its descendants' work list; no saved outside/confirmed
state supplies that exclusion. Partial batches never install a classification.
"""
from collections import defaultdict
import math

import numpy as np

from femto.geometry import exact_surface as exact
from femto.operation_diagnostics import diagnosed
from femto.planning.coverage_certificate import _Node
from femto.planning.coverage_domain import construction_cache
from femto.planning.coverage_reproof import ReproofEngine, _Partition, POLICY as V3_POLICY
from femto.planning.coverage_schedule import LayeredEngine, POLICY as LAYERED_POLICY

POLICY = dict(V3_POLICY, schedule='reprove_then_anchor_prefix_then_current_root_and_leaf_exclusions_then_new_scan',
    outside='current_all_ball_face_pixel_and_dyadic_leaf_aabb_lower_bound',
    residual_candidates='survivors_of_complete_current_group_and_root_checks_else_all_current_face_candidates',
    residual_bounds='current_exact_root_vertex_intervals_then_outward_dyadic_coefficients',
    reproof_bounds='current_root_vertex_intervals_then_paired_descendant_intervals_with_exact_ambiguous_fallback',
    root_vertex_cache_entries=4096,
    residual_order='root_index_then_candidate_group_then_leaf_index',
    residual_nonexclusion='stop_leaf_exclusion_search_at_first_current_possible_ball_overlap',
    root_group_filter='outward_union_of_current_pixel_boxes_checked_against_all_group_candidate_balls',
    shared_edge_filter='current_identical_vertices_or_positive_collinear_rays_in_same_source_face_then_exact_intervals',
    new_scan_geometry='current_homogeneous_integer_cells_and_positive_rational_subedges',
    residual_root_batch=128, residual_leaf_batch=128, residual_pair_batch=4096,
    residual_install='completed_all_candidate_batch_only_without_changing_current_witnesses')


def checked_policy(value):
    from femto.planning.planning_revision import digest
    if not isinstance(value, dict) or value != POLICY or digest(value) != digest(POLICY):
        raise ValueError('覆盖遗留复算策略无效或版本不一致')
    return value


class _ResidualPartition(_Partition):
    def __init__(self, *args):
        super().__init__(*args)
        self.outside_leaves = np.zeros(self.count, bool)

    def __getitem__(self, number):
        if number < 0:raise IndexError(number)
        if number >= self.count:return self.extra[number-self.count]
        node = self.touched.get(number)
        if node is None:
            root = int(self.roots[number]); status = int(self.root_status[root])
            if self.outside_leaves[number] and status == 7:status = 3
            node = _Node(root, int(self.depths[number]), int(self.codes[number]), status=status)
            self.touched[number] = node
        return node

    def unknown_numbers(self, check):
        status = self.root_status[self.roots].copy()
        status[self.outside_leaves & (status == 7)] = 3
        for index, (number, node) in enumerate(self.touched.items()):
            if index % 128 == 0:check()
            status[number] = node.status if node.active else 0
        return np.flatnonzero(status == 7)

    def mark_leaves_outside(self, numbers):
        self.outside_leaves[numbers] = True
        for value in numbers:
            number = int(value); node = self.touched.get(number)
            if node is not None:
                if node.status == 7:node.status = 3
                elif node.status != 3:self.outside_leaves[number] = False

    def leaf_arrays(self):
        values, remap = super().leaf_arrays()
        numbers = remap[np.flatnonzero(self.outside_leaves)]
        numbers = numbers[numbers >= 0]
        numbers = numbers[values['status'][numbers] == 7]
        values['status'][numbers] = 3
        return values, remap


def _outside_pairs(low, high, center_low, center_high, radii):
    """Strict lower distance bound; NaN/overflow ambiguity never excludes."""
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        delta = np.maximum(0, np.maximum(
            np.nextafter(low[:,None,:]-center_high, -np.inf),
            np.nextafter(center_low-high[:,None,:], -np.inf)))
        squared = np.maximum(0, np.nextafter(delta*delta, -np.inf))
        lower = np.nextafter(np.nextafter(squared[:,:,0]+squared[:,:,1], -np.inf)+squared[:,:,2], -np.inf)
        upper = np.nextafter(radii*radii, np.inf)
    return lower > upper


class _CurrentTriangles:
    """Materialize exact current XYZ only for an ambiguous paired predicate."""
    def __init__(self, geometry, numbers):self.geometry, self.numbers = geometry, numbers
    def __getitem__(self, index):return self.geometry(self.numbers[index])[0]


def _positive_shared_vertex(left,right,point):
    """One exact common vertex plus two collinear rays with positive overlap."""
    denominator = sum(point)
    def rays(triangle):
        return [(p[1]*denominator-point[1]*sum(p),p[2]*denominator-point[2]*sum(p))
                for p in triangle if p != point]
    return any(x*v == y*u and x*u+y*v > 0 for x,y in rays(left) for u,v in rays(right))


class ResidualEngine(ReproofEngine):
    def _initial_partition(self):
        count = len(self.domain.areas)
        return _ResidualPartition(np.arange(count, dtype='<i8'), np.zeros(count, 'u1'),
                                  np.zeros(count, '<u4'), self.domain.quality)

    def __init__(self, *args, guide):
        super().__init__(*args, guide=guide)
        self._root_candidates = {}
        self._root_vertices = construction_cache(self._make_root_vertices, maxsize=POLICY['root_vertex_cache_entries'])
        self.meter.counts.update(residual_roots_reconstructed=0, residual_unconstructed_roots=0,
            residual_leaves_rechecked=0, residual_outside_leaves=0, residual_ball_comparisons=0,
            residual_batches_completed=0, residual_short_circuited_leaves=0,
            residual_current_root_candidates=0, root_vertex_bounds_built=0,
            root_group_ball_comparisons=0, root_group_candidate_pairs_pruned=0,
            shared_vertex_ray_pairs=0, new_scan_integer_leaves=0)

    def _prepare(self):
        if not super()._prepare():return False
        old = self.nodes
        self.nodes = _ResidualPartition(old.roots, old.depths, old.codes, self.domain.quality)
        return True

    def _visit(self, number):
        node = self.nodes[number]
        if node.status == 3:return
        self.meter.touch()
        if self._checked_integer_root(node) is None:return
        previous = node.status,node.good
        try:
            self.meter.counts['new_scan_integer_leaves'] += 1
            face = int(self.domain.source_faces[node.root])
            triangle,bounds = self._geometry(number)
            low = np.nextafter(bounds[0].min(axis=0)-self.radius,-np.inf)
            high = np.nextafter(bounds[1].max(axis=0)+self.radius,np.inf)
            candidates = np.asarray(self.candidates_for(face),np.int64)
            self.meter.counts['spatial_candidates'] += len(candidates)
            candidates = candidates[np.all(self.points[candidates]>=low,axis=1)&np.all(self.points[candidates]<=high,axis=1)]
            self.meter.counts['aabb_ball_comparisons'] += len(candidates)
            radii = np.asarray([self.seeds[i]['inner'] for i in candidates])
            rejected = _outside_pairs(bounds[0].min(axis=0)[None,:],bounds[1].max(axis=0)[None,:],
                self.centers_low[candidates],self.centers_high[candidates],radii)[0]
            candidates = candidates[~rejected]
            if not len(candidates):
                node.status = 3; return
            centroid = (bounds[0]/2+bounds[1]/2).mean(axis=0)
            distance = np.linalg.norm(self.points[candidates]-centroid,axis=1)
            order = np.lexsort((candidates,distance)); cap = LAYERED_POLICY['max_cell_ball_candidates']
            self.meter.counts['cell_candidates_deferred'] += max(0,len(candidates)-cap)
            candidates = candidates[order[:cap]]
            for _ in candidates:self.meter.touch('comparisons')
            contained,fallbacks = exact.contains_balls(triangle,tuple(self.seeds[i]['center'] for i in candidates),
                [self.seeds[i]['inner'] for i in candidates],triangle_bounds=bounds,
                center_bounds=(self.centers_low[candidates],self.centers_high[candidates]))
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
                node.status = 1; node.good = tuple(sorted(set(node.good)|set(previous[1])))
            elif self.limited_seeds is not None and node.status not in (1,6):node.status = 7

    def _link(self, number):
        node = self.nodes[number]; face = int(self.domain.source_faces[node.root])
        for edge in self._edges(number):
            self.meter.touch()
            for other_edge,other in self.edges[edge[0]]:
                self.meter.touch('adjacencies')
                if other != number and exact.integer_subedges_overlap(edge,other_edge):
                    self.neighbors[number].add(other); self.neighbors[other].add(number)
            self.edges[edge[0]].append((edge,number))
        for seed_id in node.good:
            seed = self.seeds[seed_id]
            if face == seed['face'] and exact.integer_contains_point(self._integer_bary(number),seed['bary']):
                self._remember_proof(number,seed_id,-1)
        for other in sorted(self.neighbors[number]):
            for seed_id in node.good:
                parent = self.by_node[other].get(seed_id)
                if parent is not None:self._remember_proof(number,seed_id,parent)

    def _subdivide(self):
        while self.pending:
            self.meter.phase = 'refinement'; self.meter.touch()
            number = self.pending.popleft(); node = self.nodes[number]
            if node.witness >= 0:continue
            if self.meter.counts['added_cells']+3 > self.budget.max_added_cells:
                self.meter.flags.add('cell_budget'); break
            first = len(self.nodes)
            self.nodes.extend(_Node(node.root,node.depth+1,4*node.code+child) for child in range(4))
            node.active = False; node.bary = None; self.meter.counts['added_cells'] += 3
            for child in range(first,first+4):self._visit(child)
            self._drain(LAYERED_POLICY['proof_batch'])
        self._drain()

    def _make_root_vertices(self, root):
        integer_root = self._integer_root_cache(root)
        if integer_root is None:return None
        self.meter.counts['root_vertex_bounds_built'] += 1
        return self.domain.integer_cell(int(self.domain.source_faces[root]), integer_root.triangle)[1]

    def _shared_subedge(self,left,right):
        if left == right:return False
        a,b = self.nodes[left],self.nodes[right]
        if a.root == b.root:return exact.descendants_share_subedge(a.depth,a.code,b.depth,b.code)
        if self.domain.source_faces[a.root] == self.domain.source_faces[b.root]:
            aa,bb = self._integer_bary(left),self._integer_bary(right)
            if aa is not None and bb is not None:
                common = set(aa).intersection(bb)
                if len(common) >= 2:
                    self.meter.counts['shared_endpoint_pairs'] += 1; return True
                if common and _positive_shared_vertex(aa,bb,next(iter(common))):
                    self.meter.counts['shared_vertex_ray_pairs'] += 1; return True
        return any(exact.integer_subedges_overlap(a,b) for a in self._edges(left) for b in self._edges(right))

    def _proof(self, number, seed_id, parent):
        self.meter.touch('proofs')
        node = self.nodes[number]
        witness = len(self.proofs); self.proofs.append((number, seed_id, parent))
        if node.witness < 0:
            node.witness = witness; node.status = 1
            if self._retained_flags is not None and number < len(self._retained_flags) and self._retained_flags[number]:
                self.meter.counts['retained_leaves_rechecked'] += 1
        return witness

    @diagnosed('coverage.retained_reproof')
    def _reprove(self):
        self.meter.phase = 'retained_proofs'
        a = self.guide.arrays
        remap = np.full(len(a['proof_leaf']), -1, np.int64)
        for offset in range(0, len(self.selected_proofs), POLICY['proof_batch']):
            rows, seed_ids, roots = [], [], []
            for proof in self.selected_proofs[offset:offset+POLICY['proof_batch']]:
                self.meter.touch('comparisons')
                number = int(a['proof_leaf'][proof]); node = self.nodes[number]
                seed_id = self.seed_remap[a['proof_seed'][proof]]
                if seed_id < 0:
                    self.meter.counts['unbound_hint_witnesses'] += 1; continue
                if self.domain.quality[node.root]:continue
                if node.root not in self._reconstructed and self._checked_integer_root(node) is None:continue
                rows.append((int(proof),number,node,seed_id)); seed_ids.append(seed_id); roots.append(node.root)
            inside = np.zeros(len(rows), bool)
            if rows and self.prism_bounds is not None:
                inside = exact.boxes_in_balls(self.prism_bounds[0][roots], self.prism_bounds[1][roots],
                    [self.seeds[i]['inner'] for i in seed_ids],
                    center_bounds=(self.centers_low[seed_ids],self.centers_high[seed_ids]))
                self.meter.counts['prism_contained_pairs'] += int(inside.sum())
            pending = np.flatnonzero(~inside)
            if len(pending):
                lows, highs, numbers, depths, codes, current_seeds = [], [], [], [], [], []
                for index in pending:
                    _, number, node, seed_id = rows[index]
                    bounds = self._root_vertices(node.root)
                    lows.append(bounds[0]); highs.append(bounds[1]); numbers.append(number)
                    depths.append(node.depth); codes.append(node.code); current_seeds.append(seed_id)
                bounds = exact.descendant_vertex_bounds(lows,highs,np.asarray(depths),np.asarray(codes))
                contained, fallbacks = exact.contains_ball_pairs(_CurrentTriangles(self._geometry,numbers),
                    tuple(self.seeds[i]['center'] for i in current_seeds), [self.seeds[i]['inner'] for i in current_seeds],
                    triangle_bounds=bounds, center_bounds=(self.centers_low[current_seeds],self.centers_high[current_seeds]))
                inside[pending] = contained; self.meter.counts['exact_fallbacks'] += fallbacks
            for (proof, number, node, seed_id), contained in zip(rows,inside):
                self.meter.touch(); self.meter.counts['hint_proofs_rechecked'] += 1
                if not contained:
                    self.meter.counts['hint_containment_failures'] += 1; continue
                parent = int(a['proof_parent'][proof]); seed = self.seeds[seed_id]
                self.meter.touch('adjacencies')
                connected = (int(self.domain.source_faces[node.root]) == seed['face']
                    and exact.integer_contains_point(self._integer_bary(number),seed['bary'])) if parent == -1 else (
                    remap[parent] >= 0 and self._shared(*sorted((number,int(a['proof_leaf'][parent])))))
                if not connected:
                    self.meter.counts['hint_connectivity_failures'] += 1; continue
                if seed_id not in node.good:node.good += (seed_id,)
                known = self.by_node[number].get(seed_id)
                if known is None:
                    known = self._proof(number,seed_id,-1 if parent == -1 else int(remap[parent]))
                    self.by_node[number][seed_id] = known
                remap[proof] = known
        if self.meter.counts['retained_leaves_rechecked'] != len(self.old_confirmed):
            self.meter.flags.add('retained_proofs_unconfirmed'); return False
        return True

    def _box_groups(self, roots):
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
        return groups

    def _candidates_in_box(self, key):
        if key[0] is None:return LayeredEngine.candidates_for(self, key[1])
        found = []
        for x in range(key[0], key[3]+1):
            for y in range(key[1], key[4]+1):
                for z in range(key[2], key[5]+1):
                    self.meter.touch(); self.meter.counts['outside_index_bins_visited'] += 1
                    found.extend(self.grid.get((x,y,z), ()))
        return sorted(found)

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
                for offset in range(0,len(candidates),POLICY['outside_pair_batch']):
                    current = candidates[offset:offset+POLICY['outside_pair_batch']]
                    self.meter.touch_many(len(current),'comparisons')
                    self.meter.counts['root_group_ball_comparisons'] += len(current)
                    rejected[offset:offset+len(current)] = _outside_pairs(low,high,
                        self.centers_low[current],self.centers_high[current],radii[offset:offset+len(current)])[0]
                self.meter.counts['root_group_candidate_pairs_pruned'] += len(group)*int(rejected.sum())
                candidates,radii = candidates[~rejected],radii[~rejected]
                if not len(candidates):
                    outside[group] = True; continue
                size = min(POLICY['outside_pixel_batch'], max(1, POLICY['outside_pair_batch']//max(1,len(candidates))))
                for start in range(0, len(group), size):
                    self.meter.touch()
                    batch = group[start:start+size]
                    low, high = self.prism_bounds[0][batch], self.prism_bounds[1][batch]
                    survivors = [[] for _ in batch]
                    stride = max(1, POLICY['outside_pair_batch']//len(batch))
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
                        if remaining:self._root_candidates[root] = tuple(remaining)
                        else:outside[root] = True
        finally:
            self.nodes.mark_outside(outside)
            self.meter.counts['outside_pixel_roots_rechecked'] = int(outside.sum())

    @diagnosed('coverage.residual_recheck')
    def _outside_leaves(self):
        self.meter.phase = 'outside_leaf_recheck'
        numbers = self.nodes.unknown_numbers(self.meter.touch)
        numbers = numbers[np.argsort(self.nodes.roots[numbers], kind='stable')]
        roots, starts, counts = np.unique(self.nodes.roots[numbers], return_index=True, return_counts=True)
        for start in range(0, len(roots), POLICY['residual_root_batch']):
            groups = defaultdict(list); lows, highs = [], []
            for index in range(start, min(len(roots), start+POLICY['residual_root_batch'])):
                self.meter.touch()
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
                rows = np.concatenate([part for _,part in pieces])
                positions = np.repeat([position for position,_ in pieces], [len(part) for _,part in pieces])
                candidates = np.asarray(candidates, np.int64)
                radii = np.asarray([self.seeds[i]['inner'] for i in candidates])
                size = POLICY['residual_leaf_batch']
                for offset in range(0, len(rows), size):
                    self.meter.touch()
                    batch, root_positions = rows[offset:offset+size], positions[offset:offset+size]
                    low, high = exact.descendant_boxes(lows[root_positions], highs[root_positions],
                        self.nodes.depths[batch], self.nodes.codes[batch])
                    excluded = np.ones(len(batch), bool)
                    for index, current in enumerate(candidates):
                        active = np.flatnonzero(excluded)
                        if not len(active):break
                        self.meter.touch_many(len(active), 'comparisons')
                        self.meter.counts['residual_ball_comparisons'] += len(active)
                        rejected = _outside_pairs(low[active],high[active],self.centers_low[current:current+1],
                            self.centers_high[current:current+1],radii[index:index+1])[:,0]
                        excluded[active] = rejected
                    self.nodes.mark_leaves_outside(batch[excluded])
                    self.meter.counts['residual_leaves_rechecked'] += len(batch)
                    self.meter.counts['residual_outside_leaves'] += int(excluded.sum())
                    self.meter.counts['residual_short_circuited_leaves'] += int((~excluded).sum())
                    self.meter.counts['residual_batches_completed'] += 1

    def cells(self):
        if not self._prepare():return
        self._pixel_bounds()
        if not self._reprove():return
        self._new_neighborhoods(anchor_only=True)
        self._outside_faces()
        self._outside_pixels()
        self._outside_leaves()
        self._new_neighborhoods()
        self.pending.clear()
        for number in range(len(self.nodes)):
            self.meter.phase = 'remaining_original_cells'; self.meter.touch()
            node = self.nodes[number]
            if not node.active or node.status in (1,2,3,6):continue
            self.meter.counts['original_cells_visited'] += 1
            self._visit(number)
            if number % LAYERED_POLICY['connectivity_batch'] == 0:self._drain(LAYERED_POLICY['proof_batch'])
        self._drain(); self._subdivide()
