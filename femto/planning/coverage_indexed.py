"""Versioned exact interval index for P2 same-ball subedge connectivity.

Only impossible interval pairs are pruned. No coordinate tolerance, seed
containment, adjacency criterion or proof permission is changed. Legacy
LayeredEngine remains the replay implementation for saved v2 certificates.
"""
from __future__ import annotations

from femto.geometry import exact_surface as exact
from femto.planning.coverage_schedule import LayeredEngine, POLICY as LEGACY_POLICY

POLICY = dict(LEGACY_POLICY,
    connectivity='exact_avl_positive_subedges_same_ball',
    adjacency_search='rational_low_key_and_exact_subtree_max_high',
    candidate_order='distance_to_cell_centroid_then_seed_id',
    edge_order='positive_overlaps_in_original_insertion_order',
    adjacency_budget='each_visited_query_node',
    insertion_checks='each_avl_insert_node_and_edge',
    edge_storage='one_node_per_registered_canonical_edge_interval')


def checked_policy(value):
    if (not isinstance(value, dict) or set(value) != set(POLICY)
            or any(type(value[key]) is not type(expected) or value[key] != expected
                   for key, expected in POLICY.items())):
        raise ValueError('索引覆盖策略版本或参数无效')
    return value


def _less(left, right):
    """Exact rational comparison, without temporary Fraction operations."""
    return left[0] * right[1] < right[0] * left[1]


def _ratio(value):return (value.numerator, value.denominator)


class _IntervalNode:
    __slots__ = ('low', 'high', 'number', 'sequence', 'left', 'right', 'height', 'maximum')

    def __init__(self, low, high, number, sequence):
        self.low, self.high, self.number, self.sequence = low, high, number, sequence
        self.left = self.right = None
        self.height, self.maximum = 1, high


def _height(node):return 0 if node is None else node.height


def _refresh(node):
    node.height = 1 + max(_height(node.left), _height(node.right))
    node.maximum = node.high
    if node.left is not None and _less(node.maximum, node.left.maximum):
        node.maximum = node.left.maximum
    if node.right is not None and _less(node.maximum, node.right.maximum):
        node.maximum = node.right.maximum
    return node


def _rotate_right(node):
    top = node.left
    node.left, top.right = top.right, node
    _refresh(node)
    return _refresh(top)


def _rotate_left(node):
    top = node.right
    node.right, top.left = top.left, node
    _refresh(node)
    return _refresh(top)


def _insert(root, value, check):
    check()
    if root is None:return value
    if _less(value.low, root.low) or (value.low == root.low and value.sequence < root.sequence):
        root.left = _insert(root.left, value, check)
    else:
        root.right = _insert(root.right, value, check)
    _refresh(root)
    balance = _height(root.left) - _height(root.right)
    if balance > 1:
        if _height(root.left.left) < _height(root.left.right):
            root.left = _rotate_left(root.left)
        return _rotate_right(root)
    if balance < -1:
        if _height(root.right.right) < _height(root.right.left):
            root.right = _rotate_right(root.right)
        return _rotate_left(root)
    return root


class ExactIntervalIndex:
    """One AVL node per interval; exact bounds and logarithmic tree height."""
    def __init__(self):
        self.root, self.count = None, 0

    def add(self, low, high, number, check):
        self.root = _insert(self.root, _IntervalNode(_ratio(low), _ratio(high), number, self.count), check)
        self.count += 1

    def overlaps(self, low, high, check):
        low, high = _ratio(low), _ratio(high)
        found = []
        stack = [self.root] if self.root is not None else []
        while stack:
            node = stack.pop()
            check()
            if not _less(low, node.maximum):continue
            if not _less(node.low, high):
                if node.left is not None:stack.append(node.left)
                continue
            if _less(low, node.high):found.append((node.sequence, node.number))
            if node.right is not None:stack.append(node.right)
            if node.left is not None:stack.append(node.left)
        return [number for _, number in sorted(found)]


class IndexedLayeredEngine(LayeredEngine):
    def __init__(self, *args):
        super().__init__(*args)
        self.interval_indexes = {}

    def _link(self, number):
        node = self.nodes[number]
        face = int(self.domain.source_faces[node.root])
        ids = tuple(int(i) for i in self.domain.faces[face])
        for a, b in zip(node.bary, node.bary[1:] + node.bary[:1]):
            self.meter.touch()
            edge = exact.edge_interval(face, ids, a, b)
            if edge is None:continue
            key, low, high = edge
            index = self.interval_indexes.setdefault(key, ExactIntervalIndex())
            for other in index.overlaps(low, high, lambda: self.meter.touch('adjacencies')):
                if other != number:
                    self.neighbors[number].add(other)
                    self.neighbors[other].add(number)
            index.add(low, high, number, self.meter.touch)
        for seed_id in node.good:
            seed = self.seeds[seed_id]
            if face == seed['face'] and exact.contains_point(node.bary, seed['bary']):
                self._remember_proof(number, seed_id, -1)
        for other in sorted(self.neighbors[number]):
            for seed_id in node.good:
                parent = self.by_node[other].get(seed_id)
                if parent is not None:self._remember_proof(number, seed_id, parent)
