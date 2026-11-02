"""Exact pixel-grid routing with compact topology and bounded search state.

Horizontal runs only compress connectivity labels. Returned routes still contain
every original pixel edge; scan spacing, surface gates and tie order are unchanged.
"""
from __future__ import annotations

import heapq
from typing import Callable

import numpy as np


MAX_GRAPH_RUNS = 800_000
MAX_SEARCH_NODES = 800_000
MAX_SEARCH_EXPANSIONS = 4_000_000


class SurfacePixelRouter:
    def __init__(self, safe: np.ndarray, right: np.ndarray, down: np.ndarray,
                 check: Callable[[], None] = lambda: None):
        self.height, self.width = safe.shape
        self.right, self.down, self.check = right, down, check
        self.expanded = 0
        # A 1080p selection can contain over 800k supported pixels while a
        # connector visits only tens of them. Do not allocate a Python graph for
        # those pixels or reject them as if they were actual search states.
        starts = safe.copy()
        starts[:, 1:] &= ~right
        self.run_count = int(np.count_nonzero(starts))
        if self.run_count > MAX_GRAPH_RUNS:
            raise ValueError('有效表面过于碎片化，超过 800000 个连通行段预算；请分区规划。')
        labels = np.cumsum(starts.ravel(), dtype=np.int32).reshape(safe.shape)
        labels -= 1
        labels[~safe] = -1
        self.labels = labels
        self.parents = np.arange(self.run_count, dtype=np.int32)
        # Only certified vertical edges join runs. A plain connectedComponents
        # on safe would incorrectly bridge depth jumps and nonadjacent faces.
        for top in range(0, self.height - 1, 64):
            check()
            bottom = min(top + 64, self.height - 1)
            permitted = down[top:bottom]
            upper = labels[top:bottom][permitted].astype(np.int64)
            lower = labels[top + 1:bottom + 1][permitted]
            pairs = np.unique(upper * self.run_count + lower)
            for index, pair in enumerate(pairs):
                if index % 1024 == 0:
                    check()
                a, b = divmod(int(pair), self.run_count)
                a, b = self._root(a), self._root(b)
                if a != b:
                    self.parents[max(a, b)] = min(a, b)

    def _root(self, node: int) -> int:
        while self.parents[node] != node:
            self.parents[node] = self.parents[self.parents[node]]
            node = int(self.parents[node])
        return node

    def connect(self, start: tuple[int, int], goal: tuple[int, int]):
        self.check()
        a, b = int(self.labels[start]), int(self.labels[goal])
        if a < 0 or b < 0 or self._root(a) != self._root(b):
            return None
        width = self.width
        start_id, goal_id = start[0] * width + start[1], goal[0] * width + goal[1]
        # Row-major integer IDs have the same ordering as the old (y, x) heap
        # keys, preserving deterministic equal-cost path selection.
        queue = [(0, 0, start_id)]
        costs = {start_id: 0}
        previous = {}
        while queue:
            _, cost, node = heapq.heappop(queue)
            if cost != costs[node]:
                continue
            self.expanded += 1
            if self.expanded % 2048 == 0:
                self.check()
            if self.expanded > MAX_SEARCH_EXPANSIONS:
                raise ValueError('路线连接搜索超过 4000000 次展开预算；请分区规划，已保留当前标记。')
            if node == goal_id:
                route = [node]
                while route[-1] != start_id:
                    route.append(previous[route[-1]])
                return [divmod(item, width) for item in reversed(route)]
            y, x = divmod(node, width)
            neighbours = []
            if x < width - 1 and self.right[y, x]:
                neighbours.append(node + 1)
            if x > 0 and self.right[y, x - 1]:
                neighbours.append(node - 1)
            if y < self.height - 1 and self.down[y, x]:
                neighbours.append(node + width)
            if y > 0 and self.down[y - 1, x]:
                neighbours.append(node - width)
            for next_node in neighbours:
                next_cost = cost + 1
                if next_cost < costs.get(next_node, float('inf')):
                    if next_node not in costs and len(costs) >= MAX_SEARCH_NODES:
                        raise ValueError('单次路线连接超过 800000 个搜索状态预算；请分区规划。')
                    costs[next_node] = next_cost
                    previous[next_node] = node
                    ny, nx = divmod(next_node, width)
                    distance = abs(ny - goal[0]) + abs(nx - goal[1])
                    heapq.heappush(queue, (next_cost + distance, next_cost, next_node))
        return None
