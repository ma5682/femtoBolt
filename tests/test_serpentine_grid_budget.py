"""Large selections keep their pixel precision without unbounded route searches."""
import heapq
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d

from femto.annotation.fixed_rgb_geometry import FixedProjection
from femto.planning.serpentine_path import serpentine_plan
from femto.planning.surface_pixel_graph import SurfacePixelRouter


def links(safe):
    return safe[:, :-1] & safe[:, 1:], safe[:-1] & safe[1:]


def original_route(safe, right, down, start, goal):
    """Frozen predecessor A* as an independent tie-order compatibility oracle."""
    h, w = safe.shape
    queue = [(0, 0, start)]
    costs = {start: 0}
    parents = {}
    while queue:
        _, cost, node = heapq.heappop(queue)
        if cost != costs[node]:
            continue
        if node == goal:
            route = [node]
            while route[-1] != start:
                route.append(parents[route[-1]])
            return route[::-1]
        y, x = node
        neighbours = []
        if x < w - 1 and right[y, x]: neighbours.append((y, x + 1))
        if x > 0 and right[y, x - 1]: neighbours.append((y, x - 1))
        if y < h - 1 and down[y, x]: neighbours.append((y + 1, x))
        if y > 0 and down[y - 1, x]: neighbours.append((y - 1, x))
        for nxt in neighbours:
            nc = cost + 1
            if nc < costs.get(nxt, float('inf')):
                costs[nxt], parents[nxt] = nc, node
                heapq.heappush(queue, (nc + abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]), nc, nxt))
    return None


class SerpentineGridBudgetTests(unittest.TestCase):
    def test_million_pixel_grid_has_local_exact_search(self):
        safe = np.ones((1080, 1200), bool)
        router = SurfacePixelRouter(safe, *links(safe))
        self.assertEqual(router.run_count, 1080)
        self.assertEqual(router.connect((500, 500), (515, 500)),
                         [(y, 500) for y in range(500, 516)])
        self.assertEqual(router.expanded, 16)

    def test_large_disconnected_faces_do_not_consume_search_budget(self):
        safe = np.ones((1080, 1200), bool)
        right, down = links(safe)
        # Both sides have supported pixels but no certified shared-face edge.
        right[:, 599] = False
        router = SurfacePixelRouter(safe, right, down)
        self.assertIsNone(router.connect((500, 599), (500, 600)))
        self.assertEqual(router.expanded, 0)
        self.assertEqual(router.connect((500, 600), (501, 600)), [(500, 600), (501, 600)])

    def test_holes_and_depth_gates_preserve_exact_legacy_routes(self):
        rng = np.random.default_rng(801)
        for _ in range(5):
            safe = rng.random((18, 23)) > .18
            right, down = links(safe)
            right &= rng.random(right.shape) > .08
            down &= rng.random(down.shape) > .08
            router = SurfacePixelRouter(safe, right, down)
            points = [tuple(int(v) for v in p) for p in np.argwhere(safe)]
            for start, goal in zip(points[::17], reversed(points[::17])):
                self.assertEqual(router.connect(start, goal), original_route(safe, right, down, start, goal))

    def test_actual_search_and_fragmentation_budgets_remain_enforced(self):
        safe = np.ones((20, 20), bool)
        router = SurfacePixelRouter(safe, *links(safe))
        with patch('femto.planning.surface_pixel_graph.MAX_SEARCH_NODES', 8):
            with self.assertRaisesRegex(ValueError, '搜索状态预算'):
                router.connect((0, 0), (19, 19))
        router = SurfacePixelRouter(safe, *links(safe))
        with patch('femto.planning.surface_pixel_graph.MAX_SEARCH_EXPANSIONS', 8):
            with self.assertRaisesRegex(ValueError, '展开预算'):
                router.connect((0, 0), (19, 19))
        with patch('femto.planning.surface_pixel_graph.MAX_GRAPH_RUNS', 8):
            with self.assertRaisesRegex(ValueError, '连通行段预算'):
                SurfacePixelRouter(safe, *links(safe))

    def test_cancel_is_checked_during_topology_and_search(self):
        safe = np.ones((20, 20), bool)
        def cancel(): raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            SurfacePixelRouter(safe, *links(safe), cancel)
        router = SurfacePixelRouter(safe, *links(safe))
        router.check = cancel
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            router.connect((0, 0), (1, 0))

    def surface(self, size):
        extent = size * .0005
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector([[0, 0, .5], [extent, 0, .5],
                                      [extent, extent, .5], [0, extent, .5]]),
            o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]))
        projection = FixedProjection(mesh, np.diag([1000., 1000., 1.]),
                                     np.full((size, size), 500, np.uint16))
        return mesh, projection, dict(annotation_id='large-plane', triangle_indices=[0, 1])

    def test_full_plan_over_old_grid_limit_retains_spacing_and_each_pixel(self):
        mesh, projection, region = self.surface(1000)
        observed = []
        constructor = SurfacePixelRouter.__init__
        def record(router, safe, right, down, check):
            observed.append(int(safe.sum()))
            constructor(router, safe, right, down, check)
        with patch.object(SurfacePixelRouter, '__init__', record):
            plan = serpentine_plan(projection, mesh, region, 10., 20., boundary_margin_mm=0.)
        self.assertGreater(observed[0], 800_000)
        self.assertEqual(plan['spacing_mm'], 8.)
        self.assertEqual(plan['row_step_px'], 16)
        self.assertEqual(len(plan['segments']), 1)
        points = np.asarray([p['pixel'] for p in plan['segments'][0]])
        np.testing.assert_array_equal(np.abs(np.diff(points, axis=0)).sum(axis=1), 1)
        scan_rows = sorted({p['pixel'][1] for p in plan['segments'][0] if p['motion_kind'] == 'scan'})
        np.testing.assert_array_equal(np.diff(scan_rows), 16)
        self.assertFalse(plan['execution_ready'])
        self.assertFalse(plan['physical_accuracy_verified'])

    def test_output_budget_rejects_before_world_sample_allocation(self):
        mesh, projection, region = self.surface(600)
        with patch.object(projection, 'pixel', wraps=projection.pixel) as pixel:
            with self.assertRaisesRegex(ValueError, '200000 samples'):
                serpentine_plan(projection, mesh, region, .2, 90., boundary_margin_mm=0.)
            pixel.assert_not_called()


if __name__ == '__main__':
    unittest.main()
