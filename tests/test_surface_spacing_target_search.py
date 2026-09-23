"""Target-set Dijkstra stops early without changing any spacing measurement."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from tests.test_mesh_slice_planner import grid_mesh, project_uv
from femto.planning.surface_spacing_report import spacing_report


def chain(vertices, faces, face, level, count=5):
    triangle = vertices[faces[face]]
    points = []
    for fraction in np.linspace(0., 1., count):
        weights = np.array([.6 - .4 * fraction, .2 + .4 * fraction, .2])
        xyz = weights @ triangle
        points.append(dict(xyz_m=xyz.tolist(), pixel=project_uv(xyz[None])[0].tolist(),
                           triangle_index=face, surface_level_m=level))
    return points


def fixture(*, count=5, tilt=0., disconnected=False, missing_previous=False, mixed=False):
    source = grid_mesh(.30, .04, .01, tilt_deg=tilt)
    vertices, faces = np.array(source.vertices), np.array(source.faces)
    remote = None
    if disconnected:
        remote = len(faces)
        faces = np.vstack((faces, faces + len(vertices)))
        vertices = np.vstack((vertices, vertices + [.5, 0., .2]))
    mesh = SimpleNamespace(vertices=vertices, triangles=faces)
    segments = [chain(vertices, faces, 0, .005, count)]
    target_level = .025 if missing_previous else .015
    if mixed:
        segments.append(chain(vertices, faces, 1, target_level, count))
    segments.append(chain(vertices, faces, remote if remote is not None else 3, target_level, count))
    return mesh, {'triangle_indices': list(range(len(faces)))}, {'segments': segments}


def report(inputs, **kwargs):
    return spacing_report(*inputs, 10., 2., **kwargs)


def without_expansions(value):
    result = deepcopy(value)
    result['budget'].pop('expansions')
    return result


class SurfaceSpacingTargetSearchTests(unittest.TestCase):
    def test_default_stays_exactly_the_frozen_full_search(self):
        inputs = fixture()
        self.assertEqual(report(inputs), report(inputs, target_limited=False))

    def test_early_stop_preserves_every_measurement_and_only_reduces_expansions(self):
        for tilt in (0., 23., 57.):
            with self.subTest(tilt=tilt):
                inputs = fixture(count=67, tilt=tilt)
                full = report(inputs, target_limited=False)
                limited = report(inputs, target_limited=True)
                self.assertEqual(without_expansions(limited), without_expansions(full))
                self.assertLess(limited['budget']['expansions'], full['budget']['expansions'])
                self.assertEqual(limited['sample_count'], 32)
                self.assertEqual([row['sample_index'] for row in limited['measurements']],
                                 np.unique(np.linspace(0, 66, 32).astype(int)).tolist())
                self.assertEqual(limited['budget']['max_expansions'], 2_000_000)
                self.assertEqual(limited['budget']['max_distance_pairs'], 100_000_000)

    def test_every_target_chain_is_included_before_stopping(self):
        inputs = fixture(count=41)
        mesh, _, plan = inputs
        # A second target lies well beyond the first target's search frontier.
        plan['segments'].append(chain(mesh.vertices, mesh.triangles, 80, .015, 39))
        full = report(inputs, target_limited=False)
        limited = report(inputs, target_limited=True)
        self.assertEqual(without_expansions(limited), without_expansions(full))
        self.assertEqual(limited['sample_count'], 64)
        self.assertEqual({row['chain_index'] for row in limited['measurements']}, {1, 2})

    def test_unreachable_targets_exhaust_reachable_component_and_stay_unknown(self):
        for mixed in (False, True):
            with self.subTest(mixed=mixed):
                inputs = fixture(disconnected=True, mixed=mixed)
                full = report(inputs, target_limited=False)
                limited = report(inputs, target_limited=True)
                self.assertEqual(limited, full)
                remote = [row for row in limited['measurements'] if row['chain_index'] == (2 if mixed else 1)]
                self.assertTrue(remote)
                for row in remote:
                    self.assertEqual(row['status'], 'unmeasurable')
                    self.assertIsNone(row['surface_graph_distance_mm'])
                    self.assertIsNone(row['lower_bound_mm'])
                    self.assertIsNone(row['signed_deviation_mm'])

    def test_missing_previous_level_is_not_replaced_by_an_earlier_level(self):
        inputs = fixture(missing_previous=True)
        full = report(inputs, target_limited=False)
        limited = report(inputs, target_limited=True)
        self.assertEqual(limited, full)
        self.assertEqual(limited['budget']['expansions'], 0)
        self.assertTrue(all(row['status'] == 'unmeasurable' for row in limited['measurements']))

    def test_shared_expansion_budget_is_not_reset_or_relaxed(self):
        for limited in (False, True):
            with self.subTest(target_limited=limited):
                budget = {'expansions': 1_999_999, 'distance_pairs': 0}
                with self.assertRaisesRegex(ValueError, '200万节点展开预算'):
                    report(fixture(), target_limited=limited, budget=budget)
                self.assertEqual(budget['expansions'], 2_000_001)

    def test_exact_remaining_expansion_allowance_succeeds_but_one_less_fails(self):
        inputs = fixture()
        needed = report(inputs, target_limited=True)['budget']['expansions']
        self.assertGreater(needed, 1)
        budget = {'expansions': 2_000_000 - needed, 'distance_pairs': 0}
        result = report(inputs, target_limited=True, budget=budget)
        self.assertEqual(result['budget']['expansions'], 2_000_000)
        self.assertEqual(budget['expansions'], 2_000_000)
        with self.assertRaisesRegex(ValueError, '200万节点展开预算'):
            report(inputs, target_limited=True,
                   budget={'expansions': 2_000_001 - needed, 'distance_pairs': 0})

    def test_distance_pair_budget_still_checks_every_original_measurement(self):
        for limited in (False, True):
            with self.subTest(target_limited=limited):
                with self.assertRaisesRegex(ValueError, '1亿距离对预算'):
                    report(fixture(), target_limited=limited,
                           budget={'expansions': 0, 'distance_pairs': 100_000_000})

    def test_cancellation_inside_search_propagates_at_existing_check_boundary(self):
        budget = {'expansions': 511, 'distance_pairs': 0}
        def check():
            if budget['expansions'] >= 512:
                raise RuntimeError('cancel requested')
        with self.assertRaisesRegex(RuntimeError, 'cancel requested'):
            report(fixture(), target_limited=True, budget=budget, check=check)
        self.assertEqual(budget['expansions'], 512)
        self.assertEqual(budget['distance_pairs'], 0)

    def test_target_preparation_and_search_do_not_mutate_inputs(self):
        mesh, region, plan = fixture(count=67)
        vertices, faces = mesh.vertices.copy(), mesh.triangles.copy()
        saved_region, saved_plan = deepcopy(region), deepcopy(plan)
        report((mesh, region, plan), target_limited=True)
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.triangles, faces)
        self.assertEqual(region, saved_region)
        self.assertEqual(plan, saved_plan)


if __name__ == '__main__':
    unittest.main()
