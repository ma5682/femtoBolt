"""B2 scheduling must keep B1 proof semantics, original weights and replay."""
from copy import deepcopy
from fractions import Fraction
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from femto.planning import coverage_certificate as coverage, footprint_model
from femto.planning.coverage_domain import CoverageDomain
from test_coverage_certificate import triangle_domain, seed, reseal


def run(domain=None, records=None, width=500, **budget):
    return coverage.evaluate(domain if domain is not None else triangle_domain(),
        records if records is not None else [seed()], footprint_model.nominal(width),
        algorithm=coverage.CURRENT_ALGORITHM, budget=coverage.CoverageBudget(**budget))


class CoverageScheduleTests(unittest.TestCase):
    def test_empty_spatial_filter_leaves_cell_unconfirmed_and_counts_unchanged(self):
        from femto.planning.coverage_schedule import LayeredEngine
        engine = LayeredEngine.__new__(LayeredEngine)
        node = SimpleNamespace(root=0, bary=object(), status=7)
        engine.nodes = [node]
        bounds = (np.zeros((3, 3)), np.zeros((3, 3)))
        engine.domain = SimpleNamespace(source_faces=[0], cell=lambda face, bary: (None, bounds))
        engine.radius = 0.1
        engine.candidates_for = lambda face: [0]
        engine._candidate_ids = lambda face: np.array([0], dtype=np.int64)
        engine.points = np.array([[1., 1., 1.]])
        engine.meter = SimpleNamespace(touch=lambda *args: None,
            counts={'spatial_candidates': 0, 'aabb_ball_comparisons': 0})
        engine._visit(0)
        self.assertEqual((node.status, node.bary), (3, None))
        self.assertEqual(engine.meter.counts, {'spatial_candidates': 1, 'aabb_ball_comparisons': 0})

    def test_cached_candidate_ids_keep_complete_and_partial_proof_arrays(self):
        from femto.planning.coverage_schedule import LayeredEngine
        def evaluate(frontier=None):
            return coverage.evaluate(triangle_domain(), [seed()], footprint_model.nominal(125),
                algorithm=coverage.CURRENT_ALGORITHM,
                budget=coverage.CoverageBudget(max_depth=3), _frontier=frontier)
        original_init = LayeredEngine.__init__
        def uncached_init(engine, *args, **kwargs):
            original_init(engine, *args, **kwargs)
            engine._candidate_ids = lambda face: np.asarray(engine.candidates_for(face), np.int64)
        full = evaluate()
        stops = (None, dict(stop_reason='time_budget', steps=17),
                 dict(stop_reason='time_budget', steps=full.metadata['metrics']['steps']//2))
        for frontier in stops:
            with self.subTest(frontier=frontier):
                with patch.object(LayeredEngine, '__init__', uncached_init):
                    expected = evaluate(frontier)
                actual = evaluate(frontier)
                self.assertEqual(actual.metadata, expected.metadata)
                for key in actual.arrays:
                    np.testing.assert_array_equal(actual.arrays[key], expected.arrays[key])

    def test_whole_cell_is_confirmed_and_v1_still_replays(self):
        domain, records, model = triangle_domain(), [seed()], footprint_model.nominal(500)
        old = coverage.evaluate(domain, records, model)
        new = run()
        self.assertEqual(old.metadata['algorithm'], coverage.ALGORITHM)
        self.assertEqual(new.metadata['algorithm'], coverage.CURRENT_ALGORITHM)
        self.assertNotEqual(old.metadata['coverage_computation_sha256'], new.metadata['coverage_computation_sha256'])
        for certificate in (old, new):
            self.assertTrue(certificate.summary['original_target_coverage_complete'])
            coverage.validate(certificate, domain, records, model)

    def test_partial_circle_is_positive_and_below_independent_area(self):
        result = run(width=125, max_depth=3)
        area = result.summary['confirmed_nominal_area_m2']
        self.assertGreater(area, 0)
        self.assertLessEqual(area, math.pi*.0625**2/4)
        self.assertFalse(result.summary['original_target_coverage_complete'])
        coverage.validate(result, triangle_domain(), [seed()], footprint_model.nominal(125))

    def test_original_cells_are_visited_before_any_refinement(self):
        domain = CoverageDomain.mesh([[0, 0, 0], [.125, 0, 0], [0, .125, 0],
            [0, 0, 1], [.01, 0, 1], [0, .01, 1]], [[0, 1, 2], [3, 4, 5]])
        result = run(domain, [seed(), seed((0, 0, 1), 1)], width=125, max_added_cells=0)
        self.assertEqual(result.metadata['metrics']['original_cells_visited'], 2)
        self.assertEqual(result.arrays['status'][result.arrays['root'] == 1].tolist(), [1])
        self.assertIn('cell_budget', result.metadata['metrics']['limits_reached'])

    def test_connectivity_proofs_exist_before_a_later_budget_exit(self):
        vertices = [[x, y, 0] for y in (0, .03125) for x in (0, .03125, .0625)]
        domain = CoverageDomain.mesh(vertices, [[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]])
        result = run(domain, max_comparisons=3)
        self.assertGreaterEqual(result.summary['confirmed_leaf_count'], 2)
        self.assertTrue(np.any(result.arrays['proof_parent'] >= 0))
        self.assertEqual(result.metadata['metrics']['stop_reason'], 'comparisons_budget')

    def test_mixed_depth_neighbors_retain_one_ball_parent_chains(self):
        domain = CoverageDomain.mesh([[0, 0, 0], [.125, 0, 0], [0, .125, 0], [.125, .125, 0]],
                                      [[0, 1, 2], [1, 3, 2]])
        records = [seed(), seed((.125, .125, 0), 1)]
        result = run(domain, records, width=250, max_depth=3)
        self.assertEqual(set(result.arrays['root'][result.arrays['status'] == 1]), {0, 1})
        arrays = result.arrays
        for index, parent in enumerate(arrays['proof_parent']):
            if parent >= 0:
                self.assertLess(parent, index)
                self.assertEqual(arrays['proof_seed'][index], arrays['proof_seed'][parent])
        coverage.validate(result, domain, records, footprint_model.nominal(250))

    def test_touching_vertices_and_coincident_disconnected_mesh_are_not_welded(self):
        for vertices, faces in (
            ([[0, 0, 0], [.125, 0, 0], [0, .125, 0], [-.125, 0, 0], [0, -.125, 0]], [[0, 1, 2], [0, 3, 4]]),
            ([[0, 0, 0], [.125, 0, 0], [0, .125, 0], [.125, 0, 0], [.125, .125, 0], [0, .125, 0]], [[0, 1, 2], [3, 4, 5]])):
            result = run(CoverageDomain.mesh(vertices, faces))
            self.assertEqual(result.arrays['status'].tolist(), [1, 4])

    def test_fold_leaving_the_ball_does_not_confirm_the_close_far_end(self):
        vertices = [[0, 0, 0], [.125, 0, 0], [0, .125, 0], [2, 0, 0],
                    [2, .125, 0], [.125, 0, .03125], [0, .125, .03125], [0, 0, .03125]]
        faces = [[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4], [4, 5, 6], [5, 7, 6]]
        result = run(CoverageDomain.mesh(vertices, faces), max_depth=2)
        self.assertEqual(result.arrays['status'][result.arrays['root'] == 5].tolist(), [4])

    def test_hole_and_original_area_remain_unchanged(self):
        vertices = [[x, y, 0] for y in (0, .0625, .125, .1875) for x in (0, .0625, .125, .1875)]
        faces = []
        for y in range(3):
            for x in range(3):
                if x == y == 1:continue
                a = y*4+x; faces.extend([[a, a+1, a+4], [a+1, a+5, a+4]])
        result = run(CoverageDomain.mesh(vertices, faces), width=1000)
        self.assertEqual(result.summary['confirmed_nominal_area_m2'], 8*.0625**2)
        self.assertTrue(result.summary['original_target_coverage_complete'])

    def test_candidate_ranking_never_joins_disconnected_layers(self):
        vertices = [[x, y, z] for z in (0, .01) for x,y in ((0, 0), (.125, 0), (0, .125))]
        domain = CoverageDomain.mesh(vertices, [[0, 1, 2], [3, 4, 5]])
        records = [seed((0, 0, .01), 1)]*16 + [seed()]
        result = run(domain, records)
        self.assertEqual(result.arrays['status'].tolist(), [1, 1])
        self.assertGreater(result.metadata['metrics']['cell_candidates_deferred'], 0)
        for leaf, seed_index in zip(result.arrays['proof_leaf'], result.arrays['proof_seed']):
            self.assertEqual(result.arrays['root'][leaf], result.arrays['seed_face'][seed_index])

    def test_three_vertex_balls_do_not_prove_a_cell(self):
        result = run(records=[seed(), seed((.125, 0, 0)), seed((0, .125, 0))], width=62.5, max_depth=0)
        self.assertEqual(result.summary['confirmed_leaf_count'], 0)
        self.assertEqual(result.arrays['status'].tolist(), [5])

    def test_each_budget_keeps_original_area_and_replays(self):
        for limit in (dict(seconds=0), dict(max_seeds=0), dict(max_comparisons=0), dict(max_proofs=0),
                      dict(max_added_cells=0), dict(max_depth=0)):
            with self.subTest(limit=limit):
                result = run(width=125, **limit)
                self.assertFalse(result.summary['original_target_coverage_complete'])
                self.assertEqual(result.summary['located_nominal_area_m2'], .125**2/2)
                coverage.validate(result, triangle_domain(), [seed()], footprint_model.nominal(125))

    def test_time_budget_replays_the_exact_frontier(self):
        with patch('femto.planning.coverage_certificate.time.monotonic', side_effect=range(10000)):
            result = run(width=125, seconds=8)
        self.assertEqual(result.metadata['metrics']['stop_reason'], 'time_budget')
        coverage.validate(result, triangle_domain(), [seed()], footprint_model.nominal(125))

    def test_cancellation_returns_no_certificate(self):
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            coverage.evaluate(triangle_domain(), [seed()], footprint_model.nominal(500),
                              algorithm=coverage.CURRENT_ALGORITHM, check=cancel)

    def test_algorithm_or_policy_relabelling_cannot_reuse_a_proof(self):
        for change in ('algorithm', 'policy'):
            result = run()
            metadata = deepcopy(result.metadata)
            if change == 'algorithm':
                metadata['algorithm'] = metadata['inputs']['algorithm'] = coverage.ALGORITHM
            else:metadata['inputs']['search_policy']['max_cell_ball_candidates'] = 999
            from femto.planning.planning_revision import digest
            metadata['coverage_computation_sha256'] = digest(metadata['inputs'])
            forged = reseal(result, metadata=metadata)
            with self.assertRaises(ValueError):coverage.validate(forged, triangle_domain(), [seed()], footprint_model.nominal(500))

    def test_certificate_cannot_mutate_the_shared_search_policy(self):
        result=run()
        result.metadata['inputs']['search_policy']['max_cell_ball_candidates']=999
        other=run()
        self.assertEqual(other.metadata['inputs']['search_policy']['max_cell_ball_candidates'],8)

    def test_tiny_radius_box_pruning_is_outward_and_terminates(self):
        result = run(records=[seed((.125, 0, 0))], width=1e-310, max_depth=0)
        self.assertEqual(result.summary['confirmed_leaf_count'], 0)
        self.assertEqual(result.arrays['status'].tolist(), [5])

    def test_independent_fraction_vertex_audit_of_confirmed_leaves(self):
        result = run(width=125, max_depth=3)
        points = [(Fraction(0), Fraction(0), Fraction(0)),
                  (Fraction(1, 8), Fraction(0), Fraction(0)),
                  (Fraction(0), Fraction(1, 8), Fraction(0))]
        for depth, code, state in zip(result.arrays['depth'], result.arrays['code'], result.arrays['status']):
            if state != 1:continue
            triangle = points
            for level in range(int(depth)-1, -1, -1):
                a,b,c = triangle
                ab,bc,ca = [tuple((x+y)/2 for x,y in zip(p,q)) for p,q in ((a,b),(b,c),(c,a))]
                triangle = ((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca))[(int(code)//4**level)%4]
            for point in triangle:
                self.assertLessEqual(sum(value*value for value in point), Fraction(1,16)**2)
