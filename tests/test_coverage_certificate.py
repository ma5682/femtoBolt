"""Analytic truth cases for the independent nominal coverage prototype."""
from copy import deepcopy
from fractions import Fraction as Q
import math
import unittest
from unittest.mock import patch

import numpy as np

from femto.geometry import exact_surface as exact
from femto.planning import coverage_certificate as coverage, footprint_model
from femto.planning.coverage_domain import CoverageDomain, scan_endpoints
from femto.planning.planning_revision import array_record, digest


def triangle_domain(scale=.125):
    return CoverageDomain.mesh([[0, 0, 0], [scale, 0, 0], [0, scale, 0]], [[0, 1, 2]])


def seed(point=(0., 0., 0.), face=0, role='scan'):
    return dict(source_face=face, xyz_m=list(point), role=role)


def run(domain=None, records=None, width=500., **limits):
    return coverage.evaluate(triangle_domain() if domain is None else domain,
                             [seed()] if records is None else records,
                             footprint_model.nominal(width), budget=coverage.CoverageBudget(**limits))


def reseal(certificate, *, arrays=None, metadata=None):
    data = deepcopy(certificate.metadata) if metadata is None else metadata
    values = {k: v.copy() for k, v in certificate.arrays.items()} if arrays is None else arrays
    data['arrays'] = {k: array_record(v) for k, v in sorted(values.items())}
    data['sha256'] = digest({k: v for k, v in data.items() if k != 'sha256'})
    return coverage.Certificate(data, values)


class ExactSurfaceTests(unittest.TestCase):
    def test_boundary_and_adjacent_binary_values_use_exact_fallback(self):
        center = exact.rational([0, 0, 0])
        for radius in (.125, 2.**-100, 2.**100):
            for value, expected in ((math.nextafter(radius, 0), True), (radius, True),
                                    (math.nextafter(radius, math.inf), False)):
                tri = tuple(exact.rational(p) for p in ((value, 0, 0), (0, 0, 0), (0, radius/2, 0)))
                with self.subTest(radius=radius, value=value):
                    self.assertEqual(exact.contains_ball(tri, center, radius)[0], expected)

    def test_interval_filter_matches_independent_exact_vertex_test(self):
        random = np.random.default_rng(90210)
        for _ in range(150):
            values = random.uniform(-2, 2, (4, 3))
            tri = tuple(exact.rational(p) for p in values[:3]); center = exact.rational(values[3])
            radius = float(random.uniform(.01, 4))
            expected = all(sum((p[i]-center[i])**2 for i in range(3)) <= Q(radius)**2 for p in tri)
            self.assertEqual(exact.contains_ball(tri, center, radius)[0], expected)

    def test_batched_predicate_matches_exact_results_at_ulp_boundaries(self):
        triangle = tuple(exact.rational(p) for p in ((.125, 0, 0), (0, 0, 0), (0, .0625, 0)))
        centers = tuple(exact.rational(p) for p in ((0, 0, 0), (0, 0, .25), (0, 0, 0)))
        radii = [.125, .125, math.nextafter(.125, 0)]
        result, fallbacks = exact.contains_balls(triangle, centers, radii)
        self.assertEqual(result.tolist(), [True, False, False]); self.assertGreater(fallbacks, 0)

    def test_anchor_keeps_original_point_and_deducts_off_plane_distance(self):
        tri = tuple(exact.rational(p) for p in ((0, 0, 0), (.125, 0, 0), (0, .125, 0)))
        point = [0., 0., .03125]
        bary, center, inner = exact.anchor_ball(point, tri, .125)
        self.assertEqual(center, (0, 0, 0)); self.assertEqual(bary, (1, 0, 0))
        self.assertLessEqual(Q(inner)+Q(.03125), Q(.125))
        self.assertEqual(point, [0., 0., .03125])
        self.assertIsNone(exact.anchor_ball([.25, .25, 0], tri, .5))
        self.assertIsNone(exact.anchor_ball([0, 0, .125], tri, .125))

    def test_anchor_squared_displacement_underflow_remains_bounded(self):
        tri = tuple(exact.rational(p) for p in ((0, 0, 0), (.125, 0, 0), (0, .125, 0)))
        for displacement in (1e-200, math.nextafter(0., 1.)):
            bary, center, inner = exact.anchor_ball([0, 0, displacement], tri, .125)
            self.assertEqual(bary, (1, 0, 0)); self.assertEqual(center, (0, 0, 0))
            self.assertLessEqual(Q(inner)+Q(displacement), Q(.125))

    def test_exact_half_pixel_clip_and_dyadic_children_partition(self):
        tri = tuple(exact.rational(p) for p in ((-.125, -.125, .5), (.125, -.125, .5), (0, .125, .5)))
        polygon = exact.clip_pixel(tri, np.array([[100., 0, 50], [0, 100., 50], [0, 0, 1.]]), 50, 50)
        points = [exact.xyz(bary, tri) for bary in polygon]
        self.assertEqual(set(p[0] for p in points), {Q(-1, 400), Q(1, 400)})
        self.assertEqual(set(p[1] for p in points), {Q(-1, 400), Q(1, 400)})
        for part in exact.subdivide(exact.IDENTITY):
            self.assertEqual(abs(exact.orient(*part)), Q(1, 4))
        self.assertEqual(exact.descend(exact.IDENTITY, 2, 7), exact.subdivide(exact.subdivide(exact.IDENTITY)[1])[3])

    def test_shared_source_edge_orientation_and_partial_intervals(self):
        a = exact.edge_interval(0, (0, 1, 2), exact.IDENTITY[1], exact.IDENTITY[2])
        part = exact.subdivide(exact.IDENTITY)[0]
        b = exact.edge_interval(1, (2, 1, 3), part[0], part[1])
        self.assertEqual(a[0], b[0]); self.assertGreater(min(a[2], b[2])-max(a[1], b[1]), 0)


class CoverageCertificateTests(unittest.TestCase):
    def test_whole_triangle_has_positive_complete_proof(self):
        result = run()
        self.assertEqual(result.summary['status'], 'model_coverage_complete')
        self.assertEqual(result.summary['confirmed_nominal_area_m2'], .125**2/2)
        self.assertEqual(result.arrays['proof_parent'].tolist(), [-1])
        self.assertFalse(result.summary['execution_ready']); self.assertFalse(result.summary['physical_accuracy_verified'])
        self.assertFalse(result.arrays['root'].flags.writeable)

    def test_partial_intersection_subdivides_to_a_nonzero_subset(self):
        result = run(width=125, max_depth=3)
        # A quadrant of a disk has area pi*r^2/4. Positive certified area must
        # stay below this analytic intersection, and below the original triangle.
        area = result.summary['confirmed_nominal_area_m2']
        self.assertGreater(area, 0); self.assertLessEqual(area, math.pi*(.0625**2)/4)
        self.assertLess(area, result.summary['located_nominal_area_m2'])
        self.assertIn('depth_limit', result.metadata['metrics']['limits_reached'])
        self.assertFalse(result.summary['original_target_coverage_complete'])

    def test_three_distinct_vertex_balls_do_not_certify_whole_triangle(self):
        result = run(records=[seed(), seed((.125, 0, 0)), seed((0, .125, 0))], width=62.5, max_depth=0)
        self.assertEqual(result.summary['confirmed_leaf_count'], 0)
        self.assertEqual(result.arrays['status'].tolist(), [5])

    def test_overlapping_balls_count_each_leaf_once(self):
        result = run(records=[seed(), seed((.03125, .03125, 0)), seed()])
        self.assertEqual(result.summary['confirmed_nominal_area_m2'], .125**2/2)
        self.assertGreater(len(result.arrays['proof_leaf']), result.summary['confirmed_leaf_count'])

    def test_mixed_depth_shared_subedges_reach_second_face(self):
        domain = CoverageDomain.mesh([[0, 0, 0], [.125, 0, 0], [0, .125, 0], [.125, .125, 0]],
                                     [[0, 1, 2], [1, 3, 2]])
        result = run(domain, width=250, max_depth=3)
        self.assertGreater(result.summary['confirmed_nominal_area_m2'], .125**2/2)
        self.assertEqual(result.arrays['depth'][result.arrays['root'] == 0].tolist(), [0])
        self.assertTrue(np.any(result.arrays['depth'][result.arrays['root'] == 1] > 0))
        confirmed_faces = result.arrays['root'][result.arrays['status'] == 1]
        self.assertEqual(set(confirmed_faces.tolist()), {0, 1})

    def test_distinct_layers_are_not_welded_by_distance(self):
        vertices = [[x, y, z] for z in (0, .03125) for x, y in ((0, 0), (.125, 0), (0, .125))]
        result = run(CoverageDomain.mesh(vertices, [[0, 1, 2], [3, 4, 5]]))
        self.assertEqual(result.arrays['status'].tolist(), [1, 4])
        self.assertEqual(result.summary['confirmed_nominal_area_m2'], .125**2/2)

    def test_zero_or_positive_coordinate_gap_does_not_invent_shared_ids(self):
        for gap in (0., float(np.nextafter(np.float32(.125), np.float32(1)))-.125):
            vertices = [[0, 0, 0], [.125, 0, 0], [0, .125, 0],
                        [.125+gap, 0, 0], [.125+gap, .125, 0], [gap, .125, 0]]
            result = run(CoverageDomain.mesh(vertices, [[0, 1, 2], [3, 4, 5]]))
            self.assertEqual(result.arrays['status'].tolist(), [1, 4])

    def test_vertex_contact_alone_is_not_a_shared_subedge(self):
        vertices = [[0, 0, 0], [.125, 0, 0], [0, .125, 0], [-.125, 0, 0], [0, -.125, 0]]
        result = run(CoverageDomain.mesh(vertices, [[0, 1, 2], [0, 3, 4]]))
        self.assertEqual(result.arrays['status'].tolist(), [1, 4])

    def test_topological_fold_must_connect_inside_the_same_ball(self):
        # Both end triangles lie in r=.25, but their only mesh path passes
        # through an excursion at x=2; a spatial union would overcount the far end.
        vertices = [[0, 0, 0], [.125, 0, 0], [0, .125, 0], [2, 0, 0],
                    [2, .125, 0], [.125, 0, .03125], [0, .125, .03125], [0, 0, .03125]]
        faces = [[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4], [4, 5, 6], [5, 7, 6]]
        result = run(CoverageDomain.mesh(vertices, faces), max_depth=2)
        self.assertEqual(result.arrays['status'][result.arrays['root'] == 5].tolist(), [4])
        self.assertGreater(result.summary['confirmed_nominal_area_m2'], 0)

    def test_hole_is_preserved_and_not_added_to_area(self):
        vertices = [[x, y, 0] for y in (0, .0625, .125, .1875) for x in (0, .0625, .125, .1875)]
        faces = []
        for y in range(3):
            for x in range(3):
                if x == y == 1:continue
                a = y*4+x; faces.extend([[a, a+1, a+4], [a+1, a+5, a+4]])
        result = run(CoverageDomain.mesh(vertices, faces), width=1000)
        self.assertEqual(result.summary['confirmed_nominal_area_m2'], 8*(.0625**2))
        self.assertEqual(result.summary['status'], 'model_coverage_complete')

    def test_duplicate_reversed_faces_and_seed_alias_are_counted_once(self):
        vertices = [[0, 0, 0], [.125, 0, 0], [0, .125, 0]]*2
        domain = CoverageDomain.mesh(vertices, [[0, 1, 2], [5, 4, 3]])
        self.assertEqual(len(domain.areas), 1)
        result = run(domain, records=[seed(face=1)])
        self.assertEqual(result.summary['status'], 'model_coverage_complete')
        self.assertEqual(result.summary['confirmed_nominal_area_m2'], .125**2/2)

    def test_connectors_do_not_generate_coverage(self):
        result = run(records=[seed(role='connector')])
        self.assertEqual(result.summary['confirmed_leaf_count'], 0)
        self.assertEqual(result.metadata['metrics']['ignored_connectors'], 1)

    def test_no_seeds_and_empty_mesh_are_not_complete(self):
        for result in (run(records=[]), run(CoverageDomain.mesh(np.empty((0, 3)), np.empty((0, 3), dtype=int)), records=[])):
            self.assertFalse(result.summary['original_target_coverage_complete'])
            self.assertEqual(result.summary['confirmed_leaf_count'], 0)

    def test_invalid_and_degenerate_meshes_are_rejected(self):
        for vertices, faces in (([[0, 0, 0]]*3, [[0, 1, 2]]), ([[0, 0, 0]], [[0, 1, 2]]),
                                ([[math.nan, 0, 0]]*3, [[0, 1, 2]])):
            with self.assertRaises(ValueError):CoverageDomain.mesh(vertices, faces)

    def test_out_of_face_or_off_surface_seeds_remain_unbound(self):
        result = run(records=[seed((2, 2, 0)), seed((0, 0, 2))])
        self.assertEqual(result.metadata['metrics']['unbound_seeds'], 2)
        self.assertEqual(result.summary['confirmed_leaf_count'], 0)

    def test_tiny_positive_radius_does_not_overflow_spatial_index(self):
        result = run(records=[seed((.125, 0, 0))], width=1e-310, max_depth=0)
        self.assertEqual(result.summary['confirmed_leaf_count'], 0)
        self.assertFalse(result.summary['original_target_coverage_complete'])

    def test_budgets_preserve_original_area_and_cannot_mark_complete(self):
        options = [('seeds', dict(max_seeds=0)), ('comparisons', dict(max_comparisons=0)),
                   ('proofs', dict(max_proofs=0)), ('time', dict(seconds=0)),
                   ('cells', dict(max_added_cells=0, width=125)), ('depth', dict(max_depth=0, width=125))]
        for name, limits in options:
            with self.subTest(name=name):
                result = run(**limits)
                self.assertEqual(result.summary['status'], 'calculation_incomplete')
                self.assertEqual(result.summary['located_nominal_area_m2'], .125**2/2)
                self.assertGreaterEqual(result.summary['leaf_count'], 1)
                coverage.validate(result, triangle_domain(), [seed()], footprint_model.nominal(limits.get('width', 500)))

    def test_adjacency_budget_preserves_unconfirmed_cells(self):
        domain = CoverageDomain.mesh([[0, 0, 0], [.125, 0, 0], [0, .125, 0], [.125, .125, 0]], [[0, 1, 2], [1, 3, 2]])
        result = run(domain, max_adjacencies=0)
        self.assertIn('adjacencies_budget', result.metadata['metrics']['limits_reached'])
        self.assertFalse(result.summary['original_target_coverage_complete'])
        self.assertEqual(result.arrays['status'].tolist(), [1, 4])

    def test_each_parent_chain_keeps_one_ball_after_multiple_ball_union(self):
        domain = CoverageDomain.mesh([[0, 0, 0], [.125, 0, 0], [0, .125, 0], [.125, .125, 0]], [[0, 1, 2], [1, 3, 2]])
        result = run(domain, records=[seed(), seed((.125, .125, 0), face=1)])
        arrays = result.arrays
        self.assertEqual(result.summary['confirmed_nominal_area_m2'], .125**2)
        for i, parent in enumerate(arrays['proof_parent']):
            if parent >= 0:
                self.assertLess(parent, i)
                self.assertEqual(arrays['proof_seed'][i], arrays['proof_seed'][parent])

    def test_cancel_is_propagated_instead_of_certifying_partial_work(self):
        calls = 0
        def check():
            nonlocal calls
            calls += 1
            if calls > 1:raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            coverage.evaluate(triangle_domain(), [seed()], footprint_model.nominal(500), check=check)

    def test_time_frontier_replays_without_wall_clock_equivalence(self):
        ticks = iter([0., 0., 0., 100.])
        with patch('femto.planning.coverage_certificate.time.monotonic', side_effect=lambda: next(ticks, 100.)):
            result = run()
        self.assertEqual(result.metadata['metrics']['stop_reason'], 'time_budget')
        coverage.validate(result, triangle_domain(), [seed()], footprint_model.nominal(500))

    def test_tampering_rejected_even_after_all_public_hashes_are_recomputed(self):
        result = run()
        for field, value in (('witness', -1), ('proof_leaf', 99), ('proof_parent', 0), ('seed_face', 99), ('code', 1)):
            with self.subTest(field=field):
                arrays = {k: v.copy() for k, v in result.arrays.items()}; arrays[field][0] = value
                with self.assertRaises(ValueError):
                    coverage.validate(reseal(result, arrays=arrays), triangle_domain(), [seed()], footprint_model.nominal(500))
        data = deepcopy(result.metadata); data['summary']['confirmed_nominal_area_m2'] *= 2
        with self.assertRaises(ValueError):
            coverage.validate(reseal(result, metadata=data), triangle_domain(), [seed()], footprint_model.nominal(500))

    def test_other_geometry_seeds_or_model_cannot_reuse_certificate(self):
        result = run()
        for domain, records, model in ((triangle_domain(.25), [seed()], footprint_model.nominal(500)),
                                       (triangle_domain(), [seed((.03125, 0, 0))], footprint_model.nominal(500)),
                                       (triangle_domain(), [seed()], footprint_model.nominal(600))):
            with self.assertRaises(ValueError):coverage.validate(result, domain, records, model)

    def test_route_binding_cannot_be_replaced_while_reusing_the_same_seeds(self):
        route = dict(schema='declared-test-route', route_record_sha256='a'*64)
        result = coverage.evaluate(triangle_domain(), [seed()], footprint_model.nominal(500), route_binding=route)
        with self.assertRaises(ValueError):
            coverage.validate(result, triangle_domain(), [seed()], footprint_model.nominal(500),
                              route_binding=dict(route, route_record_sha256='b'*64))

    def test_malformed_seed_and_budget_inputs_are_rejected(self):
        for record in (seed(face=-1), seed(face=True), seed((math.nan, 0, 0)), seed(role='unknown')):
            with self.subTest(record=record), self.assertRaises(ValueError):run(records=[record])
        for budget in (dict(max_depth=9), dict(max_added_cells=-1), dict(seconds=46), dict(max_proofs=True)):
            with self.subTest(budget=budget), self.assertRaises(ValueError):run(**budget)

    def test_nominal_parameters_remain_explicitly_unknown_where_uncalibrated(self):
        model = footprint_model.nominal(20)
        self.assertEqual(model['nominal_radius_m'], .01)
        for key in ('physical_center_offset_bound_m', 'registration_uncertainty_m', 'calibration_uncertainty_m'):
            self.assertIsNone(model[key])
        for width in (0, -1, math.nan, math.inf, True, math.nextafter(0., 1.)):
            with self.subTest(width=width), self.assertRaises(ValueError):footprint_model.nominal(width)

    def test_scan_endpoint_roles_follow_incoming_edge_and_full_face_mapping(self):
        from femto.planning.surface_trajectory import ScanSegment
        from types import SimpleNamespace
        points = [dict(xyz_m=[float(i), 0., 0.], triangle_index=0, role=role)
                  for i, role in enumerate(('connector', 'scan', 'connector', 'scan'))]
        item = ScanSegment(segment_index=9, row_index=0, points=points)
        trajectory = SimpleNamespace(ordered_items=[item])
        records = scan_endpoints(trajectory, np.array([17]))
        self.assertEqual([r['xyz_m'][0] for r in records], [0., 1., 2., 3.])
        self.assertEqual([r['edge_index'] for r in records], [0, 0, 2, 2])
        self.assertEqual({r['source_face'] for r in records}, {17})

    def test_implicit_v3_scan_role_and_zero_length_edges(self):
        from femto.planning.surface_trajectory import ScanSegment
        from types import SimpleNamespace
        points = [dict(xyz_m=[x, 0., 0.], triangle_index=0) for x in (0., .125, .125)]
        records = scan_endpoints(SimpleNamespace(ordered_items=[ScanSegment(0, 0, points)]), np.array([0]))
        self.assertEqual(len(records), 2)
        self.assertEqual([r['edge_index'] for r in records], [0, 0])
        points[1]['role'] = None
        self.assertEqual(records, scan_endpoints(SimpleNamespace(ordered_items=[ScanSegment(0, 0, points)]), np.array([0])))


class CoverageOriginalTargetTests(unittest.TestCase):
    def setUp(self):
        # Import the fixture module rather than exposing its TestCase to discovery.
        import test_target_surface
        self.fixture = test_target_surface.TargetSurfaceTests()
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)

    def test_exact_pixel_domain_preserves_original_id_and_positive_area(self):
        import open3d as o3d
        from test_mesh_slice_planner import grid_mesh
        from femto.planning import target_surface, surface_workspace
        from femto.planning.effective_domain import build
        mesh = grid_mesh(width=.03, height=.03, cell=.005)
        native = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
        mask = np.zeros((100, 100), bool); mask[47:54, 47:54] = True
        model, region = self.fixture.fixture(mesh=native, mask=mask)
        ledger, revision, evidence = self.fixture.build(model, region)
        quality = build(surface_workspace.canonical_region_mesh(model.mesh, region), region, evidence, 5)
        ledger = target_surface.with_quality(ledger, revision, region, evidence, quality)
        before = deepcopy(ledger.metadata)
        domain = CoverageDomain.target(ledger, revision, evidence=evidence, effective_domain=quality)
        eligible = int(np.flatnonzero(domain.quality == 0)[0]); face = int(domain.source_faces[eligible])
        point = domain.vertices[domain.faces[face]].mean(axis=0).tolist()
        result = run(domain, records=[seed(point, face)], width=100)
        self.assertGreater(result.summary['confirmed_nominal_area_m2'], 0)
        self.assertEqual(ledger.metadata, before)
        self.assertEqual(domain.binding['target_id'], ledger.metadata['target_id'])
        self.assertIsNone(ledger.metadata['coverage_certificate'])
        target_surface.validate(ledger, revision, evidence=evidence, domain=quality, reproduce=True)

    def test_quality_exclusion_keeps_all_original_area_unconfirmed(self):
        from femto.planning import target_surface, surface_workspace
        from femto.planning.effective_domain import build
        model, region = self.fixture.fixture()
        ledger, revision, evidence = self.fixture.build(model, region)
        quality = build(surface_workspace.canonical_region_mesh(model.mesh, region), region, evidence, 5)
        filtered = target_surface.with_quality(ledger, revision, region, evidence, quality)
        domain = CoverageDomain.target(filtered, revision, evidence=evidence, effective_domain=quality)
        result = run(domain, records=[seed((0., 0., .5), 0)], width=100)
        self.assertEqual(result.summary['confirmed_nominal_area_m2'], 0)
        self.assertEqual(result.summary['located_nominal_area_m2'], ledger.summary['located_area_m2'])
        self.assertEqual(set(result.arrays['status'].tolist()), {2})

    def test_unknown_pixels_never_become_zero_area_complete(self):
        model, region = self.fixture.fixture(depth=np.zeros((100, 100), np.uint16))
        ledger, revision, evidence = self.fixture.build(model, region)
        domain = CoverageDomain.target(ledger, revision, evidence=evidence, effective_domain=None)
        result = run(domain, records=[])
        self.assertGreater(result.summary['unlocated_pixel_count'], 0)
        self.assertIsNone(result.summary['unlocated_area_m2'])
        self.assertIsNone(result.summary['total_original_target_area_m2'])
        self.assertFalse(result.summary['original_target_coverage_complete'])

    def test_other_computation_revision_is_rejected(self):
        model, region = self.fixture.fixture()
        ledger, revision, evidence = self.fixture.build(model, region)
        domain = CoverageDomain.target(ledger, revision, evidence=evidence, effective_domain=None)
        with self.assertRaisesRegex(ValueError, '计算修订'):
            coverage.evaluate(domain, [], footprint_model.nominal(100), route_binding=dict(computation_sha256='0'*64))

    def test_real_adapter_preserves_a_generated_v3_route_and_implicit_scan_samples(self):
        import test_surface_workspace
        from femto.planning import surface_workspace
        from femto.planning.coverage_domain import from_plan
        fixture = test_surface_workspace.SurfaceWorkspaceTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        region = fixture.region(); plan = surface_workspace.generate(fixture.model, region, fixture.head)
        before = deepcopy(plan['surface_trajectory']); points = deepcopy(plan['segments'])
        domain, seeds, model, route = from_plan(fixture.model, region, plan)
        self.assertGreater(len(seeds), 100)
        self.assertTrue(route['accepted_scan_edges_only'])
        self.assertEqual(route['algorithm'], 'quality-constrained-surface-v3')
        self.assertEqual(plan['surface_trajectory'], before); self.assertEqual(plan['segments'], points)
        result = coverage.evaluate(domain, seeds, model, route_binding=route, budget=coverage.CoverageBudget(seconds=0))
        coverage.validate(result, domain, seeds, model, route_binding=route)


if __name__ == '__main__':unittest.main()
