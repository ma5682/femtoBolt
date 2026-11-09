"""Independent oracles for the November 5 route hot paths and publication guards."""
from contextvars import copy_context
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from femto import io_utils, small_vectors
from femto.planning import edge_check_batch as batch, quality_constrained as quality
from femto.planning import surface_trajectory as trajectory, triangle_certificates as certificates
import test_serpentine_prepared_checks as edge_fixture
import test_surface_workspace as workspace_fixture


class BatchPredicateAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = edge_fixture.PreparedQualityChecksTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def compare(self, segments, record, width, allowed):
        fast_reference = quality.ReferenceEnvelope(record, width)
        old_reference = quality.ReferenceEnvelope(record, width)
        original = batch.evaluate
        outcomes = []

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            outcomes.append(result is not None)
            return result

        for _ in range(2):
            with patch.object(batch, 'evaluate', observed):
                actual = quality._edge_checks(segments, self.fixture.mesh, fast_reference, width,
                                             lambda: None, allowed_faces=allowed)
            with patch.object(batch, 'evaluate', return_value=None):
                expected = quality._edge_checks(segments, self.fixture.mesh, old_reference, width,
                                               lambda: None, allowed_faces=allowed)
            self.fixture.assert_same(actual, expected)
            self.assertEqual(fast_reference.sampled_pixels, old_reference.sampled_pixels)
        return any(outcomes)

    def test_batched_decisions_margins_and_repeated_residual_charges_match_scalar(self):
        rng = np.random.default_rng(4515)
        starts = np.c_[rng.uniform(-.15, .15, (128, 2)), np.full(128, .5)]
        ends = starts.copy()
        ends[:, :2] += rng.uniform(-.015, .015, (128, 2))
        segments = [self.fixture.segment(a, b) for a, b in zip(starts, ends)]
        segments += [self.fixture.segment((-.11, 0., .5), (.11, 0., .5))]
        enabled = []
        for holes in (False, True):
            if holes:
                self.fixture.mask[48:53, 48:53] = False
            record = self.fixture.record()
            for width in (1., 5., 30.):
                for allowed in (self.fixture.allowed, np.array([False, True])):
                    enabled.append(self.compare(segments, record, width, allowed))
        self.assertTrue(any(enabled), 'the optimized path must actually be exercised')

    def test_sparse_rectangle_maxima_match_direct_slices(self):
        rng = np.random.default_rng(815)
        values = rng.integers(0, 20000, (100, 130), dtype=np.int32)
        ya = rng.integers(0, 99, 160)
        xa = rng.integers(0, 129, 160)
        yb = np.array([rng.integers(y + 1, 101) for y in ya])
        xb = np.array([rng.integers(x + 1, 131) for x in xa])
        expected = [values[y:Y, x:X].max() for y, Y, x, X in zip(ya, yb, xa, xb)]
        np.testing.assert_array_equal(batch._range_maxima(values, ya, yb, xa, xb, lambda: None), expected)
        with patch.object(batch, '_TABLE_BYTES', 0):
            self.assertIsNone(batch._range_maxima(values, ya, yb, xa, xb, lambda: None))

    def test_boundary_search_respects_the_original_configured_budget(self):
        from femto.planning import boundary_clearance
        record = self.fixture.record()
        segments = [self.fixture.segment()] * 10
        with patch.object(boundary_clearance, 'MAX_DISTANCE_CHECKS', 0):
            with self.assertRaisesRegex(ValueError, '超过预算'):
                quality.edge_checks(segments, self.fixture.mesh, record, 1.)

    def test_long_boundary_declines_before_expanding_billions_of_pieces(self):
        boundary = np.array([[[0., 0., 0.], [1e6, 0., 0.]]])
        with patch.object(batch.np, 'repeat', side_effect=AssertionError('unbounded allocation')):
            self.assertIsNone(batch._segment_index(boundary))

    def test_boundary_decisions_match_exhaustive_distance_and_fall_back_near_equality(self):
        from femto.planning.region_coverage import point_segment_distance
        rng = np.random.default_rng(179)
        boundary = rng.uniform(-.04, .04, (13, 2, 3))
        points = rng.uniform(-.05, .05, (110, 3))
        expected = np.min([point_segment_distance(points, a, b) for a, b in boundary], axis=0)
        index = batch._segment_index(boundary)
        for shift in (-1e-4, 1e-4):
            limits = expected + shift
            np.testing.assert_array_equal(batch._clearance(points, limits, index, lambda: None), expected >= limits)
        self.assertIsNone(batch._clearance(points, expected, index, lambda: None))
        with patch.object(batch, '_QUERY_PAIRS', 0):
            self.assertIsNone(batch._clearance(points, expected, index, lambda: None))

    def test_cancelled_preparation_never_returns_partial_decisions(self):
        def cancel():
            raise RuntimeError('cancel boundary preparation')
        with self.assertRaisesRegex(RuntimeError, 'cancel boundary preparation'):
            batch._segment_index(np.array([[[0., 0., 0.], [.01, .01, 0.]]]), cancel)


class NumericCertificateAcceptanceTests(unittest.TestCase):
    def test_every_positive_certificate_is_accepted_by_original_scalar_geometry(self):
        rng = np.random.default_rng(51105)
        vertices = []
        points = []
        endpoints = []
        # Include nearly flat triangles, translations and points at plane tolerance.
        for scale in (1e-5, .001, .1, 10., 1000.):
            for sine in (1e-5, .00101, .01, .5):
                for _ in range(15):
                    origin = rng.uniform(-10., 10., 3)
                    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                    ab = rotation[:, 0] * scale
                    ac = (rotation[:, 0] + rotation[:, 1] * sine) * scale
                    normal = np.cross(ab, ac)
                    normal /= np.linalg.norm(normal)
                    uv = rng.dirichlet([1., 1., 1.])[:2]
                    offset = rng.choice([0., 1e-6 * (1 - 2e-6), 1e-6 * (1 + 2e-6)])
                    vertices.extend((origin, origin + ab, origin + ac))
                    points.append(origin + uv[0] * ab + uv[1] * ac + normal * offset)
                    endpoints.append(origin + .3 * ab + .3 * ac + normal * offset)
        vertices, points, endpoints = map(np.array, (vertices, points, endpoints))
        faces = np.arange(len(vertices)).reshape(-1, 3)
        ids = np.arange(len(faces))
        point_flags = certificates.points_on_triangles(points, vertices, faces, ids)
        chord_flags = certificates.chords_on_triangles(points, endpoints, vertices, faces, ids)
        self.assertTrue(point_flags.any())
        self.assertTrue(chord_flags.any())
        for i in np.flatnonzero(point_flags):
            self.assertTrue(trajectory._point_on_triangle(points[i], vertices, faces[i]), ('point', i))
        for i in np.flatnonzero(chord_flags):
            self.assertTrue(quality._supported_chord(points[i], endpoints[i], faces, vertices, [i]), ('chord', i))

    def test_small_vector_results_are_bit_identical_for_mixed_signs_and_strides(self):
        rng = np.random.default_rng(129)
        for n in (2, 3):
            left = rng.normal(size=(4096, n * 2))[:, ::2]
            right = rng.normal(size=(4096, n * 2))[:, ::2]
            left *= rng.choice([1e-160, 1e-5, 1., 1e140], (len(left), 1))
            right *= rng.choice([1e-160, 1e-5, 1., 1e140], (len(right), 1))
            expected_dots = np.array([np.dot(a, b) for a, b in zip(left, right)])
            expected_norms = np.array([np.linalg.norm(a) for a in left])
            self.assertEqual(small_vectors.row_dots(left, right).tobytes(), expected_dots.tobytes())
            self.assertEqual(small_vectors.row_norms(left).tobytes(), expected_norms.tobytes())

    def test_small_vector_reduction_retains_blas_positive_zero_seed(self):
        left = np.array([[0., 0.], [-0., -0.], [1., -1.], [1e-300, -1e-300]])
        right = np.array([[-1., -1.], [1., 1.], [0., -0.], [-1e-300, 1e-300]])
        for a, b in ((left, right), (np.c_[left, np.zeros(4)], np.c_[right, np.ones(4)])):
            expected = np.array([np.dot(x, y) for x, y in zip(a, b)])
            self.assertEqual(small_vectors.row_dots(a, b).tobytes(), expected.tobytes())


class CompactJsonAcceptanceTests(unittest.TestCase):
    def test_compact_json_checks_cancel_during_traversal_before_encoding_all_rows(self):
        visited = []

        class ObservedRows(list):
            def __iter__(self):
                for row in super().__iter__():
                    visited.append(row)
                    yield row

        rows = ObservedRows(range(20000))

        def cancel():
            if len(visited) >= 2048:
                raise RuntimeError('cancel while traversing')

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'manifest.json'
            target.write_bytes(b'previous')
            with self.assertRaisesRegex(RuntimeError, 'cancel while traversing'):
                io_utils.write_json_strict(target, {'rows': rows}, indent=None, check=cancel)
            self.assertEqual(target.read_bytes(), b'previous')
            self.assertEqual(list(Path(temporary).iterdir()), [target])
        self.assertLess(len(visited), len(rows))


class SelectionProjectionAcceptanceTests(unittest.TestCase):
    def test_many_components_do_not_retain_an_image_per_component(self):
        import weakref
        from femto.planning.selection_blocks import check_projection_inside_selection
        retained = []
        peak = []

        class Labels(np.ndarray):
            def __eq__(self, value):
                mask = np.asarray(self) == value
                retained.append(weakref.ref(mask))
                peak.append(sum(ref() is not None for ref in retained))
                return mask

        labels = np.tile(np.arange(1, 31), (10, 1)).view(Labels)
        blocks = [dict(component_id=i) for i in range(30)]
        segments = [trajectory.ScanSegment(i, i, [dict(xyz_m=[float(i), 4., 1.])], i) for i in range(30)]
        check_projection_inside_selection(segments, blocks, labels, np.eye(3), np.array([30, 10]))
        self.assertLessEqual(max(peak), 2)
        self.assertFalse(any(ref() is not None for ref in retained))


class IndependentReplayAcceptanceTests(unittest.TestCase):
    def test_fresh_fast_and_scalar_generation_and_independent_archive_replay_match(self):
        writer = r'''
import sys,json
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
from copy import deepcopy
import numpy as np
import open3d as o3d
sys.path.insert(0,'tests')
import test_surface_workspace as fixture
from femto.planning import surface_workspace as workspace, region_project, edge_check_batch, triangle_certificates
from femto import small_vectors
case=fixture.SurfaceWorkspaceTests();case.setUp()
try:
    o3d.io.write_triangle_mesh(case.model.store.source,case.model.mesh)
    region=case.region(split=True)
    region['annotation_id']='claude-acceptance-fixed-region'
    with ExitStack() as stack:
        if sys.argv[2]=='scalar':
            stack.enter_context(patch.object(edge_check_batch,'evaluate',return_value=None))
            stack.enter_context(patch.object(triangle_certificates,'points_on_triangles',side_effect=lambda points,*a,**k:np.zeros(len(points),bool)))
            stack.enter_context(patch.object(triangle_certificates,'chords_on_triangles',side_effect=lambda points,*a,**k:np.zeros(len(points),bool)))
            stack.enter_context(patch.object(small_vectors,'_PLAIN',False))
        plan=workspace.generate(case.model,region,case.head)
        native=workspace.export(case.model,region,plan)
        archive=Path(sys.argv[1])/'plan.region.zip'
        region_project.save_project(case.model,region,archive,plan)
        import shutil
        shutil.copytree(native,Path(sys.argv[1])/'native')
        payload=plan['surface_trajectory']
        fields=('ordered_items','statistics','gap_report','spacing_report','row_report','blocks')
        (Path(sys.argv[1])/'expected.json').write_text(json.dumps({k:payload[k] for k in fields},ensure_ascii=False),encoding='utf-8')
    # Keep native source bindings alive until the independent reader exits.
    import subprocess
    reader=r"""
import sys,json
from pathlib import Path
from femto.planning.trajectory_export import load_trajectory
from femto.planning.region_project import open_project
root=Path(sys.argv[1]);expected=json.loads((root/'expected.json').read_text('utf-8'))
native=load_trajectory(root/'native').to_dict()
opened=open_project(root/'plan.region.zip',workspace=root/'opened')
portable=opened.saved_plan['surface_trajectory']
assert {k:native[k] for k in expected}==expected,'native replay changed route'
assert {k:portable[k] for k in expected}==expected,'portable replay changed route'
"""
    result=subprocess.run([sys.executable,'-X','utf8','-c',reader,sys.argv[1]],capture_output=True,text=True,encoding='utf-8',timeout=90)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
finally:case.doCleanups()
'''
        with tempfile.TemporaryDirectory(prefix='femto_route_replay_') as directory:
            summaries = []
            for mode in ('scalar', 'fast'):
                root = Path(directory) / mode
                root.mkdir()
                result = subprocess.run([sys.executable, '-X', 'utf8', '-c', writer, str(root), mode],
                                        capture_output=True, text=True, encoding='utf-8', timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                summaries.append(json.loads((root/'expected.json').read_text(encoding='utf-8')))
            differences = []
            def compare(left, right, path=''):
                if type(left) is not type(right):
                    differences.append((path, left, right))
                elif isinstance(left, dict):
                    if left.keys() != right.keys():
                        differences.append((path, sorted(left), sorted(right)))
                    else:
                        for key in left:compare(left[key], right[key], path+'/'+key)
                elif isinstance(left, list):
                    if len(left) != len(right):
                        differences.append((path, len(left), len(right)))
                    else:
                        for i, (a, b) in enumerate(zip(left, right)):compare(a, b, path+'/'+str(i))
                elif left != right:
                    differences.append((path, left, right))
            compare(summaries[1], summaries[0])
            self.assertFalse(differences, repr(differences[:20]))


class ExportReceiptAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from femto.planning import surface_workspace
        cls.fixture = workspace_fixture.SurfaceWorkspaceTests()
        cls.fixture.setUp()
        cls.addClassCleanup(cls.fixture.doCleanups)
        cls.plan = surface_workspace.generate(cls.fixture.model, cls.fixture.region(), cls.fixture.head)

    def model(self):
        return trajectory.SurfaceTrajectory.from_dict(deepcopy(self.plan['surface_trajectory']))

    def test_export_scope_rejects_unvalidated_or_mutated_nested_data(self):
        for before in (False, True):
            model = self.model()
            with self.subTest(before=before), self.assertRaises(ValueError):
                if before:
                    model.statistics['point_count'] += 1
                with trajectory.exporting(model):
                    if not before:
                        model.statistics['point_count'] += 1
                    trajectory.to_player_plan(model)

    def test_closed_or_copied_export_scope_cannot_skip_validation(self):
        model = self.model()
        with trajectory.exporting(model):
            context = copy_context()
        original = trajectory.validate_trajectory
        with patch.object(trajectory, 'validate_trajectory', wraps=original) as validate:
            context.run(trajectory.to_player_plan, model)
        self.assertEqual(sum(call.args[0] is model for call in validate.call_args_list), 1)

    def test_active_export_reuses_only_unchanged_content_and_rejects_other_thread(self):
        from concurrent.futures import ThreadPoolExecutor
        model = self.model()
        original = trajectory.validate_trajectory
        with trajectory.exporting(model), patch.object(trajectory, 'validate_trajectory', wraps=original) as validate:
            expected = trajectory.to_player_plan(model)
            self.assertEqual(validate.call_count, 0)
            context = copy_context()
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertEqual(pool.submit(context.run, trajectory.to_player_plan, model).result(), expected)
            self.assertEqual(sum(call.args[0] is model for call in validate.call_args_list), 1)

    def test_export_callback_mutation_cannot_publish_any_artifact(self):
        from femto.planning.trajectory_export import export_trajectory
        model = self.model()

        def change_after_writing(folder, current, check):
            current.statistics['point_count'] += 1

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / 'exports'
            with self.assertRaisesRegex(ValueError, '发生变化'):
                export_trajectory(model, parent, include_visuals=False, augment=change_after_writing)
            self.assertEqual(list(parent.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
