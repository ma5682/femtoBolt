"""Source projection reuse preserves byte values, original face IDs and stops."""
from copy import deepcopy
from unittest.mock import patch
import unittest

import numpy as np
import open3d as o3d

from femto.annotation.fixed_rgb_geometry import GeometryTaskCancelled
from femto.geometry.pixel_clip import project
from femto.planning import target_surface as target
from test_fixed_rgb_experience import quad
import test_target_surface as fixtures


_SOURCE_SEPARATION = target._source_separation


def without_source_projection(triangles, intrinsic, source_vertex_ids=None, *, source_projection=None):
    """The prior per-face projection path, with identical prepared predicates."""
    return _SOURCE_SEPARATION(triangles, intrinsic, source_vertex_ids)


class TargetProjectionReuseTests(unittest.TestCase):
    def fixture(self, *, mesh=None):
        case = fixtures.TargetSurfaceTests()
        case.setUp(); self.addCleanup(case.doCleanups)
        mask = np.zeros((100, 100), bool)
        mask[48:53, 48:53] = True
        mask[49, 51] = False
        model, region = case.fixture(mesh=mesh, mask=mask)
        return case, model, region

    def assert_arrays_identical(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            with self.subTest(array=name):
                self.assertEqual(actual[name].dtype, expected[name].dtype)
                self.assertEqual(actual[name].shape, expected[name].shape)
                self.assertEqual(actual[name].tobytes(), expected[name].tobytes())

    def test_batch_projection_keeps_scalar_bytes_including_signed_zero(self):
        rng = np.random.default_rng(241024)
        for dtype in (np.float32, np.float64):
            for scale in (1e-20, 1e-8, 1., 1e8, 1e20):
                for focal in (1e-8, 1., 830.125, 1e8):
                    with self.subTest(dtype=dtype, scale=scale, focal=focal):
                        points = (rng.normal(size=(128, 3, 3)) * scale).astype(dtype)
                        points[..., 2] = np.abs(points[..., 2]) + scale / 32
                        points[0, 0, :2] = [-0., 0.]
                        k = np.array([[focal, 0., -0.], [0., focal*1.0625, 0.], [0., 0., 1.]])
                        for triangles in (points, points[::-1]):
                            batch = project(triangles, k)
                            scalar = np.stack([project(face, k) for face in triangles])
                            self.assertEqual(batch.dtype, scalar.dtype)
                            self.assertEqual(batch.tobytes(), scalar.tobytes())

    def test_nonconsecutive_face_ids_reuse_matching_rows_and_missing_faces_fall_back(self):
        a = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
        triangles = np.array([a, a+[3., 0., 0.], a, a+[6., 0., 0.], a+[9., 0., 0.]])
        source_ids = np.array([0, 2, 4])
        uv = project(triangles[source_ids], np.eye(3))
        source_ids.setflags(write=False); uv.setflags(write=False)
        separated = target._source_separation(
            triangles, np.eye(3), source_projection=(source_ids, uv))
        with patch.object(target, 'project', wraps=project) as projected:
            self.assertFalse(separated(0, 2))
            self.assertTrue(separated(0, 4))
            self.assertTrue(separated(2, 4))
            self.assertEqual(projected.call_count, 0)
            self.assertTrue(separated(1, 4))
            self.assertEqual(projected.call_count, 1)
            self.assertEqual(projected.call_args.args[0].tobytes(), triangles[1].tobytes())
        self.assertEqual(separated.cache_info().maxsize, 4096)

    def test_complete_ledger_and_occlusion_match_per_face_projection_bytes(self):
        front = quad(left=-.1, right=.1, top=-.1, bottom=.1)
        duplicate = o3d.geometry.TriangleMesh(front)
        duplicate.triangles = o3d.utility.Vector3iVector(np.asarray(duplicate.triangles)[:, ::-1])
        back = quad(.505, left=-.1, right=.1, top=-.1, bottom=.1)
        for mesh in (front, front+duplicate+back):
            case, model, region = self.fixture(mesh=mesh)
            actual, revision, quality = case.build(model, region)
            with patch.object(target, '_source_separation', without_source_projection):
                expected, _, _ = case.build(model, region)
            self.assertEqual(actual.metadata, expected.metadata)
            self.assert_arrays_identical(actual.arrays, expected.arrays)
            self.assertGreater(actual.metadata['metrics']['pair_checks'], 0)
            if len(mesh.triangles) > 2:
                self.assertGreater(actual.metadata['metrics']['occluded_polygons'], 0)
                self.assertEqual(actual.metadata['metrics']['duplicate_face_count'], 2)
            target.validate(actual, revision, evidence=quality, domain=None)

    def test_budget_frontiers_keep_pair_charges_and_all_unknown_intent(self):
        case, model, region = self.fixture()
        ledger, revision, _ = case.build(model, region)
        for budget, reason in ((target.TargetBudget(max_candidates=0), 'candidate_budget'),
                               (target.TargetBudget(max_pair_checks=3), 'pair_budget'),
                               (target.TargetBudget(max_units=7), 'unit_budget'),
                               (target.TargetBudget(seconds=0), 'time_budget')):
            with self.subTest(reason=reason):
                actual, counts = target._construct(ledger.arrays, revision, budget, lambda: None)
                with patch.object(target, '_source_separation', without_source_projection):
                    expected, old_counts = target._construct(ledger.arrays, revision, budget, lambda: None)
                self.assertEqual(counts, old_counts)
                self.assertEqual(counts['stop_reason'], reason)
                self.assert_arrays_identical(actual, expected)
                self.assertGreater(np.count_nonzero(actual['pixel_status'] == 9), 0)
        for override in (False, True):
            with patch.object(target.time, 'monotonic', side_effect=[0., 46.]):
                with patch.object(target, '_source_separation', without_source_projection if override else _SOURCE_SEPARATION):
                    arrays, counts = target._construct(ledger.arrays, revision, target.TargetBudget(), lambda: None)
            self.assertEqual(counts['stop_reason'], 'index_time_budget')
            if not override:
                actual, actual_counts = arrays, counts
            else:
                self.assertEqual(counts, actual_counts)
                self.assert_arrays_identical(arrays, actual)

    def test_cancellation_after_projection_and_during_pixels_keeps_original_checks(self):
        case, model, region = self.fixture()
        ledger, revision, _ = case.build(model, region)
        for stop_at in (3, 4, 5, 7, 10):
            for reuse in (False, True):
                with self.subTest(stop_at=stop_at, reuse=reuse):
                    checks = 0

                    def check():
                        nonlocal checks
                        checks += 1
                        if checks == stop_at:
                            raise GeometryTaskCancelled('projection cancellation')

                    with patch.object(target, '_source_separation', _SOURCE_SEPARATION if reuse else without_source_projection):
                        with self.assertRaises(GeometryTaskCancelled):
                            target._construct(ledger.arrays, revision, target.TargetBudget(), check)
                    self.assertEqual(checks, stop_at)

    def test_new_construction_reprojects_changed_geometry_and_intrinsics(self):
        case, model, region = self.fixture()
        ledger, revision, _ = case.build(model, region)
        seen = []

        def capture(triangles, intrinsic, source_vertex_ids=None, *, source_projection=None):
            ids, uv = source_projection
            self.assertFalse(ids.flags.writeable)
            self.assertFalse(uv.flags.writeable)
            expected = np.stack([project(triangles[face], intrinsic) for face in ids])
            self.assertEqual(uv.tobytes(), expected.tobytes())
            seen.append(uv)
            return _SOURCE_SEPARATION(triangles, intrinsic, source_vertex_ids, source_projection=source_projection)

        changed = dict(ledger.arrays, vertices=ledger.arrays['vertices'].copy())
        changed['vertices'][:, 0] += .002
        changed_revision = deepcopy(revision)
        changed_revision['inputs']['calibration']['intrinsic'][0][0] *= 1.25
        for inputs, current in ((ledger.arrays, revision), (changed, revision), (changed, changed_revision)):
            with patch.object(target, '_source_separation', capture):
                actual, counts = target._construct(inputs, current, target.TargetBudget(), lambda: None)
            with patch.object(target, '_source_separation', without_source_projection):
                expected, old_counts = target._construct(inputs, current, target.TargetBudget(), lambda: None)
            self.assert_arrays_identical(actual, expected)
            self.assertEqual(counts, old_counts)
        for left, right in zip(seen, seen[1:]):
            self.assertFalse(np.shares_memory(left, right))
            self.assertNotEqual(left.tobytes(), right.tobytes())


if __name__ == '__main__':
    unittest.main()
