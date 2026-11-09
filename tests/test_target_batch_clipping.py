"""Batch execution preserves scalar geometry, ambiguity and work frontiers."""
from unittest.mock import patch
import unittest

import numpy as np

from femto.geometry.pixel_clip import clip_triangle_planes, pixel_planes
from femto.geometry.pixel_clip_batch import MAX_BATCH, clip_triangles_planes
from femto.annotation.fixed_rgb_geometry import GeometryTaskCancelled
from femto.planning import planning_revision as revisions, target_surface as target
from femto.planning.quality_constrained import evidence
from test_fixed_rgb_experience import quad
import test_target_surface as fixtures


class TargetBatchClippingTests(unittest.TestCase):
    def assert_clips_equal(self, triangles, planes):
        actual = clip_triangles_planes(triangles, planes)
        rows = np.broadcast_to(planes, (len(triangles), 4, 3))
        expected = [clip_triangle_planes(t, p) for t, p in zip(triangles, rows)]
        self.assertEqual(len(actual), len(expected))
        for (a, exact_a), (b, exact_b) in zip(actual, expected):
            self.assertEqual(exact_a, exact_b)
            self.assertEqual(a.dtype, b.dtype)
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.tobytes(), b.tobytes())

    def test_scalar_bytes_across_sizes_scales_and_distinct_pixel_planes(self):
        rng = np.random.default_rng(261026)
        k = np.array([[1.7, 0., .2], [0., 1.3, -.4], [0., 0., 1.]])
        for scale in (1e-20, 1e-8, 1., 1e8, 1e20):
            for size in (0, 1, 7, 8, 31, 64, 65, 127, 128, 255, 256):
                with self.subTest(scale=scale, size=size):
                    triangles = rng.uniform(-2, 2, (size, 3, 3)) * scale
                    triangles[:, :, 2] = rng.uniform(.5, 2, (size, 3)) * scale
                    planes = np.array([pixel_planes(k, *xy) for xy in rng.integers(-2, 3, (size, 2))]).reshape(size, 4, 3)
                    self.assert_clips_equal(triangles, planes)
                    self.assert_clips_equal(triangles[::-1], pixel_planes(k, 0, 0))

    def test_exact_boundary_degeneracy_signed_zero_and_overflow_fall_back(self):
        boundary = np.array([[-.5, -.5, 1.], [.5, -.5, 1.], [.5, .5, 1.]])
        triangles = np.repeat(boundary[None], 64, axis=0)
        triangles[1::3, 0, 0] += 2**-48
        triangles[2::3, 0, 0] -= 2**-48
        triangles[3] = [[-0., 0., 1.], [-0., 0., 1.], [0., 0., 1.]]
        self.assert_clips_equal(triangles, pixel_planes(np.eye(3), 0, 0))
        with np.errstate(all='ignore'):
            self.assert_clips_equal(triangles * 1e200, pixel_planes(np.diag([1e200, 1e200, 1.]), 0, 0))
            self.assert_clips_equal(triangles * 1e-200, pixel_planes(np.diag([1e-200, 1e-200, 1.]), 0, 0))

    def test_fallback_rows_use_supplied_scalar_and_workspace_is_bounded(self):
        boundary = np.array([[-.5, -.5, 1.], [.5, -.5, 1.], [.5, .5, 1.]])
        triangles = np.repeat(boundary[None], 64, axis=0)
        with patch('femto.geometry.pixel_clip_batch.clip_triangle_planes', wraps=clip_triangle_planes) as scalar:
            clip_triangles_planes(triangles, pixel_planes(np.eye(3), 0, 0), scalar_clip=scalar)
        self.assertEqual(scalar.call_count, 64)
        with self.assertRaises(ValueError):
            clip_triangles_planes(np.repeat(boundary[None], MAX_BATCH+1, axis=0), pixel_planes(np.eye(3), 0, 0))

    def test_unexpected_polygon_growth_uses_scalar_before_buffer_indexing(self):
        # Inject an invalid computed count to exercise the defensive path for a
        # finite-arithmetic violation of the convex seven-vertex bound.
        triangle = np.array([[-1., -.1, 1.], [1., -.1, 1.], [0., .8, 1.]])
        triangles = np.repeat(triangle[None], 64, axis=0)
        cumulative = np.cumsum
        def oversized(*args, **kwargs):
            result = cumulative(*args, **kwargs)
            result[0, -1] = 8
            return result
        with patch('femto.geometry.pixel_clip_batch.np.cumsum', side_effect=oversized):
            self.assert_clips_equal(triangles, pixel_planes(np.eye(3), 0, 0))

    def test_preparation_cannot_cross_explicit_group_frontier(self):
        triangles = np.tile(np.array([[[-.5, -.5, 1.], [.5, -.5, 1.], [.5, .5, 1.]]]), (5, 1, 1))
        faces = np.tile(np.arange(5), 100)
        groups = np.arange(100)
        ends = np.arange(5, 501, 5)
        batches = target._PixelClipBatches(triangles, np.eye(3), 100, groups, faces, ends, 3)
        with patch.object(target, 'clip_triangles_planes', wraps=clip_triangles_planes) as clipped:
            batches.get(0)
            batches.get(14)
            self.assertEqual(clipped.call_count, 1)
            self.assertEqual(len(clipped.call_args.args[0]), 15)
            for ordinal, row in enumerate(clipped.call_args.args[1]):
                np.testing.assert_array_equal(row, pixel_planes(np.eye(3), ordinal//5, 0))
            with self.assertRaises(ValueError):
                batches.get(15)

    def test_full_ledger_and_work_budget_frontiers_match_scalar_path(self):
        case = fixtures.TargetSurfaceTests(); case.setUp(); self.addCleanup(case.doCleanups)
        mask = np.zeros((100, 100), bool); mask[47:54, 46:55] = True; mask[50, 50] = False
        meshes = (quad(left=-.1, right=.1, top=-.1, bottom=.1),
                  quad(left=-.1, right=.1, top=-.1, bottom=.1)
                  + quad(.505, left=-.1, right=.1, top=-.1, bottom=.1))
        for mesh in meshes:
            model, region = case.fixture(mesh=mesh, mask=mask)
            quality = evidence(model, region, allow_empty=True)
            for algorithm in (revisions.ALGORITHMS['target'], revisions.EXACT_TARGET):
                revision = revisions.capture(model, region, case.head,
                    algorithms=dict(revisions.ALGORITHMS, target=algorithm))
                for budget in (target.TargetBudget(), target.TargetBudget(max_units=7),
                               target.TargetBudget(max_pair_checks=3), target.TargetBudget(max_candidates=10)):
                    with self.subTest(algorithm=algorithm, budget=budget, faces=len(mesh.triangles)):
                        actual = target.build(model, region, revision, evidence=quality, budget=budget)
                        with patch.object(target, '_PixelClipBatches', return_value=None):
                            expected = target.build(model, region, revision, evidence=quality, budget=budget)
                        self.assertEqual(actual.metadata, expected.metadata)
                        for key in actual.arrays:
                            self.assertEqual(actual.arrays[key].tobytes(), expected.arrays[key].tobytes(), key)
                        target.validate(actual, revision, evidence=quality, domain=None)

    def test_original_cancellation_check_order_is_retained(self):
        case = fixtures.TargetSurfaceTests(); case.setUp(); self.addCleanup(case.doCleanups)
        mask = np.zeros((100, 100), bool); mask[47:54, 46:55] = True
        model, region = case.fixture(mask=mask)
        revision = revisions.capture(model, region, case.head)
        quality = evidence(model, region, allow_empty=True)
        counts = []
        for batched in (False, True):
            calls = []
            def check():
                calls.append(None)
            if batched:
                target.build(model, region, revision, evidence=quality, check=check)
            else:
                with patch.object(target, '_PixelClipBatches', return_value=None):
                    target.build(model, region, revision, evidence=quality, check=check)
            counts.append(len(calls))
        self.assertEqual(*counts)
        for stop_at in (1, 7, counts[0]//2, counts[0]-1):
            for batched in (False, True):
                calls = []
                def cancel():
                    calls.append(None)
                    if len(calls) == stop_at:
                        raise GeometryTaskCancelled('test cancellation')
                with self.subTest(stop_at=stop_at, batched=batched):
                    with self.assertRaises(GeometryTaskCancelled):
                        if batched:
                            target.build(model, region, revision, evidence=quality, check=cancel)
                        else:
                            with patch.object(target, '_PixelClipBatches', return_value=None):
                                target.build(model, region, revision, evidence=quality, check=cancel)
                    self.assertEqual(len(calls), stop_at)


if __name__ == '__main__':
    unittest.main()
