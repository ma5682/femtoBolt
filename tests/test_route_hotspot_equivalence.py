"""Oracles for compact triangle records, quality evidence and edge handoffs."""
import unittest
from unittest.mock import patch
from copy import deepcopy
import numpy as np

from femto.geometry import triangle_support as support
from femto.planning import edge_check_batch as edges, quality_constrained as quality
from femto.planning import surface_trajectory as trajectory
from femto.planning.surface_trajectory import ScanSegment
import test_serpentine_prepared_checks as fixtures


class RouteHotspotTests(unittest.TestCase):
    def test_binary_record_matches_unpacked_numpy_path_bit_for_bit(self):
        rng = np.random.default_rng(219)
        points = np.r_[rng.uniform(-1., 1., (96, 3)), np.array([[-0., 0., -0.], [0., -0., 0.]])]
        ids = rng.integers(0, len(points), (180, 3))
        ids[:3] = [[96, 97, 96], [97, 96, 97], [0, 0, 0]]
        for dtype, scale in [(float, 1.), (float, 1e29), (float, 1e-90), (float, 1e-150),
                             (float, 1e35), ('>f8', 1.), ('f4', 1.)]:
            vertices = (points*scale).astype(dtype)
            for i in range(len(ids)):
                reference = support._pack_geometry(support._prepare_one(vertices, ids, i))
                packed = support._prepare_one(vertices, ids, i, packed=True)
                for before, after in zip(support._unpack_geometry(reference), support._unpack_geometry(packed)):
                    self.assertEqual(np.asarray(before).tobytes(), np.asarray(after).tobytes())
                    self.assertEqual(np.shape(before), np.shape(after))

    def test_rle_copy_preserves_independence_aliases_and_subclass_hooks(self):
        values = list(range(300))
        record = dict(classes_runs=values, residual_um_runs=values, extra=[values])
        copied = quality._copy_evidence(record)
        self.assertEqual(copied, deepcopy(record))
        self.assertIs(copied['classes_runs'], copied['extra'][0])
        self.assertIs(copied['classes_runs'], copied['residual_um_runs'])
        copied['classes_runs'][0] = -1
        self.assertEqual(values[0], 0)
        class Number(int):
            def __deepcopy__(self, memo):
                return 'hook'
        values.append(Number(7))
        self.assertEqual(quality._copy_evidence(record)['extra'][0][-1], 'hook')
        with self.assertRaisesRegex(ValueError, 'orientation'):
            trajectory._forbidden_orientation_keys([0]*400 + [{'nested': [{'Quaternion': []}]}])

    def test_previous_point_checked_on_new_face_at_handoff(self):
        vertices = np.array([[0.,0.,.5], [1.,0.,.5], [0.,1.,.5],
                             [1.,0.,.5], [1.,1.,.5], [0.,1.,.5]])
        faces = np.array([[0,1,2], [3,4,5]])
        points = [dict(xyz_m=[.1,.1,.5],triangle_index=0),
                  dict(xyz_m=[.2,.2,.5],triangle_index=0),
                  dict(xyz_m=[.8,.8,.5],triangle_index=1)]
        _, _, current, prior = trajectory._scan_point_certificates(points, vertices, faces, 2.)
        self.assertTrue(current.all())
        self.assertTrue(prior[1])
        self.assertFalse(prior[2])
        for point in points:
            point['role'] = 'scan'
        points[2]['role'] = 'connector'
        _, _, current, prior = trajectory._scan_point_certificates(
            points, vertices, faces, 2., connected_rows=True)
        self.assertTrue(current.all())
        self.assertFalse(prior.any())  # scanning has no incoming connector; handoff genuinely fails
        points[1]['role'] = 'connector'
        _, _, _, prior = trajectory._scan_point_certificates(points, vertices, faces, 2., connected_rows=True)
        np.testing.assert_array_equal(prior, [False, True, False])

    def test_whole_chords_not_just_endpoints_even_when_handoffs_are_batched(self):
        f = fixtures.PreparedQualityChecksTests()
        f.setUp(); self.addCleanup(f.doCleanups)
        record = f.record()
        def segment(start, end):
            return ScanSegment(0, 0, [dict(xyz_m=start,triangle_index=0),
                                      dict(xyz_m=end,triangle_index=1)], 0)
        good = segment([0., 0., .5], [-.015, .015, .5])
        bad = segment([0., 0., .504], [-.015, .015, .5])
        segments = [good, bad]*40
        args = (segments, f.mesh, record, 1.)
        with patch.object(edges, 'evaluate', return_value=None):
            expected = quality.edge_checks(*args, allowed_faces=f.allowed)
        actual = quality.edge_checks(*args, allowed_faces=f.allowed)
        f.assert_same(actual, expected)
        self.assertTrue(actual[0][::2].all())
        self.assertFalse(actual[0][1::2].any())

    def test_small_and_vector_probes_charge_the_same_residual_budget(self):
        f = fixtures.PreparedQualityChecksTests()
        f.setUp(); self.addCleanup(f.doCleanups)
        record = f.record()
        chosen = np.arange(4)
        a = np.array([[0., 0., .5], [-.01,.02,.5], [1.,.2,.5], [.01,-.1,-.5]])
        half = np.array([.02, .0002, .001, .003])
        for used in (0, 99999999):
            ref = quality.ReferenceEnvelope(record, 1.);ref.sampled_pixels=used
            results = [edges._reference_probes(chosen, a, half, ref, lambda: None, small)
                       for small in (True, False)]
            if results[0] is None:
                self.assertIsNone(results[1])
            else:
                self.assertIsNotNone(results[1])
                for l,r in zip(*results):
                    np.testing.assert_array_equal(l,r)
            self.assertEqual(ref.sampled_pixels,used)
            self.assertIsNone(ref._residual_tiles)
