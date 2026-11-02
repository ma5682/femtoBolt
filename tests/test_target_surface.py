"""Analytic pixel-cell geometry, immutable intent and hostile archive regressions."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import unittest
import zipfile

import numpy as np
import open3d as o3d

from test_fixed_rgb_experience import make_model, quad, K
from femto.annotation.fixed_rgb_geometry import GeometryBudget, GeometryTaskCancelled
from femto.annotation.selection_intent import build_selection_intent, build_selection_support
from femto.geometry.pixel_clip import pixel_owner
from femto.planning import planning_revision as revision, target_surface as target, surface_workspace as workspace
from femto.planning.quality_constrained import evidence
from femto.planning.effective_domain import build as build_domain
from femto.planning.treatment_head import head_parameters


class TargetSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.head = head_parameters('direct_row_spacing', row_spacing_mm=7)

    def fixture(self, mesh=None, mask=None, depth=None):
        root = Path(self.temp.name) / str(len(list(Path(self.temp.name).iterdir())))
        model = make_model(root, quad(left=-.1, right=.1, top=-.1, bottom=.1) if mesh is None else mesh, depth)
        model.view_reference['undistorted_intrinsic'] = K.tolist()
        model.store.origin_reference = dict(kind='initial_color_camera', provenance='synthetic_test')
        region = model.region([[0, 0], [99, 0], [99, 99], [0, 99]], 'synthetic', '',
                              selection=dict(mask=model.projection.valid.copy(), method='work_only'))
        if mask is None:
            mask = np.zeros((100, 100), bool); mask[50, 50] = True
        # A full legacy region binding intentionally surrounds a much smaller
        # authoritative pixel selection: the inventory must clip the geometry.
        region['selection_intent'] = build_selection_intent(mask, method='work_only')
        region['selection_support'] = build_selection_support(mask, model.projection)
        return model, region

    def build(self, model, region, **kwargs):
        current = revision.capture(model, region, self.head)
        quality = evidence(model, region, allow_empty=True)
        ledger = target.build(model, region, current, evidence=quality, **kwargs)
        return ledger, current, quality

    def test_one_pixel_clips_two_large_faces_to_analytic_area(self):
        model, region = self.fixture()
        ledger, _, _ = self.build(model, region)
        self.assertAlmostEqual(ledger.summary['located_area_m2'], (.5/100)**2, places=12)
        self.assertEqual(set(ledger.arrays['source_face']), {0, 1})
        self.assertEqual(set(ledger.arrays['pixel_index']), {5050})
        self.assertEqual(ledger.summary['coverage_status'], 'not_evaluated')
        np.testing.assert_allclose(ledger.arrays['barycentric'].sum(axis=2), 1, rtol=0, atol=1e-14)

    def test_shared_pixel_and_triangle_edges_have_no_double_area(self):
        mask = np.zeros((100, 100), bool); mask[49:52, 49:52] = True
        ledger, current, quality = self.build(*self.fixture(mask=mask))
        self.assertAlmostEqual(ledger.summary['located_area_m2'], 9*(.5/100)**2, places=12)
        np.testing.assert_array_equal(pixel_owner(np.array([[49.5, 49.5], [50.5, 50.5]])), [[50, 50], [51, 51]])
        identities = np.c_[ledger.arrays['parent_id'], ledger.arrays['child_index']]
        self.assertEqual(len(np.unique(identities, axis=0)), len(identities))
        target.validate(ledger, current, evidence=quality, domain=None)

    def test_partial_pixel_keeps_unknown_remainder_without_inventing_area(self):
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(
            [[-.0025, -.0025, .5], [.0025, -.0025, .5], [-.0025, .0025, .5]]),
            o3d.utility.Vector3iVector([[0, 1, 2]]))
        model, region = self.fixture(mesh=mesh)
        ledger, _, _ = self.build(model, region)
        self.assertAlmostEqual(ledger.summary['located_area_m2'], .5*(.005**2), places=11)
        self.assertEqual(ledger.arrays['pixel_status'][50, 50], 10)
        self.assertIsNone(ledger.summary['total_target_area_m2'])
        self.assertIsNone(ledger.summary['unlocated_area_m2'])

    def test_duplicate_reordered_source_faces_are_counted_once(self):
        mesh = quad(left=-.1, right=.1, top=-.1, bottom=.1)
        duplicate = o3d.geometry.TriangleMesh(mesh)
        duplicate.triangles = o3d.utility.Vector3iVector(np.asarray(duplicate.triangles)[:, ::-1])
        ledger, _, _ = self.build(*self.fixture(mesh=mesh+duplicate))
        self.assertEqual(ledger.metadata['metrics']['duplicate_face_count'], 2)
        self.assertAlmostEqual(ledger.summary['located_area_m2'], .005**2, places=12)

    def test_occluded_back_surface_is_not_added_even_inside_depth_gate(self):
        mesh = quad(left=-.1, right=.1, top=-.1, bottom=.1) + quad(.505, left=-.1, right=.1, top=-.1, bottom=.1)
        ledger, _, _ = self.build(*self.fixture(mesh=mesh))
        self.assertAlmostEqual(ledger.summary['located_area_m2'], .005**2, places=12)
        self.assertTrue(set(ledger.arrays['source_face']) <= {0, 1})
        self.assertGreater(ledger.metadata['metrics']['occluded_polygons'], 0)

    def test_depth_and_source_pixel_unknowns_have_no_fabricated_area(self):
        for reason in ('missing', 'conflict', 'source'):
            with self.subTest(reason=reason):
                model, region = self.fixture()
                if reason == 'source':
                    model.projection.image_valid = np.ones((100, 100), bool)
                    model.projection.image_valid[50, 50] = False
                    model.projection.valid[50, 50] = False; model.projection.ids[50, 50] = -1
                    model.projection.reasons[50, 50] = 4; model.projection.z[50, 50] = np.nan
                else:
                    from femto.annotation.fixed_rgb_geometry import FixedProjection
                    depth = model.projection.depth.copy(); depth[50, 50] = 0 if reason == 'missing' else 480
                    model.projection = FixedProjection(model.mesh, K, depth)
                ledger, _, _ = self.build(model, region)
                self.assertEqual(ledger.summary['located_area_m2'], 0)
                self.assertEqual(ledger.summary['unlocated_pixel_count'], 1)
                self.assertIsNone(ledger.summary['total_target_area_m2'])

    def test_quality_filter_changes_eligibility_without_shrinking_target(self):
        model, region = self.fixture()
        ledger, current, quality = self.build(model, region)
        domain = build_domain(workspace.canonical_region_mesh(model.mesh, region), region, quality, 5)
        self.assertEqual(domain['eligible_face_count'], 0)
        filtered = target.with_quality(ledger, current, region, quality, domain)
        self.assertEqual(filtered.metadata['target_id'], ledger.metadata['target_id'])
        np.testing.assert_array_equal(filtered.arrays['barycentric'], ledger.arrays['barycentric'])
        self.assertEqual(filtered.summary['located_area_m2'], ledger.summary['located_area_m2'])
        self.assertEqual(filtered.summary['quality_eligible_area_m2'], 0)
        self.assertGreater(filtered.summary['quality_excluded_area_m2'], 0)

    def test_faces_outside_legacy_centroid_region_still_keep_pixel_intent(self):
        model, region = self.fixture()
        region['triangle_indices'] = [0]
        ledger, _, _ = self.build(model, region)
        self.assertEqual(set(ledger.arrays['source_face']), {0, 1})
        self.assertEqual(set(ledger.arrays['local_face']), {-1, 0})
        self.assertGreater(ledger.summary['outside_route_region_area_m2'], 0)
        self.assertAlmostEqual(ledger.summary['located_area_m2'], .005**2, places=12)

    def test_canonical_coordinates_and_lineage_reconstruct_each_leaf(self):
        ledger, _, _ = self.build(*self.fixture())
        a = ledger.arrays
        np.testing.assert_array_equal(a['vertices'], a['vertices'].astype('f4').astype('f8'))
        xyz = np.einsum('nij,njk->nik', a['barycentric'], a['vertices'][a['faces'][a['source_face']]])
        areas = np.linalg.norm(np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), axis=1)/2
        # Batched einsum and per-polygon matmul have different roundoff. The
        # record explicitly stores nominal float64 areas, not exact arithmetic.
        np.testing.assert_allclose(areas, a['area_m2'], rtol=1e-14, atol=0)
        np.testing.assert_array_equal(a['parent_id'], a['pixel_index']*len(a['faces'])+a['source_face'])

    def test_repeated_build_is_deterministic_and_does_not_change_model(self):
        model, region = self.fixture()
        original = deepcopy(region); depth = model.projection.depth.copy()
        a, _, _ = self.build(model, region); b, _, _ = self.build(model, region)
        self.assertEqual(a.metadata, b.metadata)
        self.assertEqual(region, original); np.testing.assert_array_equal(depth, model.projection.depth)
        self.assertTrue(model.projection.depth.flags.writeable)
        self.assertFalse(a.arrays['depth'].flags.writeable)

    def test_candidate_unit_pair_and_time_budgets_keep_unprocessed_intent(self):
        model, region = self.fixture()
        for budget, reason in ((target.TargetBudget(max_candidates=0), 'candidate_budget'),
                               (target.TargetBudget(max_units=0), 'unit_budget'),
                               (target.TargetBudget(max_pair_checks=0), 'pair_budget'),
                               (target.TargetBudget(seconds=0), 'time_budget')):
            with self.subTest(reason=reason):
                ledger, current, quality = self.build(model, region, budget=budget)
                self.assertEqual(ledger.metadata['metrics']['stop_reason'], reason)
                self.assertEqual(ledger.summary['selected_pixel_count'], 1)
                self.assertEqual(ledger.summary['unlocated_pixel_count'], 1)
                self.assertEqual(ledger.summary['located_area_m2'], 0)
                target.validate(ledger, current, evidence=quality, domain=None)

    def test_cancelled_build_never_publishes_partial_ledger(self):
        model, region = self.fixture()
        cancel = threading.Event(); cancel.set()
        with self.assertRaises(GeometryTaskCancelled):
            self.build(model, region, check=GeometryBudget(cancel).check)

    def test_asset_roundtrip_and_new_hash_cannot_hide_forged_area(self):
        ledger, current, quality = self.build(*self.fixture())
        folder = Path(self.temp.name)
        target.write(folder, ledger)
        restored = target.read(folder, ledger.metadata, current, evidence=quality, domain=None)
        self.assertEqual(restored.metadata, ledger.metadata)
        forged = {k: v.copy() for k, v in ledger.arrays.items()}
        forged['area_m2'] *= 2
        bad = target._seal(forged, current, target.TargetBudget(), ledger.metadata['metrics'], quality['sha256'], None)
        target.write(folder, bad)
        with self.assertRaisesRegex(ValueError, '不能由原输入重现'):
            target.read(folder, bad.metadata, current, evidence=quality, domain=None)

    def test_corrupt_missing_or_duplicate_asset_entries_are_rejected(self):
        ledger, current, quality = self.build(*self.fixture())
        folder = Path(self.temp.name); target.write(folder, ledger)
        (folder/target.ASSET).write_bytes(b'broken archive')
        with self.assertRaises((ValueError, zipfile.BadZipFile)):
            target.read(folder, ledger.metadata, current, evidence=quality, domain=None)
        target.write(folder, ledger)
        (folder/target.MANIFEST).unlink()
        with self.assertRaises(ValueError):
            target.read(folder, ledger.metadata, current, evidence=quality, domain=None)

    def test_mixed_revision_and_forged_completion_are_rejected(self):
        model, region = self.fixture(); ledger, current, quality = self.build(model, region)
        changed = revision.capture(model, region, head_parameters('direct_row_spacing', row_spacing_mm=8))
        with self.assertRaises(ValueError):target.validate(ledger, changed, evidence=quality, domain=None)
        metadata = deepcopy(ledger.metadata); metadata['target_completion'] = 'complete'
        metadata['sha256'] = revision.digest({k: v for k, v in metadata.items() if k != 'sha256'})
        with self.assertRaises(ValueError):target.validate_metadata(metadata, current)


if __name__ == '__main__':
    unittest.main()
