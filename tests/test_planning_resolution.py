"""Explicit coarse references retain evidence and reject ambiguous input state."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import open3d as o3d

from test_fixed_rgb_experience import make_model, K
from femto.annotation.scene_annotations import mesh_digest
from femto.planning.planning_resolution import (
    SCHEMA, create_reduced_model, profile, route_sample_step_mm,
)


class PlanningResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = make_model(self.root/'source')
        self.source.view_reference['undistorted_intrinsic'] = K.tolist()
        self.source.store.origin_reference = dict(kind='initial_color_camera', provenance='synthetic_test')
        self.source.rgb = np.arange(100*100*3, dtype=np.uint8).reshape(100, 100, 3)
        self.assertTrue(o3d.io.write_triangle_mesh(self.source.store.source, self.source.mesh))

    def test_profile_is_exact_and_legacy_defaults_stay_one(self):
        self.assertEqual(route_sample_step_mm({}), 1.)
        self.assertEqual(route_sample_step_mm({'derivation': {'schema': 'other'}}), 1.)
        for stride in (4, 8, 16):
            for step in (1., 2., 3., 5.):
                view = dict(derivation=dict(schema=SCHEMA, stride=stride,
                    route_sample_step_mm=step, profile=profile(stride, step)))
                self.assertEqual(route_sample_step_mm(view), step)
        for step in (None, True, False, '3', float('nan'), float('inf'), 0., 4.):
            with self.subTest(step=step), self.assertRaises(ValueError):
                route_sample_step_mm(dict(derivation=dict(schema=SCHEMA, stride=8,
                    route_sample_step_mm=step, profile='stride8-original-mesh-step3mm-v1')))
        for stride in (True, None, 3, 8.):
            with self.subTest(stride=stride), self.assertRaises(ValueError):profile(stride, 3.)
        with self.assertRaisesRegex(ValueError, '不一致'):
            route_sample_step_mm(dict(derivation=dict(schema=SCHEMA, stride=8,
                route_sample_step_mm=3., profile=profile(4, 3.))))

    def test_exact_sampled_rays_mesh_identity_and_source_unchanged(self):
        path = Path(self.source.store.source)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        old_reference = deepcopy(self.source.view_reference)
        model = create_reduced_model(self.source, self.root/'derived', stride=8, route_sample_step_mm=3.)
        np.testing.assert_array_equal(model.rgb, self.source.rgb[4::8, 4::8])
        np.testing.assert_array_equal(model.projection.depth, self.source.projection.depth[4::8, 4::8])
        xx, yy = np.meshgrid(np.arange(model.rgb.shape[1]), np.arange(model.rgb.shape[0]))
        k = model.projection.k
        np.testing.assert_allclose((xx-k[0, 2])/k[0, 0], (8*xx+4-K[0, 2])/K[0, 0], atol=1e-15)
        np.testing.assert_allclose((yy-k[1, 2])/k[1, 1], (8*yy+4-K[1, 2])/K[1, 1], atol=1e-15)
        self.assertEqual(mesh_digest(model.mesh), mesh_digest(self.source.mesh))
        self.assertIsNot(model.mesh, self.source.mesh)
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(old_reference, self.source.view_reference)
        self.assertEqual(model.store.regions, [])
        self.assertEqual(model.store.points, [])
        self.assertIsNone(model.saved_plan)
        self.assertEqual(route_sample_step_mm(model.view_reference), 3.)
        self.assertTrue((model.reconstruction/'planning_reference.json').is_file())

    def test_existing_selection_plan_history_or_derived_model_rejected(self):
        for attribute, value in (('regions', [{}]), ('points', [{}]), ('_history', [({}, {})])):
            original = getattr(self.source.store, attribute)
            setattr(self.source.store, attribute, value)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                create_reduced_model(self.source, self.root/'rejected')
            setattr(self.source.store, attribute, original)
        self.source.saved_plan = {'saved': True}
        with self.assertRaises(ValueError):create_reduced_model(self.source, self.root/'rejected')
        self.source.saved_plan = None
        model = create_reduced_model(self.source, self.root/'derived')
        with self.assertRaises(ValueError):create_reduced_model(model, self.root/'second')

    def test_quality_policy_is_preserved_without_aliasing(self):
        self.source.planning_quality_config = {'custom': {'depth_gate_mm': 7.}}
        model = create_reduced_model(self.source, self.root/'derived')
        self.assertEqual(model.planning_quality_config, self.source.planning_quality_config)
        model.planning_quality_config['custom']['depth_gate_mm'] = 9.
        self.assertEqual(self.source.planning_quality_config['custom']['depth_gate_mm'], 7.)

    def test_mismatched_mesh_or_reference_rejected_before_creating_destination(self):
        old = self.source.view_reference['image_size']
        self.source.view_reference['image_size'] = [99, 100]
        with self.assertRaises(ValueError):create_reduced_model(self.source, self.root/'rejected')
        self.assertFalse((self.root/'rejected').exists())
        self.source.view_reference['image_size'] = old
        points = np.asarray(self.source.mesh.vertices).copy();points[:, 2] += .001
        self.source.mesh.vertices = o3d.utility.Vector3dVector(points)
        with self.assertRaises(ValueError):create_reduced_model(self.source, self.root/'rejected')
        self.assertFalse((self.root/'rejected').exists())

    def test_cancellation_does_not_install_or_modify_source(self):
        path = Path(self.source.store.source);before = path.read_bytes()
        def cancelled():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            create_reduced_model(self.source, self.root/'cancelled', check=cancelled)
        self.assertFalse((self.root/'cancelled').exists())
        self.assertEqual(path.read_bytes(), before)

    def test_source_acquisition_assessment_is_retained_as_history_only(self):
        from femto.dataset.acquisition_completion import portable_completion, validate_portable_completion
        binding = dict(session_id=self.source.view_reference['session_id'], reference_frame_id=0)
        record = dict(schema='femto-acquisition-completion-v1', assessment_status='reference_support_met',
            confirmed_for_planning=True, execution_ready=False, physical_accuracy_verified=False,
            binding=binding, feedback=dict(binding=deepcopy(binding), status='reference_support_met'),
            end_status='reference_support_met')
        self.source.portable_acquisition = portable_completion(record, self.source.store.digest, self.source.view_reference)
        old = deepcopy(self.source.portable_acquisition)
        model = create_reduced_model(self.source, self.root/'derived')
        self.assertEqual(model.portable_acquisition['completion'], record)
        self.assertEqual(model.source_lineage['original_acquisition_wrapper'], old)
        self.assertEqual(self.source.portable_acquisition, old)
        self.assertTrue(model.view_reference['derivation']['source_assessment_only'])
        self.assertFalse(model.view_reference['derivation']['derived_geometry_physically_confirmed'])
        validate_portable_completion(model.portable_acquisition, model.store.digest, model.view_reference)

    def test_missing_source_image_pixel_keeps_intersecting_coarse_cells_invalid(self):
        self.source.projection.image_valid = np.ones((100, 100), bool)
        self.source.projection.image_valid[8, 8] = False
        model = create_reduced_model(self.source, self.root/'derived')
        # At stride 8, the coarse cells centred at source 4 and 12 both
        # intersect the original pixel cell centred at 8.
        self.assertFalse(model.projection.image_valid[:2, :2].any())
        self.assertFalse(model.projection.valid[:2, :2].any())
        np.testing.assert_array_equal(model.projection.depth, self.source.projection.depth[4::8, 4::8])


if __name__ == '__main__':unittest.main()
