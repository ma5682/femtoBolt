"""Auxiliary removal must preserve measured geometry and publication gates."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import numpy as np

from femto.geometry.local_region import load_reference, save_request
from femto.io_utils import load_json_object, write_json_strict
from femto.reconstruction_profile import LOCAL_ROUTE_PROFILE, LOCAL_ROUTE_SKIP_REASON
from test_region_first_reconstruction import dataset_fixture
from test_local_offline import result_fixture


class MinimalLocalTests(TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_single_and_multiframe_do_not_compute_auxiliaries(self):
        import open3d as o3d
        from femto.pipeline import scene_reconstruction as scene
        for count in (1, 2, 3):
            with self.subTest(frames=count):
                root = self.root / str(count)
                root.mkdir()
                _, dataset = dataset_fixture(root, count)
                reference = load_reference(dataset)
                mask = np.zeros(reference.depth.shape, np.uint8)
                mask[40:80, 60:100] = 1
                request = save_request(reference, mask, root/'request', 25)
                with (mock.patch.object(scene, '_cloud_mesh_consistency', side_effect=AssertionError('cloud diagnostic')),
                     mock.patch.object(scene, 'export_gaussian_initialization', side_effect=AssertionError('Gaussian')),
                     mock.patch.object(scene, 'export_multiview_texture', side_effect=AssertionError('texture')),
                     mock.patch.object(scene, '_camera_trajectory', side_effect=AssertionError('camera visualization'))):
                    report = scene.reconstruct_scene(dataset, root/'model', depth_min_m=.25,
                                                      depth_max_m=2.21, region_request=request)
                self.assertEqual(report['artifact_profile'], LOCAL_ROUTE_PROFILE)
                self.assertEqual(report['frame_count'], count)
                self.assertEqual(report['outputs']['mesh']['report']['diagnostics_status'], 'not_computed')
                self.assertIsNone(report['outputs']['gaussian_initialization'])
                self.assertNotIn('second_sheet_fraction', report['metrics'])
                from femto.geometry.fusion import read_geometry_safe
                cloud = read_geometry_safe(root/'model/scene_fused_points_m.ply', mesh=False)
                mesh = read_geometry_safe(root/'model/scene_tsdf_mesh_m.ply', mesh=True)
                self.assertFalse(cloud.has_normals())
                self.assertTrue(mesh.has_vertex_normals())
                self.assertGreater(len(mesh.triangles), 30)
                self.assertFalse((root/'model/gaussian').exists())
                self.assertFalse((root/'model/camera_trajectory.ply').exists())

    def test_demo_has_no_reports_or_training_export(self):
        from femto.pipeline import complete_demo as demo
        from femto.local.local_processing import LocalProcessing
        _, dataset = dataset_fixture(self.root, 2)
        reference = load_reference(dataset)
        mask = np.zeros(reference.depth.shape, np.uint8)
        mask[40:80,60:100] = 1
        request = save_request(reference, mask, self.root/'request', 25)
        with mock.patch('femto.local.local_processing.write_html', side_effect=AssertionError('HTML')):
            job = LocalProcessing(dataset, self.root/'journal', write_html_report=False)
            job.reconstruction('running')
        args = argparse.Namespace(mode='latest', dataset=dataset, datasets_root=self.root,
            outputs_root=self.root/'outputs', region_request=request, no_source_report=True,
            publish_latest=False, force_nerfstudio=True, local_processing_report=str(job.path))
        with (mock.patch.object(demo, 'convert', side_effect=AssertionError('training export')),
             mock.patch.object(demo, '_contact_sheet', side_effect=AssertionError('contact sheet')),
             mock.patch.object(demo, '_write_html_report', side_effect=AssertionError('HTML')),
             mock.patch('femto.reports.scene_quality_report.write_report', side_effect=AssertionError('aux report'))):
            result, summary = demo.run_demo(args)
        self.assertEqual(summary['status'], 'pass_local_geometry')
        self.assertEqual(summary['nerfstudio_skipped_reason'], LOCAL_ROUTE_SKIP_REASON)
        self.assertFalse(list(result.glob('*.html')))
        self.assertFalse((result/'result_quality.json').exists())
        self.assertEqual(load_json_object(result/'processing_report.json')['state'], 'complete')

    def minimal_publication(self):
        _, outputs, result, _ = result_fixture(self.root)
        summary = load_json_object(result/'demo_summary.json')
        summary.update(artifact_profile=LOCAL_ROUTE_PROFILE, local_measured=True,
            reconstruction_source_mode='moving', contributing_frame_count=3,
            nerfstudio=None, nerfstudio_skipped_reason=LOCAL_ROUTE_SKIP_REASON)
        write_json_strict(result/'demo_summary.json', summary)
        reconstruction = load_json_object(result/'reconstruction/reconstruction_report.json')
        reconstruction.update(artifact_profile=LOCAL_ROUTE_PROFILE, status='complete_local_geometry',
                              reconstruction_scope=dict(contributing_frame_count=3))
        write_json_strict(result/'reconstruction/reconstruction_report.json', reconstruction)
        for file in result.glob('*.html'):
            file.unlink()
        (result/'nerfstudio/transforms.json').unlink()
        for name in ('scene_tsdf_mesh_m.ply', 'scene_fused_points_m.ply', 'region_request.json', 'requested_mask.png'):
            (result/'reconstruction'/name).write_text('fixture', encoding='utf-8')
        return outputs, result

    def test_publication_accepts_explicit_geometry_profile_without_html(self):
        from femto.pipeline.local_offline import publish_result
        from femto.pipeline.reconstruction_sources import mode_of
        outputs, result = self.minimal_publication()
        self.assertEqual(publish_result(result, outputs, 'one')['status'], 'published')
        self.assertEqual(mode_of(result), 'moving')

    def test_missing_mesh_still_blocks_publication(self):
        from femto.pipeline.local_offline import publish_result
        outputs, result = self.minimal_publication()
        (result/'reconstruction/scene_tsdf_mesh_m.ply').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing completed result artifact'):
            publish_result(result, outputs, 'one')
        self.assertFalse(list(outputs.glob('latest*.txt')))

    def test_summary_alone_cannot_bypass_legacy_artifacts(self):
        from femto.pipeline.local_offline import publish_result
        outputs, result = self.minimal_publication()
        reconstruction = load_json_object(result/'reconstruction/reconstruction_report.json')
        reconstruction.pop('artifact_profile')
        write_json_strict(result/'reconstruction/reconstruction_report.json', reconstruction)
        with self.assertRaisesRegex(ValueError, 'Missing completed result artifact'):
            publish_result(result, outputs, 'one')
