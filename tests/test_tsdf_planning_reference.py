"""Real Open3D integration, immutable evidence and guarded TSDF replay."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import cv2
import numpy as np

from test_region_first_reconstruction import dataset_fixture
from femto.annotation.fixed_rgb_geometry import FixedRGBModel, FixedProjection, GeometryTaskCancelled
from femto.geometry.local_region import load_reference, save_request, model_selection
from femto.io_utils import imread_unicode, imwrite_unicode, load_jsonl, sha256_file
from femto.local.planning_tsdf import PROFILE, checked as check_tsdf
from femto.pipeline.scene_reconstruction import reconstruct_scene
from femto.planning.patch_reference import create, load, guard, SOURCE_ASSET


class TSDFPlanningTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, count=1, *, contributors=None, holes=False, surface='plane'):
        capture, dataset = dataset_fixture(self.root, count)
        if surface != 'plane':
            from tools.benchmark_region_workflow import set_synthetic_surface
            set_synthetic_surface(capture, dataset, surface)
        rows = load_jsonl(dataset/'frames.jsonl')
        for index, row in enumerate(rows):
            path = dataset/row['depth']
            depth = imread_unicode(path, cv2.IMREAD_UNCHANGED)
            if holes: depth[53:60,73:80] = 0
            if contributors is not None and index >= contributors: depth[:] = 800
            imwrite_unicode(path, depth)
        reference = load_reference(dataset)
        mask = np.zeros(reference.depth.shape, np.uint8)
        mask[35:85,45:110] = 1
        if holes:
            mask[45:51,58:65] = 0  # Intentional internal exclusion, separate from missing depth.
            mask[:,85:90] = 0      # Two disconnected selected surfaces.
        request = save_request(reference, mask, self.root/'request', 0)
        report = reconstruct_scene(dataset,self.root/'measurement',depth_min_m=.25,depth_max_m=2.21,
                                   region_request=request)
        return FixedRGBModel(self.root/'measurement'), mask, capture, dataset, report

    def test_single_frame_keeps_full_measurement_and_portable_guard(self):
        source, mask, capture, dataset, report = self.source(holes=True)
        files = {p:sha256_file(p) for p in capture.session.rglob('*') if p.is_file()}
        original = Path(source.store.source).read_bytes()
        model = create(source, self.root/'planning')
        self.assertEqual(report['parameters']['mesh_method'], 'single_frame_pixel_adjacency')
        d = model.view_reference['derivation']
        self.assertEqual(d['profile'], PROFILE)
        self.assertEqual(d['partition']['contributing_frame_count'], 1)
        self.assertEqual([r['passes'] for r in d['partition']['integrations']], [1])
        self.assertEqual((model.reconstruction/SOURCE_ASSET).read_bytes(), original)
        self.assertEqual(Path(source.store.source).read_bytes(), original)
        self.assertTrue(all(sha256_file(p)==h for p,h in files.items()))
        np.testing.assert_array_equal(model_selection(model), mask)
        self.assertFalse(model.projection.valid[53:60,73:80].any())
        self.assertFalse(model.projection.valid[45:51,58:65].any())
        self.assertFalse(model.projection.valid[:,85:90].any())
        self.assertTrue(model.projection.valid[mask.astype(bool)].any())
        reloaded = load(model.reconstruction)
        np.testing.assert_array_equal(reloaded.projection.valid, model.projection.valid)
        self.assertEqual(reloaded.source_surface_check, model.source_surface_check)
        # Portable replay must not need the external RGB-D or original output.
        from femto.app.qt_planning_io import extract_region, masks_from_region
        from femto.planning.region_project import save_project, open_project
        model, region = extract_region(model, mask, np.zeros_like(mask), 'test', '', None)
        save_project(model, region, self.root/'planning.region.zip')
        held = capture.session.with_name(capture.session.name+'_unavailable')
        capture.session.rename(held)
        reopened = open_project(self.root/'planning.region.zip', workspace=self.root/'reopened')
        self.assertEqual((reopened.reconstruction/SOURCE_ASSET).read_bytes(), original)
        np.testing.assert_array_equal(masks_from_region(reopened.store.regions[0], mask.shape)[0], mask)
        self.assertEqual(reopened.source_surface_check, model.source_surface_check)
        full = reopened.reconstruction/SOURCE_ASSET
        full.write_bytes(full.read_bytes()+b'tampered')
        from femto.planning.patch_reference import verify_asset
        with self.assertRaisesRegex(ValueError,'完整原表面'): verify_asset(reopened)

    def test_multiframe_integrates_actual_poses_once_each(self):
        import open3d as o3d
        source, mask, _, dataset, _ = self.source(3)
        factory = o3d.pipelines.integration.ScalableTSDFVolume
        matrices = []
        class TracedVolume:
            def __init__(self, **kwargs): self.actual = factory(**kwargs)
            def integrate(self, rgbd, intrinsic, extrinsic):
                matrices.append(extrinsic.copy()); self.actual.integrate(rgbd, intrinsic, extrinsic)
            def extract_triangle_mesh(self): return self.actual.extract_triangle_mesh()
        with mock.patch('femto.local.planning_tsdf.o3d.pipelines.integration.ScalableTSDFVolume', TracedVolume):
            model = create(source, self.root/'planning')
        from femto.geometry.rgbd_frames import load_poses
        poses = load_poses(dataset, load_jsonl(dataset/'frames.jsonl'))
        self.assertEqual(len(matrices), 3)
        for actual, pose in zip(matrices, poses): np.testing.assert_array_equal(actual, np.linalg.inv(pose))
        self.assertFalse(np.array_equal(matrices[1], np.eye(4)))
        self.assertEqual(model.view_reference['derivation']['profile'], PROFILE)
        np.testing.assert_array_equal(model_selection(model), mask)

    def test_three_admitted_one_contributor_still_integrates_only_once(self):
        source, _, _, _, _ = self.source(3, contributors=1)
        model = create(source, self.root/'planning')
        record = model.view_reference['derivation']['partition']
        self.assertEqual(record['input_frame_count'], 3)
        self.assertEqual(record['contributing_frame_count'], 1)
        self.assertEqual([r['frame'] for r in record['integrations']], [0])

    def test_curved_single_and_multi_use_same_tsdf_contract(self):
        for count in (1,3):
            with self.subTest(contributors=count), TemporaryDirectory() as temp:
                self.root=Path(temp)
                source, mask, _, _, _ = self.source(count,surface='sphere')
                original=sha256_file(source.store.source)
                model=create(source,self.root/'planning')
                self.assertEqual(model.view_reference['derivation']['profile'],PROFILE)
                self.assertTrue(model.projection.valid[mask.astype(bool)].any())
                self.assertEqual(sha256_file(model.reconstruction/SOURCE_ASSET),original)
                np.testing.assert_array_equal(model_selection(model),mask)
                np.testing.assert_array_equal(load(model.reconstruction).projection.valid,model.projection.valid)

    def test_changed_nonreference_input_is_rejected_before_publishing(self):
        source, _, _, dataset, _ = self.source(3)
        row = load_jsonl(dataset/'frames.jsonl')[1]
        path = dataset/row['depth']; data = imread_unicode(path, cv2.IMREAD_UNCHANGED)
        data[50,60] += 1; imwrite_unicode(path,data)
        with self.assertRaisesRegex(ValueError,'指纹不一致'): create(source,self.root/'planning')
        self.assertFalse((self.root/'planning').exists())

    def test_cancel_during_integration_does_not_publish_or_change_source(self):
        source, _, _, _, _ = self.source()
        digest = sha256_file(source.store.source)
        stopped = False
        def progress(message):
            nonlocal stopped
            if '融合第' in message: stopped = True
        def check():
            if stopped: raise GeometryTaskCancelled('test cancellation')
        with self.assertRaises(GeometryTaskCancelled):
            create(source,self.root/'planning',check=check,progress=progress)
        self.assertFalse((self.root/'planning').exists())
        self.assertEqual(sha256_file(source.store.source),digest)

    def test_source_changed_during_integration_is_not_published(self):
        source, _, _, dataset, _ = self.source()
        row = load_jsonl(dataset/'frames.jsonl')[0]
        path = dataset/row['depth']
        def progress(message):
            if '融合第' in message:
                depth = imread_unicode(path, cv2.IMREAD_UNCHANGED)
                depth[50,60] += 1; imwrite_unicode(path, depth)
        with self.assertRaisesRegex(ValueError,'指纹不一致'):
            create(source,self.root/'planning',progress=progress)
        self.assertFalse((self.root/'planning').exists())

    def test_depth_deviation_is_unknown_and_saved_parameters_are_checked(self):
        source, mask, _, _, _ = self.source()
        model = create(source,self.root/'planning')
        d = model.view_reference['derivation']
        bad = deepcopy(d['partition']); bad['voxel_m'] = .003
        with self.assertRaisesRegex(ValueError,'TSDF 规划表面'):
            check_tsdf(bad,source.store.local_context['reconstruction_scope'])
        candidate = deepcopy(model.mesh); candidate.translate((0,0,.004))
        p = source.projection
        model.mesh = candidate
        model.projection = FixedProjection(candidate,p.k,p.depth,p.minimum,p.maximum)
        model.projection.image_valid = p.image_valid.copy()
        guard(model,source.mesh,source_projection=p)
        self.assertFalse(model.projection.valid[mask.astype(bool)].any())
        self.assertGreater(model.source_surface_check['exceeds_tolerance_pixels'],0)
        np.testing.assert_array_equal(model_selection(model),mask)

    def test_new_surface_distance_and_legacy_axial_checks_replay_separately(self):
        import open3d as o3d
        from femto.local.planning_tsdf import LEGACY_PROFILE
        source,_,_,_,_=self.source()
        model=create(source,self.root/'planning')
        vertices=np.array([[-.1,-.1,.5],[.1,-.1,1.5],[-.1,.1,.5],[.1,.1,1.5]])
        measured=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector([[0,1,2],[1,3,2]]))
        candidate=deepcopy(measured);candidate.translate([0,0,.004])
        k=np.array([[1000.,0,25],[0,1000.,25],[0,0,1.]])
        _,x=np.indices((50,50));depth=np.rint(1000/(1-5*(x-25)/1000)).astype(np.uint16)
        original=FixedProjection(measured,k,depth,.25,2.21)
        for profile in (PROFILE,LEGACY_PROFILE):
            model.view_reference['derivation']['profile']=profile
            model.projection=FixedProjection(candidate,k,depth,.25,2.21)
            model.projection.image_valid=np.ones(depth.shape,bool)
            guard(model,measured,source_projection=original)
            if profile==PROFILE:
                self.assertTrue(model.projection.valid.all())
                self.assertEqual(model.source_surface_check['method'],'full_reference_pixel_source_surface_check_v3')
            else:
                self.assertFalse(model.projection.valid.any())
                self.assertEqual(model.source_surface_check['method'],'full_reference_pixel_source_surface_check_v2')

    def test_changed_distance_policy_is_rejected_on_portable_model_reload(self):
        import json
        source,_,_,_,_=self.source()
        model=create(source,self.root/'planning')
        path=model.reconstruction/'surface_patch_reference.json'
        record=json.loads(path.read_text())
        record['view_reference']['derivation']['source_surface_metric']='ignore_distance'
        path.write_text(json.dumps(record),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'核验版本不一致'):load(model.reconstruction)


if __name__ == '__main__': unittest.main()
