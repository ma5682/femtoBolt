"""Fixed-photo annotation UX, source eligibility and cancellation regressions.

All fixtures are temporary synthetic geometry/metadata. No camera is opened.
"""
import copy
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import open3d as o3d

from femto.annotation.fixed_rgb_geometry import FixedProjection, FixedRGBModel, GeometryTaskCancelled
from femto.annotation.scene_annotations import AnnotationStore, region_boundary
from femto.pipeline.local_history import scan_history, existing_actions
from femto.local.local_processing import empty_report


def quad(z=.5, left=-.2, right=.2, top=-.2, bottom=.2):
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices=o3d.utility.Vector3dVector([[left,top,z],[right,top,z],[right,bottom,z],[left,bottom,z]])
    mesh.triangles=o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
    return mesh


def ring():
    mesh=quad()
    mesh.vertices=o3d.utility.Vector3dVector(np.r_[np.asarray(mesh.vertices),
        [[-.05,-.05,.5],[.05,-.05,.5],[.05,.05,.5],[-.05,.05,.5]]])
    mesh.triangles=o3d.utility.Vector3iVector([[0,1,5],[0,5,4],[1,2,6],[1,6,5],
        [2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return mesh


K=np.array([[100.,0,50],[0,100.,50],[0,0,1]])
POLYGON=[[0,0],[99,0],[99,99],[0,99]]


def make_model(folder,mesh=None,depth=None):
    model=FixedRGBModel.__new__(FixedRGBModel)
    model.reconstruction=Path(folder)/'reconstruction';model.reconstruction.mkdir(parents=True)
    model.mesh=mesh if mesh is not None else quad()
    model.rgb=np.full((100,100,3),140,np.uint8)
    model.projection=FixedProjection(model.mesh,K,np.full((100,100),500,np.uint16) if depth is None else depth)
    model.store=AnnotationStore(model.mesh,model.reconstruction/'mesh.ply')
    model.view_reference=dict(reference_frame_id=0,session_id='synthetic-fixture',
                              rgb_sha256='fixture-rgb',depth_sha256='fixture-depth',image_size=[100,100])
    model.max_radius_mm=0;model._overlay_cache={}
    return model


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')


def history_fixture(folder):
    root=Path(folder);source=root/'datasets/scene_fixture_local_a'
    result=root/'outputs/complete_demo_fixture';rec=result/'reconstruction'
    ref=dict(session_id='fixture',reference_frame_id=0,coordinate_frame='world_rdf',unit='metre',transform_direction='T_W0_Ct')
    write_json(source/'derived/local_capture_report.json',dict(ref,schema='femto-local-tracking-v1',
               pose_source='online_rgbd_research_grade',accepted_frames=[0],state='stopped'))
    write_json(rec/'local_geometry.json',dict(ref,tracking_report=str(source/'derived/local_capture_report.json')))
    write_json(rec/'reconstruction_report.json',dict(source_dataset=str(source),frame_count=1,
               local_capture=ref,outputs=dict(mesh=dict(ply=str(rec/'scene_tsdf_mesh_m.ply')))))
    (rec/'scene_tsdf_mesh_m.ply').write_bytes(b'synthetic catalogue fixture, not loaded as geometry')
    write_json(source/'manifest.json',dict(local_capture=ref))
    write_json(source/'pose/0.json',dict(ref,pose_valid=True,pose_source='online_rgbd_research_grade',T_WC=np.eye(4).tolist()))
    for name in ('rgb/0.png','depth/0.png'):
        path=source/name;path.parent.mkdir();path.write_bytes(b'catalogue existence fixture')
    (source/'frames.jsonl').write_text(json.dumps(dict(frame=0,pose_valid=True,rgb='rgb/0.png',depth='depth/0.png',pose='pose/0.json'))+'\n')
    write_json(source/'calibration/color_intrinsic.json',dict(width=100,height=100,fx=100,fy=100,cx=50,cy=50))
    write_json(source/'calibration/distortion.json',dict(rgb=dict(model='BROWN_CONRADY_K6',
               k1=0,k2=0,k3=0,k4=0,k5=0,k6=0,p1=0,p2=0)))
    return source,result


class GeometryExperienceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)

    def test_five_mm_back_surface_is_occluded_without_relaxing_depth_gate(self):
        projection=FixedProjection(quad()+quad(.505),K,np.full((100,100),500,np.uint16))
        np.testing.assert_array_equal(projection.visible([[0,0,.5],[0,0,.505]]),[True,False])
        self.assertEqual(projection.region(POLYGON),[0,1])
        conflict=FixedProjection(quad(),K,np.full((100,100),480,np.uint16))
        self.assertIsNone(conflict.pixel(50,50))

    def test_boundary_with_visible_ends_is_clipped_at_middle_occluder(self):
        mesh=quad(top=0,bottom=.2)+quad(.48,left=-.03,right=.03,top=-.03,bottom=.03)
        depth=np.full((100,100),500,np.uint16);depth[44:57,44:57]=480
        projection=FixedProjection(mesh,K,depth)
        boundary=np.array([[[-.2,0,.5],[.2,0,.5]]]);before=boundary.copy()
        segments=projection.visible_boundary(boundary)
        self.assertEqual(len(segments),2)
        self.assertLess(segments[0,:,0].max(),50)
        self.assertGreater(segments[1,:,0].min(),50)
        np.testing.assert_array_equal(boundary,before)

    def test_pixel_reasons_distinguish_missing_conflict_and_invalid_source(self):
        p=FixedProjection(quad(),K,np.full((100,100),500,np.uint16))
        self.assertEqual(p.pixel_reason(-1,50),'outside_image')
        self.assertEqual(p.pixel_reason(float('nan'),50),'invalid_pixel')
        self.assertEqual(p.pixel_reason(2,2),'no_surface')
        p.image_valid=np.ones((100,100),bool);p.image_valid[50,50]=False
        self.assertEqual(p.pixel_reason(50,50),'no_source_pixel');self.assertIsNone(p.pixel(50,50))
        conflict=FixedProjection(quad(),K,np.full((100,100),300,np.uint16))
        self.assertEqual(conflict.pixel_reason(50,50),'depth_conflict')

    def test_region_pixel_support_keeps_hole_and_is_not_surface_area(self):
        model=make_model(self.folder,ring())
        item=model.region(POLYGON,'ring','')
        stats=item['geometry_support']
        self.assertEqual(stats['unit'],'pixel');self.assertEqual(stats['interpretation'],'image_pixel_support_not_surface_area')
        self.assertEqual(stats['requested_pixels'],99*99)
        expected=int(model.projection.valid[:99,:99].sum())
        self.assertEqual(stats['geometry_supported_pixels'],expected)
        self.assertAlmostEqual(stats['support_ratio'],expected/(99*99))
        self.assertGreater(stats['unsupported_pixels'],0)
        self.assertIsNone(model.projection.pixel(50,50))
        self.assertEqual(len(item['boundary_world_m']),8)

    def test_support_recomputed_on_load_and_original_json_unchanged(self):
        model=make_model(self.folder,ring())
        item=model.region(POLYGON,'ring','');expected=copy.deepcopy(item['geometry_support'])
        path=model.store.save(self.folder/'saved')
        data=json.loads(path.read_text(encoding='utf-8'))
        data['regions'][0]['geometry_support']['geometry_supported_pixels']=999999
        data['regions'][0]['geometry_support']['support_ratio']=1
        changed=self.folder/'stale-derived-fields.json';write_json(changed,data)
        before=changed.read_bytes();model.load(changed)
        self.assertEqual(model.store.regions[0]['geometry_support'],expected)
        self.assertEqual(before,changed.read_bytes());self.assertFalse(model.store.dirty)

    def test_legacy_region_has_unrecorded_support_and_wrong_pixel_unit_is_rejected(self):
        model=make_model(self.folder)
        model.region(POLYGON,'region','');path=model.store.save(self.folder/'saved')
        data=json.loads(path.read_text(encoding='utf-8'))
        data['regions'][0].pop('view_reference');legacy=self.folder/'legacy.json';write_json(legacy,data)
        model.load(legacy);self.assertNotIn('geometry_support',model.store.regions[0])
        bad=json.loads(path.read_text(encoding='utf-8'));bad['regions'][0]['screen_polygon_unit']='metre'
        changed=self.folder/'wrong-units.json';write_json(changed,bad)
        before=model.store
        with self.assertRaisesRegex(ValueError,'units'):model.load(changed)
        self.assertIs(model.store,before)

    def test_cancelled_region_and_load_leave_current_annotations_unchanged(self):
        model=make_model(self.folder);model.pick(70,40,'A','');path=model.store.save(self.folder/'saved')
        cancel=threading.Event();cancel.set();before=model.store
        with self.assertRaises(GeometryTaskCancelled):model.region(POLYGON,'B','',cancel=cancel)
        with self.assertRaises(GeometryTaskCancelled):model.load(path,cancel=cancel)
        self.assertIs(model.store,before);self.assertEqual(len(model.store.points),1);self.assertEqual(model.store.regions,[])

    def test_display_budget_rejects_before_committing_region(self):
        model=make_model(self.folder)
        with mock.patch('femto.annotation.fixed_rgb_geometry.MAX_BOUNDARY_SAMPLES',1):
            with self.assertRaisesRegex(ValueError,'budget'):model.region(POLYGON,'B','')
        with mock.patch('femto.annotation.fixed_rgb_geometry.MAX_OVERLAY_BYTES',1):
            with self.assertRaisesRegex(ValueError,'budget'):model.region(POLYGON,'B','')
        self.assertEqual(model.store.regions,[]);self.assertFalse(model.store.dirty)

    def test_large_ply_is_rejected_before_native_allocation(self):
        rec=self.folder/'reconstruction';rec.mkdir()
        write_json(rec/'reconstruction_report.json',{})
        (rec/'scene_tsdf_mesh_m.ply').write_bytes(b'ply\nformat ascii 1.0\nelement vertex 4000001\nelement face 4000001\nend_header\n')
        with mock.patch('femto.annotation.fixed_rgb_geometry.read_geometry_safe') as read:
            with self.assertRaisesRegex(ValueError,'budget'):FixedRGBModel(rec)
        read.assert_not_called()

    def test_dirty_content_save_edit_undo_and_failed_save(self):
        model=make_model(self.folder);self.assertFalse(model.store.dirty)
        model.pick(70,40,'A','');self.assertTrue(model.store.dirty)
        path=model.store.save(self.folder/'saved');before=path.read_bytes();self.assertFalse(model.store.dirty)
        model.store.update_annotation('point',0,name='B',note='changed');self.assertTrue(model.store.dirty)
        model.store.undo();self.assertFalse(model.store.dirty)
        model.store.update_annotation('point',0,name='C',note='')
        with mock.patch.object(Path,'write_text',side_effect=PermissionError('simulated disk write failure')):
            with self.assertRaises(PermissionError):model.store.save(self.folder/'failed')
        self.assertTrue(model.store.dirty);self.assertEqual(path.read_bytes(),before)

    def test_different_model_import_preserves_current_content_and_undo(self):
        a=make_model(self.folder/'a');b=make_model(self.folder/'b')
        a.pick(70,40,'A','');path=a.store.save(self.folder/'saved')
        b.pick(30,40,'B','');before=b.store.content_signature()
        with self.assertRaisesRegex(ValueError,'model'):b.load(path)
        self.assertEqual(b.store.content_signature(),before);self.assertTrue(b.store.can_undo);self.assertTrue(b.store.dirty)
