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
import tkinter as tk
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


class FixedViewerExperienceTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.model=make_model(Path(self.temp.name)/'model')
        self.make_view('zh')

    def make_view(self,language):
        from femto.planning.fixed_rgb_viewer import FixedRGBViewer
        if hasattr(self,'viewer'):self.viewer.close(force=True)
        self.root=tk.Tk()
        with mock.patch.object(FixedRGBViewer,'work'):
            self.viewer=FixedRGBViewer(self.root,self.model.reconstruction,language)
        self.viewer.loaded(self.model)
        # Legacy point-focused tests opt in explicitly; the product UI defaults
        # to the square area brush.
        self.viewer.mode.set('point');self.viewer.active_mode='point'
        self.root.update();self.viewer.draw()

    def tearDown(self):
        self.viewer.close(force=True)
        worker=getattr(self.viewer,'worker',None)
        if worker is not None:worker.join(timeout=3)
        del self.viewer,self.root
        gc.collect()

    def test_distance_experiment_dialog_and_report(self):
        v=self.viewer
        v.record_distance_test()
        self.assertIn('两个表面点',v.status.get())
        self.click_pixel(30,40);self.click_pixel(70,40)
        v.items.selection_set(0,1)
        v.record_distance_test();self.root.update()
        window=next(w for w in self.root.winfo_children() if isinstance(w,tk.Toplevel))
        entries=[w for w in window.winfo_children() if w.winfo_class()=='TEntry']
        for entry,value in zip(entries,['test A','200','1','10','ruler','static board']):entry.insert(0,value)
        button=next(w for w in window.winfo_children() if w.winfo_class()=='TButton' and w.cget('text')=='保存实验记录')
        button.invoke();self.root.update()
        self.assertEqual(len(list((self.model.reconstruction/'measurement_validation/records').glob('*.json'))),1)
        v.measurement_report();self.wait_idle()
        reports=list((self.model.reconstruction/'measurement_validation').glob('report_*/report.json'))
        self.assertEqual(len(reports),1)
        report=json.loads(reports[0].read_text(encoding='utf-8'))
        self.assertEqual(len(report['measurements']),1)
        self.assertFalse(report['physical_accuracy_verified'])

    def test_capture_hides_planning_buttons(self):
        self.assertFalse(self.viewer.planning_tools.winfo_ismapped())
        self.assertTrue(self.viewer.planning_tools.winfo_children())

    def test_exclusion_draw_clear_and_undo(self):
        v=self.viewer
        v.polygon=copy.deepcopy(POLYGON);v.finish_polygon();self.wait_idle()
        v.begin_exclusion()
        self.assertIsNotNone(v.exclusion_target)
        v.polygon=[[40,40],[60,40],[60,60],[40,60]];v.finish_polygon()
        self.assertEqual(len(self.model.store.regions[0]['exclusion_polygons']),1)
        self.assertTrue(v.canvas.find_withtag('forbidden'))
        v.clear_exclusions()
        self.assertFalse(self.model.store.regions[0]['exclusion_polygons'])
        v.undo();self.wait_idle()
        self.assertEqual(len(self.model.store.regions[0]['exclusion_polygons']),1)

    def click_pixel(self,x=70,y=40):
        xy=np.array([x,y])*self.viewer.scale+self.viewer.offset
        self.viewer.click(SimpleNamespace(x=xy[0],y=xy[1]))

    def wait_idle(self,seconds=10):
        deadline=time.monotonic()+seconds
        while self.viewer.busy and time.monotonic()<deadline:
            self.root.update();time.sleep(.01)
        self.root.update();self.assertFalse(self.viewer.busy)

    def test_selected_point_and_region_highlight_clear_delete_undo(self):
        v=self.viewer;self.click_pixel();self.click_pixel(30,40)
        selected=v.canvas.find_withtag('selected');ordinary=v.canvas.find_withtag('ordinary')
        self.assertEqual(len(selected),1);self.assertEqual(len(ordinary),1)
        self.assertNotEqual(v.canvas.itemcget(selected[0],'fill'),v.canvas.itemcget(ordinary[0],'fill'))
        v.items.selection_clear(0,'end');v.items.selection_set(0);v.select()
        self.assertEqual(v.focus,('point',0));self.assertIn('距初始原点',v.readout.get());self.assertIn('末两点距离',v.readout.get())
        v.polygon=copy.deepcopy(POLYGON);v.finish_polygon();self.wait_idle()
        self.assertEqual(v.focus,('region',0));self.assertTrue(v.canvas.find_withtag('selected'))
        self.assertIn('不是曲面面积',v.readout.get())
        v.delete();self.assertFalse(v.canvas.find_withtag('selected'));self.assertNotIn('X:',v.readout.get())
        v.undo();self.wait_idle();self.assertEqual(v.focus,('region',0))
        v.clear_selection();self.assertIsNone(v.focus);self.assertFalse(v.canvas.find_withtag('selected'))

    def test_invalid_click_clears_readout_and_reports_specific_reason(self):
        v=self.viewer;v.show_axes.set(False);v.draw()
        self.click_pixel();count=len(self.model.store.points)
        v.click(SimpleNamespace(x=-1000,y=-1000))
        self.assertEqual(len(self.model.store.points),count);self.assertIsNone(v.focus)
        self.assertIn('图像外',v.status.get());self.assertNotIn('X:',v.readout.get())
        self.click_pixel(2,2);self.assertIn('无有效三维表面',v.status.get())
        self.model.projection=FixedProjection(self.model.mesh,K,np.full((100,100),300,np.uint16))
        self.click_pixel();self.assertIn('深度不一致',v.status.get())
        self.model.projection=FixedProjection(self.model.mesh,K,np.full((100,100),500,np.uint16))
        self.model.projection.image_valid=np.ones((100,100),bool);self.model.projection.image_valid[40,70]=False
        self.click_pixel();self.assertIn('没有有效原始图像像素',v.status.get())
        self.assertEqual(len(self.model.store.points),count)

    def test_region_selection_isolates_boundaries_exclusions_and_fill(self):
        from femto.annotation.fixed_rgb_geometry import selected_surface_mask
        v=self.viewer;model=self.model
        first=model.region([[54,26],[74,26],[74,47],[54,47]],'A','')
        second=model.region([[26,54],[47,54],[47,74],[26,74]],'B','')
        model.store.set_exclusions(0,[[[55,30],[60,30],[60,35],[55,35]]])
        v.refresh_items(('region',second['annotation_id']))
        with mock.patch('femto.annotation.fixed_rgb_geometry.selected_surface_mask',wraps=selected_surface_mask) as fill:
            v.draw()
            self.assertEqual(fill.call_args.args[1],second['triangle_indices'])
        self.assertFalse(v.canvas.find_withtag(first['annotation_id']))
        self.assertTrue(v.canvas.find_withtag(second['annotation_id']))
        self.assertFalse(v.canvas.find_withtag('forbidden'))
        v.items.selection_clear(0,'end');v.items.selection_set(0);v.select()
        self.assertTrue(v.canvas.find_withtag(first['annotation_id']))
        self.assertFalse(v.canvas.find_withtag(second['annotation_id']))
        self.assertTrue(v.canvas.find_withtag('forbidden'))
        with mock.patch('femto.annotation.fixed_rgb_geometry.selected_surface_mask',wraps=selected_surface_mask) as fill:
            v.select_all()
            self.assertEqual(fill.call_args.args[1],sorted(set(first['triangle_indices']+second['triangle_indices'])))
        self.assertTrue(v.canvas.find_withtag(first['annotation_id']))
        self.assertTrue(v.canvas.find_withtag(second['annotation_id']))
        before=model.store.content_signature();v.clear_selection()
        self.assertFalse(v.canvas.find_withtag('region'))
        self.assertFalse(v.canvas.find_withtag('forbidden'))
        self.assertEqual(model.store.content_signature(),before)

    def test_mixed_batch_delete_and_single_undo(self):
        v=self.viewer
        self.click_pixel();self.click_pixel(30,40)
        v.polygon=copy.deepcopy(POLYGON);v.finish_polygon();self.wait_idle()
        original=self.model.store.content_signature()
        v.items.selection_clear(0,'end')
        v.items.selection_set(0);v.items.selection_set(2);v.select()
        self.assertIsNone(v.focus)
        v.delete()
        self.assertEqual(len(self.model.store.points),1)
        self.assertEqual(len(self.model.store.regions),0)
        v.undo();self.wait_idle()
        self.assertEqual(self.model.store.content_signature(),original)
        v.select_all();self.assertEqual(len(v.items.curselection()),3)
        v.delete();self.assertEqual(len(self.model.store.points)+len(self.model.store.regions),0)
        v.undo();self.wait_idle()
        self.assertEqual(self.model.store.content_signature(),original)

    def test_unapplied_edit_selection_cancel_then_apply(self):
        v=self.viewer;self.click_pixel();self.click_pixel(30,40);xyz=copy.deepcopy(self.model.store.points[1]['point_world_m'])
        v.name.set('Edited');v.note.set('Pending note')
        v.items.selection_clear(0,'end');v.items.selection_set(0)
        with mock.patch('femto.planning.fixed_rgb_viewer.messagebox.askyesnocancel',return_value=None):v.select()
        self.assertEqual(v.focus,('point',1));self.assertEqual(v.items.curselection(),(1,));self.assertTrue(v.pending_edits())
        v.items.selection_clear(0,'end');v.items.selection_set(0)
        with mock.patch('femto.planning.fixed_rgb_viewer.messagebox.askyesnocancel',return_value=True):v.select()
        self.assertEqual(v.focus,('point',0));self.assertEqual(self.model.store.points[1]['name'],'Edited')
        self.assertEqual(self.model.store.points[1]['point_world_m'],xyz)

    def test_close_and_load_warn_for_dirty_edit_and_polygon_without_discard_on_cancel(self):
        v=self.viewer;self.click_pixel();path=self.model.store.save(Path(self.temp.name)/'saved')
        v.name.set('Unapplied');v.polygon=[[10,10],[80,10],[80,80]]
        with mock.patch('femto.planning.fixed_rgb_viewer.messagebox.askokcancel',return_value=False) as ask:
            v.close();self.assertFalse(v.closed);self.assertIn('名称 / 备注',ask.call_args.args[1])
            with mock.patch('femto.planning.fixed_rgb_viewer.filedialog.askopenfilename',return_value=str(path)):v.load()
        self.assertEqual(v.name.get(),'Unapplied');self.assertEqual(len(v.polygon),3);self.assertFalse(v.busy)

    def test_rejected_load_preserves_current_annotations_draft_fields_and_dirty(self):
        v=self.viewer;self.click_pixel();path=self.model.store.save(Path(self.temp.name)/'saved')
        self.click_pixel(30,40);v.name.set('Keep draft edit');v.polygon=[[10,10],[80,10],[80,80]]
        data=json.loads(path.read_text(encoding='utf-8'));data['geometry_sha256']='different'
        bad=Path(self.temp.name)/'bad.json';write_json(bad,data)
        signature=self.model.store.content_signature();old_store=self.model.store;readout=v.readout.get()
        with mock.patch('femto.planning.fixed_rgb_viewer.filedialog.askopenfilename',return_value=str(bad)),mock.patch('femto.planning.fixed_rgb_viewer.messagebox.askokcancel',return_value=True):v.load()
        self.wait_idle()
        self.assertIs(self.model.store,old_store);self.assertEqual(self.model.store.content_signature(),signature)
        self.assertEqual(v.readout.get(),readout);self.assertEqual(v.name.get(),'Keep draft edit');self.assertEqual(len(v.polygon),3)
        self.assertTrue(self.model.store.dirty);self.assertIn('操作失败',v.status.get())

    def test_successful_save_and_load_clear_only_actual_unsaved_content(self):
        v=self.viewer;self.click_pixel();v.save();self.wait_idle()
        self.assertFalse(self.model.store.dirty);self.assertIn('没有未保存',v.save_state.get())
        saved=next((self.model.reconstruction/'annotations').glob('*/annotations.json'));before=saved.read_bytes()
        self.click_pixel(30,40);v.polygon=[[10,10],[80,10],[80,80]]
        with mock.patch('femto.planning.fixed_rgb_viewer.filedialog.askopenfilename',return_value=str(saved)),mock.patch('femto.planning.fixed_rgb_viewer.messagebox.askokcancel',return_value=True):v.load()
        self.wait_idle()
        self.assertEqual(len(self.model.store.points),1);self.assertFalse(self.model.store.dirty);self.assertEqual(v.polygon,[])
        self.assertEqual(v.focus,('point',0));self.assertEqual(saved.read_bytes(),before)
        v.name.set('Changed');v.edit();self.assertTrue(self.model.store.dirty)
        v.undo();self.wait_idle();self.assertFalse(self.model.store.dirty)

    def test_capture_page_has_no_trajectory_actions(self):
        self.assertFalse(hasattr(self.viewer, 'surface_trajectory_button'))
        self.assertFalse(hasattr(self.viewer, 'generate_surface_trajectory'))

    def test_save_failure_keeps_dirty_state_and_reports_failure(self):
        v=self.viewer;self.click_pixel()
        with mock.patch.object(Path,'write_text',side_effect=PermissionError('simulated write failure')):
            v.save();self.wait_idle()
        self.assertTrue(self.model.store.dirty);self.assertIn('尚未保存',v.save_state.get());self.assertIn('操作失败',v.status.get())

    def test_zoom_pan_resize_keep_xyz_and_polygon_pixels_and_minimum_layout(self):
        for language in ('zh','en'):
            if language!='zh':self.make_view(language)
            v=self.viewer;self.click_pixel()
            xyz=copy.deepcopy(self.model.store.points[-1]['point_world_m'])
            v.mode.set('region');v.change_mode()
            for x,y in ((20,20),(80,20),(80,80)):self.click_pixel(x,y)
            v.brush_end()
            polygon=copy.deepcopy(v.polygon)
            v.wheel(SimpleNamespace(x=200,y=200,delta=120));v.pan_start(SimpleNamespace(x=0,y=0));v.pan_move(SimpleNamespace(x=31,y=17))
            self.root.geometry('1000x720');self.root.update();self.root.update()
            self.assertEqual(v.polygon,polygon);self.assertEqual(self.model.store.points[-1]['point_world_m'],xyz)
            for widget in v.widgets:
                if widget.winfo_ismapped():self.assertGreaterEqual(widget.winfo_height(),widget.winfo_reqheight())
            v.side_canvas.yview_moveto(1);self.root.update()
            last=[widget for widget in v.widgets if widget.winfo_ismapped()][-1]
            self.assertLessEqual(last.winfo_rooty()+last.winfo_height(),self.root.winfo_rooty()+self.root.winfo_height())
            self.assertGreater(v.canvas.winfo_height(),100)

    def test_square_brush_erase_undo_and_finish(self):
        from femto.annotation.square_brush import polygon_mask
        v=self.viewer;v.show_axes.set(False);v.mode.set('region');v.change_mode()
        v.brush_size.set(160)
        self.click_pixel(40,40)
        xy=np.array([60,60])*v.scale+v.offset
        event=SimpleNamespace(x=xy[0],y=xy[1])
        v.brush_move(event);v.brush_end(event)
        painted=copy.deepcopy(v.polygon)
        self.assertTrue(polygon_mask(painted,(100,100))[50,50])
        v.brush_size.set(16);v.brush_erase.set(True)
        self.click_pixel(50,50);v.brush_end()
        self.assertFalse(polygon_mask(v.polygon,(100,100))[50,50])
        v.undo();self.assertEqual(v.polygon,painted)
        v.undo();self.assertEqual(v.polygon,[])
        v.brush_size.set(160);v.brush_erase.set(False)
        self.click_pixel(63,37);v.brush_end();v.finish_polygon();self.wait_idle()
        self.assertEqual(len(self.model.store.regions),1)
        self.assertEqual(v.polygon,[]);self.assertEqual(v._brush_history,[])

    def test_square_cursor_size_and_busy_guard(self):
        v=self.viewer;v.mode.set('region');v.change_mode();v.show_axes.set(False)
        xy=np.array([50,50])*v.scale+v.offset;e=SimpleNamespace(x=xy[0],y=xy[1])
        v.brush_size.set(72);v.brush_hover(e)
        rectangle=v.canvas.coords(v.canvas.find_withtag('brush_bounds')[0])
        self.assertAlmostEqual(rectangle[2]-rectangle[0],72)
        v.busy=True;v.click(e);v.brush_move(e);v.brush_end()
        self.assertEqual(v.polygon,[]);v.busy=False

    def test_brush_moves_reuse_background_and_preserve_every_segment(self):
        from femto.annotation.square_brush import stroke,mask_polygon
        v=self.viewer;v.show_axes.set(False);v.mode.set('region');v.change_mode()
        self.click_pixel(25,25)
        expected=v._brush_mask.copy();last=v._brush_last.copy()
        with mock.patch.object(v,'draw',wraps=v.draw) as redraw:
            for pixel in ([35,25],[35,45],[60,45],[60,60]):
                xy=np.asarray(pixel)*v.scale+v.offset
                v.brush_move(SimpleNamespace(x=xy[0],y=xy[1]))
                stroke(expected,last,pixel,v._brush_stroke_size,v._brush_stroke_erase);last=pixel
            redraw.assert_not_called()
            np.testing.assert_array_equal(v._brush_mask,expected)
            self.assertIsNotNone(v._brush_frame)
        expected_polygon=mask_polygon(expected)
        v.brush_end()
        self.assertIsNone(v._brush_frame)
        self.assertEqual(v.polygon,expected_polygon)
        v.undo();self.assertEqual(v.polygon,[])

    def test_brush_cache_reused_and_invalidated_by_polygon(self):
        from femto.annotation.square_brush import polygon_mask
        v=self.viewer;polygon=[[10,10],[30,10],[30,30],[10,30]]
        with mock.patch('femto.annotation.square_brush.polygon_mask',wraps=polygon_mask) as raster:
            first=v.cached_brush_mask('work',polygon)
            self.assertIs(v.cached_brush_mask('work',copy.deepcopy(polygon)),first)
            self.assertEqual(raster.call_count,1)
            changed=v.cached_brush_mask('work',[[40,40],[60,40],[60,60],[40,60]])
            self.assertEqual(raster.call_count,2)
            self.assertFalse(changed[20,20]);self.assertTrue(changed[50,50])

    def test_product_ui_hides_non_core_tools(self):
        texts=[]
        def visit(widget):
            for child in widget.winfo_children():
                try:
                    text=child.cget('text')
                    if text:texts.append(str(text))
                except tk.TclError:pass
                visit(child)
        visit(self.root)
        for hidden in ('回读标记','记录实测距离（选中两点）','同位置几何诊断（选中一点）',
                       '数据质量 / 精度测试报告','打开精度测试记录','保存方案','方案另存为…',
                       '恢复或放弃草稿…','打开已有方案…','添加禁止区域（先选主区域）','选点 / 测距',
                       '应用名称备注','删除所选 / 可撤销','清除当前选中'):
            self.assertNotIn(hidden,texts)
        self.assertIn('保存并导出当前结果 GLB / PLY',texts)
        self.assertIn('历史区域导出记录',texts)
        self.assertIn('工作区域',texts)
        self.assertIn('禁止区域',texts)
        self.assertIn('完成区域选择',texts)

    def test_single_region_submit_replaces_previous_and_save_applies_metadata(self):
        v=self.viewer;v.mode.set('region');v.change_mode()
        v.name.set('第一次');v.polygon=[[54,26],[74,26],[74,47],[54,47]]
        v.finish_polygon();self.wait_idle()
        first_id=self.model.store.regions[0]['annotation_id']
        v.name.set('最终区域');v.note.set('单次保存')
        v.polygon=[[26,54],[47,54],[47,74],[26,74]]
        v.finish_polygon();self.wait_idle()
        self.assertEqual(len(self.model.store.regions),1)
        self.assertNotEqual(self.model.store.regions[0]['annotation_id'],first_id)
        self.assertEqual(self.model.store.regions[0]['name'],'最终区域')
        v.name.set('导出名称');v.note.set('保存时自动应用')
        with mock.patch.object(self.model.store,'save',return_value=Path(self.temp.name)/'annotations.json') as save:
            v.save();self.wait_idle()
        save.assert_called_once()
        self.assertEqual(self.model.store.regions[0]['name'],'导出名称')
        self.assertEqual(self.model.store.regions[0]['note'],'保存时自动应用')

    def test_automatic_work_forbidden_combinations(self):
        v=self.viewer;v.mode.set('region');v.change_mode()
        # Case 1: work only. Triangle 0 projects near (63,37).
        v.polygon=[[54,26],[74,26],[74,47],[54,47]]
        v.finish_polygon();self.wait_idle()
        self.assertEqual(len(self.model.store.regions),1)
        region=self.model.store.regions[-1]
        self.assertEqual(region['triangle_indices'],[0])
        self.assertEqual(region['selection_method'],'work_only')
        # Case 2: forbidden only keeps every other supported surface.
        v.selection_method.set('forbidden');v.change_selection_method()
        v.polygon=[[54,26],[74,26],[74,47],[54,47]]
        v.finish_polygon();self.wait_idle();region=self.model.store.regions[-1]
        self.assertEqual(region['triangle_indices'],[1])
        self.assertEqual(region['selection_method'],'full_minus_forbidden')
        # Case 3: work and forbidden combines as work minus forbidden.
        v.selection_method.set('work');v.change_selection_method()
        v.polygon=[[0,0],[99,0],[99,99],[0,99]]
        v.selection_method.set('forbidden');v.change_selection_method()
        v.polygon=[[54,26],[74,26],[74,47],[54,47]]
        v.finish_polygon();self.wait_idle();region=self.model.store.regions[-1]
        self.assertEqual(region['triangle_indices'],[1])
        self.assertEqual(region['selection_method'],'work_minus_forbidden')
        self.assertTrue(region['work_screen_polygon']);self.assertTrue(region['forbidden_screen_polygon'])
        self.assertIn('工作区域减去禁止区域',v.status.get())

    def test_mode_change_can_preserve_or_explicitly_discard_draft(self):
        v=self.viewer;v.mode.set('region');v.change_mode();v.polygon=[[20,20],[80,20],[80,80]]
        v.mode.set('point')
        with mock.patch('femto.planning.fixed_rgb_viewer.messagebox.askokcancel',return_value=False):v.change_mode()
        self.assertEqual(v.mode.get(),'region');self.assertEqual(len(v.polygon),3)
        v.mode.set('point')
        with mock.patch('femto.planning.fixed_rgb_viewer.messagebox.askokcancel',return_value=True):v.change_mode()
        self.assertEqual(v.polygon,[]);self.assertEqual(self.model.store.regions,[])

    def test_close_cancels_worker_and_drops_late_callback(self):
        v=self.viewer;started=threading.Event();release=threading.Event();done=mock.Mock()
        def slow():started.set();release.wait(3);return 'late'
        v.operation='read';v.work(slow,done)
        self.assertTrue(started.wait(1));worker=v.worker
        v.close(force=True);release.set();worker.join(3)
        self.assertFalse(worker.is_alive());self.assertTrue(v.task_cancel.is_set());self.assertTrue(v.mailbox.empty());done.assert_not_called()

    def test_model_change_discards_old_worker_and_clears_markers_and_readout(self):
        v=self.viewer;self.click_pixel();started=threading.Event();release=threading.Event();done=mock.Mock()
        def slow():started.set();release.wait(3);return 'old model'
        v.operation='read';v.work(slow,done);self.assertTrue(started.wait(1));worker=v.worker
        replacement=make_model(Path(self.temp.name)/'replacement')
        v.loaded(replacement);release.set();worker.join(3);self.root.update()
        self.assertIs(v.model,replacement);self.assertIsNone(v.focus);self.assertNotIn('X:',v.readout.get());self.assertFalse(v.busy)
        self.assertEqual(v.items.size(),0);self.assertTrue(v.mailbox.empty());done.assert_not_called()

    def test_timeout_uses_real_elapsed_time_and_does_not_apply_late_callback(self):
        v=self.viewer;started=threading.Event();release=threading.Event();done=mock.Mock()
        def slow():started.set();release.wait(3);return 'late'
        v.operation='read';v.work(slow,done);self.assertTrue(started.wait(1))
        v.task_started=time.monotonic()-121
        self.root.after_cancel(v.timer);v.poll()
        self.assertTrue(v.task_cancel.is_set());self.assertIn('120',v.status.get())
        release.set();self.wait_idle();done.assert_not_called();self.assertIn('超时',v.status.get())


class HistoryExperienceTests(unittest.TestCase):
    def setUp(self):
        gc.collect();self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)

    def test_missing_or_unknown_provenance_disables_fixed_but_keeps_mesh_and_hashes(self):
        source,result=history_fixture(self.folder)
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.folder.rglob('*') if p.is_file()}
        rows=scan_history(self.folder/'datasets',self.folder/'outputs');self.assertTrue(existing_actions(rows[0])['fixed'])
        after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.folder.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        (source/'manifest.json').unlink()
        self.assertFalse(existing_actions(rows[0])['fixed']);self.assertTrue(existing_actions(rows[0])['mesh'])
        row=scan_history(self.folder/'datasets',self.folder/'outputs')[0]
        self.assertEqual(row['fixed_reason'],'missing_source_files');self.assertFalse(row['fixed'])

    def test_wrong_session_and_invalid_calibration_are_not_eligible(self):
        source,result=history_fixture(self.folder)
        manifest=json.loads((source/'manifest.json').read_text());manifest['local_capture']['session_id']='other';write_json(source/'manifest.json',manifest)
        row=scan_history(self.folder/'datasets',self.folder/'outputs')[0]
        self.assertFalse(row['fixed']);self.assertEqual(row['fixed_reason'],'origin_unverified')
        manifest['local_capture']['session_id']='fixture';write_json(source/'manifest.json',manifest)
        write_json(source/'calibration/color_intrinsic.json',dict(width=100,height=100,fx=0,fy=100,cx=50,cy=50))
        row=scan_history(self.folder/'datasets',self.folder/'outputs')[0]
        self.assertFalse(row['fixed']);self.assertEqual(row['fixed_reason'],'invalid_calibration')

    def test_unpublished_results_keep_raw_record_and_never_change_latest(self):
        source,result=history_fixture(self.folder)
        latest=self.folder/'outputs/latest_complete_demo.txt';latest.write_text('previous-result',encoding='utf-8')
        for status in ('pending','cancelled','failed','published'):
            write_json(result/'publication.json',dict(schema='femto-local-publication-v1',
                       status=status,session_id='fixture',result=str(result.resolve())))
            before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.folder.rglob('*') if p.is_file()}
            rows=scan_history(self.folder/'datasets',self.folder/'outputs')
            model=next(row for row in rows if row['result'])
            self.assertEqual(model['publication_status'],status)
            self.assertEqual(model['state'],'model' if status=='published' else 'unpublished')
            self.assertEqual(existing_actions(model)['fixed'],status=='published')
            self.assertTrue(existing_actions(model)['mesh'])
            self.assertEqual(len(rows),1 if status=='published' else 2)
            self.assertEqual(latest.read_text(encoding='utf-8'),'previous-result')
            after={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.folder.rglob('*') if p.is_file()}
            self.assertEqual(before,after)

    def test_missing_or_cross_session_publication_receipt_is_rejected(self):
        source,result=history_fixture(self.folder)
        write_json(result/'demo_summary.json',dict(publication_required=True,publication_id='fixture'))
        row=next(row for row in scan_history(self.folder/'datasets',self.folder/'outputs') if row['result'])
        self.assertEqual(row['publication_status'],'missing');self.assertFalse(existing_actions(row)['fixed'])
        write_json(result/'publication.json',dict(schema='femto-local-publication-v1',status='published',
                   session_id='other',result=str(result.resolve())))
        row=next(row for row in scan_history(self.folder/'datasets',self.folder/'outputs') if row['result'])
        self.assertEqual(row['publication_status'],'invalid');self.assertFalse(existing_actions(row)['fixed'])

    def test_comparison_requires_all_source_rgbd_but_fixed_view_only_needs_reference(self):
        source,result=history_fixture(self.folder)
        rows=scan_history(self.folder/'datasets',self.folder/'outputs')
        self.assertTrue(existing_actions(rows[0])['compare'])
        pose=json.loads((source/'pose/0.json').read_text());write_json(source/'pose/1.json',pose)
        with (source/'frames.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(frame=1,pose_valid=True,rgb='rgb/1.png',depth='depth/1.png',pose='pose/1.json'))+'\n')
        row=scan_history(self.folder/'datasets',self.folder/'outputs')[0]
        self.assertTrue(existing_actions(row)['fixed']);self.assertFalse(existing_actions(row)['compare'])
        with (source/'calibration/color_intrinsic.json').open('a',encoding='utf-8') as stream:stream.write(' ')
        self.assertFalse(existing_actions(row)['fixed'])

    def test_history_comparison_launches_only_for_eligible_published_model(self):
        from femto.app.app_local_history import LocalHistoryWindow
        source,result=history_fixture(self.folder)
        write_json(result/'publication.json',dict(schema='femto-local-publication-v1',status='published',
                   session_id='fixture',result=str(result.resolve())))
        root=tk.Tk();root.withdraw();window=None
        launch=mock.Mock()
        try:
            window=LocalHistoryWindow(root,self.folder/'datasets',self.folder/'outputs')
            deadline=time.monotonic()+5
            while window.loading and time.monotonic()<deadline:root.update();time.sleep(.01)
            self.assertFalse(window.loading)
            with mock.patch.dict('sys.modules',{'femto.app.app_local_comparison':SimpleNamespace(ComparisonWindow=launch)}):
                window.open('compare')
                launch.assert_called_once_with(window.window,str(result.resolve()),self.folder/'outputs','zh')
                write_json(result/'publication.json',dict(schema='femto-local-publication-v1',status='cancelled',
                           session_id='fixture',result=str(result.resolve())))
                window.open('compare')
                self.assertEqual(launch.call_count,1)
        finally:
            if window is not None:window.close()
            root.destroy()
        del root,window;gc.collect()

    def test_refresh_preserves_selected_key_and_quality_summary_at_small_size(self):
        from femto.app.app_local_history import LocalHistoryWindow
        for language in ('zh','en'):
            root=tk.Tk();root.withdraw();window=None
            def row(key):
                return dict(key=key,name=key,result=None,source=str(self.folder),session_id=key,state='recorded',count=1,
                            error=None,fixed=False,mesh=None,report=None,time=1,quality=empty_report())
            try:
                with mock.patch('femto.app.app_local_history.scan_history',return_value=[row('a'),row('b')]):
                    window=LocalHistoryWindow(root,self.folder/'datasets',self.folder/'outputs',language)
                    deadline=time.monotonic()+3
                    while window.loading and time.monotonic()<deadline:root.update();time.sleep(.01)
                window.tree.selection_set('1');window.selected()
                with mock.patch('femto.app.app_local_history.scan_history',return_value=[row('c'),row('a'),row('b')]):
                    window.refresh();deadline=time.monotonic()+3
                    while window.loading and time.monotonic()<deadline:root.update();time.sleep(.01)
                self.assertEqual(window.current()['key'],'b')
                text=window.quality_text.get('1.0','end')
                self.assertIn('未记录' if language=='zh' else 'Not recorded',text)
                window.window.geometry('760x520');root.update();root.update()
                bottom=window.window.winfo_rooty()+window.window.winfo_height()
                for button in window.buttons.values():
                    self.assertGreaterEqual(button.winfo_height(),button.winfo_reqheight())
                    self.assertLessEqual(button.winfo_rooty()+button.winfo_height(),bottom)
            finally:
                if window is not None:window.close()
                root.destroy()
            del root,window;gc.collect()

    def test_closed_history_ignores_late_scan_and_budget_can_cancel(self):
        from femto.app.app_local_history import LocalHistoryWindow
        cancel=threading.Event();cancel.set();history_fixture(self.folder)
        with self.assertRaisesRegex(RuntimeError,'cancelled'):scan_history(self.folder/'datasets',self.folder/'outputs',cancel=cancel)
        with self.assertRaises(TimeoutError):scan_history(self.folder/'datasets',self.folder/'outputs',seconds=-1)
        root=tk.Tk();root.withdraw();started=threading.Event();release=threading.Event()
        def slow(*args,**kwargs):started.set();release.wait(3);return []
        try:
            with mock.patch('femto.app.app_local_history.scan_history',side_effect=slow):
                window=LocalHistoryWindow(root,self.folder/'datasets',self.folder/'outputs')
                self.assertTrue(started.wait(1));window.close();release.set()
                deadline=time.monotonic()+1
                while time.monotonic()<deadline and not window.mailbox.empty():time.sleep(.01)
                window.poll();self.assertTrue(window.cancel.is_set());self.assertTrue(window.mailbox.empty())
        finally:release.set();root.destroy()
        del root,window;gc.collect()


if __name__=='__main__':unittest.main()
