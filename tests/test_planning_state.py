import copy
import gc
import json
import tempfile
import time
import tkinter as tk
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from femto.planning.fixed_rgb_viewer import FixedRGBViewer
from femto.planning.planning_state import state, parameters, validate
from femto.planning.region_path import raster_path, export_path, load_path
from femto.planning.region_project import save_project, open_project
from tests.test_fixed_rgb_experience import make_model


class PlanningStateTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        self.temp=tempfile.TemporaryDirectory();self.folder=Path(self.temp.name)
        self.model=make_model(self.folder)
        import open3d as o3d
        o3d.io.write_triangle_mesh(self.model.store.source,self.model.mesh)
        self.region=self.model.region([[15,15],[85,15],[85,85],[15,85]],'Region','')
        self.plan=raster_path(self.model.projection,self.model.mesh,self.region,10,10)
        self.root=tk.Tk();self.root.withdraw()
        with mock.patch.object(FixedRGBViewer,'work'):
            self.viewer=FixedRGBViewer(self.root,self.model.reconstruction)
        self.viewer.loaded(self.model)
        self.viewer.refresh_items(('region',self.region['annotation_id']))
        self.viewer.show_region_path(self.model,self.plan)

    def tearDown(self):
        self.viewer.close(force=True)
        if hasattr(self.viewer,"worker"):self.viewer.worker.join(3)
        self.viewer=None;self.root=None;gc.collect()
        self.temp.cleanup()

    def pump(self):
        deadline=time.monotonic()+3
        while self.viewer.busy and time.monotonic()<deadline:
            self.root.update();time.sleep(.01)
        self.assertFalse(self.viewer.busy)

    def test_exclusion_stops_open_demo_and_export_then_undo_restores(self):
        v=self.viewer;v.present_region_path();demo=self.root._route_demo_window
        self.assertTrue(demo.winfo_exists())
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)
        for label in ('播放','查看覆盖报告'):
            button=next(w for w in descendants(demo) if 'text' in w.keys() and w.cget('text')==label)
            button.invoke()
        reports=[w for w in demo.winfo_children() if isinstance(w,tk.Toplevel)]
        self.assertEqual(len(reports),1)
        self.model.store.set_exclusions(0,[[[30,30],[40,30],[40,40],[30,40]]])
        v.refresh_items(('region',self.region['annotation_id']))
        self.assertEqual(v.planning_state()['phase'],'invalid')
        self.assertFalse(demo.winfo_exists())
        self.assertFalse(reports[0].winfo_exists())
        self.assertIsNone(v.current_region_path())
        self.assertTrue(v._plan_buttons['export_region_path'].instate(['disabled']))
        with mock.patch('femto.planning.region_path.export_path') as export:
            v.export_region_path();export.assert_not_called()
        with self.assertRaises(ValueError):
            export_path(self.plan,self.model.store,self.folder/'bad',self.model.store.regions[0])
        self.assertFalse((self.folder/'bad').exists())
        v.undo();self.assertIs(v.current_region_path(),self.plan)

    def test_name_note_keeps_path_but_marks_save_dirty(self):
        self.model.store.mark_saved()
        self.model.store.update_annotation('region',0,name='Renamed',note='A note')
        self.viewer.refresh_items(('region',self.region['annotation_id']))
        self.assertIs(self.viewer.current_region_path(),self.plan)
        self.assertEqual(self.viewer.planning_state()['save_state'],'dirty')
        validate(self.plan,self.model.mesh,self.model.store.regions[0])

    def test_multiple_regions_switch_without_report_reuse(self):
        v=self.viewer
        other=self.model.region([[15,15],[85,15],[85,85],[15,85]],'Other','')
        v.refresh_items(('region',other['annotation_id']))
        self.assertIsNone(v.current_region_path())
        other_plan=raster_path(self.model.projection,self.model.mesh,other,20,10)
        v.show_region_path(self.model,other_plan)
        self.assertIs(v.current_region_path(),other_plan)
        v.refresh_items(('region',self.region['annotation_id']))
        self.assertIs(v.current_region_path(),self.plan)
        self.assertEqual(v.current_region_path()['plan_binding']['region_id'],self.region['annotation_id'])

    def test_failure_and_retry_preserve_previous_and_parameter_mismatch_blocks(self):
        v=self.viewer;v.operation='path_preview'
        def fail():raise ValueError('injected failure')
        v.work(fail,lambda _:self.fail('must not commit'))
        self.assertEqual(v.planning_state()['phase'],'generating')
        self.pump()
        self.assertEqual(v.planning_state()['phase'],'failed')
        self.assertIs(v.current_region_path(),self.plan)
        v._requests[self.region['annotation_id']]=dict(kind='raster',values=dict(row_step_px=20,max_step_mm=10))
        v.work(fail,lambda _:self.fail('must not commit'));self.pump()
        self.assertIsNone(v.current_region_path())
        self.assertIs(v._plans[self.region['annotation_id']],self.plan)
        v.work(lambda:raster_path(self.model.projection,self.model.mesh,self.region,20,10),
               lambda plan:v.show_region_path(self.model,plan));self.pump()
        self.assertEqual(v.planning_state()['phase'],'valid')
        self.assertEqual(v.current_region_path()['row_step_px'],20)

    def test_cancel_rejects_late_result(self):
        import threading
        release=threading.Event();v=self.viewer;v.operation='path_preview'
        v.work(lambda:(release.wait(1),self.plan)[1],lambda _:self.fail('cancelled result committed'))
        v.cancel_planning();release.set();self.pump()
        self.assertEqual(v.planning_state()['phase'],'failed')
        self.assertIs(v.current_region_path(),self.plan)

    def test_late_result_after_region_change_is_rejected_and_poll_survives(self):
        import threading
        v=self.viewer;v.operation='path_preview';release=threading.Event()
        v.work(lambda:(release.wait(1),self.plan)[1],lambda plan:v.show_region_path(self.model,plan))
        self.model.store.set_exclusions(0,[[[30,30],[40,30],[40,40],[30,40]]])
        release.set();self.pump()
        self.assertIsNone(v.current_region_path())
        self.assertEqual(v.planning_state()['phase'],'failed')
        self.assertIn('过期',v.planning_state()['reason'])

    def test_save_rename_reopen_and_parameter_roundtrip(self):
        self.model.store.update_annotation('region',0,name='New title',note='')
        region=self.model.store.regions[0]
        path=save_project(self.model,region,self.folder/'plan.zip',self.plan)
        reopened=open_project(path,self.folder/'work')
        self.assertEqual(reopened.saved_plan,self.plan)
        self.assertEqual(reopened.store.regions[0]['name'],'New title')
        self.viewer.loaded(reopened)
        self.assertEqual(self.viewer.planning_state()['save_state'],'saved')
        self.assertEqual(self.viewer.planning_state()['phase'],'valid')
        params=dict(kind='raster',values=dict(row_step_px=25,max_step_mm=10))
        path=save_project(reopened,reopened.store.regions[0],self.folder/'pending.zip',planning_parameters=params)
        pending=open_project(path,self.folder/'pending-work')
        self.assertIsNone(pending.saved_plan);self.assertEqual(pending.planning_parameters,params)

    def test_tampered_versions_reports_parameters_and_region_rejected(self):
        for change in ('algorithm','parameter','report','region'):
            plan=copy.deepcopy(self.plan);region=copy.deepcopy(self.region)
            if change=='algorithm':plan['plan_binding']['algorithm_version']='unknown-v9'
            elif change=='parameter':plan['row_step_px']=7
            elif change=='report':plan['coverage']={'sampled_coverage_percent':100}
            else:region['annotation_id']='someone-else'
            with self.subTest(change=change),self.assertRaises(ValueError):validate(plan,self.model.mesh,region)

    def test_project_cannot_rewrite_wrong_region_binding(self):
        path=save_project(self.model,self.region,self.folder/'plan.zip',self.plan)
        with zipfile.ZipFile(path) as z:assets={name:z.read(name) for name in z.namelist()}
        data=json.loads(assets['project.json']);data['region']['annotation_id']='wrong'
        assets['project.json']=json.dumps(data).encode()
        with zipfile.ZipFile(self.folder/'bad.zip','w') as z:
            for name,value in assets.items():z.writestr(name,value)
        with self.assertRaises(ValueError):open_project(self.folder/'bad.zip',self.folder/'bad-work')

    def test_old_json_requires_original_signature_and_recomputation(self):
        from femto.planning.region_path import signature
        folder=export_path(self.plan,self.model.store,self.folder/'exports')
        path=folder/'path.json';payload=json.loads(path.read_text(encoding='utf-8'))
        payload.pop('plan_binding');payload['region_signature']=signature(self.region)
        path.write_text(json.dumps(payload),encoding='utf-8')
        with self.assertWarnsRegex(UserWarning,'旧路径'):
            loaded=load_path(path,self.model.store,self.model.projection,self.region)
        self.assertIn('plan_binding',loaded)
        payload['region_signature']='cannot-verify';path.write_text(json.dumps(payload),encoding='utf-8')
        with self.assertRaises(ValueError):load_path(path,self.model.store,self.model.projection,self.region)

    def test_empty_and_pending_states_are_separate(self):
        self.assertEqual(state(None,None)['phase'],'no_region')
        result=state(self.region,None,pending=True)
        self.assertEqual(result['phase'],'not_generated');self.assertEqual(result['save_state'],'pending')


if __name__=='__main__':unittest.main()
