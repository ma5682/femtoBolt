import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest import mock
import zipfile

from planning_drafts import Journal, snapshot, read
from planning_io import TaskControl
from region_project import save_project, open_project, digest
from region_path import export_path
from tests import test_planning_state as fixtures


class RecoveryTests(unittest.TestCase):
    setUp=fixtures.PlanningStateTests.setUp
    tearDown=fixtures.PlanningStateTests.tearDown
    pump=fixtures.PlanningStateTests.pump

    def journal(self):
        self.viewer.draft_journal=Journal(self.folder/'source.zip',self.folder/'drafts')
        return self.viewer.draft_journal

    def save(self,path=None,**kwargs):
        return save_project(self.model,self.model.store.regions[0],path or self.folder/'plan.zip',self.plan,**kwargs)

    def test_replace_failure_preserves_formal_bytes_and_dirty_state(self):
        path=self.save();before=path.read_bytes()
        self.model.store.update_annotation('region',0,name='Unsaved',note='')
        with mock.patch('planning_io.publish',side_effect=OSError('simulated disk full')):
            with self.assertRaisesRegex(OSError,'disk full'):self.save(path)
        self.assertEqual(path.read_bytes(),before);self.assertTrue(self.model.store.dirty)
        self.assertEqual(open_project(path,self.folder/'work').saved_plan['segments'],self.plan['segments'])

    def test_cancel_save_before_commit_preserves_old(self):
        path=self.save();before=path.read_bytes();control=TaskControl()
        def stage(value):
            if value=='写入方案清单与压缩包':control.set()
        with self.assertRaises(RuntimeError):self.save(path,check=control.check,commit=control.publish,progress=stage)
        self.assertEqual(before,path.read_bytes());self.assertFalse(control.committed)

    def test_cancel_after_commit_reports_completed_not_cancelled(self):
        control=TaskControl();path=self.save(check=control.check,commit=control.publish)
        control.set();self.assertTrue(control.committed);self.assertFalse(control.is_set())
        self.assertTrue(path.exists())

    def test_other_region_and_external_file_change_cannot_overwrite(self):
        path=self.save();before=path.read_bytes()
        other=dict(self.region,annotation_id='other')
        with self.assertRaisesRegex(ValueError,'另一个区域'):save_project(self.model,other,path)
        self.assertEqual(path.read_bytes(),before)
        with self.assertRaisesRegex(ValueError,'其他窗口'):self.save(path,expected_digest='stale-digest')
        self.assertEqual(path.read_bytes(),before)

    def test_export_cancel_has_no_published_or_half_result(self):
        control=TaskControl();parent=self.folder/'exports'
        def progress(stage):
            if stage=='压缩完整分享包':control.set()
        with self.assertRaises(RuntimeError):
            export_path(self.plan,self.model.store,parent,check=control.check,progress=progress,commit=control.publish)
        self.assertFalse(parent.exists());self.assertFalse(list(self.folder.glob('.pending_export_*')))
        self.assertIs(self.viewer.current_region_path(),self.plan)

    def test_history_failure_is_warning_after_successful_save(self):
        with mock.patch('export_history.register_project',side_effect=OSError('index denied')):path=self.save()
        self.assertTrue(path.exists());self.assertIn('已保存',self.model.last_save_warning)
        self.assertIn('index denied',self.model.last_save_warning)

    def test_missing_corrupt_assets_and_reference_fail_without_working_copy(self):
        source=self.save()
        with zipfile.ZipFile(source) as z:assets={name:z.read(name) for name in z.namelist()}
        for damage in ('missing','corrupt','reference'):
            values=dict(assets)
            if damage=='missing':values.pop('depth.npy')
            elif damage=='corrupt':values['rgb.png']=b'broken'
            else:
                data=json.loads(values['project.json']);data['view_reference']['session_id']='wrong'
                values['project.json']=json.dumps(data).encode()
            bad=self.folder/(damage+'.zip');workspace=self.folder/(damage+'-work')
            with zipfile.ZipFile(bad,'w') as z:
                for name,value in values.items():z.writestr(name,value)
            with self.subTest(damage=damage),self.assertRaises(ValueError):open_project(bad,workspace)
            self.assertEqual(list(workspace.iterdir()),[])
        self.assertIs(self.viewer.current_region_path(),self.plan)

    def test_cancel_load_cleans_only_its_workspace(self):
        path=self.save();workspace=self.folder/'work';workspace.mkdir();keep=workspace/'keep.txt';keep.write_text('keep')
        control=TaskControl()
        def progress(stage):
            if stage=='解包并核对资产指纹':control.set()
        with self.assertRaises(RuntimeError):open_project(path,workspace,control,progress)
        self.assertEqual(list(workspace.iterdir()),[keep])

    def test_draft_restore_regions_exclusions_params_not_paths(self):
        journal=self.journal();v=self.viewer;self.model.store.mark_saved()
        self.model.store.set_exclusions(0,[[[30,30],[40,30],[40,40],[30,40]]])
        v._requests[self.region['annotation_id']]=dict(kind='raster',values=dict(row_step_px=20,max_step_mm=10))
        v._selection_polygons={'work':[[5,5],[80,5],[80,80],[5,80]],
                               'forbidden':[[30,30],[40,30],[40,40],[30,40]]}
        v.selection_method.set('forbidden');v.active_selection_method='forbidden'
        v.polygon=copy.deepcopy(v._selection_polygons['forbidden'])
        data=snapshot(v);journal.submit(data);journal.wait()
        self.model.store.undo()
        v._requests={};v.reset_brush();v.polygon=[]
        with mock.patch('fixed_rgb_viewer.messagebox.askyesnocancel',return_value=True):v.offer_recovery()
        self.pump()
        self.assertEqual(self.model.store.regions[0]['exclusion_polygons'],data['regions'][0]['exclusion_polygons'])
        self.assertEqual(v._requests,data['parameters']);self.assertIsNone(v.current_region_path())
        self.assertEqual(v.selection_polygons(),data['selection_polygons'])
        self.assertEqual(v.selection_method.get(),'forbidden')
        self.assertTrue(self.model.store.dirty);self.assertTrue(journal.path.exists())

    def test_cancel_recovery_keeps_live_annotations_and_draft(self):
        journal=self.journal();v=self.viewer;data=snapshot(v)
        data['regions'][0]['name']='recover-me';journal.submit(data);journal.wait()
        before=copy.deepcopy(self.model.store.regions);gate=threading.Event();entered=threading.Event()
        def delayed(candidate,**kwargs):entered.set();gate.wait(2)
        with mock.patch.object(type(self.model),'prepare_display',delayed),mock.patch('fixed_rgb_viewer.messagebox.askyesnocancel',return_value=True):
            v.offer_recovery();self.assertTrue(entered.wait(1));v.cancel_planning();gate.set();self.pump()
        self.assertEqual(self.model.store.regions,before);self.assertTrue(journal.path.exists())
        self.assertIs(v.current_region_path(),self.plan)

    def test_draft_write_failure_keeps_last_recoverable_snapshot(self):
        journal=self.journal();data=snapshot(self.viewer);journal.submit(data);journal.wait();before=journal.path.read_bytes()
        with mock.patch('planning_drafts.json.dump',side_effect=OSError('disk full')):
            journal.submit(dict(data,name='unsaved'));journal.wait()
        self.assertEqual(journal.path.read_bytes(),before);self.assertIn('disk full',journal.error)
        self.assertEqual(read(journal.path,self.model)['name'],data['name'])

    def test_abrupt_process_exit_preserves_previous_draft(self):
        journal=self.journal();data=snapshot(self.viewer);journal.submit(data);journal.wait();before=journal.path.read_bytes()
        payload=self.folder/'payload.json';payload.write_text(json.dumps(dict(data,name='interrupted')),encoding='utf-8')
        program="import sys,json,os; from planning_drafts import Journal; from pathlib import Path; j=Journal(sys.argv[1],sys.argv[3]); os.fsync=lambda fd:os._exit(17); j.submit(json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))); j.wait()"
        result=subprocess.run([sys.executable,'-c',program,str(self.folder/'source.zip'),str(payload),str(self.folder/'drafts')],cwd=Path(__file__).resolve().parents[1],timeout=10)
        self.assertEqual(result.returncode,17);self.assertEqual(journal.path.read_bytes(),before)
        self.assertEqual(read(journal.path,self.model)['regions'],data['regions'])

    def test_discard_draft_prevents_late_writer_resurrection(self):
        journal=self.journal();gate=threading.Event();entered=threading.Event()
        original=os.fsync
        def wait(fd):entered.set();gate.wait(2);original(fd)
        with mock.patch('planning_drafts.os.fsync',side_effect=wait):
            journal.submit(snapshot(self.viewer));self.assertTrue(entered.wait(1));journal.discard();gate.set();journal.wait()
        self.assertFalse(journal.path.exists())

    def test_recovery_decline_or_mismatched_source_keeps_current_plan(self):
        journal=self.journal();data=snapshot(self.viewer);journal.submit(data);journal.wait()
        with mock.patch('fixed_rgb_viewer.messagebox.askyesnocancel',return_value=False):self.viewer.offer_recovery()
        self.assertFalse(journal.path.exists());self.assertIs(self.viewer.current_region_path(),self.plan)
        data['source']['geometry_sha256']='wrong';journal.submit(data);journal.wait()
        with mock.patch('fixed_rgb_viewer.messagebox.askyesnocancel') as ask:self.viewer.offer_recovery();ask.assert_not_called()
        self.assertTrue(journal.path.exists());self.assertIs(self.viewer.current_region_path(),self.plan)

    def test_autosave_and_duplicate_task_guard(self):
        journal=self.journal();v=self.viewer;v._draft_due=0;v.poll_draft();journal.wait()
        self.assertTrue(journal.path.exists())
        gate=threading.Event();v.operation='export';called=[]
        v.work(lambda:gate.wait(1),lambda _:None)
        v.work(lambda:called.append(True),lambda _:None)
        v.cancel_planning();gate.set();self.pump()
        self.assertEqual(called,[]);self.assertIn('已取消',v.task_feedback.get())

    def test_capture_host_does_not_shutdown_when_workspace_declines(self):
        from femto_control_app import FemtoControlApp
        app=FemtoControlApp.__new__(FemtoControlApp)
        app.capturing=False;app.processing=False;app.root=self.root
        app.stop_event=threading.Event();app.command_queue=mock.Mock()
        with mock.patch('planning_workspace.close_workspaces',return_value=False):app.close()
        self.assertFalse(app.stop_event.is_set());app.command_queue.put.assert_not_called()
        self.assertTrue(self.root.winfo_exists())

    def test_host_close_respects_workspace_unsaved_cancel(self):
        from planning_workspace import close_workspaces
        import tkinter as tk
        window=tk.Toplevel(self.root);viewer=mock.Mock();window.viewer=viewer
        self.root._planning_windows={'test':window}
        self.assertFalse(close_workspaces(self.root));viewer.close.assert_called_once_with(force=False)
        self.assertTrue(window.winfo_exists())
        viewer.close.side_effect=lambda **kwargs:window.destroy()
        self.assertTrue(close_workspaces(self.root))

    def test_preparing_workspace_close_cancels_and_ignores_late_result(self):
        from planning_workspace import open_workspace
        gate=threading.Event();started=threading.Event();controls=[]
        def prepare(*args,**kwargs):
            controls.append(kwargs['cancel']);started.set();gate.wait(2);kwargs['cancel'].check()
        with mock.patch('region_project.project_from_annotations',side_effect=prepare),mock.patch('fixed_rgb_viewer.FixedRGBViewer') as viewer:
            window=open_workspace(self.root,self.folder/'annotations.json',region_index=0)
            self.assertTrue(started.wait(1));window.destroy();gate.set();self.root.update()
            self.assertTrue(controls[0].is_set());viewer.assert_not_called()

    def test_save_as_moves_portable_draft_entry(self):
        v=self.viewer;self.model.project_id='portable';v.project_path=self.folder/'original.zip'
        target=self.save(self.folder/'copy.zip')
        v.project_saved(target,self.model.store.regions[0],self.plan)
        self.assertEqual(v.draft_journal.path,Journal(target).path)

    def test_save_one_region_does_not_clear_other_dirty_contents(self):
        v=self.viewer
        self.model.project_id='test';self.model.region([[15,15],[85,15],[85,85],[15,85]],'Other','')
        v.project_saved(self.save(),self.region,self.plan)
        self.assertTrue(self.model.store.dirty)
        self.assertIn('未保存',v.save_state.get())


if __name__=='__main__':unittest.main()
