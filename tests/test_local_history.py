import gc
import json
from pathlib import Path
import tempfile
import time
import tkinter as tk
import unittest
from unittest import mock
from local_history import scan_history,existing_actions,set_record_deleted,set_records_deleted


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data),encoding='utf-8')


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.datasets=self.root/'datasets';self.outputs=self.root/'outputs'
        self.source=self.datasets/'scene_20260915_local_a'
        reference=dict(session_id='a',reference_frame_id=0,coordinate_frame='world_rdf',unit='metre',transform_direction='T_W0_Ct')
        write(self.source/'derived/local_capture_report.json',dict(reference,schema='femto-local-tracking-v1',
              pose_source='online_rgbd_research_grade',accepted_frames=[0],state='stopped'))
        self.result=self.outputs/'complete_demo_a';rec=self.result/'reconstruction'
        write(rec/'local_geometry.json',dict(reference,tracking_report=str(self.source/'derived/local_capture_report.json')))
        write(rec/'reconstruction_report.json',dict(source_dataset=str(self.source),frame_count=1,
              local_capture=reference,outputs=dict(mesh=dict(ply=str(rec/'scene_tsdf_mesh_m.ply')))))
        (rec/'scene_tsdf_mesh_m.ply').write_bytes(b'fixture')
        for key in ('rgb','depth'):
            path=self.source/key/'0';path.parent.mkdir(exist_ok=True);path.write_bytes(b'fixture')
        write(self.source/'pose/0',dict(reference,pose_valid=True,pose_source='online_rgbd_research_grade',
              T_WC=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
        write(self.source/'manifest.json',dict(local_capture=reference))
        (self.source/'frames.jsonl').write_text(json.dumps(dict(frame=0,pose_valid=True,rgb='rgb/0',depth='depth/0',pose='pose/0'))+'\n')
        write(self.source/'calibration/color_intrinsic.json',dict(width=100,height=100,fx=100,fy=100,cx=50,cy=50))
        write(self.source/'calibration/distortion.json',dict(rgb=dict(model='BROWN_CONRADY_K6',
              k1=0,k2=0,k3=0,k4=0,k5=0,k6=0,p1=0,p2=0)))

    def test_completed_and_failed_catalogue_is_read_only(self):
        failed=self.datasets/'scene_20260915_local_b'
        write(failed/'derived/local_failure.json',dict(error='tracking_lost',session_id='b'))
        before={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        rows=scan_history(self.datasets,self.outputs)
        self.assertEqual(len(rows),2)
        complete=next(r for r in rows if r['state']=='model')
        self.assertTrue(existing_actions(complete)['fixed'])
        failure=next(r for r in rows if r['state']=='failed')
        self.assertFalse(existing_actions(failure)['mesh']);self.assertTrue(existing_actions(failure)['report'])
        self.assertEqual(before,{p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_missing_reference_keeps_mesh_and_corrupt_report_does_not_abort_scan(self):
        (self.source/'rgb/0').unlink()
        rows=scan_history(self.datasets,self.outputs)
        self.assertFalse(rows[0]['fixed']);self.assertTrue(existing_actions(rows[0])['mesh'])
        (self.result/'reconstruction/reconstruction_report.json').write_text('broken')
        rows=scan_history(self.datasets,self.outputs)
        self.assertTrue(any(r['state']=='incomplete' for r in rows))
        self.assertEqual(scan_history(self.root/'missing',self.root/'none'),[])

    def test_delete_persists_and_undo_preserves_all_evidence(self):
        before={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        row=scan_history(self.datasets,self.outputs)[0]
        set_record_deleted(row,self.datasets,self.outputs)
        self.assertEqual(scan_history(self.datasets,self.outputs),[])
        self.assertEqual(before,{p:p.read_bytes() for p in before})
        set_record_deleted(row,self.datasets,self.outputs,False)
        self.assertEqual(scan_history(self.datasets,self.outputs)[0]['key'],row['key'])
        with self.assertRaises(ValueError):
            set_record_deleted(dict(row,key=str(self.root)),self.datasets,self.outputs)

    def test_delete_raw_record(self):
        failed=self.datasets/'scene_20260915_local_b'
        write(failed/'derived/local_failure.json',dict(error='tracking_lost',session_id='b'))
        row=next(r for r in scan_history(self.datasets,self.outputs) if r['state']=='failed')
        set_record_deleted(row,self.datasets,self.outputs)
        self.assertEqual(len(scan_history(self.datasets,self.outputs)),1)
        self.assertTrue((failed/'derived/local_failure.json').is_file())

    def test_batch_delete_selected_history_records_and_undo(self):
        failed=self.datasets/'scene_20260915_local_b'
        write(failed/'derived/local_failure.json',dict(error='tracking_lost',session_id='b'))
        from app_local_history import LocalHistoryWindow
        root=tk.Tk();root.withdraw()
        try:
            window=LocalHistoryWindow(root,self.datasets,self.outputs)
            deadline=time.monotonic()+5
            while window.loading and time.monotonic()<deadline:root.update();time.sleep(.01)
            self.assertEqual(len(window.rows),2)
            window.tree.selection_set('0','1')
            window.selected()
            self.assertEqual(len(window.selected_rows()),2)
            with mock.patch('app_local_history.messagebox.askyesno',return_value=True) as ask:
                window.buttons['delete'].invoke()
                self.assertIn('2',ask.call_args.args[1])
            deadline=time.monotonic()+5
            while window.loading and time.monotonic()<deadline:root.update();time.sleep(.01)
            self.assertEqual(window.rows,[])
            window.undo_delete()
            deadline=time.monotonic()+5
            while window.loading and time.monotonic()<deadline:root.update();time.sleep(.01)
            self.assertEqual(len(window.rows),2)
            window.close()
        finally:root.destroy()
        del root,window;gc.collect()

    def test_history_window_opens_correct_model_and_report_without_acquisition(self):
        gc.collect()
        from app_local_history import LocalHistoryWindow
        root=tk.Tk();root.withdraw()
        try:
            window=LocalHistoryWindow(root,self.datasets,self.outputs)
            deadline=time.monotonic()+5
            while window.loading and time.monotonic()<deadline:root.update();time.sleep(.01)
            self.assertFalse(window.loading)
            self.assertEqual(len(window.rows),1)
            self.assertNotIn('compare',window.buttons)
            self.assertNotIn('report',window.buttons)
            self.assertFalse(window.quality_text.winfo_ismapped())
            with mock.patch('app_local_history.subprocess.Popen') as launch:
                window.open('fixed')
                args=launch.call_args.args[0]
                self.assertTrue(args[1].endswith('fixed_rgb_viewer.py'))
                self.assertEqual(Path(args[2]),self.result/'reconstruction')
            with mock.patch('app_local_history.os.startfile') as open_file:
                window.open('report');self.assertTrue(open_file.called)
            with mock.patch('app_local_history.messagebox.askyesno',return_value=False):
                window.open('delete');self.assertEqual(len(scan_history(self.datasets,self.outputs)),1)
            self.assertIn('删除选中记录',window.buttons['delete'].cget('text'))
            with mock.patch('app_local_history.messagebox.askyesno',return_value=True):
                window.buttons['delete'].invoke()
            deadline=time.monotonic()+5
            while window.loading and time.monotonic()<deadline:root.update();time.sleep(.01)
            self.assertEqual(window.rows,[])
            self.assertEqual(str(window.buttons['delete']['state']),'disabled')
            window.undo_delete()
            deadline=time.monotonic()+5
            while window.loading and time.monotonic()<deadline:root.update();time.sleep(.01)
            self.assertEqual(len(window.rows),1)
            window.close()
        finally:root.destroy()
        del root,window;gc.collect()
