import argparse
import copy
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest import mock

from local_capture import LocalCapture
from local_offline import LocalOfflineJob, PublicationError, publish_result
from local_processing import SCHEMA, read_json


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def result_fixture(root, session='one'):
    outputs, dataset = root / 'outputs', root / 'source'
    result = outputs / 'complete_demo_test'
    write(dataset / 'manifest.json', dict(pose=dict(source='online_rgbd_research_grade'),
        local_capture=dict(session_id=session, reference_frame_id=0, measured_only=True)))
    (dataset / 'frames.jsonl').write_text(json.dumps(dict(frame=0)) + '\n', encoding='utf-8')
    write(result / 'demo_summary.json', dict(publication_required=True, publication_id=session,
        data_kind='real_femto', fixed_camera=False, frame_count=3))
    write(result / 'reconstruction/reconstruction_report.json', dict(source_dataset=str(dataset)))
    write(result / 'reconstruction/local_geometry.json', dict(session_id=session, reference_frame_id=0))
    write(result / 'publication.json', dict(schema='femto-local-publication-v1', status='pending',
        session_id=session, result=str(result)))
    quality = dict(schema=SCHEMA, state='complete', source_session_id=session, reference_frame_id=0,
        actual_dataset=str(dataset), actual_pose_source='online_rgbd_research_grade',
        reconstruction=dict(status='complete'), refinement=dict(status='not_adopted'),
        temporal=dict(status='not_adopted', applied_views=0, frames=[dict(frame=0, applied=False)]))
    write(result / 'processing_report.json', quality)
    for name in ('demo_report.html', 'processing_report.html', 'result_quality.html', 'nerfstudio/transforms.json'):
        path = result / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}', encoding='utf-8')
    return dataset, outputs, result, quality


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_pending_results_are_not_restored_until_publication_commits(self):
        from reconstruction_sources import mode_of, restore
        _, outputs, result, _ = result_fixture(self.root)
        self.assertIsNone(mode_of(result))
        self.assertIsNone(restore(outputs)['moving'])
        publication = publish_result(result, outputs, 'one')
        self.assertEqual(publication['status'], 'published')
        self.assertEqual(mode_of(result), 'moving')
        self.assertEqual(restore(outputs)['moving'], result)
        self.assertEqual((outputs / 'latest_complete_demo.txt').read_text(), str(result / 'reconstruction'))

    def test_damaged_or_cancelled_receipts_cannot_break_or_bypass_restore(self):
        from reconstruction_sources import mode_of,restore
        _,outputs,result,_=result_fixture(self.root)
        publication=publish_result(result,outputs,'one')
        for bad in ([],dict(publication,schema='unsupported-v999'),dict(publication,status='cancelled')):
            write(result/'publication.json',bad)
            self.assertIsNone(mode_of(result))
            self.assertIsNone(restore(outputs)['moving'])
        summary=read_json(result/'demo_summary.json')
        summary['publication_required']=False
        write(result/'demo_summary.json',summary)
        self.assertIsNone(mode_of(result))
        write(result/'demo_summary.json',[])
        self.assertIsNone(restore(outputs)['moving'])

    def test_publication_rejects_cross_session_and_missing_final_report(self):
        _, outputs, result, _ = result_fixture(self.root)
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            publish_result(result, outputs, 'another')
        (result / 'result_quality.html').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            publish_result(result, outputs, 'one')
        self.assertFalse(list(outputs.glob('latest*.txt')))

    def test_partial_publication_failure_restores_existing_indexes(self):
        import local_offline
        _, outputs, result, _ = result_fixture(self.root)
        before = {}
        for name in ('latest_complete_demo.txt', 'latest_moving_reconstruction.txt'):
            path = outputs / name
            path.write_text('old-' + name, encoding='utf-8')
            before[path] = path.read_bytes()
        replace = local_offline.os.replace
        def fail_second(source, destination):
            if Path(destination).name == 'latest_moving_reconstruction.txt':
                raise OSError('simulated marker disk failure')
            return replace(source, destination)
        with mock.patch.object(local_offline.os, 'replace', side_effect=fail_second):
            with self.assertRaises(PublicationError) as raised:
                publish_result(result, outputs, 'one')
        self.assertFalse(raised.exception.details['successful_model_published'])
        self.assertEqual([str(outputs / 'latest_complete_demo.txt')], raised.exception.details['rolled_back_markers'])
        for path, original in before.items():
            self.assertEqual(path.read_bytes(), original)
        self.assertEqual(read_json(result / 'publication.json')['status'], 'failed')

    def test_cancel_during_publication_rolls_back_before_completion_receipt(self):
        import local_offline
        _,outputs,result,_=result_fixture(self.root)
        marker=outputs/'latest_complete_demo.txt'
        marker.write_text('previous',encoding='utf-8')
        cancelled=threading.Event()
        replace=local_offline.os.replace
        def signal_after_first(source,destination):
            value=replace(source,destination)
            if Path(destination)==marker:cancelled.set()
            return value
        with mock.patch.object(local_offline.os,'replace',side_effect=signal_after_first):
            with self.assertRaises(PublicationError) as raised:
                publish_result(result,outputs,'one',cancelled=cancelled.is_set)
        self.assertEqual(raised.exception.status,'cancelled')
        self.assertEqual(marker.read_text(),'previous')
        self.assertFalse((outputs/'latest_moving_reconstruction.txt').exists())
        self.assertEqual(read_json(result/'publication.json')['status'],'cancelled')

    def test_application_shutdown_cancels_even_before_sdk_owner_returns(self):
        from test_local_capture import make_camera
        from femto_control_app import FemtoControlApp
        from types import SimpleNamespace
        stop=threading.Event()
        session_cancel=threading.Event()
        with mock.patch.object(LocalCapture,'_run'):
            capture=LocalCapture(make_camera('wide_detail',10),self.root/'data',self.root/'outputs',queue.Queue(),
                                 shutdown_event=stop,cancel_event=session_cancel)
            capture.thread.join(2)
        app=SimpleNamespace(capturing=False,processing=True,stop_event=stop,command_queue=queue.Queue(),
            temporal_collecting=False,point_cloud_preview=None,root=mock.Mock(),_t=lambda key:key)
        with mock.patch('femto_control_app.messagebox.askyesno',return_value=True):
            FemtoControlApp.close(app)
        self.assertFalse(session_cancel.is_set())  # SDK owner has not handled its finally/queue yet.
        self.assertTrue(capture._cancelled())
        self.assertTrue(app.root.destroy.called)
        _,outputs,result,_=result_fixture(self.root/'model')
        from local_comparison import StageStopped
        with self.assertRaises(StageStopped):
            publish_result(result,outputs,'one',cancelled=capture._cancelled)
        self.assertFalse(list(outputs.glob('latest*.txt')))

    def test_cancel_after_capture_finish_reaches_offline_job_but_not_after_commit(self):
        capture = LocalCapture.__new__(LocalCapture)
        capture.lock = threading.Lock()
        capture.closing = threading.Event()
        capture.processing_cancel = threading.Event()
        capture.successful_model_published = False
        capture.emit = mock.Mock()
        capture.finish(True)
        self.assertTrue(capture.reconstruct)
        capture.finish(False)
        self.assertTrue(capture.processing_cancel.is_set())
        capture.processing_cancel.clear()
        capture.successful_model_published = True
        capture.finish(False)
        self.assertFalse(capture.processing_cancel.is_set())

    def test_late_completed_worker_after_cancel_never_publishes(self):
        from local_comparison import StageStopped
        dataset, outputs, result, quality = result_fixture(self.root)
        cancel = threading.Event()
        job = LocalOfflineJob(dataset, self.root / 'job', outputs, 'one', cancel.is_set)
        def completed(*args, **kwargs):
            write(job.path, quality)
            write(job.output / 'created_result.json', dict(result=str(result), session_id='one'))
            write(job.output / 'worker_result.json', dict(status='complete', result=str(result), session_id='one'))
            cancel.set()
            return dict(elapsed_s=.1, returncode=0)
        with mock.patch('local_comparison.supervise', side_effect=completed):
            with self.assertRaises(StageStopped) as raised:
                job.run()
        self.assertEqual(raised.exception.status, 'cancelled')
        self.assertFalse(list(outputs.glob('latest*.txt')))
        self.assertEqual(read_json(result / 'publication.json')['status'], 'cancelled')

    def test_child_sidecar_cannot_edit_a_different_existing_result(self):
        dataset, outputs, foreign, quality = result_fixture(self.root, session='foreign')
        publish_result(foreign, outputs, 'foreign')
        original={p:p.read_bytes() for p in foreign.rglob('*') if p.is_file()}
        job=LocalOfflineJob(dataset,self.root/'job',outputs,'current',lambda:False)
        def wrong_result(*args,**kwargs):
            write(job.path,dict(quality,source_session_id='current'))
            write(job.output/'created_result.json',dict(result=str(foreign),session_id='current'))
            write(job.output/'worker_result.json',dict(status='complete',result=str(foreign),session_id='current'))
            return dict(elapsed_s=.1,returncode=0)
        with mock.patch('local_comparison.supervise',side_effect=wrong_result):
            with self.assertRaisesRegex(ValueError,'ownership'):
                job.run()
        self.assertIsNone(job.result)
        for path,data in original.items():self.assertEqual(path.read_bytes(),data)
        self.assertEqual((outputs/'latest_moving_reconstruction.txt').read_text(),str(foreign))

    def test_reporting_disk_failure_cannot_hide_capture_terminal_event(self):
        from test_local_capture import make_camera
        with mock.patch.object(LocalCapture, '_run'):
            capture = LocalCapture(make_camera('wide_detail', 10), self.root / 'data', self.root / 'out', queue.Queue())
            capture.thread.join(2)
        capture.closing.set()
        capture.reconstruct = True
        capture.finished_at = time.monotonic()
        capture.records = [dict(state='tracking', received_monotonic=capture.finished_at)]
        capture.accepted = [dict(frame_id=i) for i in range(3)]
        tracker = mock.Mock(lost=False, reference_id=0, max_gap_s=2., loss_context=None, last_rejection=None)
        offline = mock.Mock(path=self.root / 'missing/report.json', report=dict(state='failed'))
        offline.run.side_effect = OSError('original processing failure')
        offline.terminal.side_effect = OSError('journal disk failure')
        with mock.patch('local_capture.LiveTracker', return_value=tracker), \
             mock.patch('local_capture.PairEstimator'), \
             mock.patch('local_live_preview.LivePreviewWorker'), \
             mock.patch('femto_control_app.finalize_scan'), \
             mock.patch.object(capture, '_write_tracking_report'), \
             mock.patch.object(capture, '_export_poses', return_value=self.root / 'posed'), \
             mock.patch('local_offline.LocalOfflineJob', return_value=offline):
            capture._run()
        events = []
        while not capture.events.empty():
            events.append(capture.events.get_nowait())
        terminal = [payload for kind, payload in events if kind == 'local_error']
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]['error'], 'original processing failure')
        self.assertIn('journal disk failure', terminal[0]['failure']['processing_reporting_errors'])
        self.assertFalse(terminal[0]['failure']['successful_model_published'])

    def test_processing_exception_keeps_original_error_when_journal_fails(self):
        from local_processing import LocalProcessing
        dataset, _, _, _ = result_fixture(self.root)
        job = LocalProcessing(dataset, self.root / 'processing')
        with mock.patch.object(job, 'publish', side_effect=OSError('journal failure')):
            with self.assertRaisesRegex(OSError, 'journal failure'):
                job.run()
        self.assertEqual(job.report['state'], 'failed')
        self.assertEqual(job.report['reporting_error'], 'journal failure')

    def test_real_offline_subprocess_finishes_without_touching_sources_or_latest(self):
        from test_local_comparison import make_fixture
        from local_comparison import changed_files, fingerprints
        dataset = make_fixture(self.root / 'fixture')
        manifest = read_json(dataset / 'manifest.json')
        before = fingerprints(p for p in (self.root / 'fixture').rglob('*') if p.is_file())
        outputs = self.root / 'results'
        outputs.mkdir()
        latest = outputs / 'latest_complete_demo.txt'
        latest.write_text('previous-model', encoding='utf-8')
        events = []
        job = LocalOfflineJob(dataset, self.root / 'offline', outputs,
            manifest['local_capture']['session_id'], lambda: False, events.append,
            timeout_s=90, memory_mb=2048)
        result, summary = job.run()
        self.assertEqual(changed_files(before), [])
        self.assertEqual(latest.read_text(), 'previous-model')
        self.assertTrue(read_json(job.output / 'source_integrity.json')['source_unchanged'])
        self.assertEqual(read_json(result / 'publication.json')['status'], 'pending')
        self.assertEqual(summary['data_kind'], 'synthetic')
        self.assertEqual(job.report['reconstruction']['status'], 'complete')
        self.assertTrue((result / 'reconstruction/initial_observed_points_m.ply').is_file())
        self.assertTrue(events)
        publication = publish_result(result, outputs, manifest['local_capture']['session_id'])
        self.assertEqual(publication['status'], 'published')
        self.assertEqual(latest.read_text(), str(result / 'reconstruction'))

    def test_live_processing_cancel_control_and_late_events_are_session_bound(self):
        import gc
        import tkinter as tk
        import femto_control_app as control
        gc.collect()
        with mock.patch.object(control.FemtoControlApp, '_start_camera'), \
             mock.patch.object(control.FemtoControlApp, '_load_latest_paths'):
            root = tk.Tk()
            root.withdraw()
            try:
                app = control.FemtoControlApp(root)
                app.camera_ready = True
                panel = app.local_panel
                panel.start()
                start_command=next(payload for command,payload in list(app.command_queue.queue) if command=='local_start')
                self.assertIs(start_command['cancel_event'],panel.cancel_event)
                panel.count = 3
                panel.finish(True)
                panel.event('local_processing', dict(session_id=panel.session_id))
                self.assertEqual(panel.buttons['cancel'].cget('state'), 'normal')
                old_session = panel.session_id
                panel.finish(False)
                self.assertTrue(panel.cancelling)
                self.assertTrue(start_command['cancel_event'].is_set())
                self.assertEqual(panel.buttons['cancel'].cget('state'), 'disabled')
                panel.event('local_cancelled', dict(session_id=old_session))
                panel.start()
                self.assertIsNot(start_command['cancel_event'],panel.cancel_event)
                self.assertFalse(panel.cancel_event.is_set())
                panel.event('local_finished', dict(session_id=old_session, result='stale'))
                self.assertTrue(panel.active)
                self.assertIsNone(panel.result)
            finally:
                for timer in root.tk.call('after', 'info'):
                    root.after_cancel(timer)
                root.destroy()
        del app, panel, root
        gc.collect()


if __name__ == '__main__':
    unittest.main()
