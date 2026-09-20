import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from local_processing import LocalProcessing, empty_report, format_quality, result_quality, temporal_summary


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root/'input'
        write(self.dataset/'manifest.json', dict(pose=dict(source='online_rgbd_research_grade'),
            local_capture=dict(measured_only=True, session_id='one', reference_frame_id=7)))

    def test_rejected_stages_report_actual_unchanged_input_and_separate_reasons(self):
        before = (self.dataset/'manifest.json').read_bytes()
        events = []
        refinement = dict(accepted=False, frames=[dict(frame=9,
            reasons=['correction_exceeds_5mm_or_1deg'], before=dict(forward=dict(median_m=.008, p90_m=.020)),
            after=dict(forward=dict(median_m=.004, p90_m=.018)))])
        temporal = dict(applied_frames=0, frames=[
            dict(frame=7, applied=False, reason='no_stationary_five_frame_window'),
            dict(frame=9, applied=False, source_frames=[5,6,7,8,9], reason='image_motion_rejected')])
        with mock.patch('local_pose_refinement.refine', return_value=(self.dataset, refinement)), \
             mock.patch('local_temporal_support.prepare', return_value=(self.dataset, temporal)):
            job = LocalProcessing(self.dataset, self.root/'job', events.append)
            chosen, report = job.run()
        self.assertEqual(chosen, self.dataset)
        self.assertEqual(report['actual_pose_source'], 'online_rgbd_research_grade')
        self.assertEqual(report['refinement']['status'], 'not_adopted')
        self.assertEqual(report['temporal']['windows_found'], 1)
        self.assertEqual(report['temporal']['applied_views'], 0)
        self.assertEqual(before, (self.dataset/'manifest.json').read_bytes())
        self.assertTrue(any(e['refinement']['status']=='running' for e in events))
        self.assertIn('8.0/20.0 → 4.0/18.0', format_quality(report))
        self.assertIn('not physical measurement error', format_quality(report, 'en'))
        self.assertFalse(list(self.root.rglob('latest*.txt')))

    def test_failed_stage_and_inconsistent_adoption_never_claim_success(self):
        job = LocalProcessing(self.dataset, self.root/'job')
        with mock.patch('local_pose_refinement.refine', return_value=(self.dataset, dict(accepted=True))):
            with self.assertRaisesRegex(ValueError, 'decision'):
                job.run()
        recorded = json.loads(job.path.read_text(encoding='utf-8'))
        self.assertEqual(recorded['state'], 'failed')
        self.assertEqual(recorded['refinement']['status'], 'failed')
        self.assertEqual(recorded['temporal']['status'], 'not_run')

    def test_removed_statistics_only_count_masks_actually_used(self):
        report = dict(frames=[
            dict(applied=True, valid_pixels=100, removed_pixels=10, source_frames=[1,2,3,4,5]),
            dict(applied=False, valid_pixels=100, removed_pixels=90, source_frames=[1,2,3,4,5])])
        actual = temporal_summary(report)
        self.assertEqual(actual['removed_pixels'], 10)
        self.assertEqual(actual['removed_ratio'], .1)
        self.assertEqual(actual['windows_found'], 2)
        self.assertEqual(actual['applied_views'], 1)
        old = temporal_summary(dict(frames=[dict(applied=True, removed_pixels=10)]))
        self.assertIsNone(old['removed_ratio'])

    def test_legacy_missing_reports_are_unknown_and_cross_session_binding_rejected(self):
        result = self.root/'model'
        write(result/'reconstruction/reconstruction_report.json', dict(source_dataset=str(self.dataset)))
        write(result/'reconstruction/local_geometry.json', dict(session_id='one', reference_frame_id=7))
        report = result_quality(result)
        self.assertEqual(report['refinement']['status'], 'not_recorded')
        self.assertIn('未记录', format_quality(report))
        self.assertNotIn('0.0%', format_quality(report))
        value = dict(empty_report(), actual_dataset=str(self.dataset),
            source_session_id='another', reference_frame_id=7, refinement=dict(status='adopted'))
        write(result/'processing_report.json', value)
        rejected = result_quality(result)
        self.assertIn('binding mismatch', rejected['read_error'])
        self.assertEqual(rejected['refinement']['status'], 'not_recorded')

    def test_quality_events_cannot_update_another_or_finished_session(self):
        import gc
        import tkinter as tk
        import femto_control_app as control
        gc.collect()
        with mock.patch.object(control.FemtoControlApp, '_start_camera'), \
             mock.patch.object(control.FemtoControlApp, '_load_latest_paths'):
            root=tk.Tk()
            root.withdraw()
            try:
                app=control.FemtoControlApp(root)
                app.camera_ready=True
                panel=app.local_panel
                panel.start()
                report=dict(empty_report(), refinement=dict(status='running'))
                payload=dict(session_id='old', quality=report)
                panel.event('local_processing', payload)
                self.assertIsNone(panel.quality)
                payload['session_id']=panel.session_id
                panel.event('local_processing', payload)
                self.assertEqual(panel.quality, report)
                panel.event('local_cancelled', dict(session_id=panel.session_id))
                payload['quality']=dict(empty_report(), refinement=dict(status='adopted'))
                panel.event('local_processing', payload)
                self.assertEqual(panel.quality, report)
            finally:
                for timer in root.tk.call('after','info'):
                    root.after_cancel(timer)
                root.destroy()
        del app,panel,root
        gc.collect()

