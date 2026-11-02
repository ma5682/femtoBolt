"""Committed-prefix stage snapshots, isolation, cancellation and continuity gates."""
import hashlib
import json
from pathlib import Path
import queue
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

import numpy as np

from femto.capture.local_capture import LocalCapture
from femto.capture.local_checkpoint import freeze_stage
from femto.capture.scan_session import save_scan_frame
from femto.dataset.local_reference import validate_reference_provenance
from femto.pipeline.local_offline import _source_paths
from test_segmented_capture import make_camera, make_frame


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CheckpointTests(unittest.TestCase):
    def test_no_gain_or_round_budget_stops_keyframe_requests_without_resetting_reference(self):
        manager=self.capture.checkpoints
        accepted=list(self.capture.accepted)
        manager.last_feedback=({'stop_reason':'repeated_no_observed_reference_support_gain'},None)
        self.assertFalse(self.capture.request())
        self.assertFalse(self.capture.check_current())
        self.assertIsNone(self.capture.requested)
        self.assertFalse(self.capture.closing.is_set())
        self.assertEqual(self.capture.accepted,accepted)
        self.assertEqual(manager.stop_reason,'checkpoint_no_gain_limit')
        manager.last_feedback=None;manager.rounds=8
        self.assertEqual(manager.stop_reason,'checkpoint_round_limit')
        self.assertFalse(self.capture.request())
        self.assertFalse(self.capture.closing.is_set())

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        with mock.patch.object(LocalCapture, '_run'):
            self.capture = LocalCapture(make_camera('wide_detail', 10), root/'raw',
                                        root/'results', queue.Queue())
            self.capture.thread.join()
        self.tracker = SimpleNamespace(lost=False, relocalizing=False,
            last_valid_received=time.monotonic(), max_gap_s=2., reference_id=0, loss_context=None)
        for index in range(3):
            self.add_frame(index)
        self.capture.scan_roi = dict(status='bound', session_id=self.capture.session_id,
            reference_frame_id=0, image_size=[10,10], rect_norm=[0.,0.,1.,1.],
            calibration_sha256=self.capture.calibration['sha256'], raw_rgbd_cropped=False)

    def add_frame(self, index):
        save_scan_frame(self.capture.scan, make_frame([500]*10, index+1), lossless=True)
        raw = json.loads(self.capture.scan.frames_log.read_text().splitlines()[-1])
        record = dict(frame_id=index, state='tracking', T_WC=np.eye(4).tolist(),
            reference_frame_id=0, capture_timestamp_us=raw['camera_timestamp_us'],
            color_timestamp_us=raw['color_timestamp_us'], depth_timestamp_us=raw['depth_timestamp_us'],
            latency_ms=1., selected_for_fusion=True, session_id=self.capture.session_id)
        self.capture.accepted.append(record)
        self.capture.records.append(record)

    def freeze(self):
        self.tracker.last_valid_received = time.monotonic()
        return freeze_stage(self.capture, self.tracker, uuid.uuid4().hex)

    def test_frozen_metadata_stays_fixed_as_live_source_grows(self):
        original = {p:sha(p) for p in self.capture.session.rglob('*') if p.is_file()}
        parent, posed, request = self.freeze()
        self.assertFalse(self.capture.closing.is_set())
        self.assertFalse((self.capture.session/'manifest.json').exists())
        self.assertFalse((self.capture.session.parent/'latest_scene.txt').exists())
        self.assertFalse((self.capture.outputs/'latest_complete_demo.txt').exists())
        self.assertTrue(all(sha(p)==digest for p,digest in original.items()))
        frozen = {p:sha(p) for p in _source_paths(posed)}
        self.add_frame(3)
        self.assertTrue(all(sha(p)==digest for p,digest in frozen.items()))
        self.assertEqual(request['accepted_frame_ids'], [0,1,2])
        rows = [json.loads(line) for line in (posed/'frames.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),3)
        row=rows[0]; pose=json.loads((posed/row['pose']).read_text())
        proof=validate_reference_provenance(request['reference_provenance'], posed, row, pose)
        self.assertEqual(proof['session_id'],self.capture.session_id)
        self.assertEqual((posed/row['rgb']).resolve(), self.capture.session/'rgb/000000.png')
        tracking=json.loads((Path(request['frozen_source'])/'derived/local_capture_report.json').read_text())
        self.assertEqual(tracking['reference_provenance'],proof)
        second, _, next_request=self.freeze()
        self.assertNotEqual(parent,second)
        self.assertEqual(next_request['accepted_frame_ids'],[0,1,2,3])
        self.assertTrue(all(sha(p)==digest for p,digest in frozen.items()))

    def test_untrusted_pose_unbound_target_and_changed_calibration_reject(self):
        self.tracker.lost=True
        with self.assertRaisesRegex(ValueError,'pose_untrusted'):self.freeze()
        self.tracker.lost=False
        self.capture.scan_roi['status']='pending_reference'
        with self.assertRaisesRegex(ValueError,'target_unbound'):self.freeze()
        self.capture.scan_roi['status']='bound'
        calibration=self.capture.session/'calibration/color_intrinsic.json'
        calibration.write_text('{}')
        with self.assertRaisesRegex(ValueError,'calibration_changed'):self.freeze()

    def test_failure_or_cancellation_does_not_install_or_publish_result(self):
        manager=self.capture.checkpoints
        parent,posed,request=self.freeze()
        previous=dict(stage_id='previous',result='previous-result')
        manager.last_result=previous
        with mock.patch('femto.pipeline.local_offline.LocalOfflineJob') as job:
            job.return_value.run.side_effect=RuntimeError('reconstruction failed')
            manager._run(parent,posed,request,threading.Event())
            self.assertIs(manager.last_result,previous)
            job.return_value.run.side_effect=None
            job.return_value.run.return_value=(parent/'unused-result',{})
            cancel=threading.Event();cancel.set()
            manager._run(parent,posed,request,cancel)
            self.assertIs(manager.last_result,previous)
        self.assertFalse((self.capture.outputs/'latest_complete_demo.txt').exists())

    def test_worker_receives_fixed_prefix_while_session_remains_open(self):
        manager=self.capture.checkpoints
        model=SimpleNamespace(view_reference={'session_id':self.capture.session_id},
            store=SimpleNamespace(source=self.capture.session/'calibration/color_intrinsic.json'))
        with mock.patch('femto.pipeline.local_offline.LocalOfflineJob') as job, \
                mock.patch('femto.annotation.fixed_rgb_geometry.FixedRGBModel',return_value=model), \
                mock.patch('femto.capture.scan_roi.assess_model',return_value=(
                    {'status':'checked'},np.zeros((10,10,3),np.uint8),np.ones((10,10),np.uint8))):
            job.return_value.run.return_value=(self.capture.outputs/'stage-model',{})
            self.assertTrue(self.capture.check_current())
            self.assertFalse(self.capture.check_current())
            manager.poll(self.tracker);manager.thread.join(timeout=3)
            self.assertFalse(manager.thread.is_alive())
            self.assertEqual(manager.last_result['accepted_frame_ids'],[0,1,2])
            self.assertFalse(self.capture.closing.is_set())
            self.assertTrue(self.capture.check_current())
            manager.poll(self.tracker)
            self.assertEqual(manager.rounds,1)
            self.assertFalse(self.capture.closing.is_set())

    def test_cancel_queued_stage_releases_ui_without_ending_capture(self):
        self.assertTrue(self.capture.check_current())
        self.capture.cancel_check()
        events=[]
        while not self.capture.events.empty():events.append(self.capture.events.get()[0])
        self.assertIn('local_checkpoint_cancelled',events)
        self.assertFalse(self.capture.closing.is_set())

    def test_new_unknown_stage_replaces_old_task_feedback_without_inventing_comparison(self):
        from femto.capture.capture_exchange import current_tasks
        manager=self.capture.checkpoints
        model=SimpleNamespace(view_reference={'session_id':self.capture.session_id},
            store=SimpleNamespace(source=self.capture.session/'calibration/color_intrinsic.json'))
        with mock.patch('femto.pipeline.local_offline.LocalOfflineJob') as job, \
                mock.patch('femto.annotation.fixed_rgb_geometry.FixedRGBModel',return_value=model), \
                mock.patch('femto.capture.scan_roi.assess_model') as assessment:
            job.return_value.run.return_value=(self.capture.outputs/'stage-model',{})
            for classes in (np.full((10,10),3,np.uint8),None,np.ones((10,10),np.uint8)):
                assessment.return_value=({'status':'checked'},np.zeros((10,10,3),np.uint8),classes)
                parent,posed,request=self.freeze()
                manager._run(parent,posed,request,threading.Event())
                self.assertEqual(manager.last_result['stage_id'],request['stage_id'])
                self.assertEqual(current_tasks(self.capture)['task_batch_id'],request['stage_id'])
                if classes is None:
                    self.assertIsNone(manager.last_feedback[1])
                    self.assertEqual(current_tasks(self.capture)['tasks'],[])
                    self.assertEqual(manager.last_feedback[0]['status'],'cannot_determine')
            self.assertIsNone(manager.last_result['feedback']['comparison'])
            self.assertEqual(manager.last_result['feedback']['stagnant_rounds'],0)


if __name__=='__main__':unittest.main()
