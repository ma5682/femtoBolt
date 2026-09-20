import copy
import json
import threading
import unittest
from unittest import mock

import numpy as np

from femto.capture.local_capture import LocalCapture
from femto.tracking.local_stability_feedback import CandidateStability, format_stability
from femto.local.local_temporal_support import stationary_window


def record(index, *, x=.015, received=None):
    pose=np.eye(4)
    pose[0,3]=x
    timestamp=1_000_001 + index*400_000
    return dict(frame_id=index, session_id='session', reference_frame_id=0, state='tracking',
        T_WC=pose.tolist(), capture_timestamp_us=timestamp, color_timestamp_us=timestamp,
        depth_timestamp_us=timestamp, received_monotonic=10+index*.4 if received is None else received,
        step_m=0.,step_deg=0.)


class CandidateTests(unittest.TestCase):
    def test_five_candidates_match_offline_pose_window_without_keyframe_side_effects(self):
        monitor=CandidateStability('session')
        monitor.request(10.)
        records=[record(i) for i in range(5)]
        original=copy.deepcopy(records)
        for i,r in enumerate(records):
            value=monitor.observe(r,r['received_monotonic']+.1,accepted=i==0)
            self.assertEqual(value['count'],i+1)
        self.assertEqual(value['state'],'candidate_ready')
        self.assertEqual(value['accepted_frame_id'],0)
        self.assertEqual(value['image_depth_audit'],'pending_offline')
        self.assertEqual(value['keyframes_added_by_monitor'],0)
        self.assertAlmostEqual(value['observed_pose_fps'],2.5)
        self.assertIsNotNone(stationary_window(records,0))
        self.assertEqual(records,original)
        self.assertIn('not a denoising approval',format_stability(value,'en'))

    def test_motion_after_acceptance_does_not_move_anchor_or_create_window(self):
        monitor=CandidateStability('session')
        monitor.observe(record(0),10.1,accepted=True)
        value=monitor.observe(record(1,x=.017),10.5)
        self.assertEqual(value['count'],0)
        self.assertEqual(value['reason'],'motion_after_capture')
        self.assertTrue(value['reason_triggered_this_frame'])
        carried=monitor.observe(record(2,x=.017),10.9)
        self.assertFalse(carried['reason_triggered_this_frame'])
        self.assertEqual(carried['reason_trigger_frame_id'],1)
        self.assertIsNotNone(carried['adjacent_pose_delta_median_mm'])
        self.assertIn('carried from prior frame',format_stability(carried,'en'))
        self.assertEqual(value['keyframes_added_by_monitor'],0)

    def test_invalid_lost_outside_nonfinite_stale_and_unsynced_reset(self):
        cases = {
            'lost': dict(state='lost',T_WC=None),
            'outside': dict(T_WC=record(1,x=.020001)['T_WC']),
            'nonfinite': dict(T_WC=[[float('nan')]*4]*4),
            'stale': dict(received_monotonic=7.),
            'sync': dict(color_timestamp_us=1_900_000),
            'session': dict(session_id='other'),
            'reference': dict(reference_frame_id=99),
        }
        for label,changes in cases.items():
            with self.subTest(label=label):
                monitor=CandidateStability('session')
                monitor.observe(record(0),10.1,accepted=True)
                r=record(1)
                r.update(changes)
                value=monitor.observe(r,10.5)
                self.assertEqual(value['count'],0)
                self.assertNotEqual(value['state'],'candidate_ready')

    def test_gap_duplicate_and_five_frame_memory_budget(self):
        for timestamp in (1_000_001,2_100_002):
            monitor=CandidateStability('session')
            monitor.observe(record(0),10.1,accepted=True)
            r=record(1)
            r['capture_timestamp_us']=timestamp
            value=monitor.observe(r,10.5)
            self.assertEqual(value['reason'],'timestamp_gap')
            self.assertEqual(value['count'],0)
        monitor=CandidateStability('session')
        for i in range(100):
            r=record(i)
            r['color_bgr']=np.zeros((100,100,3),np.uint8)
            value=monitor.observe(r,r['received_monotonic']+.1,accepted=i==0)
            self.assertLessEqual(len(monitor.frames),5)
        self.assertLess(len(json.dumps(list(monitor.frames))),4096)
        self.assertFalse(any('color_bgr' in r for r in monitor.frames))

    def test_request_pause_timeout_cancel_and_new_request_reset_counts(self):
        monitor=CandidateStability('session')
        monitor.request(10)
        self.assertEqual(monitor.observe(record(0,received=9.9),10.1)['count'],0)
        monitor.observe(record(1),10.5)
        self.assertEqual(len(monitor.frames),1)
        monitor.observe(record(2),10.9,paused=True)
        self.assertEqual(monitor.snapshot()['state'],'paused')
        self.assertEqual(monitor.snapshot()['count'],0)
        monitor.request(11)
        monitor.expire(14.1)
        self.assertEqual(monitor.snapshot()['reason'],'request_timeout')
        monitor.reset('cancelled')
        self.assertEqual(monitor.snapshot()['count'],0)
        monitor.request(15)
        self.assertEqual(monitor.snapshot()['state'],'waiting')

    def test_single_frame_capture_still_accepts_before_five_and_duplicate_remains_rejected(self):
        capture=LocalCapture.__new__(LocalCapture)
        capture.session_id='session'
        capture.lock=threading.Lock()
        capture.closing=threading.Event()
        capture.accepted=[record(0,x=0.)]
        capture.requested=10.
        capture.paused=False
        capture.emit=mock.Mock()
        capture.stability=CandidateStability('session')
        capture.stability.request(10.)
        r=record(1)
        with mock.patch('femto.capture.local_capture.time.monotonic',return_value=10.5):
            selected=capture._select(r,10.4)
        self.assertTrue(selected)
        value=capture.stability.observe(r,10.5,accepted=selected)
        self.assertEqual(value['count'],1)
        self.assertIsNone(capture.requested)
        capture.accepted.append(r)
        capture.requested=10.8
        with mock.patch('femto.capture.local_capture.time.monotonic',return_value=11.3):
            self.assertFalse(capture._select(record(3),11.2))
        self.assertEqual(capture.stability.snapshot()['state'],'duplicate')
        self.assertEqual(capture.accepted[0]['T_WC'],record(0,x=0.)['T_WC'])
