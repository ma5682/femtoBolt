"""The scan request stays tied to the exact first full RGB-D frame."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
import queue

import numpy as np

from femto.capture.local_capture import LocalCapture
from femto.capture.scan_roi import assess_model, depth_check, pixel_bounds, prepare, read
from femto.hardware.orbbec_rgbd import AlignedFrame
from tests.test_segmented_capture import make_camera
from tests.test_fixed_rgb_experience import history_fixture, make_model, ring


def frame(depth=None):
    return AlignedFrame(
        color_bgr=np.full((64, 64, 3), 127, np.uint8),
        depth_mm=(np.full((64, 64), 1000, np.uint16) if depth is None else depth),
        timestamp_us=123, system_timestamp_us=123, frame_index=7,
        color_timestamp_us=123, color_system_timestamp_us=123,
        depth_timestamp_us=123, depth_system_timestamp_us=123,
    )


class ScanRoiTests(TestCase):
    def test_range_needs_real_aligned_depth_and_stays_image_space(self):
        selected = [.125, .125, .875, .875]
        self.assertEqual(pixel_bounds(selected, [64, 64]), (8, 8, 56, 56))
        record = prepare(selected, frame(), 250, 2210)
        self.assertEqual(record['image_size'], [64, 64])
        self.assertEqual(record['depth_check']['valid_fraction'], 1.)
        self.assertIsNone(record['reference_frame_id'])
        self.assertFalse(record['raw_rgbd_cropped'])
        with self.assertRaises(ValueError):
            prepare([0., 0., .1, .1], frame(), 250, 2210)
        sparse = np.zeros((64, 64), np.uint16)
        sparse[8:30, 8:56] = 1000
        self.assertLess(depth_check(sparse, selected, 250, 2210)['valid_fraction'], .5)
        with self.assertRaisesRegex(ValueError, '有效深度不足'):
            prepare(selected, frame(sparse), 250, 2210)

    def test_seed_frame_cannot_be_replaced_by_later_camera_queue(self):
        first = frame()
        roi = prepare([.125, .125, .875, .875], first, 250, 2210)
        with TemporaryDirectory() as directory, mock.patch.object(LocalCapture, '_run'):
            capture = LocalCapture(make_camera('wide_detail', 64),
                Path(directory)/'raw', Path(directory)/'results', queue.Queue(),
                initial_frame=first, scan_roi=roi)
            capture.thread.join()
            first.color_bgr[:] = 0
            for _ in range(3):
                capture.submit(frame())
            self.assertEqual(capture._initial_frame[0].color_bgr[0, 0, 0], 127)
            self.assertEqual(capture._initial_frame[0].depth_mm.shape, (64, 64))
            self.assertEqual(read(capture.session), roi)
            self.assertEqual(json.loads((capture.session/'derived'/'scan_roi.json').read_text())['status'],
                             'pending_reference')

    def test_model_with_different_reference_cannot_inherit_scan_range(self):
        roi = prepare([.125, .125, .875, .875], frame(), 250, 2210)
        roi.update(status='bound', session_id='A', reference_frame_id=0)
        model = mock.Mock(view_reference=dict(session_id='B', reference_frame_id=0,
                                              image_size=[64, 64]))
        with self.assertRaisesRegex(ValueError, '首帧不一致'):
            assess_model(model, roi)

    def test_actual_reference_projection_keeps_missing_pixels_visible(self):
        with TemporaryDirectory() as directory:
            source, _ = history_fixture(directory)
            model = make_model(directory, ring())
            model.store.origin_reference = {'source_dataset': str(source)}
            roi = dict(status='bound', session_id='synthetic-fixture',
                       reference_frame_id=0, image_size=[100, 100],
                       rect_norm=[.1, .1, .9, .9])
            summary, preview = assess_model(model, roi, preview=True)
            self.assertEqual(summary['status'], 'checked')
            self.assertGreater(summary['supported_pixels'], 0)
            self.assertGreater(summary['pending_pixels'], 0)
            self.assertEqual(preview.shape, model.rgb.shape)
            np.testing.assert_array_equal(preview[0, 0], model.rgb[0, 0])
