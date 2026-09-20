from dataclasses import asdict
import queue
import threading
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import open3d as o3d

from femto.viewer.live_point_cloud import CalibratedCloudBuilder, LivePointCloudPreview, focus_bounds
from femto.geometry.pose_geometry import Intrinsics
from femto.capture.scene_segmented_capture import depth_cloud
from femto.app.femto_control_app import FemtoControlApp


class LivePointCloudTests(unittest.TestCase):
    def setUp(self):
        self.intr = Intrinsics(6, 1, 5.0, 5.0, 2.5, 0.0)
        self.dist = {"model": "BROWN_CONRADY_K6", **dict.fromkeys(("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"), 0.0)}
        self.dist["k1"] = 0.05
        self.packet = {"rgb": np.arange(18, dtype=np.uint8).reshape(1, 6, 3), "depth": np.array([[0, 249, 250, 2210, 2211, np.nan]], np.float32), "intrinsics": asdict(self.intr), "distortion": self.dist}

    def test_live_geometry_matches_saved_point_cloud_and_recalibrates(self):
        builder = CalibratedCloudBuilder()
        for cx in (2.5, 1.5):
            self.packet["intrinsics"]["cx"] = cx
            intr = Intrinsics(**self.packet["intrinsics"])
            live = builder.build(self.packet)
            saved = depth_cloud(self.packet["depth"], self.packet["rgb"], intr, self.dist, 250, 2210)
            np.testing.assert_allclose(np.asarray(live.points), np.asarray(saved.points), atol=1e-6)
            np.testing.assert_array_equal(np.asarray(live.colors), np.asarray(saved.colors))
            np.testing.assert_allclose(np.asarray(live.points)[:, 2], [0.25, 2.21])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            builder.build({**self.packet, "rgb": np.zeros((2, 6, 3), np.uint8)})

    def test_focus_uses_main_surface_without_removing_background_points(self):
        xyz = np.r_[np.tile([[0.1, 0.2, 0.35]], (1000, 1)), np.tile([[2, 2, 2.0]], (50, 1))]
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        bounds = focus_bounds(cloud)
        self.assertLess(bounds.get_center()[2], 0.5)
        np.testing.assert_array_equal(np.asarray(cloud.points), xyz)
        empty = CalibratedCloudBuilder().build({**self.packet, "depth": np.zeros((1, 6), np.float32)})
        self.assertEqual(len(empty.points), 0)

    def test_busy_renderer_drops_display_submission_without_touching_source(self):
        preview = LivePointCloudPreview.__new__(LivePointCloudPreview)
        preview.frames = queue.Queue(maxsize=1)
        preview.frames.put("pending frame")
        preview.stopped = threading.Event()
        preview.closed = False
        preview.last_sent = 0
        preview.last_frame_key = None
        frame = SimpleNamespace(frame_index=1, depth_timestamp_us=10, color_bgr=self.packet["rgb"], depth_mm=self.packet["depth"])
        before = frame.depth_mm.copy()
        self.assertFalse(preview.submit(frame, self.intr, self.dist))
        np.testing.assert_array_equal(frame.depth_mm, before)
        self.assertEqual(preview.frames.get_nowait(), "pending frame")
        self.assertTrue(preview.submit(frame, self.intr, self.dist))
        self.assertFalse(preview.submit(frame, self.intr, self.dist))
        preview.stopped.set()
        self.assertFalse(preview.submit(frame, self.intr, self.dist))


    def test_paused_fixed_multiframe_modes_are_blocked_before_reading_or_reconstructing(self):
        app = FemtoControlApp.__new__(FemtoControlApp)
        app.language = "en"
        app.event_queue = queue.Queue()
        with mock.patch("femto.app.femto_control_app.run_demo") as reconstruct, mock.patch("femto.app.femto_control_app.build_fixed_keyframe_dataset") as keyframe:
            for mode in ("static", "wide"):
                app._reconstruction_worker(Path("unused_source"), mode)
                event, message = app.event_queue.get_nowait()
                self.assertEqual(event, "reconstruction_error")
                self.assertIn("paused", message)
            reconstruct.assert_not_called()
            keyframe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
