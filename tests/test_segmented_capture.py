from pathlib import Path
import json
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import cv2
import numpy as np
import open3d as o3d

from femto.capture.scan_session import save_rgbd_photo
from femto.geometry.fusion import read_geometry_safe
from femto.hardware.orbbec_rgbd import AlignedFrame, DEPTH_MODE_SPECS
from femto.geometry.pose_geometry import Intrinsics
from femto.capture.scene_segmented_capture import TUNED_DEPTH_MODES, audit_static_views, capture_three_modes, merge_segments, save_segment
from femto.geometry.depth_cloud import depth_cloud


DISTORTION = {
    "model": "BROWN_CONRADY_K6",
    **{key: 0.0 for key in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")},
}


def make_frame(depth, index=1):
    depths = np.asarray(depth, dtype=np.float32).reshape(1, -1)
    return AlignedFrame(
        color_bgr=np.full((*depths.shape, 3), 100, dtype=np.uint8),
        depth_mm=depths, timestamp_us=index, system_timestamp_us=index,
        frame_index=index, color_timestamp_us=index, color_system_timestamp_us=index,
        depth_timestamp_us=index, depth_system_timestamp_us=index,
    )


def make_camera(mode, width):
    spec = DEPTH_MODE_SPECS[mode]
    return mock.Mock(
        min_depth_mm=spec["min_depth_mm"], max_depth_mm=spec["max_depth_mm"],
        intrinsics=Intrinsics(width, 1, 10.0, 10.0, 0.0, 0.0),
        calibration={
            "rgb_distortion": DISTORTION, "depth_distortion": DISTORTION,
            "depth_intrinsic": {}, "depth_to_color": {},
        },
        metadata={"depth_mode": spec["name"]},
    )


def make_cloud(points, color):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))
    return cloud


class SegmentedCaptureTests(unittest.TestCase):
    def test_tuned_capture_saves_native_depth_and_skips_settling_frames(self):
        def camera_factory(mode, **kwargs):
            self.assertEqual(kwargs["fps"], 15)
            self.assertTrue(kwargs["retain_native_depth"])
            self.assertEqual(kwargs["color_size"], (1280, 960) if mode in ("near", "far") else (1920, 1080))
            camera = make_camera(mode, 3)
            frames = []
            for index in range(1, 26):
                frame = make_frame([0, camera.min_depth_mm, camera.max_depth_mm], index)
                frame.native_depth_mm = np.array([[0, 400, 600]], dtype=np.float32)
                frames.append(frame)
            camera.read.side_effect = frames
            return camera
        with TemporaryDirectory() as directory, mock.patch(
            "femto.capture.scene_segmented_capture.OrbbecAlignedCamera", side_effect=camera_factory,
        ), mock.patch("femto.capture.scene_segmented_capture.common_rgb_view", return_value=np.zeros((320, 480), np.uint8)), mock.patch(
            "femto.capture.scene_segmented_capture.audit_static_views", return_value={"passed": True},
        ):
            output = Path(directory) / "capture"
            report = capture_three_modes(output, tuned=True)
            self.assertEqual(report["capture_modes"], list(TUNED_DEPTH_MODES))
            self.assertEqual(report["capture_preset"], "official_detail_coverage")
            self.assertEqual(len(list(output.glob("segments/*/raw/*_native_depth_mm.npy"))), 40)
            for mode in TUNED_DEPTH_MODES:
                records = [json.loads(line) for line in (output / f"segments/{mode}/raw/frames.jsonl").read_text().splitlines()]
                self.assertEqual([record["frame_index"] for record in records], list(range(16, 26)))
                self.assertEqual(records[0]["native_depth"], "frame_001_native_depth_mm.npy")

    def test_static_gate_blocks_moving_and_inconclusive_pairs(self):
        views = [np.zeros((320, 480), np.uint8)] * 3
        fixed = {"reliable": True, "appears_fixed": True}
        for rejected in ({"reliable": True, "appears_fixed": False}, {"reliable": False, "appears_fixed": None}):
            with mock.patch("femto.capture.scene_segmented_capture.assess_rgb_motion_pair", side_effect=[fixed, rejected, fixed, fixed]):
                self.assertFalse(audit_static_views(views, views[0])["passed"])

    def test_empty_optional_detail_retains_evidence_and_falls_back_to_three_modes(self):
        def camera_factory(mode, **kwargs):
            camera = make_camera(mode, 3)
            camera.read.side_effect = [make_frame(
                [0, 0, 0] if mode == "wide_detail" else [0, camera.min_depth_mm, camera.max_depth_mm], index,
            ) for index in range(1, 26)]
            return camera
        with TemporaryDirectory() as directory, mock.patch(
            "femto.capture.scene_segmented_capture.OrbbecAlignedCamera", side_effect=camera_factory,
        ), mock.patch("femto.capture.scene_segmented_capture.common_rgb_view", return_value=np.zeros((320, 480), np.uint8)), mock.patch(
            "femto.capture.scene_segmented_capture.audit_static_views", return_value={"passed": True},
        ):
            output = Path(directory) / "capture"
            report = capture_three_modes(output, tuned=True)
            self.assertFalse(report["enhancement_applied"])
            self.assertEqual(report["capture_modes"], ["wide", "near", "far"])
            self.assertEqual(report["skipped_modes"][0]["mode"], "wide_detail")
            self.assertEqual(len(list((output / "segments/wide_detail/raw").glob('*_depth_mm.npy'))), 10)

    def test_four_mode_merge_prioritizes_native_detail_without_averaging(self):
        clouds = [
            make_cloud([[0, 0, .3], [0, 0, 1]], [1, 1, 0]),
            make_cloud([[.001, 0, .3], [.1, 0, 2.8]], [0, 1, 0]),
            make_cloud([[.001, 0, 1], [.0001, 0, 1]], [1, 0, 0]),
            make_cloud([[0, 0, 5]], [0, 0, 1]),
        ]
        reports = [{"mode": mode, "point_count": len(cloud.points)} for mode, cloud in zip(TUNED_DEPTH_MODES, clouds)]
        with TemporaryDirectory() as directory:
            report = merge_segments(Path(directory), clouds, reports)
            merged = read_geometry_safe(Path(directory) / report["file"], mesh=False)
            self.assertEqual(report["overlap_selection"]["priority"], ["near", "wide_detail", "wide", "far"])
            self.assertEqual(len(merged.points), 5)
            np.testing.assert_array_equal(np.asarray(merged.points)[:2], np.asarray(clouds[2].points))
            np.testing.assert_array_equal(np.asarray(merged.points)[2], [0, 0, .3])

    def test_all_profiles_include_endpoints_and_reject_invalid_depth(self):
        for mode, spec in DEPTH_MODE_SPECS.items():
            with self.subTest(mode=mode):
                minimum, maximum = spec["min_depth_mm"], spec["max_depth_mm"]
                frame = make_frame([0, np.nan, np.inf, minimum - 1, minimum, maximum, maximum + 1])
                camera = make_camera(mode, 7)
                cloud = depth_cloud(frame.depth_mm, frame.color_bgr, camera.intrinsics,
                                    DISTORTION, minimum, maximum)
                np.testing.assert_allclose(np.asarray(cloud.points)[:, 2], [minimum / 1000, maximum / 1000])

    def test_full_range_photo_saves_evidence_and_points_outside_legacy_nfov(self):
        for mode, depth in (("wide", [250, 300, 2880, 4000]), ("far", [250, 500, 5000, 5460])):
            with TemporaryDirectory() as directory:
                frame = make_frame(depth)
                camera = make_camera(mode, 4)
                output = save_rgbd_photo(frame, camera, Path(directory))
                self.assertTrue((output / "color.jpg").is_file())
                restored = cv2.imdecode(np.fromfile(output / "depth_mm.png", np.uint8), cv2.IMREAD_UNCHANGED)
                np.testing.assert_array_equal(restored, frame.depth_mm)
                points = read_geometry_safe(output / "point_cloud_m.ply", mesh=False)
                self.assertEqual(len(points.points), 3)
                self.assertGreaterEqual(np.asarray(points.points)[:, 2].min(), camera.min_depth_mm / 1000)
                self.assertEqual(np.asarray(points.points)[:, 2].max(), camera.max_depth_mm / 1000)
                manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["image"]["depth_range_m"],
                                 [camera.min_depth_mm / 1000, camera.max_depth_mm / 1000])

    def test_temporal_fusion_uses_full_mode_and_rejects_single_frame_noise(self):
        camera = make_camera("wide", 4)
        frames = [make_frame([300, 1500 + index, 2800, 1000 if index == 1 else 0], index)
                  for index in range(1, 6)]
        for frame in frames:
            frame.color_bgr[:] = frame.frame_index * 10
        with TemporaryDirectory() as directory:
            cloud, report = save_segment(Path(directory), "wide", frames, camera)
            self.assertEqual(report["point_count"], 3)
            self.assertEqual(report["minimum_support_frames"], 3)
            np.testing.assert_allclose(np.asarray(cloud.points)[:, 2], [.3, 1.503, 2.8])
            np.testing.assert_allclose(np.asarray(cloud.colors)[1], [30 / 255] * 3)
            self.assertTrue((Path(directory) / "segments/wide/derived/support_count.png").exists())

    def test_merge_preserves_high_resolution_coordinates_colors_and_gap_points(self):
        normal = make_cloud([[0, 0, 1], [.0004, 0, 1], [0, 0, 3.86]], [1, 0, 0])
        wide = make_cloud([[0, 0, .25], [.001, 0, 1], [.1, 0, 1.5]], [0, 1, 0])
        far = make_cloud([[.002, 0, 1], [.2, 0, 2], [0, 0, 5], [0, 0, 5.46]], [0, 0, 1])
        reports = [{"mode": mode, "point_count": len(cloud.points)}
                   for mode, cloud in zip(("wide", "near", "far"), (wide, normal, far))]
        with TemporaryDirectory() as directory:
            output = Path(directory) / "capture"
            output.mkdir()
            report = merge_segments(output, [wide, normal, far], reports)
            merged = read_geometry_safe(output / report["file"], mesh=False)
            points = np.asarray(merged.points)
            np.testing.assert_array_equal(points[:3], np.asarray(normal.points))
            np.testing.assert_array_equal(np.asarray(merged.colors)[:3], np.asarray(normal.colors))
            self.assertEqual(len(points), 8)
            self.assertEqual(report["points_before_merge"], 10)
            self.assertIn(.25, points[:, 2])
            self.assertIn(5.0, points[:, 2])
            self.assertIn(5.46, points[:, 2])
            self.assertTrue(np.any(np.all(points == [.1, 0, 1.5], axis=1)))
            self.assertTrue(np.any(np.all(points == [.2, 0, 2], axis=1)))
            unmerged = read_geometry_safe(output / report["unmerged_file"], mesh=False)
            self.assertEqual(len(unmerged.points), 10)

    def test_capture_preserves_fifteen_raw_frames_and_each_calibration(self):
        cameras = []
        def camera_factory(mode, **kwargs):
            camera = make_camera(mode, 3)
            minimum, maximum = camera.min_depth_mm, camera.max_depth_mm
            camera.read.side_effect = [make_frame([0, minimum, maximum], index) for index in range(1, 6)]
            cameras.append((mode, kwargs, camera))
            return camera
        with TemporaryDirectory() as directory, mock.patch(
            "femto.capture.scene_segmented_capture.OrbbecAlignedCamera", side_effect=camera_factory,
        ):
            output = Path(directory) / "capture"
            report = capture_three_modes(output)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(len(list(output.glob("segments/*/raw/*_depth_mm.npy"))), 15)
            self.assertEqual(len(list(output.glob("segments/*/calibration/factory_calibration.json"))), 3)
            for mode, kwargs, camera in cameras:
                camera.stop.assert_called_once()
                self.assertEqual(kwargs["color_size"], None if mode == "wide" else (1280, 960))
                depth = np.load(output / f"segments/{mode}/raw/frame_001_depth_mm.npy")
                self.assertEqual(float(depth[0, 0]), 0)
                self.assertEqual(float(depth[0, -1]), camera.max_depth_mm)

    def test_failed_profile_does_not_publish_partial_result(self):
        with TemporaryDirectory() as directory, mock.patch(
            "femto.capture.scene_segmented_capture.OrbbecAlignedCamera",
        ) as camera_type, mock.patch("femto.capture.scene_segmented_capture.time.sleep"):
            camera_type.return_value.start.side_effect = RuntimeError("camera disconnected")
            output = Path(directory) / "capture"
            marker = Path(directory) / "latest_full_range_cloud.txt"
            marker.write_text("previous good capture", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "disconnected"):
                capture_three_modes(output)
            self.assertEqual(camera_type.return_value.stop.call_count, 3)
            self.assertEqual(marker.read_text(encoding="utf-8"), "previous good capture")
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "failed")



if __name__ == "__main__":
    unittest.main()
