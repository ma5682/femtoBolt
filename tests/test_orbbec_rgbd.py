from __future__ import annotations

import unittest

from femto.hardware.orbbec_rgbd import FRAME_METADATA_FIELDS, OrbbecAlignedCamera, read_frame_metadata
from femto.geometry.pose_geometry import Intrinsics


class _FakeFrame:
    def __init__(self) -> None:
        self.values = {
            FRAME_METADATA_FIELDS["exposure"]: 250,
            FRAME_METADATA_FIELDS["gain"]: 4,
        }

    def has_metadata(self, metadata_type: object) -> bool:
        return metadata_type in self.values

    def get_metadata_value(self, metadata_type: object) -> int:
        return self.values[metadata_type]


class OrbbecMetadataTests(unittest.TestCase):
    def test_wide_unbinned_rejects_unsupported_30_fps(self):
        with self.assertRaisesRegex(ValueError, "frame rate"):
            OrbbecAlignedCamera("wide_detail", fps=30)
        camera = OrbbecAlignedCamera("wide_detail", fps=15, color_size=(1920, 1080))
        self.assertEqual(camera.depth_spec["width"], 1024)
        self.assertEqual(camera.max_depth_mm, 2210)
        self.assertEqual(camera.depth_spec["fps"], 15)

    def test_available_frame_metadata_is_preserved(self) -> None:
        metadata = read_frame_metadata(_FakeFrame())
        self.assertEqual(metadata, {"exposure": 250, "gain": 4})

    def test_aligned_read_failure_reasons_are_counted(self) -> None:
        class FakePipeline:
            def __init__(self) -> None:
                self.values = iter([None, object()])

            def wait_for_frames(self, _timeout_ms: int):
                return next(self.values)

        class FakeAlign:
            def process(self, _frames: object):
                return None

        camera = OrbbecAlignedCamera()
        camera.pipeline = FakePipeline()
        camera.align = FakeAlign()
        camera.intrinsics = Intrinsics(8, 6, 8.0, 8.0, 4.0, 3.0)
        self.assertIsNone(camera.read(10))
        self.assertIsNone(camera.read(10))
        self.assertEqual(
            camera.read_diagnostics_snapshot(),
            {
                "attempts": 2,
                "successful_aligned_frames": 0,
                "last_status": "align_none",
                "failure_counts": {"align_none": 1, "timeout": 1},
            },
        )


if __name__ == "__main__":
    unittest.main()
