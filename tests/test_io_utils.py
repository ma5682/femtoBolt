"""Contract tests for the shared unicode-safe IO helpers."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from femto.io_utils import (
    ImageCodecError,
    append_jsonl,
    imread_unicode,
    imread_unicode_or_none,
    imwrite_unicode,
    load_json,
    load_json_object,
    load_jsonl,
    write_json,
    write_json_strict,
    write_jsonl,
)


class JsonHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # Non-ASCII directory mirrors the repository path convention.
        self.root = Path(temporary.name) / "中文目录"
        self.root.mkdir()

    def test_write_json_is_atomic_creates_parents_and_keeps_unicode(self) -> None:
        target = self.root / "nested" / "deeper" / "report.json"
        write_json(target, {"名称": "值", "n": 1.5})
        self.assertTrue(target.is_file())
        self.assertFalse(list(target.parent.glob("*.tmp")))
        text = target.read_text(encoding="utf-8")
        self.assertIn("名称", text)  # ensure_ascii=False
        self.assertIn("\n  ", text)  # indent=2
        self.assertEqual(load_json(target), {"名称": "值", "n": 1.5})

    def test_load_json_tolerates_bom_and_load_json_object_rejects_non_objects(self) -> None:
        with_bom = self.root / "marker.json"
        with_bom.write_text(json.dumps({"k": 1}), encoding="utf-8-sig")
        self.assertEqual(load_json(with_bom), {"k": 1})
        self.assertEqual(load_json_object(with_bom), {"k": 1})
        as_list = self.root / "list.json"
        as_list.write_text("[1, 2]", encoding="utf-8")
        self.assertEqual(load_json(as_list), [1, 2])
        with self.assertRaisesRegex(ValueError, "Expected report object"):
            load_json_object(as_list, message="Expected report object")

    def test_strict_writer_rejects_nan_and_leaves_no_partial_file(self) -> None:
        target = self.root / "strict.json"
        with self.assertRaises(ValueError):
            write_json_strict(target, {"value": math.nan})
        self.assertFalse(target.exists())
        self.assertFalse(list(self.root.glob("*.tmp")))
        write_json(target, {"value": math.nan})  # default writer keeps json.dumps semantics
        self.assertIn("NaN", target.read_text(encoding="utf-8"))

    def test_jsonl_round_trip_and_blank_line_tolerance(self) -> None:
        target = self.root / "rows.jsonl"
        write_jsonl(target, [{"frame": 0}, {"frame": 1}])
        append_jsonl(target, {"frame": 2, "文本": "中"})
        text = target.read_text(encoding="utf-8")
        self.assertEqual(text.count("\n"), 3)
        self.assertNotIn(" ", text.splitlines()[0])  # compact separators
        target.write_text(text + "\n   \n", encoding="utf-8")
        self.assertEqual([row["frame"] for row in load_jsonl(target)], [0, 1, 2])


class ImageHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "图像目录"
        self.root.mkdir()

    def test_png_round_trip_preserves_uint16_depth_under_unicode_path(self) -> None:
        depth = np.arange(12, dtype=np.uint16).reshape(3, 4) * 300
        target = self.root / "sub" / "depth_mm.png"
        imwrite_unicode(target, depth)
        restored = imread_unicode(target, cv2.IMREAD_UNCHANGED)
        self.assertEqual(restored.dtype, np.uint16)
        np.testing.assert_array_equal(restored, depth)
        colour = np.zeros((2, 2, 3), np.uint8)
        colour[..., 0] = 255
        imwrite_unicode(self.root / "rgb.png", colour, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        np.testing.assert_array_equal(imread_unicode(self.root / "rgb.png", cv2.IMREAD_COLOR), colour)

    def test_decode_failures_raise_codec_error_compatible_with_both_legacy_types(self) -> None:
        empty = self.root / "empty.png"
        empty.write_bytes(b"")
        garbage = self.root / "garbage.png"
        garbage.write_bytes(b"not an image")
        for path in (empty, garbage):
            with self.subTest(path=path.name):
                with self.assertRaises(ImageCodecError) as caught:
                    imread_unicode(path, cv2.IMREAD_UNCHANGED)
                self.assertIsInstance(caught.exception, ValueError)
                self.assertIsInstance(caught.exception, RuntimeError)
                self.assertIsNone(imread_unicode_or_none(path, cv2.IMREAD_UNCHANGED))
        with self.assertRaises(OSError):
            imread_unicode(self.root / "missing.png", cv2.IMREAD_UNCHANGED)
        self.assertIsNone(imread_unicode_or_none(self.root / "missing.png", cv2.IMREAD_UNCHANGED))

    def test_encode_failure_raises_codec_error(self) -> None:
        with mock.patch("cv2.imencode", return_value=(False, None)):
            with self.assertRaises(ImageCodecError):
                imwrite_unicode(self.root / "bad.png", np.zeros((2, 2), np.uint8))
        self.assertFalse((self.root / "bad.png").exists())


if __name__ == "__main__":
    unittest.main()
