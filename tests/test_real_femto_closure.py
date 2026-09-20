"""Regression for provenance, input gates and failure evidence (not hardware acceptance)."""
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import cv2
import numpy as np
from PIL import Image

import femto.gsplat.gsplat_backend as gsplat_backend
from femto.app.femto_control_app import (
    ActiveScan,
    harvest_capture_motion,
    update_capture_motion_state,
    wait_for_aligned_frame,
)
from femto.gsplat.scene_gsplat_acceptance import build_acceptance_report, input_provenance
from femto.gsplat.scene_gsplat_data import load_gsplat_dataset, opengl_c2w_to_rdf
from femto.gsplat.scene_gsplat_trainer import assess_training_input
from femto.dataset.scene_gsplat_split import make_split, file_sha256
from femto.tracking.scene_pose_estimator import assess_background_motion, assess_rgb_motion_pair, estimate_poses


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_input(root, *, fixed=False, device="synthetic_test_camera", moving=True):
    raw = root / "dataset"
    write_json(raw / "manifest.json", {"schema": "femto-full-scene-rgbd-v1", "frame_count": 12,
               "camera": {"device": device}, "parameters": {"fixed_camera": fixed}})
    target = root / "nerfstudio"
    target.mkdir()
    frames = []
    for i in range(12):
        Image.fromarray(np.full((6, 8, 3), 100, np.uint8)).save(target / f"rgb{i}.png")
        Image.fromarray(np.full((6, 8), 1000, np.uint16)).save(target / f"depth{i}.png")
        pose = np.diag([1., -1., -1., 1.])
        pose[0, 3] = i * .01 if moving else 0
        frames.append({"file_path": f"rgb{i}.png", "depth_file_path": f"depth{i}.png", "transform_matrix": pose.tolist()})
    (target / "seed.ply").write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n0 0 1 128 128 128\n", encoding="ascii")
    split = make_split(len(frames))
    split['seed_sha256'] = file_sha256(target / 'seed.ply')
    write_json(target / "transforms.json", {"camera_model": "OPENCV", "w": 8, "h": 6, "fl_x": 8,
               "fl_y": 8, "cx": 4, "cy": 3, "depth_unit_scale_factor": .001, "ply_file_path": "seed.ply", "frames": frames,
               'dataset_split': split})
    write_json(target / "conversion_manifest.json", {"source_dataset": str(raw), "frame_count": 12,
               "translation_span_m": 1., "rotation_span_deg": 20.})
    return target


class RealFemtoClosureTests(unittest.TestCase):
    def test_live_capture_motion_feedback_is_sticky_and_not_a_pose(self):
        unreliable = {"reliable": False, "appears_fixed": None}
        fixed = {"reliable": True, "appears_fixed": True}
        moved = {"reliable": True, "appears_fixed": False}
        self.assertEqual(update_capture_motion_state("unknown", unreliable), "unknown")
        self.assertEqual(update_capture_motion_state("unknown", fixed), "not_moved")
        self.assertEqual(update_capture_motion_state("not_moved", moved), "moved")
        self.assertEqual(update_capture_motion_state("moved", fixed), "moved")

    def test_motion_worker_result_is_polled_without_blocking_and_written_back(self):
        future: Future[dict] = Future()
        scan = ActiveScan(
            session=Path("session"),
            frames_log=Path("frames.jsonl"),
            started=0.0,
            next_sample=0.0,
            duration_s=1.0,
            fps=1.0,
            max_frames=2,
            fixed_camera=False,
            processing_mode="moving",
            depth_min_mm=500.0,
            depth_max_mm=3860.0,
            motion_future=future,
        )
        self.assertFalse(harvest_capture_motion(scan))
        self.assertIs(scan.motion_future, future)
        future.set_result({"reliable": True, "appears_fixed": False})
        self.assertTrue(harvest_capture_motion(scan))
        self.assertEqual(scan.motion_state, "moved")
        self.assertFalse(scan.motion_report["appears_fixed"])
        self.assertIsNone(scan.motion_future)

    def test_motion_worker_failure_stays_an_unreliable_hint(self):
        future: Future[dict] = Future()
        future.set_exception(RuntimeError("synthetic worker failure"))
        scan = ActiveScan(
            session=Path("session"),
            frames_log=Path("frames.jsonl"),
            started=0.0,
            next_sample=0.0,
            duration_s=1.0,
            fps=1.0,
            max_frames=2,
            fixed_camera=False,
            processing_mode="moving",
            depth_min_mm=500.0,
            depth_max_mm=3860.0,
            motion_future=future,
        )
        self.assertTrue(harvest_capture_motion(scan))
        self.assertEqual(scan.motion_state, "not_moved")
        self.assertFalse(scan.motion_report["reliable"])
        self.assertIn("synthetic worker failure", scan.motion_report["error"])
        self.assertIsNone(scan.motion_future)

    def test_motion_worker_bounded_wait_timeout_preserves_pending_future(self):
        future: Future[dict] = Future()
        previous_report = {"reliable": True, "appears_fixed": True}
        scan = ActiveScan(
            session=Path("session"),
            frames_log=Path("frames.jsonl"),
            started=0.0,
            next_sample=0.0,
            duration_s=1.0,
            fps=1.0,
            max_frames=2,
            fixed_camera=False,
            processing_mode="moving",
            depth_min_mm=500.0,
            depth_max_mm=3860.0,
            motion_state="not_moved",
            motion_report=previous_report,
            motion_future=future,
        )
        self.assertFalse(harvest_capture_motion(scan, wait_timeout_s=0.0))
        self.assertEqual(scan.motion_state, "not_moved")
        self.assertIs(scan.motion_report, previous_report)
        self.assertIs(scan.motion_future, future)

    def test_harvest_keeps_moved_state_sticky_and_clears_completed_future(self):
        movement_evidence = {"reliable": True, "appears_fixed": False}
        future: Future[dict] = Future()
        future.set_result({"reliable": True, "appears_fixed": True})
        scan = ActiveScan(
            session=Path("session"),
            frames_log=Path("frames.jsonl"),
            started=0.0,
            next_sample=0.0,
            duration_s=1.0,
            fps=1.0,
            max_frames=2,
            fixed_camera=False,
            processing_mode="moving",
            depth_min_mm=500.0,
            depth_max_mm=3860.0,
            motion_state="moved",
            motion_report=movement_evidence,
            motion_future=future,
        )
        self.assertTrue(harvest_capture_motion(scan))
        self.assertEqual(scan.motion_state, "moved")
        self.assertIs(scan.motion_report, movement_evidence)
        self.assertFalse(scan.motion_report["appears_fixed"])
        self.assertIsNone(scan.motion_future)

    def test_harvest_keeps_not_moved_evidence_across_unreliable_result(self):
        fixed_evidence = {"reliable": True, "appears_fixed": True}
        future: Future[dict] = Future()
        future.set_result(
            {"reliable": False, "appears_fixed": None, "reason": "synthetic blur"}
        )
        scan = ActiveScan(
            session=Path("session"),
            frames_log=Path("frames.jsonl"),
            started=0.0,
            next_sample=0.0,
            duration_s=1.0,
            fps=1.0,
            max_frames=2,
            fixed_camera=False,
            processing_mode="moving",
            depth_min_mm=500.0,
            depth_max_mm=3860.0,
            motion_state="not_moved",
            motion_report=fixed_evidence,
            motion_future=future,
        )
        self.assertTrue(harvest_capture_motion(scan))
        self.assertEqual(scan.motion_state, "not_moved")
        self.assertIs(scan.motion_report, fixed_evidence)
        self.assertTrue(scan.motion_report["appears_fixed"])
        self.assertIsNone(scan.motion_future)

    def test_aligned_frame_wait_uses_a_deadline_not_an_attempt_count(self):
        expected = object()

        class FakeCamera:
            def __init__(self, frames):
                self.frames = iter(frames)
                self.timeouts = []

            def read(self, timeout_ms):
                self.timeouts.append(timeout_ms)
                return next(self.frames, None)

        successful = FakeCamera([None, expected])
        success_clock = mock.Mock(side_effect=[0.0, 0.1, 0.2])
        self.assertIs(
            wait_for_aligned_frame(successful, timeout_s=1.0, clock=success_clock),
            expected,
        )
        self.assertEqual(len(successful.timeouts), 2)

        expired = FakeCamera([None, None, None])
        expiry_clock = mock.Mock(side_effect=[0.0, 0.2, 0.7, 1.1])
        self.assertIsNone(wait_for_aligned_frame(expired, timeout_s=1.0, clock=expiry_clock))
        self.assertEqual(len(expired.timeouts), 2)

    @staticmethod
    def _textured_rgb() -> np.ndarray:
        rng = np.random.default_rng(20260904)
        image = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        for y in range(20, 230, 30):
            for x in range(20, 310, 30):
                cv2.circle(image, (x, y), 4, (255, 255, 255), -1)
        return image

    @staticmethod
    def _write_motion_sequence(root: Path, images: list[np.ndarray]) -> list[dict]:
        frames = []
        (root / "rgb").mkdir(parents=True)
        for index, image in enumerate(images):
            path = root / "rgb" / f"{index:06d}.png"
            self_written = cv2.imwrite(str(path), image)
            if not self_written:
                raise RuntimeError(f"Failed to write test image: {path}")
            frames.append({"frame": index, "rgb": path.relative_to(root).as_posix()})
        return frames

    def test_background_motion_fixed_sequence_is_blocked(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            base = self._textured_rgb()
            frames = self._write_motion_sequence(root, [base, base.copy(), base.copy()])
            report = assess_background_motion(root, frames, scale=0.5)
            self.assertTrue(report["reliable"])
            self.assertTrue(report["appears_fixed"])
            self.assertEqual(report["fixed_pair_count"], report["reliable_pair_count"])
            self.assertEqual(len(report["pair_details"]), 3)

    def test_background_motion_detects_middle_excursion_that_returns_to_start(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            base = self._textured_rgb()
            shifted = cv2.warpAffine(
                base,
                np.asarray([[1.0, 0.0, 18.0], [0.0, 1.0, 0.0]], dtype=np.float32),
                (base.shape[1], base.shape[0]),
                borderMode=cv2.BORDER_REFLECT,
            )
            frames = self._write_motion_sequence(root, [base, shifted, base.copy()])
            endpoint_only = assess_rgb_motion_pair(base, base.copy(), scale=0.5)
            report = assess_background_motion(root, frames, scale=0.5)
            self.assertTrue(endpoint_only["appears_fixed"])
            self.assertTrue(report["reliable"])
            self.assertFalse(report["appears_fixed"])
            self.assertGreater(report["max_background_displacement_px"], 10.0)
            self.assertEqual(report["pair_details"][-1]["source_frame_ordinal"], 0)
            self.assertEqual(report["pair_details"][-1]["target_frame_ordinal"], 2)

    def test_background_motion_checks_unsampled_short_excursion_before_return(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            base = self._textured_rgb()
            shifted = cv2.warpAffine(
                base,
                np.asarray([[1.0, 0.0, 18.0], [0.0, 1.0, 0.0]], dtype=np.float32),
                (base.shape[1], base.shape[0]),
                borderMode=cv2.BORDER_REFLECT,
            )
            # The previous five-ordinal sampler selected 0/2/4/7/9 and missed
            # the one-frame excursion at ordinal 1 entirely.
            images = [base, shifted] + [base.copy() for _ in range(8)]
            frames = self._write_motion_sequence(root, images)
            endpoint_only = assess_rgb_motion_pair(base, base.copy(), scale=0.5)
            report = assess_background_motion(root, frames, scale=0.5)
            self.assertTrue(endpoint_only["appears_fixed"])
            self.assertFalse(report["appears_fixed"])
            self.assertEqual(report["pair_count"], len(frames))
            self.assertTrue(
                any(
                    item["source_frame_ordinal"] == 0
                    and item["target_frame_ordinal"] == 1
                    and item["appears_fixed"] is False
                    for item in report["pair_details"]
                )
            )

    def test_low_homography_consensus_is_inconclusive_not_reliable_motion(self):
        base = self._textured_rgb()
        shifted = cv2.warpAffine(
            base,
            np.asarray([[1.0, 0.0, 18.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            (base.shape[1], base.shape[0]),
            borderMode=cv2.BORDER_REFLECT,
        )

        def low_consensus(source, _target, *_args, **_kwargs):
            mask = np.zeros((len(source), 1), dtype=np.uint8)
            mask[: max(1, len(source) // 10)] = 1
            return np.eye(3, dtype=np.float64), mask

        with mock.patch("femto.tracking.scene_pose_estimator.cv2.findHomography", side_effect=low_consensus):
            report = assess_rgb_motion_pair(base, shifted, scale=0.5)
        self.assertFalse(report["reliable"])
        self.assertIsNone(report["appears_fixed"])
        self.assertLess(report["homography_inlier_ratio"], 0.85)
        self.assertGreater(report["median_background_displacement_px"], 10.0)
        self.assertIn("inlier ratio", report["reason"])

    def test_unreliable_middle_excursion_cannot_be_declared_fixed(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            base = self._textured_rgb()
            featureless = np.zeros_like(base)
            frames = self._write_motion_sequence(
                root,
                [base, base.copy(), featureless, base.copy(), base.copy()],
            )
            report = assess_background_motion(root, frames, scale=0.5)
            self.assertIsNot(report["appears_fixed"], True)
            self.assertGreater(report["unreliable_pair_count"], 0)
            for pair in report["pair_details"]:
                self.assertIn("tracked_features", pair)
                self.assertIn("homography_inlier_ratio", pair)
                self.assertIn("median_background_displacement_px", pair)
                if not pair["reliable"]:
                    self.assertIsNone(pair["tracked_features"])
                    self.assertIsNone(pair["homography_inlier_ratio"])
                    self.assertIsNone(pair["median_background_displacement_px"])

    def test_fixed_subset_plus_unreliable_pairs_is_unknown_not_fixed(self):
        fixed = {
            "reliable": True,
            "appears_fixed": True,
            "tracked_features": 100,
            "homography_inlier_ratio": 1.0,
            "median_background_displacement_px": 0.1,
        }
        unreliable = {"reliable": False, "appears_fixed": None, "reason": "synthetic blur"}
        with TemporaryDirectory() as temp:
            root = Path(temp)
            base = self._textured_rgb()
            frames = self._write_motion_sequence(root, [base] * 5)
            with mock.patch(
                "femto.tracking.scene_pose_estimator.assess_rgb_motion_pair",
                side_effect=[fixed, unreliable, unreliable, fixed, fixed],
            ):
                report = assess_background_motion(root, frames, scale=0.5)
            self.assertFalse(report["reliable"])
            self.assertIsNone(report["appears_fixed"])
            self.assertEqual(report["reliable_pair_count"], 3)
            self.assertEqual(report["unreliable_pair_count"], 2)
            self.assertIn("cannot declare camera fixed", report["reason"])

    def test_conversion_presence_cannot_make_synthetic_real(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            target = make_input(root)
            run = root / "run"
            write_json(run / "training_report.json", {"dataset_root": str(target), "accepted": True})
            report = build_acceptance_report(run)
            self.assertEqual(report["data_kind"], "synthetic")
            self.assertFalse(report["real_femto_rendering_accepted"])
            self.assertEqual(report["end_to_end_status"], "BLOCKED")

    def test_missing_source_manifest_is_unknown(self):
        with TemporaryDirectory() as temp:
            target = Path(temp)
            write_json(target / "conversion_manifest.json", {"source_dataset": str(target / "missing")})
            self.assertEqual(input_provenance(target)["data_kind"], "unknown")

    def test_nested_derived_lineage_reaches_raw_frame_count(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            target = make_input(root, device="Orbbec Femto Bolt")
            posed = root / "dataset" / "derived" / "posed"
            write_json(posed / "manifest.json", {"source_dataset": str(root / "dataset"), "frame_count": 10,
                       "pose": {"source": "open3d_sequential_hybrid_rgbd_odometry"}})
            write_json(target / "conversion_manifest.json", {"source_dataset": str(posed), "frame_count": 10})
            report = input_provenance(target)
            self.assertEqual(report["data_kind"], "real_femto")
            self.assertEqual(report["raw_frame_count"], 12)
            self.assertEqual(report["valid_view_count"], 10)

    def test_stale_span_cannot_admit_identity_poses(self):
        with TemporaryDirectory() as temp:
            target = make_input(Path(temp), moving=False)
            report = assess_training_input(target)
            self.assertFalse(report["ready"])
            self.assertEqual(report["translation_span_m"], 0.)

    def test_invalid_seed_and_mismatched_geometry_lineage_block(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            target = make_input(root)
            write_json(root / "reconstruction" / "reconstruction_report.json", {"source_dataset": str(root / "another_dataset")})
            self.assertFalse(assess_training_input(target)["ready"])
            (target / "seed.ply").write_bytes(b"not a PLY")
            self.assertFalse(assess_training_input(target)["ready"])

    def test_fixed_capture_cannot_admit_moving_matrices(self):
        with TemporaryDirectory() as temp:
            self.assertFalse(assess_training_input(make_input(Path(temp), fixed=True))["ready"])

    def test_real_ui_gate_rejects_synthetic_without_changing_regression_gate(self):
        with TemporaryDirectory() as temp:
            target = make_input(Path(temp))
            self.assertTrue(assess_training_input(target)["ready"])
            self.assertFalse(gsplat_backend.assess_training_input(target, require_real=True)["ready"])

    def test_nonrigid_camera_and_incorrect_depth_units_are_rejected(self):
        transform = np.eye(4); transform[0, 0] = 2
        with self.assertRaises(ValueError):
            opengl_c2w_to_rdf(transform)
        with TemporaryDirectory() as temp:
            target = make_input(Path(temp))
            meta = json.loads((target / "transforms.json").read_text())
            meta["depth_unit_scale_factor"] = .01
            write_json(target / "transforms.json", meta)
            with self.assertRaises(ValueError):
                load_gsplat_dataset(target)

    def test_depth_8bit_and_calibration_size_mismatch_are_rejected(self):
        with TemporaryDirectory() as temp:
            target = make_input(Path(temp))
            Image.fromarray(np.ones((6, 8), np.uint8)).save(target / "depth0.png")
            with self.assertRaises(ValueError):
                load_gsplat_dataset(target)
            Image.fromarray(np.ones((7, 8), np.uint16)).save(target / "depth0.png")
            with self.assertRaises(ValueError):
                load_gsplat_dataset(target)

    def test_pose_rejection_writes_evidence_and_preserves_raw_bytes(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "manifest.json", {"schema": "femto-full-scene-rgbd-v1"})
            (root / "frames.jsonl").write_text('{"frame":0}\n{"frame":1}\n')
            before = hashlib.sha256((root / "frames.jsonl").read_bytes()).hexdigest()
            output = root / "derived" / "rejected"
            motion = {"reliable": True, "homography_inlier_ratio": 1., "median_background_displacement_px": .1}
            with mock.patch("femto.tracking.scene_pose_estimator.assess_background_motion", return_value=motion):
                with self.assertRaises(ValueError):
                    estimate_poses(root, output)
            report = json.loads((output / "pose_quality_report.json").read_text(encoding="utf-8-sig"))
            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(report["source_frame_count"], 2)
            self.assertFalse((output / "pose").exists())
            self.assertFalse((root / "derived" / "latest_posed_dataset.txt").exists())
            self.assertEqual(before, hashlib.sha256((root / "frames.jsonl").read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                estimate_poses(root, output)


if __name__ == "__main__":
    unittest.main()
