"""Immutable-source tests for local binary masks; no physical accuracy claim."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from femto.dataset.local_reference import calibration_identity, reference_provenance
from femto.local.local_temporal_support import MASK_SOURCE_SCHEMA, filtered_depth, prepare
from femto.io_utils import imread_unicode as _read_image, imwrite_unicode
import cv2


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')


def make_source(root, *, depth=None, accepted=(0, 2, 4), session='temporal-source-fixture',
                outlier=True, outlier_index=2):
    """Five synthetic stationary frames with complete files, timestamps and W0 proof.

    These are filter/reconstruction fixtures, not keyframes from a live capture.
    Calibration is an explicit synthetic pinhole model for these new arrays.
    """
    depth = np.full((20, 20), 500, np.uint16) if depth is None else depth.copy()
    height, width = depth.shape
    raw = root/'raw'
    posed = raw/'derived/local_posed'
    for directory in ('rgb', 'depth', 'pose', 'derived'):
        (raw/directory).mkdir(parents=True, exist_ok=True)
    intrinsic = dict(width=width, height=height, fx=width*.86, fy=width*.86,
                     cx=(width-1)/2., cy=(height-1)/2.)
    distortion = dict(model='OBCameraDistortionModel.BROWN_CONRADY_K6',
                      k1=0., k2=0., k3=0., k4=0., k5=0., k6=0., p1=0., p2=0.)
    extrinsic = dict(rotation=np.eye(3).tolist(), translation_mm=[0., 0., 0.])
    for filename, value in {
        'color_intrinsic.json': intrinsic, 'depth_intrinsic.json': intrinsic,
        'depth_to_color.json': extrinsic, 'distortion.json': dict(rgb=distortion, depth=distortion),
        'camera_metadata.json': dict(device='synthetic_test_camera'),
        'factory_calibration.json': dict(source='synthetic_ground_truth',
            color_intrinsic=intrinsic, depth_intrinsic=intrinsic, rgb_distortion=distortion,
            depth_distortion=distortion, depth_to_color=extrinsic),
    }.items():
        write_json(raw/'calibration'/filename, value)
    rgb = np.random.default_rng(17).integers(0, 256, (height, width, 3), dtype=np.uint8)
    frames, records = [], []
    for index in range(5):
        timestamp = 1000000+index*400000
        frame = dict(frame=index, camera_timestamp_us=timestamp, color_timestamp_us=timestamp,
                     depth_timestamp_us=timestamp, rgb_depth_timestamp_delta_us=0,
                     rgb=f'rgb/{index:06}.png', depth=f'depth/{index:06}.png',
                     pose=f'pose/{index:06}.json', pose_valid=False)
        sample = depth.copy()
        if outlier and index == outlier_index:
            sample[0, 0] = 550
        imwrite_unicode(raw/frame['rgb'], rgb)
        imwrite_unicode(raw/frame['depth'], sample)
        write_json(raw/frame['pose'], dict(frame=index, pose_valid=False, T_WC=None,
                                         pose_source='synthetic_fixture_raw_unposed'))
        frames.append(frame)
        records.append(dict(frame_id=index, state='tracking', session_id=session,
                            reference_frame_id=0, T_WC=np.eye(4).tolist(),
                            capture_timestamp_us=timestamp, color_timestamp_us=timestamp,
                            depth_timestamp_us=timestamp, latency_ms=10., quality=dict(usable=True)))
    write_rows(raw/'frames.jsonl', frames)
    write_rows(raw/'derived/live_tracking.jsonl', records)
    manifest = dict(schema='femto-full-scene-rgbd-v1', source='synthetic_test_camera')
    manifest.update(frame_count=len(frames), data_kind='synthetic',
                    purpose='Synthetic stationary mask-source test; no hardware operation',
                    parameters=dict(depth_min_mm=250., depth_max_mm=2210., fixed_camera=False))
    write_json(raw/'manifest.json', manifest)
    proof = reference_provenance(session, records[0], calibration_identity(raw))
    write_json(raw/'derived/local_capture_report.json', dict(schema='femto-local-tracking-v1',
               session_id=session, reference_frame_id=0, accepted_frames=list(accepted), reference_provenance=proof))
    (posed/'pose').mkdir(parents=True)
    shutil.copytree(raw/'calibration', posed/'calibration')
    selected = []
    for ordinal, index in enumerate(accepted):
        frame = copy.deepcopy(frames[index])
        frame.update(pose=f'pose/{ordinal:06}.json', pose_valid=True)
        for key in ('rgb', 'depth'):
            frame[key] = str((raw/frame[key]).resolve())
        write_json(posed/frame['pose'], dict(frame=index, pose_valid=True,
                   T_WC=np.eye(4).tolist(), pose_source='synthetic_ground_truth',
                   session_id=session, reference_frame_id=0, reference_provenance=proof,
                   capture_timestamp_us=frame['camera_timestamp_us']))
        selected.append(frame)
    manifest.update(frame_count=len(selected), source_dataset=str(raw),
                    pose=dict(valid=True, source='synthetic_ground_truth'),
                    local_capture=dict(measured_only=True, session_id=session, reference_frame_id=0,
                        reference_provenance=proof, tracking_report=str(raw/'derived/local_capture_report.json')))
    write_json(posed/'manifest.json', manifest)
    write_rows(posed/'frames.jsonl', selected)
    write_json(posed/'pose_quality_report.json', dict(frame_count=len(selected), accepted_frame_ordinals=list(accepted)))
    return posed, raw


class TemporalBindingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.raw = make_source(self.root)

    def prepare(self, name='masks'):
        with mock.patch('femto.local.local_temporal_support.audit_temporal_motion',
                        return_value=(np.zeros((20, 20), bool), dict(passed=True))):
            return prepare(self.source, self.root/name)

    def chosen(self, dataset, index=2):
        row = next(row for row in read_rows(dataset/'frames.jsonl') if row['frame'] == index)
        return row, _read_image(dataset/row['depth'], cv2.IMREAD_UNCHANGED)

    def test_new_binding_preserves_original_pixels_pose_and_source_files(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file()}
        dataset, report = self.prepare()
        row, depth = self.chosen(dataset)
        before_depth = depth.copy()
        result = filtered_depth(dataset, row, depth)
        self.assertEqual(row['local_temporal_support']['source_binding']['schema'], MASK_SOURCE_SCHEMA)
        self.assertEqual(result[0, 0], 0)
        np.testing.assert_array_equal(result[1:], depth[1:])
        np.testing.assert_array_equal(depth, before_depth)
        self.assertEqual(report['applied_frames'], 3)
        self.assertTrue(report['source_unchanged'])
        for path, digest in before.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertFalse(list(self.root.rglob('latest*.txt')))
        self.assertEqual(read_json(dataset/'pose/000000.json')['T_WC'], np.eye(4).tolist())

    def test_same_shape_depth_file_change_and_changed_in_memory_values_are_rejected(self):
        dataset, _ = self.prepare()
        row, depth = self.chosen(dataset)
        altered = depth.copy()
        altered[1, 1] += 7
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            filtered_depth(dataset, row, altered)
        imwrite_unicode(dataset/row['depth'], altered)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            filtered_depth(dataset, row, altered)

    def test_selected_rgb_and_other_window_frame_changes_are_rejected(self):
        dataset, _ = self.prepare()
        row, depth = self.chosen(dataset)
        for path in (dataset/row['rgb'], self.raw/'depth/000001.png'):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                picture = _read_image(path, cv2.IMREAD_UNCHANGED)
                picture[1, 1] += 1
                imwrite_unicode(path, picture)
                with self.assertRaises(ValueError):
                    filtered_depth(dataset, row, depth)
                path.write_bytes(original)

    def test_mask_cannot_be_transplanted_to_another_frame_or_session(self):
        dataset, _ = self.prepare()
        row, depth = self.chosen(dataset)
        other, other_depth = self.chosen(dataset, index=4)
        other['local_temporal_support'] = copy.deepcopy(row['local_temporal_support'])
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            filtered_depth(dataset, other, other_depth)
        foreign, _ = make_source(self.root/'foreign', session='another-session')
        foreign_row, foreign_depth = self.chosen(foreign)
        foreign_row['local_temporal_support'] = copy.deepcopy(row['local_temporal_support'])
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            filtered_depth(foreign, foreign_row, foreign_depth)

    def test_timestamp_grid_unit_and_calibration_mismatch_rejected(self):
        dataset, _ = self.prepare()
        row, depth = self.chosen(dataset)
        for change in ('timestamp', 'grid', 'unit', 'reference'):
            with self.subTest(change=change):
                altered = copy.deepcopy(row)
                if change == 'timestamp':
                    altered['camera_timestamp_us'] += 1
                elif change == 'grid':
                    altered['local_temporal_support']['source_binding']['pixel_grid'] = 'undistorted'
                elif change == 'unit':
                    altered['local_temporal_support']['source_binding']['depth']['unit'] = 'metre'
                else:
                    altered['local_temporal_support']['source_binding']['reference_frame_id'] = 99
                with self.assertRaises(ValueError):
                    filtered_depth(dataset, altered, depth)
        path = dataset/'calibration/color_intrinsic.json'
        changed = read_json(path)
        changed['fx'] += 1
        write_json(path, changed)
        with self.assertRaises(ValueError):
            filtered_depth(dataset, row, depth)

    def test_legacy_mask_with_original_report_source_hashes_remains_compatible(self):
        dataset, _ = self.prepare()
        rows = read_rows(dataset/'frames.jsonl')
        for row in rows:
            del row['local_temporal_support']['source_binding']
        write_rows(dataset/'frames.jsonl', rows)
        row, depth = self.chosen(dataset)
        result = filtered_depth(dataset, row, depth)
        self.assertEqual(result[0, 0], 0)
        np.testing.assert_array_equal(result[1:], depth[1:])

    def test_legacy_without_report_or_exact_applied_window_refuses_new_filtering(self):
        dataset, _ = self.prepare()
        rows = read_rows(dataset/'frames.jsonl')
        for row in rows:
            del row['local_temporal_support']['source_binding']
        write_rows(dataset/'frames.jsonl', rows)
        row, depth = self.chosen(dataset)
        altered = copy.deepcopy(row)
        altered['local_temporal_support']['source_frames'] = [10, 11, 12, 13, 14]
        with self.assertRaisesRegex(ValueError, 'Legacy temporal mask'):
            filtered_depth(dataset, altered, depth)
        manifest = read_json(dataset/'manifest.json')
        del manifest['local_capture']['temporal_support_report']
        write_json(dataset/'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'Legacy temporal mask'):
            filtered_depth(dataset, row, depth)

    def test_legacy_hash_tamper_or_missing_source_hash_is_not_silently_ignored(self):
        dataset, _ = self.prepare()
        row, depth = self.chosen(dataset)
        del row['local_temporal_support']['source_binding']
        path = self.root/'masks/temporal_support_report.json'
        report = read_json(path)
        del report['source_sha256'][str((dataset/row['depth']).resolve())]
        write_json(path, report)
        with self.assertRaisesRegex(ValueError, 'source provenance'):
            filtered_depth(dataset, row, depth)

    def test_lost_dynamic_or_unsynced_tracking_changes_invalidate_saved_window(self):
        dataset, _ = self.prepare()
        row, depth = self.chosen(dataset)
        path = self.raw/'derived/live_tracking.jsonl'
        original = read_rows(path)
        for change in ('lost', 'dynamic', 'sync', 'session'):
            with self.subTest(change=change):
                records = copy.deepcopy(original)
                if change == 'lost': records[1]['state'] = 'lost'
                elif change == 'dynamic': records[1]['quality']['usable'] = False
                elif change == 'sync': records[1]['color_timestamp_us'] += 6000
                else: records[1]['session_id'] = 'foreign'
                write_rows(path, records)
                with self.assertRaises(ValueError):
                    filtered_depth(dataset, row, depth)
        write_rows(path, original)

    def test_change_during_audit_does_not_rebase_the_expected_source_hash(self):
        def audit(_images):
            path = self.raw/'depth/000001.png'
            depth = _read_image(path, cv2.IMREAD_UNCHANGED)
            depth[1, 1] += 1
            imwrite_unicode(path, depth)
            return np.zeros((20, 20), bool), dict(passed=True)
        with mock.patch('femto.local.local_temporal_support.audit_temporal_motion', side_effect=audit):
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                prepare(self.source, self.root/'changed')

    def test_no_mask_preserves_existing_unfiltered_workflow(self):
        row, depth = self.chosen(self.source)
        self.assertIs(filtered_depth(self.source, row, depth), depth)


if __name__ == '__main__':
    unittest.main()
