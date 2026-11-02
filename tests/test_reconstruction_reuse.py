import copy
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from femto.geometry.multiview_texture import export_multiview_texture
from femto.geometry.reconstruction_reuse import (
    MAX_JUMP_MASKS, MAX_RAW_VIEWS, MAX_RETAINED_BYTES, ReconstructionReuse,
    depth_jump_mask, texture_view_indices,
)
from femto.io_utils import imread_unicode
from femto.local.local_geometry import retain_observed_triangles
from femto.pipeline.scene_reconstruction import _frame_metrics


def legacy_jump(depth):
    jump = np.zeros(depth.shape, bool)
    horizontal = ((depth[:, 1:] > 0) & (depth[:, :-1] > 0)
                  & (np.abs(depth[:, 1:].astype(np.int32) - depth[:, :-1].astype(np.int32)) > 30))
    vertical = ((depth[1:] > 0) & (depth[:-1] > 0)
                & (np.abs(depth[1:].astype(np.int32) - depth[:-1].astype(np.int32)) > 30))
    jump[:, 1:] |= horizontal
    jump[:, :-1] |= horizontal
    jump[1:] |= vertical
    jump[:-1] |= vertical
    return jump


class ReconstructionReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / '原始证据'
        self.root.mkdir()
        self.shape = (30, 40)
        self.frame = dict(frame=7, rgb='彩色.png', depth='深度.png')
        self.rgb = np.random.default_rng(771).integers(0, 256, (*self.shape, 3), dtype=np.uint8)
        self.depth = np.full(self.shape, 1000, np.uint16)
        self.depth[0, 0] = 0
        Image.fromarray(self.rgb).save(self.root / self.frame['rgb'])
        Image.fromarray(self.depth).save(self.root / self.frame['depth'])

    def reuse(self, **kwargs):
        return ReconstructionReuse(self.root, [0], **kwargs)

    def test_original_decode_sha_readonly_and_source_bytes_unchanged(self):
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.root.iterdir()}
        reuse = self.reuse()
        with patch('femto.geometry.reconstruction_reuse.cv2.imdecode', wraps=cv2.imdecode) as decode:
            color, depth = reuse.load_rgbd(0, self.frame, self.shape)
            snapshot = reuse.get_rgbd(self.root, 0, self.frame)
            self.assertEqual(decode.call_count, 2)
        np.testing.assert_array_equal(color, self.rgb[:, :, ::-1])
        np.testing.assert_array_equal(depth, self.depth)
        self.assertEqual(snapshot.rgb_sha256, hashlib.sha256(before[self.root / self.frame['rgb']][0]).hexdigest())
        self.assertEqual(snapshot.depth_sha256, hashlib.sha256(before[self.root / self.frame['depth']][0]).hexdigest())
        for array in (color, depth, snapshot.color, snapshot.depth):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        self.assertEqual(reuse.retained_bytes, color.nbytes + depth.nbytes)
        self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.root.iterdir()})
        self.assertIsNone(self.reuse().get_rgbd(self.root, 0, self.frame))

    def test_changed_raw_content_with_same_size_and_restored_mtime_is_rejected(self):
        # Uncompressed PNG has deterministic length for equal-sized pixels.
        for key, original in [('rgb', self.rgb), ('depth', self.depth)]:
            with self.subTest(key=key):
                path = self.root / self.frame[key]
                Image.fromarray(original).save(path, compress_level=0)
                reuse = self.reuse()
                reuse.load_rgbd(0, self.frame, self.shape)
                old = path.stat()
                changed = original.copy()
                changed.flat[10] += 1
                Image.fromarray(changed).save(path, compress_level=0)
                os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
                self.assertEqual(path.stat().st_size, old.st_size)
                # On Windows ctime can mean birth time. Force the metadata fast
                # check to pass on every platform to exercise SHA verification.
                snapshot = reuse._raw[0]
                identities = iter(snapshot.identities)
                with patch('femto.geometry.reconstruction_reuse._file_identity', side_effect=lambda path: next(identities)):
                    self.assertIsNone(reuse.get_rgbd(self.root, 0, self.frame))
                self.assertEqual(reuse.retained_bytes, 0)

    def test_path_dataset_replacement_and_missing_input_invalidate(self):
        for mutation in ('path', 'dataset', 'replacement', 'missing'):
            with self.subTest(mutation=mutation):
                path = self.root / self.frame['rgb']
                Image.fromarray(self.rgb).save(path)
                reuse = self.reuse()
                reuse.load_rgbd(0, self.frame, self.shape)
                frame = dict(self.frame)
                dataset = self.root
                if mutation == 'path':
                    frame['rgb'] = './' + frame['rgb']
                elif mutation == 'dataset':
                    dataset = self.root.parent
                elif mutation == 'replacement':
                    replacement = self.root / 'replacement.png'
                    replacement.write_bytes(path.read_bytes())
                    os.replace(replacement, path)
                else:
                    path.unlink()
                self.assertIsNone(reuse.get_rgbd(dataset, 0, frame))
                self.assertEqual(reuse.retained_bytes, 0)

    def test_budget_nonselected_and_large_encoded_input_use_original_reader(self):
        for case in ('budget', 'nonselected', 'encoded'):
            with self.subTest(case=case):
                reuse = self.reuse(max_bytes=1 if case == 'budget' else MAX_RETAINED_BYTES)
                ordinal = 1 if case == 'nonselected' else 0
                limit = 1 if case == 'encoded' else 32 * 1024 * 1024
                with patch('femto.geometry.reconstruction_reuse.MAX_ENCODED_FILE_BYTES', limit), \
                     patch('femto.geometry.reconstruction_reuse.imread_unicode', wraps=imread_unicode) as read:
                    color, depth = reuse.load_rgbd(ordinal, self.frame, self.shape)
                self.assertEqual(read.call_count, 2)
                np.testing.assert_array_equal(color, self.rgb[:, :, ::-1])
                np.testing.assert_array_equal(depth, self.depth)
                self.assertEqual(reuse.retained_bytes, 0)
                self.assertIsNone(reuse.get_rgbd(self.root, ordinal, self.frame))

    def test_metadata_entry_counts_and_shared_payload_budget_are_bounded(self):
        reuse = ReconstructionReuse(self.root, range(MAX_RAW_VIEWS + 1), max_bytes=MAX_RETAINED_BYTES * 2)
        self.assertEqual(reuse.max_bytes, MAX_RETAINED_BYTES)
        for ordinal in range(MAX_RAW_VIEWS + 1):
            reuse.load_rgbd(ordinal, self.frame, self.shape)
        self.assertEqual(len(reuse._raw), MAX_RAW_VIEWS)
        self.assertIsNone(reuse.get_rgbd(self.root, MAX_RAW_VIEWS, self.frame))
        one = np.ones((1, 1), np.uint16)
        for ordinal in range(MAX_JUMP_MASKS + 1):
            reuse.remember_jump(ordinal, one, depth_jump_mask(one))
        self.assertEqual(len(reuse._jumps), MAX_JUMP_MASKS)
        raw_bytes = self.depth.size * 5
        tight = self.reuse(max_bytes=raw_bytes + self.depth.size - 1)
        tight.load_rgbd(0, self.frame, self.shape)
        tight.remember_jump(0, self.depth, depth_jump_mask(self.depth))
        self.assertIsNone(tight.get_jump(0, self.depth))
        self.assertEqual(tight.retained_bytes, raw_bytes)

    def test_change_during_capture_falls_back_and_cancellation_propagates(self):
        reuse = self.reuse()
        decode = cv2.imdecode
        calls = []

        def change_during_decode(*args):
            image = decode(*args)
            if not calls:
                Image.fromarray(np.zeros_like(self.rgb)).save(self.root / self.frame['rgb'])
            calls.append(True)
            return image

        with patch('femto.geometry.reconstruction_reuse.cv2.imdecode', side_effect=change_during_decode):
            color, _ = reuse.load_rgbd(0, self.frame, self.shape)
        np.testing.assert_array_equal(color, np.zeros_like(self.rgb))
        self.assertEqual(reuse.retained_bytes, 0)
        with patch('femto.geometry.reconstruction_reuse.np.fromfile', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                reuse.load_rgbd(0, self.frame, self.shape)
        self.assertEqual(reuse.retained_bytes, 0)

    def test_selected_view_ordinals_keep_existing_rule(self):
        for count in (1, 5, 12, 13, 99, 1000):
            for limit in (1, 4, 12, 20):
                np.testing.assert_array_equal(texture_view_indices(count, limit),
                    np.unique(np.linspace(0, count - 1, min(limit, count), dtype=int)))

    def test_jump_masks_equal_legacy_at_boundaries_invalid_values_and_random_depth(self):
        fixtures = [np.array([[0, 1, 31, 62, 65535, 0]], np.uint16),
                    np.array([[0], [1000], [1030], [1061], [65535]], np.uint16),
                    np.zeros((1, 1), np.uint16)]
        rng = np.random.default_rng(300)
        for shape in ((30, 40), (257, 251)):
            depth = rng.integers(0, 65536, shape, dtype=np.uint16)
            depth[rng.random(shape) < .25] = 0
            fixtures.append(depth)
        for depth in fixtures:
            original = depth.copy()
            np.testing.assert_array_equal(depth_jump_mask(depth), legacy_jump(depth))
            np.testing.assert_array_equal(depth, original)

    def test_jump_source_mutation_identity_and_readonly_behavior(self):
        reuse = self.reuse()
        jump = depth_jump_mask(self.depth)
        reuse.remember_jump(0, self.depth, jump)
        actual = reuse.get_jump(0, self.depth)
        np.testing.assert_array_equal(actual, legacy_jump(self.depth))
        with self.assertRaises(ValueError):
            actual.setflags(write=True)
        self.depth[1, 1] += 50
        self.assertIsNone(reuse.get_jump(0, self.depth))
        self.assertEqual(reuse.retained_bytes, 0)
        reuse.remember_jump(0, self.depth, depth_jump_mask(self.depth))
        self.assertIsNone(reuse.get_jump(0, self.depth.copy()))
        self.assertEqual(reuse.retained_bytes, 0)
        reuse.remember_jump(0, self.depth, depth_jump_mask(self.depth))
        self.depth.dtype = np.int16
        self.assertIsNone(reuse.get_jump(0, self.depth))
        self.assertEqual(reuse.retained_bytes, 0)

    def test_frame_metrics_and_observed_triangles_are_identical_and_mask_is_reused(self):
        self.depth[:, 20:] = 1040
        self.depth[11:15, 8:12] = 0
        original = self.depth.copy()
        old_metrics, new_metrics = [], []
        args = (dict(frame=7), np.eye(4), self.depth, self.depth, 321,
                o3d.geometry.PointCloud(), 0, .005, False)
        _frame_metrics(*args, old_metrics, legacy_jump(self.depth))
        jump = depth_jump_mask(self.depth)
        _frame_metrics(*args, new_metrics, jump)
        self.assertEqual(old_metrics, new_metrics)
        vertices = np.array([[-.1, -.08, 1], [-.02, -.08, 1], [-.1, .08, 1],
                             [.1, .08, 1], [-.1, -.08, 1.2], [.1, -.08, 1.2], [-.1, .08, 1.2]])
        faces = np.array([[0, 2, 1], [1, 2, 3], [4, 6, 5]])
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
        reference, actual = copy.deepcopy(mesh), copy.deepcopy(mesh)
        k = np.array([[100, 0, 20], [0, 100, 15], [0, 0, 1]], float)
        maps = (None, None, k, 40, 30)
        kwargs = dict(prepared_depths=[self.depth])
        with patch('femto.local.local_geometry.depth_jump_mask', side_effect=legacy_jump):
            expected = retain_observed_triangles(reference, self.root, [self.frame], [np.eye(4)], maps, .5, 2, **kwargs)
        reuse = self.reuse()
        reuse.remember_jump(0, self.depth, jump)
        with patch('femto.local.local_geometry.depth_jump_mask', side_effect=AssertionError('mask recomputed')):
            result = retain_observed_triangles(actual, self.root, [self.frame], [np.eye(4)], maps, .5, 2,
                                              reconstruction_reuse=reuse, **kwargs)
        self.assertEqual(expected, result)
        self.assertGreater(result['triangles_after'], 0)
        self.assertLess(result['triangles_after'], result['triangles_before'])
        for field in ('vertices', 'triangles', 'vertex_normals'):
            np.testing.assert_array_equal(np.asarray(getattr(reference, field)), np.asarray(getattr(actual, field)))
        np.testing.assert_array_equal(self.depth, original)
        missed = copy.deepcopy(mesh)
        with patch('femto.local.local_geometry.depth_jump_mask', wraps=depth_jump_mask) as fallback:
            result = retain_observed_triangles(missed, self.root, [self.frame], [np.eye(4)], maps, .5, 2,
                                              reconstruction_reuse=self.reuse(max_bytes=0), **kwargs)
        self.assertEqual(fallback.call_count, 1)
        self.assertEqual(expected, result)

    def test_atlas_uv_obj_glb_and_quality_equal_without_second_decode(self):
        calibration = self.root / 'calibration'
        calibration.mkdir()
        (calibration / 'color_intrinsic.json').write_text(json.dumps(dict(width=40, height=30, fx=100, fy=100, cx=20, cy=15)))
        dist = dict(model='BROWN_CONRADY_K6', **dict.fromkeys(('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'), 0))
        (calibration / 'distortion.json').write_text(json.dumps(dict(rgb=dist)))
        vertices = np.array([[-.08, -.06, 1], [.08, -.06, 1], [-.08, .06, 1],
                             [-.08, -.06, 1.2], [.08, -.06, 1.2], [-.08, .06, 1.2]])
        faces = np.array([[0, 2, 1], [3, 5, 4]])
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
        mesh.paint_uniform_color([.2, .3, .4])
        output = self.root / 'derived'
        output.mkdir()
        exports = dict(ply=str(output / 'mesh.ply'), obj=str(output / 'mesh.obj'), glb=str(output / 'mesh.glb'))
        frames, poses = [self.frame] * 3, [np.eye(4)] * 3
        sources_before = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        export_multiview_texture(mesh, self.root, frames, poses, output, exports)
        expected = {path: path.read_bytes() for path in output.iterdir()}
        reuse = ReconstructionReuse(self.root, texture_view_indices(3))
        for ordinal, frame in enumerate(frames):
            reuse.load_rgbd(ordinal, frame, self.shape)
        with patch('femto.geometry.multiview_texture.imread_unicode', side_effect=AssertionError('raw image reread')):
            export_multiview_texture(mesh, self.root, frames, poses, output, exports, reconstruction_reuse=reuse)
        self.assertEqual(expected, {path: path.read_bytes() for path in output.iterdir()})
        np.testing.assert_array_equal(vertices, np.asarray(mesh.vertices))
        np.testing.assert_array_equal(faces, np.asarray(mesh.triangles))
        self.assertEqual(sources_before, {path: path.read_bytes() for path in sources_before})


if __name__ == '__main__':
    unittest.main()
