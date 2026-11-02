import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d
from PIL import Image
import trimesh

from femto.geometry.multiview_texture import _EncodedPngTexture, export_multiview_texture


class MultiviewTextureEncodingTests(unittest.TestCase):
    def test_one_png_encoding_supplies_repeated_exports_and_preserves_pixels(self):
        pixels = np.random.default_rng(19).integers(0, 256, (37, 53, 3), dtype=np.uint8)
        original = pixels.copy()
        calls = []
        save = Image.Image.save

        def recorded(image, *args, **kwargs):
            calls.append(kwargs.get('format'))
            return save(image, *args, **kwargs)

        with patch.object(Image.Image, 'save', recorded):
            texture = _EncodedPngTexture(pixels)
            pixels[:] = 0
            for spelling in ('png', 'PNG'):
                with io.BytesIO() as stream:
                    texture.save(stream, format=spelling)
                    self.assertEqual(stream.getvalue(), texture.png_bytes)
            self.assertEqual(calls, ['PNG'])
        np.testing.assert_array_equal(np.asarray(texture), original)
        self.assertFalse(np.asarray(texture).flags.writeable)
        with self.assertRaises(ValueError):
            np.asarray(texture).setflags(write=True)
        with Image.open(io.BytesIO(texture.png_bytes)) as decoded:
            np.testing.assert_array_equal(np.asarray(decoded), original)
        with io.BytesIO() as expected:
            Image.fromarray(original).save(expected, format='PNG')
            self.assertEqual(texture.png_bytes, expected.getvalue())

    def test_requested_encoder_options_follow_pillow_and_keep_cached_default(self):
        pixels = np.random.default_rng(23).integers(0, 256, (31, 47, 3), dtype=np.uint8)
        texture = _EncodedPngTexture(pixels)
        with io.BytesIO() as expected, io.BytesIO() as actual:
            Image.fromarray(pixels).save(expected, format='PNG', compress_level=0)
            texture.save(actual, format='PNG', compress_level=0)
            self.assertEqual(actual.getvalue(), expected.getvalue())
        with io.BytesIO() as actual:
            texture.save(actual, format='PNG')
            self.assertEqual(actual.getvalue(), texture.png_bytes)

    def test_multiview_glb_is_byte_identical_to_ordinary_pillow_material(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)/'中文编码回归'
            (root/'calibration').mkdir(parents=True)
            intrinsic = dict(width=80, height=60, fx=100, fy=100, cx=40, cy=30)
            distortion = dict(model='BROWN_CONRADY_K6',
                              **dict.fromkeys(('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'), 0))
            (root/'calibration/color_intrinsic.json').write_text(json.dumps(intrinsic))
            (root/'calibration/distortion.json').write_text(json.dumps(dict(rgb=distortion)))
            frames = []
            rng = np.random.default_rng(31)
            for index in range(3):
                rgb = f'rgb_{index}.png'
                depth = f'depth_{index}.png'
                Image.fromarray(rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)).save(root/rgb)
                Image.fromarray(np.full((60, 80), 1000, np.uint16)).save(root/depth)
                frames.append(dict(rgb=rgb, depth=depth))
            vertices = np.array([[-.1, -.1, 1], [.1, -.1, 1], [-.1, .1, 1],
                                 [-.1, -.1, 1.2], [.1, -.1, 1.2], [-.1, .1, 1.2]])
            faces = np.array([[0, 2, 1], [3, 5, 4]])
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                             o3d.utility.Vector3iVector(faces))
            mesh.paint_uniform_color([.2, .3, .4])
            exports = dict(ply=str(root/'mesh.ply'), obj=str(root/'mesh.obj'), glb=str(root/'mesh.glb'))
            report = export_multiview_texture(mesh, root, frames, [np.eye(4)]*3, root, exports)
            with Image.open(report['texture']) as image:
                atlas = np.asarray(image).copy()
            uv = np.load(report['triangle_uv'], allow_pickle=False)
            material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(atlas),
                                                           metallicFactor=0, roughnessFactor=1, doubleSided=True)
            visual = trimesh.visual.texture.TextureVisuals(uv=uv.reshape(-1, 2), material=material)
            reference = trimesh.Trimesh(vertices=vertices[faces].reshape(-1, 3),
                                       faces=np.arange(len(faces)*3).reshape(-1, 3), visual=visual, process=False)
            self.assertEqual(Path(report['glb']).read_bytes(), reference.export(file_type='glb'))
            with io.BytesIO() as expected:
                Image.fromarray(atlas).save(expected, format='PNG')
                self.assertEqual(Path(report['texture']).read_bytes(), expected.getvalue())
            actual = trimesh.load(report['glb'], force='mesh', process=False)
            np.testing.assert_array_equal(np.asarray(actual.visual.material.baseColorTexture), atlas)
            np.testing.assert_allclose(actual.vertices[actual.faces], vertices[faces], atol=1e-6)
            np.testing.assert_allclose(actual.visual.uv[actual.faces], uv, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
