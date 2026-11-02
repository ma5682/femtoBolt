import hashlib
import io
import json
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import numpy as np
from PIL import Image, PngImagePlugin, UnidentifiedImageError
import trimesh

from femto.annotation.region_texture import source_vertex_normals, textured_region
from femto.geometry.png_texture import _plain_rgb_png, load_source_texture


def png_bytes(image, **options):
    with io.BytesIO() as stream:
        image.save(stream, format='PNG', **options)
        return stream.getvalue()


def chunk(kind, data):
    return (struct.pack('>I', len(data)) + kind + data +
            struct.pack('>I', zlib.crc32(data, zlib.crc32(kind)) & 0xffffffff))


def chunks(encoded):
    offset = 8
    result = []
    while offset < len(encoded):
        length = struct.unpack_from('>I', encoded, offset)[0]
        result.append((encoded[offset + 4:offset + 8], encoded[offset + 8:offset + 8 + length]))
        offset += 12 + length
    return result


class SourcePngTextureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / '完整纹理'
        self.root.mkdir()
        self.image_path = self.root / 'texture_multiview.png'
        self.source = self.root / 'scene_tsdf_mesh_m.ply'
        self.pixels = np.random.default_rng(611).integers(0, 256, (29, 41, 3), dtype=np.uint8)
        self.mesh = SimpleNamespace(
            vertices=np.array([[0., 0., .5], [1., 0., .5], [1., 1., .7], [0., 1., .5]]),
            triangles=np.array([[0, 1, 2], [0, 2, 3]]))
        self.uv = np.array([[[0, 0], [1, 0], [1, 1]], [[.1, .1], [.8, .8], [0, 1]]])
        np.save(self.root / 'mesh_triangle_uv.npy', self.uv)

    def write_png(self, image=None, **options):
        encoded = png_bytes(Image.fromarray(self.pixels) if image is None else image, **options)
        self.image_path.write_bytes(encoded)
        return encoded

    def old_region(self, ids):
        # Frozen pre-optimization behavior, including Pillow image metadata.
        with Image.open(self.image_path) as image:
            atlas = image.convert('RGB').copy()
        ids = np.asarray(sorted(set(ids)), dtype=int)
        faces, vertices = self.mesh.triangles, self.mesh.vertices
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=atlas, metallicFactor=0, roughnessFactor=1, doubleSided=True)
        visual = trimesh.visual.texture.TextureVisuals(uv=self.uv[ids].reshape(-1, 2), material=material)
        return trimesh.Trimesh(vertices=vertices[faces[ids]].reshape(-1, 3),
                               faces=np.arange(len(ids) * 3).reshape(-1, 3), visual=visual,
                               process=False,
                               vertex_normals=source_vertex_normals(vertices, faces)[faces[ids]].reshape(-1, 3))

    @staticmethod
    def export(mesh):
        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name='region_W0_RDF', transform=np.diag([1., -1., -1., 1.]))
        return scene.export(file_type='glb')

    def assert_glb_equal(self, actual_bytes, expected_bytes, *, byte_equal=True):
        if byte_equal:
            self.assertEqual(actual_bytes, expected_bytes)
        actual = trimesh.load(io.BytesIO(actual_bytes), file_type='glb', force='scene', process=False)
        expected = trimesh.load(io.BytesIO(expected_bytes), file_type='glb', force='scene', process=False)
        self.assertEqual(list(actual.geometry), list(expected.geometry))
        self.assertEqual(actual.graph.to_flattened(), expected.graph.to_flattened())
        for name in actual.geometry:
            left, right = actual.geometry[name], expected.geometry[name]
            for field in ('vertices', 'faces', 'vertex_normals'):
                np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
            np.testing.assert_array_equal(left.visual.uv, right.visual.uv)
            left_material, right_material = left.visual.material, right.visual.material
            np.testing.assert_array_equal(np.asarray(left_material.baseColorTexture),
                                          np.asarray(right_material.baseColorTexture))
            for field in ('metallicFactor', 'roughnessFactor', 'doubleSided', 'alphaMode'):
                self.assertEqual(getattr(left_material, field), getattr(right_material, field))

    def test_canonical_rgb_skips_the_only_image_encoding_and_keeps_complete_glb_bytes(self):
        encoded = self.write_png()
        expected = self.export(self.old_region([1, 0, 1]))
        with patch.object(Image.Image, 'save', side_effect=AssertionError('unexpected PNG re-encoding')):
            result, metadata = textured_region(self.mesh, [1, 0, 1], self.source)
            actual = self.export(result)
        self.assertEqual(metadata['texture_sha256'], hashlib.sha256(encoded).hexdigest())
        self.assertEqual(metadata['source_triangle_indices'], [0, 1])
        self.assert_glb_equal(actual, expected)

    def test_alternative_rgb_compression_preserves_every_decoded_glb_field(self):
        for level in (0, 1, 9):
            with self.subTest(compress_level=level):
                encoded = self.write_png(compress_level=level)
                expected = self.export(self.old_region([1]))
                result, _ = textured_region(self.mesh, [1], self.source)
                actual = self.export(result)
                self.assert_glb_equal(actual, expected, byte_equal=False)
                json_length = struct.unpack_from('<I', actual, 12)[0]
                tree = json.loads(actual[20:20 + json_length])
                view = tree['bufferViews'][tree['images'][0]['bufferView']]
                start = 20 + json_length + 8 + view.get('byteOffset', 0)
                self.assertEqual(actual[start:start + view['byteLength']], encoded)

    def test_non_rgb_metadata_and_non_png_sources_keep_old_pillow_glb_bytes(self):
        text = PngImagePlugin.PngInfo()
        text.add_text('source', '保留原编码回退')
        palette = Image.fromarray(self.pixels).quantize(colors=32)
        variants = [
            ('rgba', Image.fromarray(np.dstack((self.pixels, np.full(self.pixels.shape[:2], 119, np.uint8)))), {}),
            ('palette', palette, {}),
            ('gray', Image.fromarray(self.pixels[:, :, 0]), {}),
            ('16bit', Image.fromarray(np.arange(29 * 41, dtype=np.uint16).reshape(29, 41)), {}),
            ('icc', Image.fromarray(self.pixels), {'icc_profile': b'test source ICC profile'}),
            ('transparency', Image.fromarray(self.pixels), {'transparency': (1, 2, 3)}),
            ('text', Image.fromarray(self.pixels), {'pnginfo': text}),
        ]
        for name, image, options in variants:
            with self.subTest(name=name):
                self.write_png(image, **options)
                expected = self.export(self.old_region([0, 1]))
                result, _ = textured_region(self.mesh, [0, 1], self.source)
                self.assertIsNone(result.visual.material.baseColorTexture._encoded)
                self.assert_glb_equal(self.export(result), expected)
        with io.BytesIO() as stream:
            Image.fromarray(self.pixels).save(stream, format='JPEG')
            self.image_path.write_bytes(stream.getvalue())
        expected = self.export(self.old_region([0]))
        result, _ = textured_region(self.mesh, [0], self.source)
        self.assertIsNone(result.visual.material.baseColorTexture._encoded)
        self.assert_glb_equal(self.export(result), expected)

    def test_full_structure_lengths_crc_and_tail_are_required_for_direct_embedding(self):
        encoded = self.write_png()
        parts = chunks(encoded)
        self.assertTrue(_plain_rgb_png(encoded, (41, 29)))
        for cut in range(len(encoded)):
            self.assertFalse(_plain_rgb_png(encoded[:cut], (41, 29)))
        self.assertFalse(_plain_rgb_png(encoded + b'tail', (41, 29)))
        self.assertFalse(_plain_rgb_png(encoded, (40, 29)))
        for index in range(len(parts)):
            edited = [chunk(*part) for part in parts]
            corrupt = bytearray(edited[index])
            corrupt[-1] ^= 1
            edited[index] = bytes(corrupt)
            self.assertFalse(_plain_rgb_png(encoded[:8] + b''.join(edited), (41, 29)))
        header = parts[0][1]
        bad_headers = [header[:8] + bytes([depth, color, compression, filtering, interlace])
                       for depth, color, compression, filtering, interlace in
                       ((16, 2, 0, 0, 0), (8, 6, 0, 0, 0), (8, 2, 1, 0, 0),
                        (8, 2, 0, 1, 0), (8, 2, 0, 0, 1))]
        for header in bad_headers:
            altered = encoded[:8] + chunk(b'IHDR', header) + b''.join(chunk(*part) for part in parts[1:])
            self.assertFalse(_plain_rgb_png(altered, (41, 29)))
        for extra in (chunk(b'tEXt', b'x\0y'), chunk(b'IHDR', parts[0][1]),
                      struct.pack('>I', 0x80000000) + b'IDAT'):
            self.assertFalse(_plain_rgb_png(encoded[:33] + extra + encoded[33:], (41, 29)))

    def test_uncertain_but_loadable_pngs_reencode_instead_of_bypassing_pillow(self):
        encoded = self.write_png()
        parts = chunks(encoded)
        corrupted_end = bytearray(encoded)
        corrupted_end[-1] ^= 1
        variants = [encoded + b'trailing bytes', bytes(corrupted_end),
                    encoded[:33] + chunk(b'vpAg', b'private ancillary data') + encoded[33:]]
        # Pillow permits an IDAT CRC mismatch on load; direct byte reuse must not.
        edited = [chunk(*part) for part in parts]
        idat_index = next(index for index, part in enumerate(parts) if part[0] == b'IDAT')
        corrupt = bytearray(edited[idat_index])
        corrupt[-1] ^= 1
        edited[idat_index] = bytes(corrupt)
        variants.append(encoded[:8] + b''.join(edited))
        for data in variants:
            with self.subTest(data_sha=hashlib.sha256(data).hexdigest()):
                self.image_path.write_bytes(data)
                expected = self.export(self.old_region([0]))
                result, _ = textured_region(self.mesh, [0], self.source)
                self.assertIsNone(result.visual.material.baseColorTexture._encoded)
                self.assert_glb_equal(self.export(result), expected)

    def test_bad_images_fail_the_complete_original_decoder(self):
        encoded = self.write_png()
        for data in (b'not an image', encoded[:40], encoded[:len(encoded) // 2]):
            with self.subTest(length=len(data)):
                self.image_path.write_bytes(data)
                with self.assertRaises((OSError, ValueError, UnidentifiedImageError)):
                    self.old_region([0])
                with self.assertRaises((OSError, ValueError, UnidentifiedImageError)):
                    textured_region(self.mesh, [0], self.source)

    def test_incomplete_extra_and_wrong_length_zlib_streams_never_embed_directly(self):
        encoded = self.write_png()
        parts = chunks(encoded)
        compressed = b''.join(payload for kind, payload in parts if kind == b'IDAT')
        rows = zlib.decompress(compressed)
        self.assertEqual(len(rows), (41 * 3 + 1) * 29)
        variants = {
            'missing_adler_byte': compressed[:-1],
            'missing_adler_trailer': compressed[:-4],
            'junk_after_member': compressed + b'not-part-of-zlib-stream',
            'extra_zlib_member': compressed + zlib.compress(b'another member'),
            'short_scanlines': zlib.compress(rows[:-1]),
            'long_scanlines': zlib.compress(rows + b'\0'),
            'large_excess_scanlines': zlib.compress(rows + b'\0' * (256 * 1024)),
        }
        for name, stream in variants.items():
            with self.subTest(name=name):
                malformed = (encoded[:8] + chunk(b'IHDR', parts[0][1]) +
                             chunk(b'IDAT', stream) + chunk(b'IEND', b''))
                self.assertFalse(_plain_rgb_png(malformed, (41, 29)))
                self.image_path.write_bytes(malformed)
                try:
                    old = self.old_region([0, 1])
                except OSError as error:
                    # Some Pillow versions refuse short scanlines outright;
                    # the optimization must retain that decoder refusal too.
                    with self.assertRaises(type(error)):
                        textured_region(self.mesh, [0, 1], self.source)
                    continue
                expected = self.export(old)
                result, _ = textured_region(self.mesh, [0, 1], self.source)
                self.assertIsNone(result.visual.material.baseColorTexture._encoded)
                self.assert_glb_equal(self.export(result), expected)

    def test_zlib_validation_bounds_input_output_and_handles_split_idat(self):
        original_factory = zlib.decompressobj
        observed = []

        class BoundedInflater:
            def __init__(self):
                self.inner = original_factory()

            def decompress(self, data, max_length=0):
                self_outer.assertLessEqual(len(data), 16 * 1024)
                self_outer.assertGreater(max_length, 0)
                self_outer.assertLessEqual(max_length, 64 * 1024)
                output = self.inner.decompress(data, max_length)
                self_outer.assertLessEqual(len(output), max_length)
                observed.append((len(data), len(output)))
                return output

            def __getattr__(self, name):
                return getattr(self.inner, name)

        self_outer = self
        images = [np.zeros((500, 511, 3), np.uint8),
                  np.random.default_rng(612).integers(0, 256, (128, 257, 3), dtype=np.uint8)]
        for pixels in images:
            image = Image.fromarray(pixels)
            encoded = png_bytes(image)
            parts = chunks(encoded)
            compressed = b''.join(payload for kind, payload in parts if kind == b'IDAT')
            for split in (1, 16383, len(compressed)):
                with self.subTest(shape=pixels.shape, split=split):
                    # Empty chunks and a trailer split across IDATs are valid.
                    blocks = [chunk(b'IDAT', compressed[start:start + split])
                              for start in range(0, len(compressed), split)]
                    rebuilt = (encoded[:8] + chunk(b'IHDR', parts[0][1]) + chunk(b'IDAT', b'') +
                               b''.join(blocks) + chunk(b'IDAT', b'') + chunk(b'IEND', b''))
                    with patch('femto.geometry.png_texture.zlib.decompressobj', BoundedInflater):
                        self.assertTrue(_plain_rgb_png(rebuilt, image.size))
        self.assertTrue(any(output == 64 * 1024 for _, output in observed))
        self.assertTrue(any(input == 16 * 1024 for input, _ in observed))
        observed.clear()
        encoded = png_bytes(Image.fromarray(self.pixels))
        header = chunks(encoded)[0][1]
        excessive = (encoded[:8] + chunk(b'IHDR', header) +
                     chunk(b'IDAT', zlib.compress(b'\0' * (1024 * 1024))) + chunk(b'IEND', b''))
        with patch('femto.geometry.png_texture.zlib.decompressobj', BoundedInflater):
            self.assertFalse(_plain_rgb_png(excessive, (41, 29)))
        self.assertEqual(sum(output for _, output in observed), (41 * 3 + 1) * 29 + 1)

    def test_pixels_are_immutable_and_options_keep_original_pillow_encoding(self):
        self.write_png()
        texture, _ = load_source_texture(self.image_path)
        pixels = np.asarray(texture)
        np.testing.assert_array_equal(pixels, self.pixels)
        with self.assertRaises(ValueError):
            pixels.setflags(write=True)
        copy = np.array(texture, copy=True)
        copy[:] = 0
        np.testing.assert_array_equal(np.asarray(texture), self.pixels)
        with self.assertRaises(ValueError):
            texture.__array__(dtype=np.float64, copy=False)
        for format, options in (('PNG', {'compress_level': 0}), ('BMP', {})):
            with self.subTest(format=format), io.BytesIO() as actual, io.BytesIO() as expected:
                texture.save(actual, format=format, **options)
                Image.fromarray(self.pixels).save(expected, format=format, **options)
                self.assertEqual(actual.getvalue(), expected.getvalue())

    def test_same_path_same_length_restored_mtime_never_reuses_old_pixels(self):
        first = self.write_png(compress_level=0)
        first_stat = self.image_path.stat()
        result, metadata = textured_region(self.mesh, [0], self.source)
        replacement = png_bytes(Image.fromarray(self.pixels ^ 255), compress_level=0)
        self.assertEqual(len(first), len(replacement))
        self.image_path.write_bytes(replacement)
        os.utime(self.image_path, ns=(first_stat.st_atime_ns, first_stat.st_mtime_ns))
        self.assertEqual(self.image_path.stat().st_mtime_ns, first_stat.st_mtime_ns)
        with self.assertRaisesRegex(ValueError, 'Source texture changed'):
            self.export(result)
        current, current_metadata = textured_region(self.mesh, [0], self.source)
        self.assertNotEqual(metadata['texture_sha256'], current_metadata['texture_sha256'])
        self.assertEqual(current_metadata['texture_sha256'], hashlib.sha256(replacement).hexdigest())
        np.testing.assert_array_equal(np.asarray(current.visual.material.baseColorTexture), self.pixels ^ 255)
        self.assert_glb_equal(self.export(current), self.export(self.old_region([0])), byte_equal=False)

    def test_source_changes_during_decode_or_during_output_are_rejected(self):
        first = self.write_png()
        replacement = png_bytes(Image.fromarray(self.pixels ^ 255))
        original = _plain_rgb_png

        def replace_while_loading(data, size):
            self.image_path.write_bytes(replacement)
            return original(data, size)

        with patch('femto.geometry.png_texture._plain_rgb_png', side_effect=replace_while_loading):
            with self.assertRaisesRegex(ValueError, 'Source texture changed'):
                load_source_texture(self.image_path)
        self.image_path.write_bytes(first)
        texture, _ = load_source_texture(self.image_path)
        source = self.image_path

        class ChangingOutput(io.BytesIO):
            def write(self, value):
                source.write_bytes(replacement)
                return super().write(value)

        with ChangingOutput() as stream:
            with self.assertRaisesRegex(ValueError, 'Source texture changed'):
                texture.save(stream, format='PNG')

    def test_fallback_image_also_rejects_source_change_before_export(self):
        self.write_png(Image.fromarray(self.pixels).convert('RGBA'))
        result, _ = textured_region(self.mesh, [0], self.source)
        self.write_png()
        with self.assertRaisesRegex(ValueError, 'Source texture changed'):
            self.export(result)


if __name__ == '__main__':
    unittest.main()
