"""Embed a fully decoded, ordinary source PNG without encoding it again.

This is a per-image byte snapshot, not a cache or a validation receipt. Special
PNG features and other image formats keep the existing Pillow RGB encoding.
"""
import hashlib
import io
from pathlib import Path
import struct
import zlib

import numpy as np
from PIL import Image

from femto.io_utils import sha256_file


def _complete_rgb_stream(data, width, height):
    """Check the complete zlib member without retaining an expanded image.

    This is called only after chunk structure/CRCs have been checked. Pillow
    can decode pixels even when the Adler trailer is missing or extra data
    follows the zlib member, so successful image.load alone is insufficient.
    """
    expected = (width * 3 + 1) * height
    decoded = 0
    inflater = zlib.decompressobj()
    view = memoryview(data)
    offset = 8
    try:
        while offset < len(data):
            length = struct.unpack_from('>I', data, offset)[0]
            if data[offset + 4:offset + 8] == b'IDAT':
                end = offset + 8 + length
                for start in range(offset + 8, end, 16 * 1024):
                    pending = view[start:min(start + 16 * 1024, end)]
                    while True:
                        if inflater.eof:
                            if pending:
                                return False
                            break
                        limit = min(64 * 1024, expected - decoded + 1)
                        output = inflater.decompress(pending, limit)
                        decoded += len(output)
                        if decoded > expected or inflater.unused_data:
                            return False
                        pending = inflater.unconsumed_tail
                        # A full output block may leave buffered output even
                        # without input tail; drain it with the same bound.
                        if not pending and len(output) < limit:
                            break
            offset += 12 + length
    except zlib.error:
        return False
    return inflater.eof and not inflater.unused_data and decoded == expected


def _plain_rgb_png(data, size):
    """Accept only complete, CRC-checked 8-bit noninterlaced RGB PNGs."""
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):
        return False
    view = memoryview(data)
    offset = 8
    seen_header = seen_data = False
    while offset + 12 <= len(data):
        length = struct.unpack_from('>I', data, offset)[0]
        end = offset + 12 + length
        if length > 0x7fffffff or end > len(data):
            return False
        kind = data[offset + 4:offset + 8]
        payload = view[offset + 8:offset + 8 + length]
        crc = zlib.crc32(payload, zlib.crc32(kind)) & 0xffffffff
        if crc != struct.unpack_from('>I', data, end - 4)[0]:
            return False
        if not seen_header:
            if kind != b'IHDR' or length != 13:
                return False
            width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', payload)
            if ((width, height) != size or not width or not height or
                    (depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0)):
                return False
            seen_header = True
        elif kind == b'IDAT':
            seen_data = True
        elif kind == b'IEND':
            return (seen_data and length == 0 and end == len(data) and
                    _complete_rgb_stream(data, width, height))
        else:
            return False
        offset = end
    return False


class _SourceTexture:
    """Trimesh's pixel-array/save protocols, with a current-source guard."""
    format = 'PNG'

    def __init__(self, source, digest, atlas, encoded):
        self._source = source
        self._digest = digest
        self._encoded = encoded
        self._image = atlas if encoded is None else None
        # A bytes-backed array cannot be made writable through an exposed view.
        self._pixels = (np.frombuffer(atlas.tobytes(), dtype=np.uint8)
                        .reshape(atlas.height, atlas.width, 3)) if encoded is not None else None

    def _require_current(self):
        if sha256_file(self._source) != self._digest:
            raise ValueError('Source texture changed during region export')

    def __array__(self, dtype=None, copy=None):
        pixels = self._pixels if self._pixels is not None else np.asarray(self._image)
        if copy is False and dtype is not None and np.dtype(dtype) != pixels.dtype:
            raise ValueError('A dtype conversion requires a texture copy')
        value = np.asarray(pixels, dtype=dtype)
        return value.copy() if copy else value.view()

    def save(self, stream, format=None, **options):
        self._require_current()
        if (self._encoded is not None and isinstance(format, str) and
                format.upper() == 'PNG' and not options):
            stream.write(self._encoded)
        else:
            atlas = self._image if self._image is not None else Image.fromarray(self._pixels)
            atlas.save(stream, format=format, **options)
        # Also reject a source edit occurring while the encoder/writer runs.
        self._require_current()


def load_source_texture(path):
    """Return RGB texture and SHA of the same bytes actually decoded.

    Always perform Pillow's complete load and the existing RGB conversion. Bad
    images still fail there; an uncertain byte layout only disables direct PNG
    embedding. Each call rereads its source, and each export rehashes it.
    """
    path = Path(path)
    encoded = path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    with Image.open(io.BytesIO(encoded)) as image:
        image.load()
        ordinary = (image.format == 'PNG' and image.mode == 'RGB' and
                    not image.info and _plain_rgb_png(encoded, image.size))
        atlas = image.convert('RGB').copy()
    texture = _SourceTexture(path, digest, atlas, encoded if ordinary else None)
    texture._require_current()
    return texture, digest
