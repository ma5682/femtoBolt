"""Bounded, operation-local reuse of original RGB-D and depth diagnostics.

The 128 MiB budget covers retained NumPy payloads (not the existing pipeline's
buffers or transient image decoding). At most 12 raw views and 128 jump masks
are retained. Encoded files above 32 MiB bypass capture. No result is persisted
or accepted across reconstruction calls; a miss follows the original path.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import cv2
import numpy as np

from femto.io_utils import ImageCodecError, imread_unicode


MAX_RETAINED_BYTES = 128 * 1024 * 1024
MAX_ENCODED_FILE_BYTES = 32 * 1024 * 1024
MAX_RAW_VIEWS = 12
MAX_JUMP_MASKS = 128


def texture_view_indices(frame_count, max_views=MAX_RAW_VIEWS):
    return np.unique(np.linspace(0, frame_count - 1, min(max_views, frame_count), dtype=int))


def depth_jump_mask(depth):
    """The existing >30 mm rule, valid on both sides, marking both endpoints."""
    signed = np.asarray(depth).astype(np.int32)
    valid = np.asarray(depth) > 0
    horizontal = valid[:, 1:] & valid[:, :-1] & (np.abs(signed[:, 1:] - signed[:, :-1]) > 30)
    vertical = valid[1:] & valid[:-1] & (np.abs(signed[1:] - signed[:-1]) > 30)
    jump = np.zeros(depth.shape, bool)
    jump[:, 1:] |= horizontal
    jump[:, :-1] |= horizontal
    jump[1:] |= vertical
    jump[:-1] |= vertical
    return jump


def _file_identity(path):
    # Include the resolved target and file identity, not only size/mtime: a
    # replaced file or retargeted link must not inherit a previous decode.
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return (str(resolved), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def _readonly_view(array):
    array.setflags(write=False)
    return array.view()


def _depth_digest(depth):
    return hashlib.sha256(np.ascontiguousarray(depth).view(np.uint8)).digest()


def _encoded_digest(path):
    # Bounded streaming verification also catches edits that restore mtime,
    # including Windows files whose ctime denotes creation rather than change.
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@dataclass(frozen=True)
class _RawRgbd:
    paths: tuple
    identities: tuple
    color: np.ndarray
    depth: np.ndarray
    rgb_sha256: str
    depth_sha256: str

    @property
    def nbytes(self):
        return self.color.nbytes + self.depth.nbytes


class ReconstructionReuse:
    """One reconstruction owns this instance; no global or cross-task cache.

    Source identities are checked before/after capture and again before use;
    encoded SHA verification at use catches changes even with restored times.
    RGB SHA is computed from the very bytes decoded, before they are released.
    Changed/missing sources evict the snapshot; texture export reads them anew.
    Masks also validate source array identity and content, so later in-memory
    edits cannot reuse obsolete diagnostics. Exception/cancellation propagates.
    """
    def __init__(self, dataset, selected_views, max_bytes=MAX_RETAINED_BYTES):
        self.dataset = Path(dataset).resolve()
        self.selected_views = frozenset(int(i) for i in selected_views)
        self.max_bytes = min(MAX_RETAINED_BYTES, max(0, int(max_bytes)))
        self.retained_bytes = 0
        self._raw = {}
        self._jumps = {}

    def _paths(self, frame):
        return tuple(str(frame[key]) for key in ('rgb', 'depth'))

    def load_rgbd(self, ordinal, frame, expected_shape):
        """Load raw integration inputs, retaining only selected original views."""
        paths = self._paths(frame)
        targets = tuple(self.dataset / path for path in paths)
        predicted_bytes = int(np.prod(expected_shape)) * 5  # uint8 BGR + uint16 depth
        capture = (ordinal in self.selected_views and len(self._raw) < MAX_RAW_VIEWS
                   and self.retained_bytes + predicted_bytes <= self.max_bytes)
        if capture:
            identities = tuple(_file_identity(path) for path in targets)
            capture = all(identity[3] <= MAX_ENCODED_FILE_BYTES for identity in identities)
        if not capture:
            return (imread_unicode(targets[0], cv2.IMREAD_COLOR),
                    imread_unicode(targets[1], cv2.IMREAD_UNCHANGED))
        images = []
        digests = []
        for index, (path, flags) in enumerate(zip(targets, (cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED))):
            encoded = np.fromfile(path, dtype=np.uint8, count=identities[index][3] + 1)
            decoded = cv2.imdecode(encoded, flags) if encoded.size else None
            if decoded is None:
                raise ImageCodecError(f'Cannot decode image: {path}')
            digests.append(hashlib.sha256(encoded).hexdigest())
            images.append(decoded)
            del encoded
        # A concurrently replaced/edited input is never retained under an old
        # signature. Follow the original reader if capture was not stable.
        if identities != tuple(_file_identity(path) for path in targets):
            return (imread_unicode(targets[0], cv2.IMREAD_COLOR),
                    imread_unicode(targets[1], cv2.IMREAD_UNCHANGED))
        nbytes = sum(image.nbytes for image in images)
        if (images[0].shape != (*expected_shape, 3) or images[1].shape != tuple(expected_shape)
                or images[1].dtype != np.uint16 or self.retained_bytes + nbytes > self.max_bytes):
            return tuple(images)
        snapshot = _RawRgbd(paths, identities, *(_readonly_view(image) for image in images), *digests)
        previous = self._raw.pop(int(ordinal), None)
        if previous is not None:
            self.retained_bytes -= previous.nbytes
        self._raw[int(ordinal)] = snapshot
        self.retained_bytes += snapshot.nbytes
        return snapshot.color, snapshot.depth

    def get_rgbd(self, dataset, ordinal, frame):
        snapshot = self._raw.get(int(ordinal))
        if snapshot is None:
            return None
        matches = Path(dataset).resolve() == self.dataset and self._paths(frame) == snapshot.paths
        if matches:
            try:
                targets = tuple(self.dataset / path for path in snapshot.paths)
                matches = snapshot.identities == tuple(_file_identity(path) for path in targets)
                if matches:
                    matches = ((snapshot.rgb_sha256, snapshot.depth_sha256)
                               == tuple(_encoded_digest(path) for path in targets)
                               and snapshot.identities == tuple(_file_identity(path) for path in targets))
            except OSError:
                matches = False
        if not matches:
            self.retained_bytes -= self._raw.pop(int(ordinal)).nbytes
            return None
        return snapshot

    def remember_jump(self, ordinal, depth, jump):
        if (len(self._jumps) >= MAX_JUMP_MASKS or int(ordinal) in self._jumps
                or self.retained_bytes + jump.nbytes > self.max_bytes):
            return
        # Keep the original source alive to prevent Python id reuse, but do not
        # duplicate the prepared-depth buffer already retained by the pipeline.
        self._jumps[int(ordinal)] = (depth, depth.dtype.str, _depth_digest(depth), _readonly_view(jump))
        self.retained_bytes += jump.nbytes

    def get_jump(self, ordinal, depth):
        entry = self._jumps.get(int(ordinal))
        if entry is None:
            return None
        source, dtype, digest, jump = entry
        if (source is not depth or source.shape != jump.shape or depth.dtype.str != dtype
                or digest != _depth_digest(depth)):
            self.retained_bytes -= self._jumps.pop(int(ordinal))[3].nbytes
            return None
        return jump
