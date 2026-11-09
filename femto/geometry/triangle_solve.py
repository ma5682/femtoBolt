"""Shared, guarded least-squares maps for interior triangle predicates.

Only an immutable matrix decomposition is retained; query coordinates and
acceptance results are not cached. The map can certify points well inside the
old predicates. Uncertain, singular and poorly conditioned cases are left to
the original ``lstsq`` call and its original comparisons.
"""
from __future__ import annotations

import math
import numpy as np

_EPS = np.finfo(np.float64).eps
_SQRT3 = float(np.sqrt(3.))


class TriangleSolve:
    """Lazily prepare one 3-by-2 matrix, shared by point and chord queries."""
    __slots__ = ('matrix', '_ready', '_inverse', '_guard', '_inverse_norm', '_matrix_norm',
                 '_shared_preparation')

    def __init__(self, matrix, *, _shared_preparation=None):
        self.matrix = matrix
        self._ready = False
        self._inverse = None
        self._guard = self._inverse_norm = self._matrix_norm = 0.
        self._shared_preparation = _shared_preparation

    def _prepare(self):
        if self._ready:
            return
        self._ready = True
        if self._shared_preparation is not None:
            bound_matrix, obtain = self._shared_preparation
            if self.matrix is bound_matrix:
                try:
                    record = obtain()
                    if record is not None:
                        inverse, guard, inverse_norm, matrix_norm = record
                        inverse = (None if inverse is None else
                                   np.frombuffer(inverse, dtype=np.float64).reshape(2, 3))
                        self._inverse, self._guard = inverse, guard
                        self._inverse_norm, self._matrix_norm = inverse_norm, matrix_norm
                        return
                except MemoryError:
                    # The bounded shared copy is optional. Original local
                    # preparation remains available without publishing a row.
                    pass
        # The envelope describes binary64 arithmetic. Keep other input dtypes
        # on their original NumPy path (notably float32 rank/rounding choices).
        if self.matrix.dtype != np.dtype(np.float64):
            return
        a0, b0, a1, b1, a2, b2 = self.matrix.ravel().tolist()
        if not all(math.isfinite(value) for value in (a0, a1, a2, b0, b1, b2)):
            return
        # Thin QR for exactly two columns. Scalar arithmetic avoids starting a
        # new tiny SVD and allocating its work arrays for every new mesh face.
        # hypot scales its inputs, retaining the old path for unrepresentable
        # or singular factors. Only a well-conditioned full-rank map is used.
        r00 = math.hypot(a0, a1, a2)
        if not math.isfinite(r00) or r00 <= 0:
            return
        q0, q1, q2 = a0/r00, a1/r00, a2/r00
        r01 = q0*b0 + q1*b1 + q2*b2
        v0, v1, v2 = b0-r01*q0, b1-r01*q1, b2-r01*q2
        r11 = math.hypot(v0, v1, v2)
        if not math.isfinite(r11) or r11 <= 0:
            return
        second = ((v0/r11)/r11, (v1/r11)/r11, (v2/r11)/r11)
        first = ((q0-r01*second[0])/r00, (q1-r01*second[1])/r00, (q2-r01*second[2])/r00)
        if not all(math.isfinite(value) for value in (*first, *second)):
            return
        # Frobenius norm product is an upper bound on the 2-norm condition
        # number (kappa + 1/kappa for an exact rank-two map). The deliberately
        # conservative condition-squared envelope covers QR, the independent
        # legacy SVD solve and subsequent products. Near-rank-deficient faces
        # cannot pass this gate and retain the original rcond/rank behavior.
        condition = math.hypot(a0, a1, a2, b0, b1, b2) * math.hypot(*first, *second)
        # Inflate before using the computed upper bound; the extra term covers
        # roundoff in the two-column factorization and norm product itself.
        condition = math.nextafter(condition*(1 + 64*_EPS*condition*condition), math.inf)
        guard = 4096 * _EPS * condition * condition
        if not math.isfinite(guard) or guard > 1e-6:
            return
        inverse = np.array((first, second), dtype=np.float64)
        inverse.setflags(write=False)
        self._inverse = inverse
        self._guard = guard
        self._inverse_norm = max(sum(map(abs, first)), sum(map(abs, second)))
        self._matrix_norm = max(abs(a0)+abs(b0), abs(a1)+abs(b1), abs(a2)+abs(b2))

    def coordinates(self, offsets):
        """Return N-by-2 coefficients and per-query uncertainty, or None.

        A batch shares one decomposition without changing the scalar fallback.
        The additive unit term covers coefficients close to zero; the infinity
        norms cover amplification from large offsets and cancellation.
        """
        self._prepare()
        if self._inverse is None:
            return None
        offsets = np.asarray(offsets)
        if offsets.dtype != np.dtype(np.float64) or not np.isfinite(offsets).all():
            return None
        coordinates = offsets @ self._inverse.T
        uncertainty = self._guard * (1 + np.max(np.abs(offsets), axis=-1) * self._inverse_norm)
        return coordinates, uncertainty

    def point_inside(self, xyz, prepared, tolerance_m):
        """Certify strictly interior points; False means use the old predicate."""
        xyz = np.asarray(xyz, dtype=np.float64)
        offsets = xyz - prepared.origin
        result = self.coordinates(offsets)
        if result is None:
            return np.zeros(xyz.shape[:-1], dtype=bool)
        coordinates, uncertainty = result
        tolerance = tolerance_m / max(prepared.minimum_edge_length, tolerance_m)
        closest = (prepared.origin + coordinates[..., 0, None] * prepared.ab
                   + coordinates[..., 1, None] * prepared.ac)
        rounding = 64 * _EPS * (
            np.max(np.abs(xyz), axis=-1) + np.max(np.abs(prepared.origin))
            + self._matrix_norm * np.max(np.abs(coordinates), axis=-1))
        residual_guard = _SQRT3 * (2 * self._matrix_norm * uncertainty + rounding)
        # The original separate plane check also remains mandatory. A batch
        # uses a guarded dot product; a point near its threshold falls back.
        normal_dot = np.abs(offsets @ prepared.normal)
        normal_guard = 32 * _EPS * (np.abs(offsets) @ np.abs(prepared.normal))
        return ((prepared.normal_length > 0)
                & np.isfinite(prepared.normal_length)
                & (normal_dot + normal_guard < tolerance_m * prepared.normal_length)
                & (np.min(coordinates, axis=-1) > -tolerance + uncertainty)
                & (coordinates.sum(axis=-1) < 1 + tolerance - 2 * uncertainty)
                & (np.linalg.norm(closest - xyz, axis=-1) + residual_guard < tolerance_m))

    def point_inside_one(self, xyz, offset, prepared, tolerance_m):
        """Scalar companion used after the unchanged scalar plane check."""
        self._prepare()
        if self._inverse is None:
            return False
        u, v = self._inverse @ offset
        magnitude = max(abs(float(offset[0])), abs(float(offset[1])), abs(float(offset[2])))
        uncertainty = self._guard * (1 + magnitude * self._inverse_norm)
        tolerance = tolerance_m / max(prepared.minimum_edge_length, tolerance_m)
        if not (min(u, v) > -tolerance + uncertainty and u+v < 1+tolerance-2*uncertainty):
            return False
        rounding = 64 * _EPS * (
            max(abs(float(x)) for x in xyz) + max(abs(float(x)) for x in prepared.origin)
            + self._matrix_norm * max(abs(u), abs(v)))
        residual_guard = _SQRT3 * (2*self._matrix_norm*uncertainty + rounding)
        closest = prepared.origin + prepared.ab*u + prepared.ac*v
        return bool(np.linalg.norm(closest-xyz) + residual_guard < tolerance_m)

    def chord_inside_one(self, offsets):
        """Scalar two-RHS companion after the original chord plane check."""
        self._prepare()
        if self._inverse is None or offsets.dtype != np.dtype(np.float64):
            return False
        coordinates = self._inverse @ offsets
        for i in (0, 1):
            u, v = coordinates[:, i]
            magnitude = max(abs(float(offsets[0, i])), abs(float(offsets[1, i])), abs(float(offsets[2, i])))
            uncertainty = self._guard*(1 + magnitude*self._inverse_norm)
            if not (min(u, v) > -1e-8+uncertainty and u+v < 1+1e-8-2*uncertainty):
                return False
        return True

    def chord_inside(self, starts, ends, prepared):
        """Certify a whole chord on this one face, never a cross-face shortcut.

        Barycentric coordinates are affine along a chord, so both endpoints
        strictly within every old half-space imply the complete interval [0,1].
        Cases needing an interval union still use the original scalar routine.
        """
        starts, ends = np.asarray(starts), np.asarray(ends)
        offsets = np.stack((starts-prepared.origin, ends-prepared.origin), axis=1)
        result = self.coordinates(offsets)
        if result is None:
            return np.zeros(len(starts), dtype=bool)
        coordinates, uncertainty = result
        normal_dot = np.abs(offsets @ prepared.normal)
        normal_guard = 32 * _EPS * (np.abs(offsets) @ np.abs(prepared.normal))
        inside = ((np.min(coordinates, axis=-1) > -1e-8 + uncertainty)
                  & (coordinates.sum(axis=-1) < 1 + 1e-8 - 2 * uncertainty)
                  & (normal_dot + normal_guard < 1e-6 * prepared.normal_length))
        return ((prepared.normal_length > 1e-15)
                & np.isfinite(prepared.normal_length) & inside.all(axis=1))
