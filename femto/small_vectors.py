"""Exact batched forms of NumPy's dot and norm for 2/3-element float64 rows.

``np.dot``/``np.linalg.norm`` on one short vector call BLAS ``ddot``; whether
its scalar tail loop fuses multiply-adds is fixed by the BLAS build and CPU
kernel. A process-wide self-test compares that call with the plain
left-to-right sums on vectors where fused and unfused results differ. Only
when every probe matches are the batched elementwise forms used; otherwise the
original per-row calls run. Results are bit-identical either way.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

_PLAIN: bool | None = None


def _fused_differs(values):
    plain = 0.
    exact = Fraction(0)
    for value in values:
        plain += value*value
        exact = Fraction(float(exact)) + Fraction(value)*Fraction(value)
    return plain != float(exact)


def _probe() -> bool:
    rng = np.random.default_rng(20261105)
    tested = 0
    with np.errstate(all='ignore'):
        for size in (2, 3):
            rows = rng.standard_normal((1024, size))*rng.choice([1e-4, 1e-2, 1., 1e2], (1024, 1))
            strided = np.empty((1024, size*2))
            strided[:, ::2] = rows
            for row, view in zip(rows, strided):
                values = row.tolist()
                plain = 0.
                for value in values:
                    plain += value*value
                if float(np.dot(row, row)) != plain or float(np.dot(view[::2], view[::2])) != plain:
                    return False
                if float(np.linalg.norm(row)) != math.sqrt(plain):
                    return False
                if _fused_differs(values):
                    tested += 1
    # The probe must have exercised cases where fusion would be visible.
    return tested >= 256


def plain_small_dot() -> bool:
    global _PLAIN
    if _PLAIN is None:
        try:
            _PLAIN = _probe()
        except Exception:
            _PLAIN = False
    return _PLAIN


def _usable(rows) -> bool:
    return (type(rows) is np.ndarray and rows.dtype == np.float64 and rows.ndim == 2
            and rows.shape[1] in (2, 3) and np.geterr() == dict(divide='warn', over='warn', under='ignore', invalid='warn')
            and (not rows.size or (np.isfinite(rows).all() and np.abs(rows).max() < 1e150)))


def row_dots(left, right) -> np.ndarray:
    """``[np.dot(a, b) for a, b in zip(left, right)]`` as one float64 array."""
    if (plain_small_dot() and _usable(left) and _usable(right) and left.shape == right.shape):
        # BLAS starts its reduction at +0, including an all-negative-zero
        # product vector. Omitting that seed changes serialized float bytes.
        total = 0. + left[:, 0]*right[:, 0]
        total = total+left[:, 1]*right[:, 1]
        if left.shape[1] == 3:
            total = total+left[:, 2]*right[:, 2]
        return total
    return np.array([np.dot(a, b) for a, b in zip(left, right)], dtype=np.float64)


def row_norms(rows) -> np.ndarray:
    """``[np.linalg.norm(row) for row in rows]`` as one float64 array."""
    if plain_small_dot() and _usable(rows):
        return np.sqrt(row_dots(rows, rows))
    return np.array([np.linalg.norm(row) for row in rows], dtype=np.float64)
