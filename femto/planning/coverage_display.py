"""Bounded display-only projections of already verified coverage evidence.

Float triangles are visual approximations of the exact proof leaves. Reference
pixels turn green only when every associated leaf is confirmed and the pixel
has no unlocated remainder. Neither representation is fed back into planning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femto.planning.coverage_tasks import CoverageTasks

NAMES = {0:'全部状态', 1:'模型内已确认', 2:'质量排除或待检查', 3:'种子内球未触及',
         4:'球内连通未确认', 5:'细分到限', 6:'目标构造未确认', 7:'预算内未处理', 8:'目标含未定位范围'}
COLORS = np.array([[0,0,0,0], [30,198,137,165], [170,109,226,170], [245,156,64,165],
                   [244,188,72,175], [235,128,67,175], [217,95,191,175], [169,177,192,175],
                   [200,114,230,185]], dtype=np.uint8)
CHILDREN = np.array([[[1,0,0],[.5,.5,0],[.5,0,.5]],
                     [[.5,.5,0],[0,1,0],[0,.5,.5]],
                     [[.5,0,.5],[0,.5,.5],[0,0,1]],
                     [[.5,.5,0],[0,.5,.5],[.5,0,.5]]])


@dataclass(frozen=True)
class CoverageDisplay:
    certificate_sha256: str
    triangles: np.ndarray
    status: np.ndarray
    source_faces: np.ndarray
    pixel_bits: np.ndarray
    pixel_state: np.ndarray
    tasks: CoverageTasks
    reference_status: np.ndarray

    def mask(self, reason=0, *, pixel_reason=None):
        if pixel_reason is not None:
            if reason != 8 or type(pixel_reason) is not int or not 2 <= pixel_reason <= 10:
                raise ValueError('未定位像素原因无效')
            return self.mask(8) & (self.reference_status == pixel_reason)
        if reason == 0:return self.pixel_state != 0
        if reason == 1:return self.pixel_state == 1
        return (self.pixel_bits & (1 << reason)) != 0

    def rgba(self, reason=0, *, pixel_reason=None):
        if reason == 0 and pixel_reason is None:return COLORS[self.pixel_state]
        result = np.zeros((*self.pixel_state.shape, 4), np.uint8)
        result[self.mask(reason, pixel_reason=pixel_reason)] = COLORS[reason]
        return result


def build(ledger, certificate, check=lambda:None):
    from femto.planning.coverage_tasks import build as build_tasks
    tasks = build_tasks(ledger, certificate, check)
    if certificate.metadata['inputs']['domain'].get('target_sha256') != ledger.metadata['sha256']:
        raise ValueError('覆盖显示资产与原目标台账不一致')
    arrays = certificate.arrays
    root, depth, code = arrays['root'], arrays['depth'], arrays['code']
    check()
    source_faces = ledger.arrays['source_face'][root]
    original = ledger.arrays['vertices'][ledger.arrays['faces'][source_faces]]
    triangles = np.einsum('nij,njk->nik', ledger.arrays['barycentric'][root], original)
    for level in range(int(depth.max(initial=0))):
        check()
        ids = np.flatnonzero(depth > level)
        digit = (code[ids] >> (2*(depth[ids].astype(np.int64)-level-1))) & 3
        triangles[ids] = np.einsum('nij,njk->nik', CHILDREN[digit], triangles[ids])
    pixel_bits = np.zeros(ledger.arrays['mask'].shape, '<u2')
    pixels = ledger.arrays['pixel_index'][root]
    for state in range(1, 8):
        check()
        np.bitwise_or.at(pixel_bits.ravel(), pixels[arrays['status'] == state], 1 << state)
    unknown = ledger.arrays['mask'] & (ledger.arrays['pixel_status'] != 1)
    pixel_bits[unknown] |= 1 << 8
    pixel_state = np.zeros(pixel_bits.shape, 'u1')
    # Later reasons take precedence, so partial coverage never paints a pixel
    # fully green. Filtering still exposes each reason independently via bits.
    for state in (1,3,4,5,7,2,6,8):pixel_state[(pixel_bits & (1 << state)) != 0] = state
    status = arrays['status'].copy()
    reference_status = ledger.arrays['pixel_status'].copy()
    for value in (triangles, status, source_faces, pixel_bits, pixel_state, reference_status):value.setflags(write=False)
    check()
    return CoverageDisplay(certificate.metadata['sha256'], triangles, status, source_faces, pixel_bits, pixel_state, tasks, reference_status)
