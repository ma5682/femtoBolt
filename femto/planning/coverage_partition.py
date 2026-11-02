"""Compare dyadic coverage sets without subtracting differently partitioned areas."""
from __future__ import annotations

import math
import numpy as np

from femto.planning.planning_revision import digest

DEPTH = 8
UNITS = 4**DEPTH
MAX_UNION_CERTIFICATES = 4


def intervals(certificate, roots: int, check=lambda: None) -> list[tuple]:
    """A quadtree prefix is an exact integer interval, independent of XYZ rounding."""
    a = certificate.arrays
    fields=[a.get(key) for key in ('root','depth','code','status')]
    if (any(not isinstance(value,np.ndarray) or value.ndim!=1 or not np.issubdtype(value.dtype,np.integer)
            for value in fields) or len({len(value) for value in fields})!=1):
        raise ValueError('覆盖增益比较分区数组长度或类型无效')
    result = []
    previous = (0, 0)
    for index, (root, depth, code, state) in enumerate(zip(a['root'], a['depth'], a['code'], a['status'])):
        if index % 1024 == 0:check()
        root, depth, code, state = int(root), int(depth), int(code), int(state)
        if not 0 <= root < roots or not 0 <= depth <= DEPTH or not 0 <= code < 4**depth or state not in range(1, 8):
            raise ValueError('覆盖增益比较遇到无效叶单元')
        length = 4**(DEPTH-depth)
        lo, hi = code*length, (code+1)*length
        if (root, lo) != previous:raise ValueError('覆盖增益比较要求完整且不重叠的原目标分区')
        result.append((root, lo, hi, state == 1))
        previous = (root+1, 0) if hi == UNITS else (root, hi)
    if previous != (roots, 0):raise ValueError('覆盖增益比较缺少原目标范围')
    return result


def compare(before, after, areas: np.ndarray, check=lambda: None, *, new_witnesses: np.ndarray | None = None) -> dict:
    """Intersect integer intervals; numerical areas are weights, not physical bounds."""
    left = intervals(before, len(areas), check)
    right = intervals(after, len(areas), check)
    retained, gained, lost, new_scan = [np.zeros(len(areas), np.int64) for _ in range(4)]
    if new_witnesses is not None and (new_witnesses.shape != (len(right),) or new_witnesses.dtype != bool):
        raise ValueError('新增扫描见证与覆盖叶单元不一致')
    i = j = 0
    while i < len(left) and j < len(right):
        if (i+j) % 1024 == 0:check()
        root, lo, hi, old = left[i]
        other, low, high, new = right[j]
        if root != other:raise ValueError('覆盖比较的原目标身份不一致')
        size = min(hi, high)-max(lo, low)
        if size <= 0:raise ValueError('覆盖比较分区顺序无效')
        if old and new:retained[root] += size
        elif new:
            gained[root] += size
            if new_witnesses is not None and new_witnesses[j]:new_scan[root] += size
        elif old:lost[root] += size
        if hi <= high:i += 1
        if high <= hi:j += 1
    def area(weights):return math.fsum(float(a)*(int(w)/UNITS) for a, w in zip(areas, weights))
    return dict(method='common-dyadic-prefix-partition-v1', comparison_depth=DEPTH,
                previous_set_sha256=digest([r[:3] for r in left if r[3]]),
                trial_set_sha256=digest([r[:3] for r in right if r[3]]),
                retained_nominal_area_m2=area(retained), gained_nominal_area_m2=area(gained),
                lost_nominal_area_m2=area(lost), retained_subset=not bool(lost.any()),
                gained_weight_units=int(gained.sum()), lost_weight_units=int(lost.sum()),
                new_scan_witness_gain_m2=area(new_scan), new_scan_witness_weight_units=int(new_scan.sum()),
                area_semantics='original_nominal_root_weights_not_physical_accuracy')


def compare_union(before, after, areas: np.ndarray, check=lambda: None, *, include_pending_leaves=False) -> dict:
    """Compare one old set with a union of independently supplied sets.

    This is only an exact partition operation. Callers must separately replay
    every proof and bind it to the same authoritative target, current route,
    seeds and footprint; matching array sizes do not establish that identity.
    Unknown cells remain unknown and no certificate is created here.
    """
    if (type(include_pending_leaves) is not bool
            or not isinstance(after, (list, tuple)) or not 1 <= len(after) <= MAX_UNION_CERTIFICATES
            or not isinstance(areas, np.ndarray) or areas.ndim != 1
            or not (np.issubdtype(areas.dtype, np.integer) or np.issubdtype(areas.dtype, np.floating))
            or not np.isfinite(areas).all()
            or np.any(areas < 0)):
        raise ValueError('覆盖合并比较的证据数量或原目标面积无效')
    streams = [intervals(c, len(areas), check) for c in [before, *after]]
    cursor = [0]*len(streams)
    retained, gained, pending = [np.zeros(len(areas), np.int64) for _ in range(3)]
    pending_leaves=[]
    steps = 0
    while all(cursor[i] < len(streams[i]) for i in range(len(streams))):
        if steps % 1024 == 0:check()
        parts = [stream[cursor[i]] for i, stream in enumerate(streams)]
        root = parts[0][0]
        if any(part[0] != root for part in parts):raise ValueError('覆盖合并比较的原目标根单元不一致')
        lo, hi = max(part[1] for part in parts), min(part[2] for part in parts)
        if hi <= lo:raise ValueError('覆盖合并比较分区顺序无效')
        old, new = parts[0][3], any(part[3] for part in parts[1:])
        if old and new:retained[root] += hi-lo
        elif old:
            pending[root] += hi-lo
            # If any part of an old leaf is unconfirmed, retain the whole old
            # leaf as a conservative candidate for witness-based follow-up.
            if include_pending_leaves and (not pending_leaves or pending_leaves[-1]!=cursor[0]):
                pending_leaves.append(cursor[0])
        elif new:gained[root] += hi-lo
        for i, part in enumerate(parts):
            if part[2] == hi:cursor[i] += 1
        steps += 1
    if any(cursor[i] != len(streams[i]) for i in range(len(streams))):
        raise ValueError('覆盖合并比较缺少完整原目标分区')
    def area(weights):return math.fsum(float(a)*(int(w)/UNITS) for a, w in zip(areas, weights))
    result=dict(method='common-dyadic-union-partition-v1', comparison_depth=DEPTH,
                union_partition_count=len(after),
                previous_set_sha256=digest([r[:3] for r in streams[0] if r[3]]),
                retained_nominal_area_m2=area(retained), gained_nominal_area_m2=area(gained),
                previously_confirmed_not_reconfirmed_nominal_area_m2=area(pending),
                retained_weight_units=int(retained.sum()), gained_weight_units=int(gained.sum()),
                not_reconfirmed_weight_units=int(pending.sum()), retained_subset=not bool(pending.any()),
                semantics='partition_comparison_only_requires_independent_proof_replay_and_identity_binding',
                area_semantics='original_nominal_root_weights_not_physical_accuracy')
    if include_pending_leaves:result['pending_old_leaf_indices']=pending_leaves
    return result
