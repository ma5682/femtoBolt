"""P5 nominal, piecewise-linear half-width bounds with explicit provenance.

The independent parameter is the normalized accumulated float64 chord length
of each saved continuous polyline, including its connectors. It is a numerical
model definition, not a measured device response or a motion command.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import date, datetime, timezone
from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Any

from femto.planning.planning_revision import digest

SCHEMA = 'femto-nominal-piecewise-ball-v1'
PARAMETER = 'normalized_saved_float64_chord_length_per_chain_including_connectors_v1'
MAX_KNOTS = 128
MAX_BYTES = 65536


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + ' 必须是有限数值')
    return float(value)


def _date(value: str | None) -> str | None:
    if value is None:return None
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('模型有效日期必须为 YYYY-MM-DD')
    return value


def nominal(knots: list[dict], *, name: str, version: str, interpolation: str = 'piecewise_linear',
            valid_from: str | None = None, valid_until: str | None = None) -> dict:
    if not isinstance(name, str) or not name.strip() or len(name) > 128:
        raise ValueError('请填写模型来源名称（最多 128 字符）')
    if not isinstance(version, str) or not version.strip() or len(version) > 64:
        raise ValueError('请填写模型版本（最多 64 字符）')
    if interpolation not in ('piecewise_linear', 'samples_only'):
        raise ValueError('未知的宽度插值规则')
    start, end = _date(valid_from), _date(valid_until)
    if start is not None and end is not None and start > end:raise ValueError('模型有效日期倒置')
    if not isinstance(knots, list) or not 2 <= len(knots) <= MAX_KNOTS:
        raise ValueError(f'变宽模型需要 2–{MAX_KNOTS} 个节点')
    rows = []
    for row in knots:
        if not isinstance(row, dict) or set(row) != {'position', 'lower_half_width_m', 'upper_half_width_m'}:
            raise ValueError('宽度节点必须包含位置与米制半宽上下界')
        position, lower, upper = (_number(row[k], k) for k in ('position', 'lower_half_width_m', 'upper_half_width_m'))
        if not 0 <= position <= 1 or not 0 <= lower <= upper <= .05:
            raise ValueError('节点位置须为 0–1，半宽须满足 0 ≤ 下界 ≤ 上界 ≤ 0.05 m')
        if rows and position <= rows[-1]['position']:raise ValueError('节点位置必须严格递增')
        rows.append(dict(position=position, lower_half_width_m=lower, upper_half_width_m=upper))
    if rows[0]['position'] != 0 or rows[-1]['position'] != 1:
        raise ValueError('节点必须覆盖每条连续路线的 0 和 1 两端')
    record = dict(schema=SCHEMA, model='piecewise_half_width_ball', mode='nominal',
                  parameter=PARAMETER, coordinate_frame='world_rdf', unit='metre', knots=rows,
                  interpolation=interpolation, source=dict(kind='user_declared_nominal_model', name=name, version=version),
                  valid_from=start, valid_until=end, center_offset_assumed_m=0.0,
                  offset_evidence=dict(kind='unknown', deterministic_bound_m=None, statistical_interval_m=None,
                                       source=None, calibration_batch=None, operating_conditions=None),
                  physical_center_offset_bound_m=None, registration_uncertainty_m=None, calibration_uncertainty_m=None,
                  active_edge_roles=['scan'], connector_effect_assumed=False,
                  seed_rule='accepted_scan_edge_endpoints_with_whole_edge_lower_bound',
                  anchor_rule='exact_surface_projection_with_displacement_deducted_inner_radius',
                  coverage_claim='nominal_model_surface_geometry_only',
                  outer_acceptance='per_edge_upper_bound_with_preserved_fixed_generation_envelope',
                  execution_ready=False, physical_accuracy_verified=False)
    record['sha256'] = digest(record)
    return record


def validate(record: dict) -> dict:
    from femto.planning.surface_trajectory import canonical_json_bytes
    try:
        source = record['source']
        expected = nominal(record['knots'], name=source['name'], version=source['version'],
                           interpolation=record['interpolation'], valid_from=record['valid_from'], valid_until=record['valid_until'])
        if canonical_json_bytes(record) != canonical_json_bytes(expected):raise ValueError('变宽模型内容、单位或摘要不一致')
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('变宽模型格式不完整') from error
    return record


def require_current(record: dict, *, on: date | None = None) -> None:
    validate(record)
    today = (on or datetime.now(timezone.utc).date()).isoformat()
    if record['valid_from'] is not None and today < record['valid_from']:
        raise ValueError('变宽模型尚未生效；原版本可只读回看')
    if record['valid_until'] is not None and today > record['valid_until']:
        raise ValueError('变宽模型已过期；请更新模型，或恢复应用模型前路线')


def load(path: Path) -> dict:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:raise ValueError('变宽模型含重复字段')
            value[key] = item
        return value
    with Path(path).open('rb') as stream:raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:raise ValueError('变宽模型文件超过 64 KiB')
    value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=unique)
    validate(value)
    return value


def _outward(value: Fraction, upper: bool) -> float:
    result = float(value)
    if (Fraction(result) < value if upper else Fraction(result) > value):
        result = math.nextafter(result, math.inf if upper else -math.inf)
    return result


def interval_bounds(record: dict, start: Fraction, end: Fraction) -> tuple[float, float] | None:
    """Include every interior knot; discrete observations never imply bounds."""
    if not 0 <= start <= end <= 1:raise ValueError('变宽模型区间超出原连续路线')
    if record['interpolation'] != 'piecewise_linear':return None
    rows = record['knots'];positions = [Fraction(row['position']) for row in rows]
    def value(t, key):
        i = min(len(rows)-2, max(0, bisect_right(positions, t)-1))
        weight = (t-positions[i])/(positions[i+1]-positions[i])
        return Fraction(rows[i][key])*(1-weight)+Fraction(rows[i+1][key])*weight
    candidates = [start, end] + [p for p in positions if start < p < end]
    lower = min(value(t, 'lower_half_width_m') for t in candidates)
    upper = max(value(t, 'upper_half_width_m') for t in candidates)
    return _outward(lower, False), _outward(upper, True)


def encode_fraction(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def decode_fraction(value: Any) -> Fraction:
    if (not isinstance(value, list) or len(value) != 2 or any(type(v) is not int or v.bit_length() > 4096 for v in value)
            or value[1] <= 0):raise ValueError('变宽区间有理参数无效')
    result = Fraction(*value)
    if encode_fraction(result) != value:raise ValueError('变宽区间参数未规范化')
    return result


def edge_intervals(trajectory: Any, check=lambda: None) -> list[dict]:
    from femto.planning.surface_trajectory import ScanSegment
    import numpy as np
    values = []
    for segment in trajectory.ordered_items:
        if not isinstance(segment, ScanSegment):continue
        check()
        points = segment.points
        lengths = [Fraction(float(np.linalg.norm(np.asarray(b['xyz_m'])-a['xyz_m']))) for a,b in zip(points,points[1:])]
        total = sum(lengths)
        if total <= 0:raise ValueError('连续路线长度为零，不能定义变宽参数')
        accumulated = Fraction(0)
        for i, length in enumerate(lengths):
            if i % 128 == 0:check()
            start, end = accumulated/total, (accumulated+length)/total
            values.append(dict(segment_index=segment.segment_index, edge_index=i,
                               interval=[encode_fraction(start), encode_fraction(end)],
                               role=points[i+1].get('role', 'scan')))
            accumulated += length
    if len(values) > 200000:raise ValueError('变宽模型超过 200,000 条边预算')
    return values


def edge_bounds(trajectory: Any, record: dict, check=lambda: None) -> list[dict]:
    validate(record)
    rows = edge_intervals(trajectory, check)
    for i, row in enumerate(rows):
        if i % 128 == 0:check()
        bounds = interval_bounds(record, *(decode_fraction(v) for v in row['interval']))
        row.update(lower_half_width_m=None if bounds is None else bounds[0],
                   upper_half_width_m=None if bounds is None else bounds[1],
                   status='unconfirmed_interpolation' if bounds is None else 'bounded_nominal')
    return rows


def seed_radius(record: dict, seed: dict) -> float:
    bounds = interval_bounds(record, *(decode_fraction(v) for v in seed['width_interval']))
    if bounds is None:raise ValueError('离散宽度样本不能产生覆盖内包球')
    if seed.get('lower_half_width_m') != bounds[0]:raise ValueError('覆盖种子与逐段半宽下界不一致')
    return bounds[0]


def summary_text(record: dict) -> str:
    lower = min(row['lower_half_width_m'] for row in record['knots'])*1000
    upper = max(row['upper_half_width_m'] for row in record['knots'])*1000
    state = ''
    try:require_current(record)
    except ValueError as error:state = '；'+str(error)
    return (f"名义变宽模型：{record['source']['name']} / {record['source']['version']}；"
            f'半宽范围 {lower:g}–{upper:g} mm；设备偏移上界未知'+state+'。')
