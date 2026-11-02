"""Small, reproducible next-step tasks derived from verified P1/P2 assets.

This report is guidance, never a planning input or an additional certificate.
Callers must have replayed/computed the evidence before building the view. No
source arrays are copied, no unknown area is invented and no task grants
execution permission. Historical reports remain bound to their own revision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path

import numpy as np

from femto.planning.planning_revision import digest

SCHEMA = 'femto-coverage-next-step-tasks-v1'
FIXED_BASES = ('quality-constrained-surface-v3', 'constructed-surface-route-v1')
REFINED = ('coverage-refined-surface-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-refined-surface-v1')
VARIABLE = 'nominal-variable-width-surface-v1'
PIXEL_REASON_LABELS = {
    'missing_reference_depth': '参考深度缺失', 'depth_outside_range': '深度超范围',
    'no_source_pixel': '参考像素不可用', 'no_mesh_surface': '无网格表面',
    'depth_conflict': '深度冲突', 'occlusion_or_multiple_surface_ambiguity': '遮挡或多表面歧义',
    'numerically_unconfirmed': '数值未确认', 'budget_unprocessed': '目标建账预算未处理',
    'partly_located_pixel_cell': '像素单元部分定位',
}
# Reason codes retain P2's meanings; 8 is P1's unlocated pixel remainder.
GUIDANCE = {
    1: ('keep', '保留已确认范围', '后续计算必须重新证明并保留当前已确认范围。'),
    2: ('inspect_geometry', '检查几何证据', '核对参考深度与质量排除原因；存在遮挡、无效深度或缺失时补采真实数据，再重新生成和核验。'),
    3: ('refine', '补充规划', '仅在合格几何上尝试新增扫描边；保留原确认集合且有新增扫描见证时才采用。'),
    4: ('recalculate', '重新核验连通', '重新检查球内连通证据；未确认连通不能直接归为行距不足或自动跨越断点。'),
    5: ('recalculate', '重新核验细分边界', '在当前限额内继续核验；若仍到细分上限，保留未知状态并检查边界与模型。'),
    6: ('inspect_target', '检查原目标构造', '核对目标映射、数值边界与源几何；先解决构造证据，再重新建立目标并核验。'),
    7: ('recalculate', '继续核验未处理范围', '在当前限额内重新核验；原确认集合重新证明后才允许替换，未处理范围继续标为未知。'),
    8: ('inspect_reference', '检查未定位范围', '在参考图定位缺失或部分定位像素，检查深度、遮挡和目标映射；需要时补采，不能补造三维坐标或面积。'),
}


@dataclass(frozen=True)
class CoverageTask:
    reason: int
    leaf_count: int
    pixel_count: int
    nominal_area_m2: float | None
    unknown_by_reason: tuple[tuple[str, int], ...] = ()
    calculation_limits_reached: tuple[str, ...] = ()

    @property
    def present(self):return bool(self.leaf_count or self.pixel_count or self.calculation_limits_reached)

    @property
    def guidance(self):
        base = GUIDANCE[self.reason][2]
        if self.reason != 8 or not self.unknown_by_reason:return base
        details = '；'.join(f'{PIXEL_REASON_LABELS.get(name, name)} {count} 像素'
                           for name, count in self.unknown_by_reason if count)
        if not details:return base
        advice = []
        reasons = {name for name, count in self.unknown_by_reason if count}
        if reasons & {'missing_reference_depth', 'depth_outside_range', 'no_source_pixel',
                      'no_mesh_surface', 'depth_conflict', 'occlusion_or_multiple_surface_ambiguity'}:
            advice.append('在参考图定位这些范围，复核取景和深度；确认实测缺失后补采')
        if 'budget_unprocessed' in reasons:
            advice.append('预算未处理范围优先检查目标建账限额与计算进度，不直接作为补采缺失')
        if reasons & {'numerically_unconfirmed', 'partly_located_pixel_cell'}:
            advice.append('数值与部分定位范围复核像素裁剪和几何对应')
        return details + '。' + '；'.join(advice) + '。未定位范围不能补造三维坐标或面积。'


@dataclass(frozen=True)
class CoverageTasks:
    target_id: str
    target_sha256: str
    computation_sha256: str
    certificate_sha256: str
    route_sha256: str
    original_target_coverage_complete: bool
    items: tuple[CoverageTask, ...]

    def task(self, reason):
        return next((item for item in self.items if item.reason == reason), None)

    def require_plan(self, plan):
        from femto.planning.coverage_workflow import KEY, check_binding
        check_binding(plan)
        metadata = plan.get(KEY)
        target = (plan.get('planning_context') or {}).get('target', {})
        if (metadata is None or metadata['sha256'] != self.certificate_sha256
                or target.get('sha256') != self.target_sha256 or target.get('target_id') != self.target_id
                or target.get('computation_sha256') != self.computation_sha256
                or metadata['inputs']['route']['route_semantics_sha256'] != self.route_sha256):
            raise ValueError('后续任务已过期；请使用当前路线与原目标重新核验后的任务')

    def to_dict(self):
        from femto.planning.coverage_display import NAMES
        binding = {key:getattr(self, key) for key in ('target_id', 'target_sha256', 'computation_sha256',
                                                     'certificate_sha256', 'route_sha256')}
        rows = []
        for item in self.items:
            if not item.present:continue
            action, title, _ = GUIDANCE[item.reason]
            row = asdict(item)
            row.update(task_id=digest(dict(binding=binding, reason=item.reason)), reason_name=NAMES[item.reason],
                       next_action=action, title=title, guidance=item.guidance,
                       status='confirmed_keep' if item.reason == 1 else 'needs_review',
                       locator=dict(kind='original_target_pixels' if item.reason == 8 else 'coverage_leaf_status',
                                    reason=item.reason))
            row['unknown_by_reason'] = dict(item.unknown_by_reason)
            row['calculation_limits_reached'] = list(item.calculation_limits_reached)
            rows.append(row)
        result = dict(schema=SCHEMA, binding=binding, tasks=rows,
                      original_target_coverage_complete=self.original_target_coverage_complete,
                      scope='derived_guidance_from_verified_nominal_geometry',
                      imported_report_is_authority=False, physical_accuracy_verified=False, execution_ready=False)
        return dict(result, sha256=digest(result))


def operation(task, algorithm):
    """Resolve an executable software operation without changing any inputs."""
    if task is None or not task.present or task.reason == 1:return None, '当前状态无需后续处理'
    action, label, _ = GUIDANCE[task.reason]
    if action in ('refine', 'recalculate'):
        if algorithm in REFINED:return 'restore', '恢复补充前路线后处理'
        if action == 'refine' and algorithm == VARIABLE:return 'refine', '变宽补充规划'
        if action == 'refine' and algorithm not in FIXED_BASES:return None, '当前版本需重新生成路线'
    return action, label


def build(ledger, certificate, check=lambda:None):
    """Aggregate a complete partition with O(leaves) time and bounded memory."""
    from femto.planning.coverage_certificate import _checked_assets
    from femto.planning.coverage_partition import DEPTH, UNITS
    _checked_assets(certificate, check)
    metadata = certificate.metadata
    domain = metadata['inputs']['domain']
    if (domain.get('scope') != 'p1_original_reference_selection'
            or domain.get('target_sha256') != ledger.metadata['sha256']
            or domain.get('computation_sha256') != ledger.metadata['computation_sha256']):
        raise ValueError('后续任务与原目标台账不属于同一修订')
    a = certificate.arrays
    areas = ledger.arrays['area_m2']
    # Check prefix identity without retaining another 200k-element interval list.
    expected = (0, 0)
    for index, (root, depth, code, state) in enumerate(zip(a['root'], a['depth'], a['code'], a['status'])):
        if index % 1024 == 0:check()
        root, depth, code, state = int(root), int(depth), int(code), int(state)
        if not 0 <= root < len(areas) or not 0 <= depth <= DEPTH or not 0 <= code < 4**depth or state not in range(1,8):
            raise ValueError('后续任务遇到无效原目标叶单元')
        lo, hi = code*4**(DEPTH-depth), (code+1)*4**(DEPTH-depth)
        if (root, lo) != expected:raise ValueError('后续任务缺少完整且不重叠的原目标分区')
        expected = (root+1, 0) if hi == UNITS else (root, hi)
    if expected != (len(areas), 0):raise ValueError('后续任务缺少原目标范围')
    weights = np.ldexp(areas[a['root']], -2*a['depth'].astype(np.int32))
    pixels = ledger.arrays['pixel_index'][a['root']]
    items = []
    for reason in range(1,8):
        check();mask = a['status'] == reason
        items.append(CoverageTask(reason, int(mask.sum()), int(len(np.unique(pixels[mask]))),
                                  math.fsum(weights[mask]), calculation_limits_reached=
                                  tuple(metadata['metrics']['limits_reached']) if reason == 7 else ()))
    unknown = ledger.arrays['mask'] & (ledger.arrays['pixel_status'] != 1)
    from femto.planning.target_surface import PIXEL_STATES
    details = tuple((name, int(np.count_nonzero(unknown & (ledger.arrays['pixel_status'] == code))))
                    for code, name in PIXEL_STATES.items() if code > 1)
    items.append(CoverageTask(8, 0, int(unknown.sum()), None, details))
    if items[-1].pixel_count != metadata['summary']['unlocated_pixel_count']:
        raise ValueError('后续任务未定位范围与覆盖摘要不一致')
    check()
    return CoverageTasks(ledger.metadata['target_id'], ledger.metadata['sha256'],
        ledger.metadata['computation_sha256'], metadata['sha256'],
        metadata['inputs']['route']['route_semantics_sha256'],
        metadata['summary']['original_target_coverage_complete'], tuple(items))


def write(path, tasks, plan, check=lambda:None):
    """Publish guidance atomically; it cannot be imported as coverage evidence."""
    from femto.planning.surface_trajectory import canonical_json_bytes
    from femto.planning.planning_io import staging_directory, durable, windows_file_retry
    tasks.require_plan(plan);check()
    path = Path(path)
    with staging_directory(path.parent, '.coverage_tasks_') as staging:
        temporary = staging/'tasks.json'
        temporary.write_bytes(canonical_json_bytes(tasks.to_dict()))
        durable(temporary);check();tasks.require_plan(plan)
        windows_file_retry(lambda:os.replace(temporary, path))
