"""Plain-language failure presentation using the current worker's evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from femto.operation_diagnostics import OperationSummary

_STAGES = {
    'qt_worker': '后台处理',
    'native.save': '保存三维路线',
    'native.read': '读取三维路线',
    'portable.save': '保存可编辑工程',
    'portable.read': '读取可编辑工程',
    'portable.manifest': '写入工程清单',
    'portable.validation_manifest': '准备工程回读核验',
    'portable.archive': '打包可编辑工程',
    'target.construct': '建立原目标台账',
}


def _matches(error,summary,current_task):
    return summary is not None and (current_task or (summary.error_type==type(error).__name__
        and summary.reason==str(error)[:4096] and summary.status!='complete'))


def brief_failure_text(error,summary,*,has_model,current_task=False):
    """Compact persistent notice; publication facts come only from matched evidence."""
    matched=_matches(error,summary,current_task)
    stage=summary.stage if matched else None
    discarded='Discarded' in type(error).__name__;cancelled='Cancel' in type(error).__name__
    title='结果未安装' if discarded else '任务已取消' if cancelled else '操作未完成'
    leaf=error;seen=set()
    while leaf.__cause__ is not None and id(leaf) not in seen:
        seen.add(id(leaf));leaf=leaf.__cause__
    reason=' '.join(str(leaf).split())
    if len(reason)>160:reason=reason[:157]+'…'
    lines=[title+' · '+_STAGES.get(stage,'当前操作')]
    if reason not in (title,'Geometry task cancelled','操作已取消','任务已取消'):
        lines.append('原因：'+reason)
    retained=summary.retained if matched else ()
    if retained:
        if cancelled or discarded:
            lines.append('原输入的文件已发布，未安装为当前方案；请独立核验。')
        elif any(item.kind=='三维路线包' for item in retained) and not any(item.kind=='可编辑工程' for item in retained):
            lines.append('三维路线已发布；可编辑工程保存未完成。当前方案保留，可重试。')
        else:lines.append('已发布文件保留，请在详情中核对。')
    else:
        lines.append('当前区域、参数与方案保留，可重试。' if has_model else '尚未载入可用数据，请查看详情后重试。')
    if matched and summary.recording_error:lines.append('诊断记录未写全，详情中可查看原因。')
    return '\n'.join(lines)


def failure_text(error: BaseException, summary: OperationSummary | None, *,
                 retained_current: str, next_steps: tuple[str, ...], current_task: bool = False) -> str:
    """Keep cancellation and discarded results distinct from backend failures.

    Only the generation-checked worker callback can mark a summary current.
Cancellation/discard may replace its error after publication. Other callers
must match the recorded error; paths are never parsed from exception strings.
    """
    matched = _matches(error,summary,current_task)
    stage = summary.stage if matched else None
    if 'Discarded' in type(error).__name__:
        title, label = '结果未安装', '核对当前输入'
    elif 'Cancel' in type(error).__name__:
        title, label = '任务已取消', _STAGES.get(stage, stage or '后台处理')
    else:
        title, label = '操作未完成', _STAGES.get(stage, stage or '当前操作')
    lines = [f'{title} · {label}', '原因：' + str(error), '已保留：' + retained_current]
    if matched:
        if summary.reason and summary.reason != str(error)[:4096]:
            lines.append('后台记录：' + summary.reason)
        for artifact in summary.retained:
            lines.append(f'已发布的{artifact.kind}：{artifact.path}')
        if summary.retained and ('Discarded' in type(error).__name__ or 'Cancel' in type(error).__name__):
            lines.append('这些包属于刚刚结束任务的原输入；未安装为当前方案，需要独立核验。')
        if summary.directory:
            lines.append('诊断记录：' + summary.directory)
        if summary.recording_error:
            lines.append('诊断写入未完成：' + summary.recording_error)
    lines.extend('下一步：' + step for step in next_steps)
    return '\n'.join(lines)


def capture_failure_text(failure: dict[str, Any], quality: dict[str, Any] | None,
                         source: str | Path | None) -> str:
    """Show existing capture/worker facts without treating a partial model as ready."""
    report = quality if isinstance(quality, dict) else {}
    stages = {'refinement': '位姿精配准', 'temporal': '时域处理', 'reconstruction': '模型重建'}
    failed = [title for key, title in stages.items()
              if isinstance(report.get(key), dict) and report[key].get('status') in ('failed', 'cancelled', 'timeout', 'memory_limit')]
    phase = ' / '.join(failed) or ('结果发布' if failure.get('publication') else '采集或建模（阶段未记录）')
    reason = str(failure.get('error') or report.get('error') or '失败原因未记录')
    lines = ['失败阶段：' + phase, '原因：' + reason]
    lines.append('已记录数据的位置：' + str(source) if source else '尚无已记录的数据目录。')
    if failure.get('processing_report'):
        lines.append('处理报告：' + str(failure['processing_report']))
    resources = report.get('worker_resources')
    if isinstance(resources, dict) and resources.get('failure_diagnostics_recorded') is True and resources.get('failure_diagnostics'):
        lines.append('后台失败日志：' + str(resources['failure_diagnostics']))
    lines.append('下一步：查看“查看本次报告”或“状态详情”，检查连接、取景与跟踪原因；'
                 '恢复有效画面并确认范围后，点击“重新建立扫描参考并开始”。')
    if failure.get('successful_model_published'):
        lines.append('报告记录有模型发布残留；请独立核验该记录，不能将本次失败当作完整成功。')
    return '\n'.join(lines)
