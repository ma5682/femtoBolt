"""Result-bound local optimization evidence; no geometry or latest-index writes."""
from __future__ import annotations

import copy
import html
import json
import math
from pathlib import Path
import time

from femto.io_utils import load_json_object, write_json_strict

SCHEMA = 'femto-local-processing-v1'
STATUS = {
    'not_recorded': ('未记录', 'Not recorded'),
    'not_run': ('未执行', 'Not run'),
    'running': ('执行中', 'Running'),
    'adopted': ('已采用', 'Adopted'),
    'not_adopted': ('未采用', 'Not adopted'),
    'failed': ('失败', 'Failed'),
    'cancelled': ('已取消', 'Cancelled'),
    'complete': ('已完成', 'Complete'),
}
REASONS = {
    'correction_exceeds_5mm_or_1deg': ('修正超过 5 mm / 1°', 'Correction exceeds 5 mm / 1 degree'),
    'bidirectional_disagreement': ('前后向求解不一致', 'Forward/reverse disagreement'),
    'outside_workspace': ('候选超出局部范围', 'Candidate outside workspace'),
    'missing_validation': ('验证指标缺失或非有限', 'Missing/nonfinite validation'),
    'overlap_degraded': ('重叠不足或下降', 'Insufficient/degraded overlap'),
    'insufficient_heldout_improvement': ('验证采样改善不足 5%', 'Held-out improvement below 5%'),
    'tail_error_degraded': ('尾部残差恶化', 'Tail residual degraded'),
    'color_alignment_degraded': ('颜色一致性恶化', 'Color consistency degraded'),
    'insufficient_3d_structure': ('空间结构不足或近似平面', 'Insufficient 3D structure / near planar'),
    'more_than_20_views_budget': ('超过 20 视角优化预算', 'Over the 20-view refinement budget'),
    'no_stationary_five_frame_window': ('未找到合格五帧候选窗口', 'No eligible five-frame window'),
    'image_motion_rejected': ('图像运动审核未通过', 'Image motion audit rejected'),
    'less_than_80_percent_retained': ('可保留像素不足 80%', 'Less than 80% of valid pixels retained'),
}
WARNINGS = (
    ('内部一致性指标不是物理测量误差；实物精度未验收。',
     'Internal consistency is not physical measurement error; physical accuracy unverified.'),
    ('已生成模型不等于精度达标；RGB 完整不等于几何完整。',
     'A generated model does not establish accuracy; complete RGB does not imply complete geometry.'),
    ('预览点数增长不等于补洞数量或覆盖率。',
     'Preview point growth is not a hole-fill count or coverage ratio.'),
)


def empty_report():
    return dict(schema=SCHEMA, state='not_recorded', actual_pose_source=None,
                actual_dataset=None, refinement=dict(status='not_recorded'),
                temporal=dict(status='not_recorded'), physical_accuracy_verified=False)


def temporal_summary(report):
    rows = report.get('frames')
    if not isinstance(rows, list) or not rows or any(not isinstance(r,dict) or type(r.get('applied')) is not bool for r in rows):
        return dict(status='not_recorded', accepted_views=None, windows_found=None,
                    applied_views=None, removed_pixels=None, removed_ratio=None, frames=[])
    applied = [r for r in rows if r.get('applied') is True]
    if 'applied_frames' in report and report['applied_frames'] != len(applied):
        raise ValueError('Temporal applied count disagrees with frame evidence')
    counts = [r.get('removed_pixels') for r in applied]
    removed = sum(counts) if all(type(n) is int and n >= 0 for n in counts) else None
    denominators = [r.get('valid_pixels') for r in applied]
    known = removed is not None and bool(applied) and all(type(n) is int and n > 0 for n in denominators)
    windows = [(r['window_found'] if type(r.get('window_found')) is bool
                else True if len(r.get('source_frames', [])) == 5
                else False if r.get('reason') == 'no_stationary_five_frame_window'
                else None) for r in rows]
    return dict(status='adopted' if applied else 'not_adopted',
                accepted_views=len(rows),
                windows_found=sum(windows) if all(w is not None for w in windows) else None,
                applied_views=len(applied), removed_pixels=removed,
                removed_ratio=removed / sum(denominators) if known else (0.0 if not applied else None),
                ratio_definition='Removed / valid raw pixels in views whose masks were applied; zero when no mask applied',
                frames=rows)


class LocalProcessing:
    """One journal per job. All I/O is called by a worker, not Tk."""
    def __init__(self, dataset, output, progress=None):
        self.dataset = Path(dataset).resolve()
        self.output = Path(output).resolve()
        if self.output == self.dataset or self.dataset in self.output.parents:
            raise ValueError('Processing output must be outside its input dataset')
        self.output.mkdir(parents=True, exist_ok=False)
        self.path = self.output / 'processing_report.json'
        manifest = load_json_object(self.dataset / 'manifest.json')
        local = manifest.get('local_capture', {})
        if not local.get('measured_only'):
            raise ValueError('Local observed-depth dataset required')
        self.progress = progress
        self.report = dict(schema=SCHEMA, state='not_run', source_dataset=str(self.dataset),
            capture_scope=local.get('capture_scope', 'complete_capture'),
            interrupted_prefix=local.get('interrupted_prefix'),
            source_session_id=local.get('session_id'), reference_frame_id=local.get('reference_frame_id'),
            coordinate_frame='world_rdf', unit='metre', transform_direction='T_W0_Ct',
            actual_dataset=str(self.dataset), actual_pose_source=manifest.get('pose', {}).get('source'),
            refinement=dict(status='not_run'), temporal=dict(status='not_run'),
            reconstruction=dict(status='not_run'), report_path=str(self.path),
            physical_accuracy_verified=False)
        self.publish()

    def publish(self):
        write_json_strict(self.path, self.report)
        write_html(self.output / 'processing_report.html', self.report)
        if self.progress:
            self.progress(copy.deepcopy(self.report))

    def run(self):
        from femto.local.local_pose_refinement import refine
        from femto.local.local_temporal_support import prepare
        stage = 'refinement'
        try:
            self.report['state'] = 'running'
            self.report[stage] = dict(status='running')
            self.publish()
            started = time.perf_counter()
            refined, report = refine(self.dataset, self.output / 'refinement')
            refined = Path(refined).resolve()
            adopted = report.get('accepted') is True
            if adopted == (refined == self.dataset):
                raise ValueError('Refinement decision does not match the actual input dataset')
            self.report[stage] = dict(status='adopted' if adopted else 'not_run' if report.get('reason') == 'more_than_20_views_budget' else 'not_adopted',
                report=str(self.output / 'refinement/refinement_report.json'),
                elapsed_s=time.perf_counter()-started, actual_dataset=str(refined), details=report)
            self.report['actual_dataset'] = str(refined)
            self.report['actual_pose_source'] = load_json_object(refined / 'manifest.json').get('pose', {}).get('source')
            self.publish()
            stage = 'temporal'
            self.report[stage] = dict(status='running')
            self.publish()
            started = time.perf_counter()
            selected, temporal = prepare(refined, self.output / 'temporal')
            selected = Path(selected).resolve()
            if bool(temporal.get('applied_frames')) == (selected == refined):
                raise ValueError('Temporal decision does not match the actual input dataset')
            self.report[stage] = dict(temporal_summary(temporal),
                report=str(self.output / 'temporal/temporal_support_report.json'),
                elapsed_s=time.perf_counter()-started)
            self.report['actual_dataset'] = str(selected)
            self.report['state'] = 'complete'
            self.publish()
            return selected, copy.deepcopy(self.report)
        except Exception as error:
            self.report['state'] = 'failed'
            self.report[stage].update(status='failed', error=str(error))
            try:
                self.publish()
            except Exception as report_error:
                self.report['reporting_error'] = str(report_error)
            raise

    def reconstruction(self, status, error=None):
        self.report['reconstruction'] = dict(status=status, error=error)
        self.report['state'] = 'running' if status == 'running' else status
        self.publish()


def result_quality(result):
    """Read only reports explicitly bound to this geometry version; never guess from newer attempts."""
    root = Path(result).resolve()
    if root.name == 'reconstruction':
        root = root.parent
    rec = root / 'reconstruction'
    quality = empty_report()
    try:
        reconstruction = load_json_object(rec / 'reconstruction_report.json')
        actual = Path(reconstruction['source_dataset']).resolve()
        context = load_json_object(rec / 'local_geometry.json')
        manifest = load_json_object(actual / 'manifest.json')
        quality.update(actual_dataset=str(actual), actual_pose_source=manifest.get('pose', {}).get('source'))
        linked = root / 'processing_report.json'
        if linked.is_file():
            value = load_json_object(linked)
            if (value.get('schema') != SCHEMA or Path(value.get('actual_dataset', '')).resolve() != actual
                    or value.get('source_session_id') != context.get('session_id')
                    or value.get('reference_frame_id') != context.get('reference_frame_id')
                    or value.get('actual_pose_source') != manifest.get('pose',{}).get('source')):
                raise ValueError('Processing report / model binding mismatch')
            if value.get('refinement',{}).get('status') == 'adopted':
                if value['refinement'].get('details',{}).get('accepted') is not True:
                    raise ValueError('Adopted refinement lacks an accepted decision')
                _validate_refinement_binding(value['refinement']['details'],manifest,context)
            if value.get('temporal',{}).get('status') in ('adopted', 'not_adopted'):
                _validate_temporal_binding(value['temporal'],actual)
            quality = value
        else:
            # Legacy adopted outputs sometimes retain stage report pointers. Missing
            # pointers are not evidence that an attempted optimization succeeded.
            local = manifest.get('local_capture', {})
            path = local.get('refinement_report')
            if path:
                report = load_json_object(path)
                if report.get('schema') != 'femto-local-refinement-v1':
                    raise ValueError('Unknown refinement report schema')
                _validate_refinement_binding(report,manifest,context,actual)
                quality['refinement'] = dict(status='adopted' if report.get('accepted') is True else 'not_adopted',
                                             report=path, details=report)
            path = local.get('temporal_support_report')
            if path:
                report=load_json_object(path)
                if report.get('schema') != 'femto-local-temporal-support-v1':
                    raise ValueError('Unknown temporal report schema')
                source=load_json_object(Path(report['source_dataset'])/'manifest.json')
                if any(source.get('local_capture',{}).get(k) != context.get(k) for k in ('session_id','reference_frame_id')):
                    raise ValueError('Temporal report source session mismatch')
                temporal=temporal_summary(report)
                if temporal['status']=='adopted':_validate_temporal_binding(temporal,actual)
                quality['temporal'] = dict(temporal, report=path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        quality['read_error'] = str(error)
        # Never leave unverified success visible after a binding/file failure.
        quality['refinement'] = dict(status='not_recorded')
        quality['temporal'] = dict(status='not_recorded')
    return quality


def _validate_refinement_binding(report, manifest, context, actual=None):
    source=Path(report['source_dataset']).resolve()
    if report.get('accepted') is True:
        if (manifest.get('pose',{}).get('source') != 'offline_robust_icp_research_grade'
                or not manifest.get('refinement_source_dataset')
                or Path(manifest['refinement_source_dataset']).resolve() != source):
            raise ValueError('Refinement adoption / actual pose source mismatch')
    elif actual is not None and source != actual:
        raise ValueError('Refinement report belongs to a different input')
    source_manifest=load_json_object(source/'manifest.json')
    if any(source_manifest.get('local_capture',{}).get(k) != context.get(k)
           for k in ('session_id','reference_frame_id')):
        raise ValueError('Refinement report source session mismatch')


def _validate_temporal_binding(temporal, actual):
    frames=[json.loads(s) for s in (actual/'frames.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    used={f['frame'] for f in frames if f.get('local_temporal_support')}
    reported={f['frame'] for f in temporal.get('frames',[]) if f.get('applied')}
    if (bool(used) != (temporal.get('status') == 'adopted')
            or used != reported or temporal.get('applied_views') != len(used)):
        raise ValueError('Temporal adoption / actual mask input mismatch')


def format_quality(report, language='zh'):
    zh = language == 'zh'
    choose = lambda a, b: a if zh else b
    missing = choose('未记录', 'Not recorded')
    state = lambda key: STATUS.get(key, (key, key))[0 if zh else 1]
    reason = lambda key: REASONS.get(str(key), (str(key), str(key)))[0 if zh else 1]
    def number(value, factor=1.0, suffix=''):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return missing
        return f'{value*factor:.1f}{suffix}'
    refinement = report.get('refinement', {})
    temporal = report.get('temporal', {})
    lines = [
        choose('模型处理：', 'Model processing: ') + state(report.get('reconstruction',{}).get('status',report.get('state','not_recorded'))),
        choose('精配准：', 'Refinement: ') + state(refinement.get('status', 'not_recorded')),
        choose('实际位姿来源：', 'Actual pose source: ') + (report.get('actual_pose_source') or missing),
    ]
    if report.get('capture_scope') == 'interrupted_prefix':
        lines.insert(0, choose('中断前数据建模：仅使用最后可信时刻之前的已接纳视角；没有恢复跟踪。',
                               'Interrupted-prefix reconstruction: admitted views before the last trustworthy time; tracking was not resumed.'))
    details = refinement.get('details', {})
    if details.get('reason'):
        lines.append(reason(details['reason']))
    for row in details.get('frames', []):
        prefix = choose('帧 ', 'Frame ') + str(row.get('frame', '?'))
        if row.get('reasons'):
            lines.append(prefix + ': ' + '; '.join(reason(r) for r in row['reasons']))
        if row.get('correction_mm') is not None:
            lines.append(prefix+choose(' 修正：',' correction: ')+number(row.get('correction_mm'),1,' mm')+
                         ' / '+number(row.get('correction_deg'),1,'°')+
                         (choose('；仅修正上限未通过','; only correction limit rejected') if row.get('only_correction_limit_rejected') else ''))
        if row.get('diagnostic_reasons'):
            lines.append(prefix+choose(' 诊断：',' diagnostic: ')+'; '.join(reason(r) for r in row['diagnostic_reasons']))
        for direction in ('forward', 'reverse'):
            before = row.get('before', {}).get(direction, {})
            after = row.get('after', {}).get(direction, {})
            if before or after:
                lines.append(prefix + choose(' 正向' if direction == 'forward' else ' 反向',
                                               ' forward' if direction == 'forward' else ' reverse')
                    + choose(' 内部中位/P90(mm)：', ' internal median/P90(mm): ')
                    + number(before.get('median_m'), 1000) + '/' + number(before.get('p90_m'), 1000)
                    + ' → ' + number(after.get('median_m'), 1000) + '/' + number(after.get('p90_m'), 1000))
    if refinement.get('error'):
        lines.append(refinement['error'])
    lines.append(choose('五帧二值筛选：', 'Five-frame binary filtering: ') + state(temporal.get('status', 'not_recorded')))
    count = lambda key: str(temporal[key]) if temporal.get(key) is not None else missing
    lines.append(choose('接纳视角 / 找到窗口 / 实际应用：', 'Accepted views / windows found / masks applied: ')
                 + ' / '.join(count(k) for k in ('accepted_views', 'windows_found', 'applied_views')))
    lines.append(choose('实际剔除像素：', 'Actually removed pixels: ') + count('removed_pixels')
                 + ' (' + number(temporal.get('removed_ratio'), 100, '%') + ')')
    for row in temporal.get('frames', []):
        if not row.get('applied') and row.get('reason'):
            lines.append(choose('帧 ', 'Frame ') + str(row.get('frame', '?')) + ': ' + reason(row['reason']))
    if temporal.get('error'):
        lines.append(temporal['error'])
    interpretation=temporal.get('details',{}).get('window_miss_interpretation') or temporal.get('window_miss_interpretation')
    if interpretation=='stationary_windows_exist_but_do_not_contain_accepted_frames':
        lines.append(choose('会话存在合格静止窗口，但均不含接纳帧；这是接纳时刻与静止期错开，不代表全程手抖。',
                            'Valid stationary windows exist, but none contains an accepted frame; acceptance timing missed the still periods.'))
    depth_quality=report.get('reconstruction',{}).get('depth_quality') or {}
    if depth_quality.get('local_measured_repairs_never_integrated'):
        lines.append(choose('局部测量链路只融合原始深度；补洞像素从未进入TSDF。',
                            'Local measurement geometry integrates raw depth only; repaired pixels never enter TSDF.'))
        for row in depth_quality.get('frames',[]):
            lines.append(choose('深度帧 ','Depth frame ')+str(row.get('frame','?'))+': '+
                number(row.get('original_invalid_rate'),100,'%')+', policy='+str(row.get('policy',missing)))
    if report.get('read_error'):
        lines.append(choose('报告缺失或绑定异常：', 'Report unavailable / binding error: ') + report['read_error'])
    lines.extend(pair[0 if zh else 1] for pair in WARNINGS)
    return '\n'.join(lines)


def write_html(path, report):
    sections = []
    for lang, heading in [('zh', '局部优化与质量'), ('en', 'Local processing quality')]:
        sections.append('<h2>' + heading + '</h2><pre>' + html.escape(format_quality(report, lang)) + '</pre>')
    links = []
    if report.get('result'):
        links.append('<li><a href="result_quality.html">综合质量报告 / General quality report</a></li>')
    for stage in ('refinement', 'temporal'):
        target = report.get(stage, {}).get('report')
        if target:
            links.append('<li><a href="' + html.escape(Path(target).resolve().as_uri(), quote=True) + '">'
                         + stage + ' JSON</a></li>')
    text = ('<!doctype html><meta charset="utf-8"><title>Femto local quality</title>'
            '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6}</style>'
            + ''.join(sections) + '<ul>' + ''.join(links) + '</ul><pre>'
            + html.escape(json.dumps(report, ensure_ascii=False, indent=2)) + '</pre>')
    Path(path).write_text(text, encoding='utf-8')
