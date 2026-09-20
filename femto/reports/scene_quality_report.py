'''Reconstruction review and evidence-based physical distance acceptance.'''
from __future__ import annotations
from typing import Any

import argparse
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import numpy as np
from femto.io_utils import load_json, write_json_strict
from femto.dataset.scene_data_kind import classify_dataset_data_kind, load_dataset_manifest_chain

SPEC_SCHEMA = 'femto-field-acceptance-spec-v1'

def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def numeric_summary(values: Any) -> Any:
    values = np.asarray([v for v in values if v is not None], dtype=float)
    values = values[np.isfinite(values)]
    return dict(count=len(values), median=float(np.median(values)), p95=float(np.quantile(values,.95)), maximum=float(np.max(values))) if len(values) else None

def capture_identity(dataset: Any) -> tuple[str, str]:
    chain, complete = load_dataset_manifest_chain(dataset)
    if not complete or not chain: raise ValueError('Incomplete source lineage')
    raw = chain[-1][0]
    records = [json.loads(line) for line in (raw/'frames.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not records: raise ValueError('Empty raw capture')
    h = hashlib.sha256()
    for record in records:
        for key in ('rgb','depth'): h.update(digest(raw/record[key]).encode('ascii'))
    return str(raw), h.hexdigest()

def positive(value: Any, name: str) -> float:
    if isinstance(value,bool) or value is None or not np.isfinite(float(value)) or float(value)<=0:
        raise ValueError(name+' must be a positive finite number')
    return float(value)

def evaluate_field_spec(spec: Any, base: Any) -> dict[str, Any]:
    from femto.reports.scene_scale_validation import resolve_reconstruction, validate_scale
    if spec.get('schema') != SPEC_SCHEMA: raise ValueError('Unsupported field acceptance spec')
    targets, trials = spec.get('targets', []), spec.get('trials', [])
    pending, rows, seen_captures, target_map = [], [], set(), {}
    for target in targets:
        name = target.get('id')
        if not isinstance(name,str) or not name.strip() or name in target_map:
            raise ValueError('Target IDs must be nonempty and unique')
        target_map[name] = target
        for key in ('ground_truth_distance_mm','absolute_tolerance_mm','repeatability_range_tolerance_mm'):
            if target.get(key) is None: pending.append(name+': missing '+key)
            else: positive(target[key],key)
        if not str(target.get('ground_truth_source') or '').strip(): pending.append(name+': missing physical measurement source')
    if not targets: pending.append('No physical targets specified')
    for index, trial in enumerate(trials):
        reconstruction = resolve_reconstruction((base/Path(trial['reconstruction'])).resolve())
        record = load_json(reconstruction/'reconstruction_report.json')
        dataset = Path(record['source_dataset'])
        if classify_dataset_data_kind(dataset) != 'real_femto': raise ValueError('Field acceptance requires real Femto lineage')
        raw, fingerprint = capture_identity(dataset)
        if fingerprint in seen_captures: raise ValueError('Repeated raw capture cannot count as an independent trial')
        seen_captures.add(fingerprint)
        found = set()
        for measurement in trial.get('measurements',[]):
            name = measurement['target_id']
            if name not in target_map or name in found: raise ValueError('Unknown or duplicate measurement target')
            found.add(name); target = target_map[name]
            if any(measurement.get(k) is None for k in ('point_a_world_m','point_b_world_m')):
                pending.append(f'Trial {index+1}, {name}: pick both surface endpoints'); continue
            if target.get('ground_truth_distance_mm') is None or not target.get('ground_truth_source'): continue
            measured = validate_scale(reconstruction,[dict(label=name,
                ground_truth_distance_mm=float(target['ground_truth_distance_mm']),ground_truth_source=target['ground_truth_source'],
                point_a_world_m=measurement['point_a_world_m'],point_b_world_m=measurement['point_b_world_m'],
                snap_radius_mm=measurement.get('snap_radius_mm',2.0))],geometry_source=trial.get('geometry_source','tsdf_mesh'))
            row = measured['measurements'][0]
            row.update(trial=index+1, target_id=name, raw_capture=raw, capture_sha256=fingerprint)
            rows.append(row)
        for name in set(target_map)-found: pending.append(f'Trial {index+1}, {name}: missing measurement')
    results = []
    for name,target in target_map.items():
        items = [r for r in rows if r['target_id']==name]
        distances = [r['measured_distance_mm'] for r in items]
        maximum_error = max((r['absolute_error_mm'] for r in items),default=None)
        spread = float(np.ptp(distances)) if len(distances)>=2 else None
        std = float(np.std(distances,ddof=1)) if len(distances)>=2 else None
        if len(items)<3: pending.append(name+': need at least 3 independent measured captures')
        accuracy_ok = maximum_error <= float(target['absolute_tolerance_mm']) if maximum_error is not None and target.get('absolute_tolerance_mm') is not None else None
        repeat_ok = spread <= float(target['repeatability_range_tolerance_mm']) if spread is not None and len(items)>=3 and target.get('repeatability_range_tolerance_mm') is not None else None
        results.append(dict(target_id=name, trial_count=len(items), max_absolute_error_mm=maximum_error,
            distance_range_mm=spread,distance_sample_std_mm=std, accuracy_passed=accuracy_ok,repeatability_passed=repeat_ok))
    failed = any(r['accuracy_passed'] is False or r['repeatability_passed'] is False for r in results)
    status = 'FAIL' if failed else ('PENDING' if pending else 'PASS')
    return dict(status=status,accepted=False if failed else (True if status=='PASS' else None),
        pending=sorted(set(pending)),targets=results,measurements=rows, criteria=targets,
        scope='Specified landmark distances and repeatability only; not full surface or absolute world-coordinate accuracy')

def build_report(location: str | Path, field_spec: str | Path | None=None) -> tuple[Any, dict[str, Any]]:
    location = Path(location).resolve()
    if (location/'reconstruction_report.json').exists(): reconstruction,output = location,location.parent
    elif (location/'reconstruction/reconstruction_report.json').exists(): reconstruction,output=location/'reconstruction',location
    else: reconstruction,output = None,location
    geometry = load_json(reconstruction/'reconstruction_report.json') if reconstruction else None
    dataset = Path(geometry['source_dataset']) if geometry else location
    pose_path=dataset/'pose_quality_report.json'
    pose=load_json(pose_path) if pose_path.exists() else {}
    if not geometry and pose.get('source_dataset'): dataset=Path(pose['source_dataset'])
    chain,complete=load_dataset_manifest_chain(dataset)
    capture_path=chain[-1][0]/'derived/capture_quality.json' if chain else None
    capture=load_json(capture_path) if capture_path and capture_path.exists() else None
    paths=[pose_path,*[p/'manifest.json' for p,_ in chain]]
    if reconstruction: paths.append(reconstruction/'reconstruction_report.json')
    if capture_path and capture_path.exists(): paths.append(capture_path)
    evidence=[dict(path=str(p),sha256=digest(p)) for p in paths if p.exists()]
    quality=pose.get('frame_quality',[]); pairs=pose.get('pairs',[]); selected=pose.get('keyframe_selection',{})
    limitations=[]
    if not pose: limitations.append('缺少位姿质量报告，不能判断跟踪稳定性。')
    if not capture: limitations.append('此采集没有新版现场质量反馈记录。')
    if not quality: limitations.append('缺少逐帧图像质量记录，不能推算模糊帧数量。')
    limitations.extend(['有效深度比例是图像像素比例，不是目标表面覆盖率。','双向配准残差只表示内部自洽，不是实物尺寸误差。','尚无回环/全局位姿优化；未验收表面厚度、绝对坐标精度与目标完整覆盖。'])
    physical=dict(status='PENDING',accepted=None,pending=['待提供实测参考尺寸、容差、表面端点及至少3次独立采集'],targets=[],measurements=[])
    if field_spec:
        spec_path=Path(field_spec).resolve();physical=evaluate_field_spec(load_json(spec_path),spec_path.parent)
        spec=load_json(spec_path)
        if reconstruction is None:
            raise ValueError('Physical acceptance requires a reconstruction')
        from femto.reports.scene_scale_validation import resolve_reconstruction
        trial_paths={resolve_reconstruction((spec_path.parent/Path(t['reconstruction'])).resolve()) for t in spec.get('trials',[])}
        if reconstruction.resolve() not in trial_paths:
            raise ValueError('Acceptance trials do not include this reconstruction')
        evidence.append(dict(path=str(spec_path),sha256=digest(spec_path)))
    blocked=pose.get('status')=='BLOCKED'
    return output,dict(schema='femto-result-quality-v1',created_local=datetime.now().astimezone().isoformat(),
        data_kind=classify_dataset_data_kind(dataset),source_dataset=str(dataset),reconstruction=str(reconstruction) if reconstruction else None,
        pipeline_status='BLOCKED' if blocked else ('RECONSTRUCTED' if geometry else 'POSE_ONLY'),physical_acceptance=physical,
        capture=dict(source_frames=pose.get('source_frame_count',chain[-1][1].get('frame_count') if chain else None),feedback=capture),
        online_tracking=pose.get('online_tracking'),
        tracking=dict(tracked_frames=len(selected['tracked_frame_ordinals']) if 'tracked_frame_ordinals' in selected else (len(pose['accepted_frame_ordinals']) if 'accepted_frame_ordinals' in pose else None),
            keyframes=pose.get('frame_count',(geometry or {}).get('frame_count')) if not blocked else 0,
            redundant_frames=len(selected['redundant_frame_ordinals']) if 'redundant_frame_ordinals' in selected else None,
            rejected_frames=pose.get('rejected_frames',[]),failure=pose.get('error'),
            translation_span_m=pose.get('translation_span_m'),rotation_span_deg=pose.get('rotation_span_deg'),
            cycle_translation_mm=numeric_summary([p['cycle_consistency']['translation_m']*1000 for p in pairs if p.get('cycle_consistency',{}).get('translation_m') is not None]),
            cycle_rotation_deg=numeric_summary([p.get('cycle_consistency',{}).get('rotation_deg') for p in pairs]),
            rmse_mm=numeric_summary([p['inlier_rmse_m']*1000 for p in pairs if 'inlier_rmse_m' in p])),
        scene_context=dict(planned_scene='床位＋平躺人体',bed_scale_status=physical['status'],
            human_roi_coverage_status='PENDING',respiration_or_body_motion_status='PENDING',
            surface_ghosting_status='PENDING',
            note='床体刚性尺寸用于尺度与重复性；人体呼吸/体动不等于相机测量误差，未知区域不计为完整覆盖。'),
        frame_quality=quality,geometry=(geometry or {}).get('metrics',{}),limitations=limitations,evidence=evidence)

def render_report(report: Any) -> str:
    esc=lambda v:html.escape(str(v))
    def value(v):return '未提供' if v is None else (f'{v:.3f}' if isinstance(v,float) else str(v))
    def table(headers,rows):
        return '<table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(value(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    tracking=report['tracking'];physical=report['physical_acceptance']
    cards=table(['原始帧','跟踪帧','关键帧','重复帧','被拒绝帧'],[[report['capture']['source_frames'],tracking['tracked_frames'],tracking['keyframes'],tracking['redundant_frames'],len(tracking['rejected_frames']) if report['frame_quality'] or tracking['tracked_frames'] is not None else None]])
    residuals=table(['内部一致性指标','中位数','P95','最大值'],[[label,*[(tracking[key] or {}).get(k) for k in ('median','p95','maximum')]] for key,label in [('cycle_translation_mm','双向平移残差 mm'),('cycle_rotation_deg','双向旋转残差 °'),('rmse_mm','配准内点RMSE mm')]])
    targets=table(['目标','独立实测次数','最大尺寸误差 mm','距离极差 mm','距离样本标准差 mm','尺寸达标','重复性达标'],[[r.get(k) for k in ('target_id','trial_count','max_absolute_error_mm','distance_range_mm','distance_sample_std_mm','accuracy_passed','repeatability_passed')] for r in physical['targets']])
    frames=table(['源帧序号','有效深度比例','清晰度值','状态'],[[f.get(k) for k in ('frame_ordinal','valid_depth_fraction','sharpness','state')] for f in report['frame_quality']])
    rejections=table(['源帧序号','原因'],[[r.get('frame_ordinal'),r.get('reason')] for r in tracking['rejected_frames']])
    geometry=table(['重建指标','数值'],[[k,v] for k,v in report['geometry'].items()])
    files=table(['证据文件','SHA-256'],[[e['path'],e['sha256']] for e in report['evidence']])
    pending=''.join('<li>'+esc(item)+'</li>' for item in physical.get('pending',[]))
    status={'PASS':'指定距离项目通过','FAIL':'指定距离项目未通过','PENDING':'待实测'}[physical['status']]
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>实拍验收与结果质量报告</title>
<style>body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:0;background:#f3f6fa;color:#233247}}main{{max-width:1160px;margin:auto;padding:28px}}section{{background:white;border:1px solid #dbe2eb;border-radius:12px;padding:22px;margin:18px 0}}h1{{margin-bottom:8px}}h2{{font-size:21px}}p,li{{line-height:1.7}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #e4e9ef;overflow-wrap:anywhere}}th{{background:#eef3fa}}.badge{{display:inline-block;padding:7px 12px;background:#fff0cd;border-radius:7px}}details{{margin:16px 0}}code{{overflow-wrap:anywhere}}@media(max-width:650px){{main{{padding:12px}}section{{padding:12px}}table{{font-size:12px}}th,td{{padding:5px}}}}</style><main>
<h1>实拍验收与结果质量报告</h1><p>{esc(report['created_local'])} · 数据来源：{esc(report['data_kind'])}</p>
<section><h2>验收结论：<span class="badge">{status}</span></h2><p>处理阶段：{esc(report['pipeline_status'])}。流程完成不等于实物精度合格。</p><p>{esc(tracking.get('failure') or '')}</p><ul>{pending}</ul><p>实物验收只针对指定端点距离与重复性；完整表面、绝对坐标及机器人定位不在此次自动判定范围内。</p></section>
<section><h2>床位＋平躺人体：待拍验收计划</h2><p>以下是后续验收计划，不表示当前历史数据拍摄了该场景。床体：用固定标记的尺量距离验证尺寸误差与重复性。人体：选择要观察的体表区域，单独检查缺面、床单遮挡、重影及呼吸/体动。</p><p>人体区域覆盖、呼吸/体动、表面重影：均待现场检查。床体尺寸通过不代表人体表面已验收；本报告未运行人体运动检测或目标区域覆盖分析。</p></section>
<section><h2>采集与关键帧</h2>{cards}<p>所有原始帧保留；重复帧仍可参与连续跟踪。</p><details><summary>逐帧质量与筛选原因</summary>{frames}{rejections}</details></section>
<section><h2>位姿内部一致性</h2>{residuals}<p>这些数值不等于实物误差。平移跨度 {esc(value(tracking['translation_span_m']))} m；旋转跨度 {esc(value(tracking['rotation_span_deg']))}°。</p></section>
<section><h2>实物尺寸与重复性</h2>{targets}<p>同一目标至少3次独立采集；每次重新选取对应端点。不同采集的局部世界坐标不直接相减，以同一端点距离比较重复性。</p><details><summary>如何完成实拍验收</summary><ol><li>选静止、有纹理的刚性物体，在两端做可辨认标记；用尺或卡尺测量两标记间直线距离，记录工具与读数。</li><li>先确定允许尺寸误差及重复测量极差，不根据本次结果反推容差。</li><li>分别开始至少3次独立移动采集，每次在模型上选同一对端点。记录缺面、重影及边缘变厚现象。</li><li>填写 field_acceptance_spec.json，运行“生成实拍验收报告.cmd”并传入结果目录和该文件，重新计算验收结论。</li></ol><p>模板中的空值表示待填，不能使用估计尺寸代替尺量真值。</p></details></section>
<section><h2>重建几何</h2>{geometry}<ul>{''.join('<li>'+esc(item)+'</li>' for item in report['limitations'])}</ul></section>
{('<section><h2>在线局部补采记录</h2><pre>'+esc(json.dumps(report['online_tracking'],ensure_ascii=False,indent=2))+'</pre><p>延迟为实际处理记录，非实物定位精度。</p></section>') if report.get('online_tracking') else ''}
<section><h2>来源与复核</h2><p><code>{esc(report['source_dataset'])}</code></p><p><a href="result_quality.json">完整JSON报告</a> · <a href="field_acceptance_spec.json">实拍验收填写模板</a></p><details><summary>证据文件与校验值</summary>{files}</details></section></main></html>'''

def write_report(location: Any, field_spec: Any=None) -> Any:
    output,report=build_report(location,field_spec)
    output.mkdir(parents=True,exist_ok=True)
    write_json_strict(output/'result_quality.json',report)
    (output/'result_quality.html').write_text(render_report(report),encoding='utf-8')
    template=output/'field_acceptance_spec.json'
    if not template.exists():
        write_json_strict(template,dict(schema=SPEC_SCHEMA,targets=[dict(id='床体固定标记A到B',ground_truth_distance_mm=None,
            ground_truth_source='',absolute_tolerance_mm=None,repeatability_range_tolerance_mm=None)],
            trials=[dict(reconstruction=report['reconstruction'],geometry_source='tsdf_mesh',
                measurements=[dict(target_id='床体固定标记A到B',point_a_world_m=None,point_b_world_m=None)])] if report['reconstruction'] else []))
    return output/'result_quality.html'

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result',type=Path);parser.add_argument('--spec',type=Path)
    args=parser.parse_args();print(write_report(args.result,args.spec))

if __name__=='__main__':main()
