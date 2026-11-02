"""Isolated automatic preflight and read-only history replay; never starts camera streams."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PREFIX = 'BODY_REPLAY_RESULT='


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): value.update(chunk)
    return value.hexdigest()


def evidence_paths(source):
    """Fingerprint inputs, never recursively scan arbitrary report/output directories."""
    source = Path(source).resolve()
    if source.is_file(): return [source]
    root = source/'reconstruction' if (source/'reconstruction').is_dir() else source
    paths = [p for p in root.iterdir() if p.is_file()]
    paths.extend(root.parent/name for name in ('demo_summary.json', 'publication.json', 'task_context.json'))
    report = read_json(root/'reconstruction_report.json')
    dataset = Path(report['source_dataset']).resolve()
    manifest = read_json(dataset/'manifest.json')
    allowed = [dataset]
    if manifest.get('source_dataset'):
        original = Path(manifest['source_dataset']).resolve()
        if dataset.is_relative_to(original) and (original/'manifest.json').is_file():
            allowed.append(original)
            paths.append(original/'manifest.json')
            paths.extend(p for p in (original/'calibration').glob('*') if p.is_file())
    paths.extend(dataset/name for name in ('frames.jsonl', 'manifest.json'))
    paths.extend(p for p in (dataset/'calibration').glob('*') if p.is_file())
    for line in (dataset/'frames.jsonl').read_text(encoding='utf-8-sig').splitlines():
        if not line.strip(): continue
        frame = json.loads(line)
        for key in ('rgb', 'depth', 'pose'):
            path = (dataset/frame[key]).resolve()
            if not any(path.is_relative_to(root) for root in allowed): raise ValueError('帧资产越过已声明的数据集血缘目录')
            paths.append(path)
    return sorted(set(paths))


def fingerprints(paths):
    return {str(path): digest(path) if path.is_file() else None for path in paths}


def replay(source):
    from femto.app.qt_planning_io import load_source, masks_from_region
    from femto.app.region_quality import assess_selection, rescan_context
    from femto.planning.planning_state import validate
    source = Path(source).resolve()
    paths = evidence_paths(source); before = fingerprints(paths)
    model = load_source(source, threading.Event(), lambda _: None)
    region = model.store.regions[0] if model.store.regions else None
    plan = getattr(model, 'saved_plan', None)
    result = dict(status='passed', source=str(source), scope='saved_plan' if plan else 'saved_region' if region else 'capture_model',
                  vertices=len(model.mesh.vertices), triangles=len(model.mesh.triangles),
                  reference_frame_id=model.view_reference.get('reference_frame_id'),
                  origin=model.store.origin_reference.get('kind'),
                  task_context=getattr(model, 'task_context', None),
                  task_warning=getattr(model, 'task_warning', None),
                  confirmed_body_available=rescan_context(model) is not None,
                  body_status='matched' if rescan_context(model) else 'not_available_or_not_supported_for_new_capture',
                  saved_region=region is not None, saved_route=plan is not None,
                  quality=dict(status='not_available', message='此历史模型未包含已保存工作区域；不代替医师自动选区。'))
    capture = getattr(model, 'capture_source', None)
    if capture and (Path(capture).parent/'demo_summary.json').is_file():
        summary = read_json(Path(capture).parent/'demo_summary.json')
        result.update(declared_data_kind=summary.get('data_kind'), historical_frame_count=summary.get('frame_count'))
    if region:
        assessment = assess_selection(model, *masks_from_region(region, model.rgb.shape[:2]))
        result['quality'] = {k:v for k,v in assessment.items() if k not in ('classes', 'rgba')}
    if plan:
        validate(plan, model.mesh, region)
        result['route_segment_count'] = len(plan.get('segments', []))
        result['route_break_count'] = len(plan.get('chain_breaks', []))
    result['inputs_unchanged'] = before == fingerprints(paths)
    result['input_sha256'] = before
    if not result['inputs_unchanged']: result['status'] = 'failed'
    # Successful reading is not a certification of the current quality of a saved plan.
    result['review_required'] = (not result['confirmed_body_available'] or
        result['quality']['status'] != 'checked' or bool(result['quality'].get('pending')))
    return result


def enumerate_devices():
    # Use the repository's sole SDK adapter; enumeration never constructs a pipeline.
    from femto.hardware.orbbec_rgbd import Context
    context = Context(); devices = context.query_devices()
    names = [devices.get_device_by_index(i).get_device_info().get_name() for i in range(devices.get_count())]
    return dict(status='detected' if names else 'not_detected', count=len(names), names=names,
                stream_started=False, note='仅设备枚举；未验证 RGB-D 帧、同步、跟踪或移动采集。')


def native_render(source, screenshot):
    import numpy as np
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
    from vtkmodules.vtkIOImage import vtkPNGWriter
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from femto.app.qt_vtk_viewport import VTKViewport
    from femto.app.qt_planning_io import load_source
    app = QApplication([])
    if app.platformName() in ('offscreen', 'minimal'): raise RuntimeError('未获得原生图形平台')
    paths=evidence_paths(source);before=fingerprints(paths)
    model = load_source(source, threading.Event(), lambda _: None)
    vertices = np.asarray(model.mesh.vertices).copy(); faces = np.asarray(model.mesh.triangles).copy()
    region = model.store.regions[0] if model.store.regions else None
    plan = getattr(model, 'saved_plan', None)
    view = VTKViewport(); view.setAttribute(Qt.WA_DontShowOnScreen, True); view.resize(960, 720)
    try:
        view.show(); app.processEvents(); view.set_scene(model, region, plan); app.processEvents()
        camera = view.renderer.GetActiveCamera()
        initial = camera.GetPosition()
        camera.Azimuth(15); camera.Elevation(7); view._render()
        moved = camera.GetPosition()
        view.set_scene(model, region, plan, reset_camera=False)
        kept = np.allclose(camera.GetPosition(), moved) and not np.allclose(initial, moved)
        view.set_scene(model, region, plan, reset_camera=True)
        reset = np.allclose(camera.GetPosition(), initial)
        render_window = view.widget.GetRenderWindow()
        width, height = render_window.GetSize()
        picked = None
        for y in range(height//5, height, max(1, height//5)):
            for x in range(width//5, width, max(1, width//5)):
                if view.picker.Pick(x,y,0,view.renderer):
                    data=view.picker.GetDataSet();ids=data.GetCellData().GetArray('original_triangle_id')
                    cell=view.picker.GetCellId()
                    if ids is not None and 0 <= cell < ids.GetNumberOfTuples():
                        picked=int(ids.GetValue(cell));break
            if picked is not None:break
        image=vtkWindowToImageFilter();image.SetInput(render_window);image.SetInputBufferTypeToRGB()
        image.ReadFrontBufferOff();image.Update()
        pixels=vtk_to_numpy(image.GetOutput().GetPointData().GetScalars())
        nonempty=bool(pixels.size and np.any(np.ptp(pixels.astype(np.int16),axis=0)>5))
        writer=vtkPNGWriter();writer.SetInputConnection(image.GetOutputPort());writer.WriteToMemoryOn();writer.Write()
        Path(screenshot).write_bytes(vtk_to_numpy(writer.GetResult()).tobytes())
        unchanged=np.array_equal(vertices,np.asarray(model.mesh.vertices)) and np.array_equal(faces,np.asarray(model.mesh.triangles))
        checks=dict(nonempty_framebuffer=nonempty,camera_preserved=bool(kept),camera_reset=bool(reset),
                    original_triangle_picked=picked is not None and 0<=picked<len(faces),geometry_unchanged=unchanged,
                    inputs_unchanged=before==fingerprints(paths))
        capabilities=render_window.ReportCapabilities()
        return dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
                    qt_platform=app.platformName(),render_window=render_window.GetClassName(),size=[width,height],
                    screenshot=str(screenshot),picked_triangle=picked,
                    opengl=[line for line in capabilities.splitlines() if 'OpenGL' in line][:4])
    finally:
        view.shutdown();view.shutdown();view.close();app.processEvents()


def run_worker(kind, source=None, screenshot=None, timeout=180):
    from femto.paths import workspace_runtime_environment
    executable=ROOT/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    command=[str(executable),'-I','-B',str(Path(__file__).resolve()),'--worker',kind]
    if source is not None:command+=['--source',str(source)]
    if screenshot is not None:command+=['--screenshot',str(screenshot)]
    environment=dict(os.environ);environment.pop('PYTHONHOME',None);environment.pop('PYTHONPATH',None)
    environment.update(workspace_runtime_environment())
    if kind=='native':environment['QT_QPA_PLATFORM']='windows' if os.name=='nt' else environment.get('QT_QPA_PLATFORM','xcb')
    try:
        process=subprocess.run(command,cwd=ROOT,env=environment,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        lines=[line[len(PREFIX):] for line in process.stdout.splitlines() if line.startswith(PREFIX)]
        if process.returncode or not lines:return dict(status='failed',error=f'子进程未正常完成（退出码 {process.returncode}）',detail=process.stderr[-1500:])
        value=json.loads(lines[-1])
        if not isinstance(value,dict) or not isinstance(value.get('status'),str):
            return dict(status='failed',error='子进程返回了无效检查记录')
        return value
    except subprocess.TimeoutExpired:return dict(status='failed',error=f'{kind} 检查超过 {timeout} 秒，子进程已终止')
    except (OSError,ValueError) as error:return dict(status='failed',error=str(error))


def make_report(runtime, devices, records):
    good=bool(records) and runtime.get('ready') is True and devices.get('status') in ('detected','not_detected') and all(row['replay'].get('status')=='passed' and
         row.get('native',{}).get('status') in ('passed','not_run') for row in records)
    return dict(schema='femto-body-surface-replay-check-v1',created_at=datetime.now().astimezone().isoformat(),
        workspace=str(ROOT),automatic_status='passed' if good else 'failed',
        scope='automatic_preflight_and_existing_history_only',runtime=runtime,devices=devices,records=records,
        new_capture_performed=False,field_acceptance='not_run',physical_accuracy_verified=False,execution_ready=False,
        pending=['现场静止目标与多视角移动采集','实际跟踪丢失与重新定位','带明确部位和体位的新记录端到端保存恢复','实物尺度端点测量与误差验收'])


def markdown(report):
    lines=['# 自动预检与历史回放验收', '', '自动检查：'+report['automatic_status'],
           '范围：现有历史回放；未启动新采集，现场验收未执行。',
           '通过表示环境及历史读取/显示检查通过，不表示选区质量合格、人体采集能力或实物精度已验证。','',
           '环境：'+('通过' if report['runtime'].get('ready') else '未通过'),
           '设备枚举：'+str(report['devices'].get('status')),'']
    for row in report['records']:
        replayed=row['replay']
        lines.extend(['## '+str(row['source']), '', '回放：'+str(replayed.get('status')),
                      '原生三维显示：'+str(row['native'].get('status')),
                      '已确认支持部位：'+str(replayed.get('confirmed_body_available')),
                      '已保存区域 / 路线：'+str(replayed.get('saved_region'))+' / '+str(replayed.get('saved_route')),
                      '检查前后输入相同：'+str(replayed.get('inputs_unchanged'))])
        quality=replayed.get('quality',{})
        lines.append('当前选区检查：'+str(quality.get('status','未执行')))
        if quality.get('status')=='checked':
            lines.append(f"选区 {quality['selected']} 像素；网格与参考深度一致 {quality['ratio']:.2%}；待检查 {quality['pending']} 像素。")
        elif quality.get('message'):lines.append(str(quality['message']))
        if replayed.get('review_required'):lines.append('需要人工核对；缺失的任务/部位信息和几何证据不自动补全。')
        for item in (replayed,row['native']):
            if item.get('error'):lines.append('错误：'+str(item['error']))
        lines+=['']
    return '\n'.join(lines+['## 待现场验证','']+['- '+item for item in report['pending']]+[''])


def main(argv=None):
    parser=argparse.ArgumentParser(description='自动预检与现有历史回放；不开始采集，不覆盖历史文件')
    parser.add_argument('--source',action='append',type=Path)
    parser.add_argument('--native-render',action='store_true')
    parser.add_argument('--output',type=Path,help='新的报告目录；已存在时拒绝覆盖')
    parser.add_argument('--worker',choices=['replay','devices','native'],help=argparse.SUPPRESS)
    parser.add_argument('--screenshot',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:
        try:
            value=enumerate_devices() if args.worker=='devices' else native_render(args.source[0],args.screenshot) if args.worker=='native' else replay(args.source[0])
        except Exception as error:value=dict(status='failed',error=type(error).__name__+': '+str(error))
        print(PREFIX+json.dumps(value,ensure_ascii=True));return 0
    if not args.source:parser.error('请用 --source 指定要核验的现有记录，不自动猜测任务或部位')
    output=(args.output or ROOT/'outputs'/'body_surface_acceptance'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])).resolve()
    # Report writes may never be placed inside an input record.
    for source in args.source:
        protected=source.resolve() if source.is_dir() else source.resolve().parent
        if output.is_relative_to(protected):parser.error('报告目录必须位于输入记录目录之外')
    output.mkdir(parents=True,exist_ok=False)
    from tools.check_workspace_runtime import check_runtime
    runtime=check_runtime('capture',ROOT/'.venv')
    devices=run_worker('devices',timeout=20)
    records=[]
    for index,source in enumerate(args.source):
        print('正在核验：'+str(source),flush=True)
        item=run_worker('replay',source)
        native=run_worker('native',source,output/f'native_{index+1}.png') if args.native_render and item['status']=='passed' else dict(status='not_run')
        records.append(dict(source=str(source.resolve()),replay=item,native=native))
    report=make_report(runtime,devices,records)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'report.md').write_text(markdown(report),encoding='utf-8')
    print('报告：'+str(output/'report.md'));print('自动检查：'+report['automatic_status']+'；现场验收：not_run')
    return 0 if report['automatic_status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
