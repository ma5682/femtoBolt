"""Catalog of exported regions, trajectories, and portable projects.

The catalog is deliberately limited to filesystem metadata and validation.  UI layers
Both Qt pages and headless workers import these functions without creating UI.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
import zipfile

from femto.asset_storage import publish_json, resolve_relative


def local_asset(folder: str | Path, name: Any) -> Path | None:
    """Resolve a declared asset below *folder* and reject path traversal."""
    try:
        path = resolve_relative(folder, name)
    except (ValueError, OSError):
        return None
    return path if path.is_file() else None


def register_project(reconstruction: str | Path, path: str | Path,
                     check: Callable[..., Any] = lambda: None) -> Path:
    """Index a portable project without unpacking or modifying it."""
    path = Path(path).resolve()
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo('project.json')
        if info.file_size > 100 * 1024**2:
            raise ValueError('Project metadata too large')
        data = json.loads(archive.read('project.json'))
    if data.get('schema') != 'femto-region-project-v1':
        raise ValueError('请选择区域工程 ZIP，不是展示分享包。')
    from femto.planning.region_library import digest
    folder = Path(reconstruction) / 'region_project_history'
    folder.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(path).casefold().encode()).hexdigest()
    record = dict(project_file=str(path), name=data.get('region', {}).get('name') or '未命名区域',
                  project_id=data['project_id'], revision_id=data['revision_id'],
                  has_plan=bool(data.get('plan')), sha256=digest(path, check))
    record.update(task_context=data.get('task_context'),parent_revision_id=data.get('parent_revision_id'))
    publish_json(folder / (key + '.json'), record, check)
    return path


def scan_region_exports(reconstruction: str | Path,
                        check: Callable[..., Any] = lambda: None) -> list[dict[str, Any]]:
    """Read concrete region exports; ignore projects, routes, and the library index."""
    root = Path(reconstruction)
    rows: list[dict[str, Any]] = []
    for source in (root / 'annotations').glob('*/annotations.json'):
        check()
        try:
            stamp = source.stat().st_mtime
            data = json.loads(source.read_text(encoding='utf-8-sig'))
            for index, region in enumerate(data.get('regions', [])):
                asset = local_asset(source.parent, region.get('glb_file')) or local_asset(source.parent, region.get('mesh_file'))
                rows.append(dict(time=datetime.fromtimestamp(stamp).strftime('%Y-%m-%d %H:%M:%S'),
                                 stamp=stamp, kind='区域',
                                 name=region.get('name') or region.get('annotation_id') or f'区域 {index + 1}',
                                 folder=source.parent, source=source, asset=asset,
                                 region_index=index,
                                 status='可预览' if asset else '区域模型文件缺失'))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            rows.append(dict(time='', stamp=0, kind='异常记录', name=source.parent.name,
                             folder=source.parent, source=source, asset=None,
                             status='读取失败：' + str(error)))
    return sorted(rows, key=lambda row: row['stamp'], reverse=True)


def scan_exports(reconstruction: str | Path, include_hidden: bool = False,
                 check: Callable[..., Any] = lambda: None) -> list[dict[str, Any]]:
    """Build a catalog of annotations, route exports, trajectories, and projects."""
    root = Path(reconstruction)
    rows: list[dict[str, Any]] = []
    sources = (list((root / 'annotations').glob('*/annotations.json')) +
               list((root / 'annotations').glob('*/trajectory_*/trajectory.json')) +
               list((root / 'surface_paths').glob('*/trajectory.json')) +
               list((root / 'region_paths').glob('*/path.json')))
    for source in sources:
        check()
        try:
            stamp = source.stat().st_mtime
            data = json.loads(source.read_text(encoding='utf-8-sig'))
            if source.name == 'annotations.json':
                items = [('区域', r.get('name') or r.get('annotation_id', ''),
                          r.get('glb_file'), r.get('mesh_file')) for r in data['regions']]
                if not items:
                    items = [('标记', '仅点标记', None, None)]
            elif data.get('schema') == 'femto-surface-trajectory-v1':
                items = [('三维表面轨迹', data.get('source', {}).get('region_id') or source.parent.name,
                          'region_with_trajectory.glb', None)]
            else:
                items = [('区域＋路径', data.get('region', {}).get('name') or data.get('region_id', ''),
                          data.get('visualization', {}).get('file'), None)]
            for region_index, (kind, name, glb, ply) in enumerate(items):
                asset = local_asset(source.parent, glb) or local_asset(source.parent, ply)
                rows.append(dict(time=datetime.fromtimestamp(stamp).strftime('%Y-%m-%d %H:%M:%S'),
                                 stamp=stamp, kind=kind, name=name, folder=source.parent,
                                 source=source, asset=asset,
                                 diagram=(local_asset(source.parent, 'route_diagram.png') or
                                          local_asset(source.parent, 'depth_diagram.png'))
                                 if data.get('schema') == 'femto-surface-trajectory-v1'
                                 else local_asset(source.parent, data.get('route_diagram')),
                                 archive=local_asset(source.parent, data.get('share_archive')),
                                 region_index=region_index if kind == '区域' else None,
                                 status='可预览' if asset else '无三维文件 / 文件缺失'))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            rows.append(dict(time='', stamp=0, kind='异常记录', name=source.parent.name,
                             folder=source.parent, source=source, asset=None,
                             status='读取失败：' + str(error)))
    for source in (root / 'region_project_history').glob('*.json'):
        check()
        try:
            data = json.loads(source.read_text(encoding='utf-8'))
            from femto.dataset.task_context import checked_task
            task=checked_task(data.get('task_context'))
            path = Path(data['project_file'])
            exists = path.is_file()
            stamp = path.stat().st_mtime if exists else source.stat().st_mtime
            rows.append(dict(time=datetime.fromtimestamp(stamp).strftime('%Y-%m-%d %H:%M:%S'), stamp=stamp,
                             kind='区域工程 / 方案', name=((task or {}).get('name','')+' · '+data['name']).strip(' ·'), folder=path.parent,
                             task_context=data.get('task_context'),project_id=data.get('project_id'),
                             revision_id=data.get('revision_id'),parent_revision_id=data.get('parent_revision_id'),
                             project_location=path,
                             source=source, asset=None, project=path if exists else None,
                             archive=path if exists else None,
                             status=('可打开继续规划（有保存路径）' if exists and data.get('has_plan')
                                     else '可打开继续规划' if exists
                                     else '工程已移动或删除，请重新导入')))
        except (OSError, ValueError, KeyError, TypeError):
            rows.append(dict(time='', stamp=0, kind='异常工程记录', name=source.name,
                             folder=source.parent, source=source, asset=None,
                             status='工程索引读取失败'))
    from femto.planning.region_library import enrich, read_index
    known = {str(r['source'].resolve()).casefold() for r in rows}
    for entry in read_index(reconstruction)['records'].values():
        snapshot = entry.get('snapshot')
        if snapshot and str(Path(snapshot['source']).resolve()).casefold() not in known:
            row = dict(snapshot)
            for key in ('source', 'folder', 'asset', 'diagram', 'archive'):
                if row.get(key):
                    row[key] = Path(row[key])
            row['status'] = '文件丢失'
            rows.append(row)
    rows = enrich(rows, reconstruction, check)
    for row in rows:
        task = row.get('task_context')
        if task:
            row['name'] = task['name'] + ' · ' + row.get('name', '')
    return sorted([r for r in rows if include_hidden or not r['hidden']],
                  key=lambda row: row['stamp'], reverse=True)


def history_record_status(row: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """Explain saved region/route progress without weakening asset checks."""
    if (row.get('kind') != '区域' or row.get('file_status') != '导出资产指纹一致'
            or not row.get('export_manifest') or row.get('details', {}).get('plan_binding')):
        return row.get('status', '')
    children = [r for r in rows if r.get('parent_manifest') == row['export_manifest']]
    if not children:
        return '区域已保存 · 尚无轨迹'
    invalid = [r for r in children if r.get('file_status') in ('损坏', '文件丢失')]
    unverified = [r for r in children if r.get('file_status') not in ('损坏', '文件丢失', '轨迹资产指纹一致')]
    valid = [r for r in children if r.get('file_status') == '轨迹资产指纹一致']
    if not valid:
        return ('区域已保存 · 关联轨迹待验证' if unverified else
                '区域已保存 · 关联轨迹损坏或文件缺失') + ('；另有轨迹损坏或文件缺失' if invalid and unverified else '')
    partial = any(not r.get('result_complete', True) for r in valid)
    complete = any(r.get('result_complete', True) for r in valid)
    status = ('区域已保存 · 已有轨迹（另有未完成结果）' if partial and complete else
              '区域已保存 · 轨迹未完成' if partial else '区域已保存 · 已有轨迹')
    return (status + ('；另有轨迹损坏或文件缺失' if invalid else '')
            + ('；另有轨迹待验证' if unverified else ''))


def scan_saved_versions(outputs,check=lambda:None):
    """Include older capture/workspace catalogs as well as the new global index."""
    outputs=Path(outputs)
    roots=[outputs/'saved_projects']
    roots.extend(outputs.glob('complete_demo_*/reconstruction'))
    roots.extend(p.parent for p in (outputs/'region_workspaces').glob('*/region_project_history'))
    roots.extend(p.parent for p in (outputs/'planning_models').glob('*/region_project_history'))
    rows=[];seen=set()
    for root in roots:
        check()
        for row in scan_exports(root,check=check):
            if row.get('kind') not in ('区域工程 / 方案','三维表面轨迹','区域＋路径'):continue
            if row.get('kind')=='三维表面轨迹' and (Path(row['folder'])/'planning.region.zip').is_file():continue
            # A missing/corrupt project no longer has ``project``, but its
            # registered location still identifies the same archive across
            # capture, workspace and global catalogs.
            path=(row.get('project_location') or row.get('project') or
                  row.get('target') or row.get('source'))
            key=str(Path(path).resolve()).casefold()
            if key not in seen:rows.append(row);seen.add(key)
    return sorted(rows,key=lambda row:row['stamp'],reverse=True)


def scan_capture_exports(reconstruction, *, outputs=None, check=lambda:None):
    """Join saved derivatives to their exact capture, never by task/display name."""
    root = Path(reconstruction).resolve()
    rows = scan_exports(root, check=check)
    outputs = Path(outputs) if outputs is not None else root.parent.parent
    seen = {str(Path(row.get('project_location') or row.get('target') or row['source']).resolve()).casefold()
            for row in rows}
    mesh = root/'scene_tsdf_mesh_m.ply'
    for row in scan_saved_versions(outputs, check=check):
        check()
        details = row.get('details') or {}
        lineage = details.get('source_lineage') or {}
        source = lineage.get('source_mesh') or details.get('source_mesh')
        if not source or Path(source).resolve() != mesh:
            continue
        key = str(Path(row.get('project_location') or row.get('target') or row['source']).resolve()).casefold()
        if key not in seen:
            rows.append(row); seen.add(key)
    return sorted(rows, key=lambda row: row['stamp'], reverse=True)


__all__ = ['local_asset', 'register_project', 'scan_region_exports', 'scan_exports', 'scan_saved_versions',
           'scan_capture_exports', 'history_record_status']
