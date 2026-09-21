"""Rebuildable region library; metadata operations never remove source assets."""
from __future__ import annotations
from typing import Any, Callable

import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

SCHEMA = 'femto-region-library-v1'


def digest(path: str | Path, check: Callable[..., Any]=lambda: None) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            check(); h.update(block)
    return h.hexdigest()


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_index(root: str | Path) -> Any:
    path = Path(root) / 'region_library.json'
    if not path.exists():
        return dict(schema=SCHEMA, records={})
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema') != SCHEMA or not isinstance(data.get('records'), dict):
        raise ValueError('方案库索引损坏；原始工程仍可导入，请备份索引后重建')
    return data


def write_index(root: str | Path, data: Any, commit: Any=None) -> None:
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.library_', dir=root)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush(); os.fsync(stream.fileno())
        action=lambda:os.replace(temporary, root / 'region_library.json')
        if commit:commit(action)
        else:action()
    finally:
        if Path(temporary).exists(): Path(temporary).unlink()


def project_metadata(path: Any, check: Callable[..., Any]=lambda:None, verify: bool=False) -> Any:
    with zipfile.ZipFile(path) as archive:
        if archive.getinfo('project.json').file_size > 100 * 1024**2:
            raise ValueError('工程清单超出预算')
        data = json.loads(archive.read('project.json'))
        if verify:
            entries=archive.infolist(); names=[e.filename for e in entries]
            if len(names)!=len(set(names)) or sum(e.file_size for e in entries)>1024**3:
                raise ValueError('工程重复资产或超过 1 GiB 核验预算')
            required={'scene_tsdf_mesh_m.ply','rgb.png','depth.npy','image_valid.npy'}
            if not required<=set(data.get('files',{})):
                raise ValueError('工程缺少必要资产清单')
            for name, expected in data['files'].items():
                if name not in names: raise ValueError('工程缺少资产：'+name)
                h=hashlib.sha256()
                with archive.open(name) as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):
                        check();h.update(block)
                if h.hexdigest()!=expected:raise ValueError('工程资产损坏：'+name)
    if data.get('schema') != 'femto-region-project-v1':
        raise ValueError('请选择区域工程，不是分享包')
    return data


def enrich(rows: Any, root: Any, check: Callable[..., Any]=lambda: None) -> Any:
    """Relations require region ID AND source geometry/origin, never a name."""
    index = read_index(root)
    for row in rows:
        check()
        revision=None
        if row['kind']=='区域工程 / 方案':
            try:revision=json.loads(row['source'].read_text(encoding='utf-8')).get('revision_id')
            except (OSError,ValueError):pass
        row['key'] = identity([str(row['source'].resolve()).casefold(), row.get('region_index'),revision])
        saved = index['records'].get(row['key'], {})
        row.update(note=saved.get('note', ''), hidden=saved.get('hidden', False))
        row['region_key'] = row['key']; row['version_key'] = row['key']
        row['details'] = saved.get('details', {}); row['file_status'] = '损坏'
        row['region_key'] = saved.get('region_key', row['key'])
        row['version_key'] = saved.get('version_key', row['key'])
        try:
            source = row['source']
            history_binding = None
            details_files = None
            if row['kind'] == '区域工程 / 方案':
                record = json.loads(source.read_text(encoding='utf-8'))
                target = Path(saved.get('location', record['project_file']))
                row.update(project=target if target.is_file() else None,
                           archive=target if target.is_file() else None, folder=target.parent)
                row['target'] = target
                expected = (saved.get('sha256') if saved.get('location') else record.get('sha256')) or saved.get('sha256')
                row['expected_sha256'] = expected
                if not target.is_file():
                    row['file_status'] = '文件丢失'; row['status'] = '文件丢失：可重新定位（必须有原指纹）'
                    continue
                if expected and digest(target, check) != expected:
                    raise ValueError('工程内容与登记指纹不一致，请作为新版本重新导入')
                data = project_metadata(target,check,verify=True)
                region = data.get('region', {})
                row['name'] = region.get('name') or row['name']
                row['file_status'] = '工程资产指纹一致'
            else:
                target = Path(saved.get('location', source))
                row['target'] = target
                row['expected_sha256'] = saved.get('sha256')
                if not target.is_file():
                    row['file_status'] = '文件丢失'; row['status'] = '文件丢失'; continue
                if saved.get('sha256') and digest(target, check) != saved['sha256']:
                    raise ValueError('清单内容与登记指纹不一致')
                data = json.loads(target.read_text(encoding='utf-8-sig'))
                from femto.planning.export_history import local_asset
                is_trajectory = data.get('schema') == 'femto-surface-trajectory-v1'
                if is_trajectory:
                    from femto.planning.surface_trajectory import SurfaceTrajectory
                    SurfaceTrajectory.from_dict(data)
                    trajectory_source = data.get('source', {})
                    region = {'annotation_id': trajectory_source.get('region_id'), 'name': row['name']}
                    row.update(folder=target.parent,
                        asset=local_asset(target.parent, 'region_with_trajectory.glb'),
                        diagram=local_asset(target.parent, 'route_diagram.png') or
                                local_asset(target.parent, 'depth_diagram.png'), archive=None)
                else:
                    region = (data.get('regions', [])[row['region_index']] if row.get('region_index') is not None
                              else data.get('region', {}))
                    row.update(folder=target.parent,asset=local_asset(target.parent,region.get('glb_file')) or local_asset(target.parent,region.get('mesh_file')) or local_asset(target.parent,data.get('visualization',{}).get('file')),
                        diagram=local_asset(target.parent,data.get('route_diagram')),archive=local_asset(target.parent,data.get('share_archive')))
                row['file_status'] = '旧版本待验证'
                files = data.get('files', {})
                if is_trajectory:
                    file_manifest = target.parent / 'trajectory_files.json'
                    if not file_manifest.is_file():
                        raise ValueError('轨迹资产清单丢失')
                    manifest_data = json.loads(file_manifest.read_text(encoding='utf-8-sig'))
                    if (manifest_data.get('schema') != 'femto-surface-trajectory-files-v1' or
                            manifest_data.get('result_sha256') != data.get('result_sha256')):
                        raise ValueError('轨迹资产清单身份不一致')
                    files = manifest_data.get('files')
                    if not isinstance(files, dict) or 'trajectory.json' not in files:
                        raise ValueError('轨迹资产清单内容无效')
                manifest=region.get('region_manifest_file') if row['kind']=='区域' else None
                if manifest:
                    manifest_path=(target.parent/manifest).resolve()
                    if not manifest_path.is_relative_to(target.parent.resolve()):raise ValueError('区域清单越界')
                    if manifest_path.is_file():
                        manifest_data=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
                        if manifest_data.get('region',{}).get('annotation_id')!=region.get('annotation_id'):raise ValueError('区域清单身份不一致')
                        files=manifest_data.get('file_sha256',{})
                    else:raise ValueError('区域资产清单丢失')
                for name, expected in files.items():
                    check(); asset = (target.parent / name).resolve()
                    if not asset.is_relative_to(target.parent.resolve()): raise ValueError('清单资产越界')
                    if not asset.is_file():
                        row['file_status'] = '文件丢失'; break
                    if digest(asset, check) != expected: raise ValueError('资产指纹不符：' + name)
                else:
                    if files: row['file_status'] = '轨迹资产指纹一致' if is_trajectory else '导出资产指纹一致'
                declared = ('region_with_trajectory.glb' if is_trajectory else
                            region.get('glb_file') or region.get('mesh_file') or data.get('visualization', {}).get('file'))
                if not declared and not is_trajectory:
                    row['file_status'] = '仅点标记' if row['kind'] == '标记' else '无模型导出'
                elif not row.get('asset'): row['file_status'] = '文件丢失'
                if is_trajectory:
                    history_binding = dict(schema='femto-trajectory-binding-v1',
                        result_sha256=data.get('result_sha256'), region_id=trajectory_source.get('region_id'),
                        geometry_sha256=trajectory_source.get('geometry_sha256'),
                        algorithm_version=data.get('planner', {}).get('algorithm_version'),
                        execution_ready=False, physical_accuracy_verified=False)
                    details_files = files
            binding = history_binding or (data.get('plan') or data).get('plan_binding') or {}
            rid = region.get('annotation_id') or binding.get('region_id') or data.get('region_id')
            geometry = (data.get('source', {}).get('geometry_sha256') if history_binding else None) or data.get('geometry_sha256') or binding.get('geometry_sha256')
            if rid and geometry and data.get('origin_reference'):
                row['region_key'] = identity([rid, geometry, data['origin_reference']])
            row['version_key'] = binding.get('result_sha256') or data.get('revision_id') or row['key']
            row['details'] = dict(region_id=rid, geometry_sha256=geometry,
                origin_reference=data.get('origin_reference'), source_lineage=data.get('source_lineage'),
                source_mesh=data.get('source_mesh'), project_id=data.get('project_id'),
                revision_id=data.get('revision_id'), plan_binding=binding,
                files=details_files if details_files is not None else data.get('files', {}), location=str(row['target']))
            row['status'] = row['file_status'] + ('；轨迹只读、不可直接执行' if history_binding else
                '；路径关联待工作台重算' if binding else '；旧版本/无路径关联待验证')
        except TimeoutError:raise
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, zipfile.BadZipFile) as error:
            row['file_status'] = '损坏'; row['status'] = '损坏：' + str(error)
            row.update(asset=None, project=None, archive=None, diagram=None)
    return rows


def edit_records(root: Any, rows: Any, *, hidden: Any=None, note: str | None=None, check: Callable[..., Any]=lambda:None,commit: Any=None) -> None:
    """Validate whole selection first; one atomic index change, no file deletion."""
    from femto.planning.export_history import scan_exports
    current = {r['key']: r for r in scan_exports(root, include_hidden=True,check=check)}
    keys = [r['key'] for r in rows]
    if not keys or any(key not in current for key in keys): raise ValueError('选择不属于当前方案库')
    data = read_index(root)
    for key in keys:
        entry = data['records'].setdefault(key, {})
        if hidden is not None: entry['hidden'] = bool(hidden)
        if note is not None: entry['note'] = str(note)
    check(); write_index(root, data,commit)


def relocate(root: Any, row: Any, candidate: str | Path, check: Callable[..., Any]=lambda:None,commit: Any=None) -> None:
    """A lost old file without a prior digest cannot be identified by its name."""
    from femto.planning.export_history import scan_exports
    current = next((r for r in scan_exports(root, include_hidden=True,check=check) if r['key'] == row['key']), None)
    if current is None: raise ValueError('记录不属于当前方案库')
    expected = current.get('expected_sha256')
    if not expected: raise ValueError('旧记录没有原文件指纹，不能可靠重定位；请单独导入并验证')
    candidate = Path(candidate).resolve()
    if digest(candidate,check) != expected: raise ValueError('内容指纹不一致；同名文件不能重新关联')
    if current['kind'] == '区域工程 / 方案':project_metadata(candidate,check,verify=True)
    else:
        metadata=json.loads(candidate.read_text(encoding='utf-8-sig'))
        manifests=[metadata]
        for region in metadata.get('regions',[]):
            if region.get('region_manifest_file'):
                asset=(candidate.parent/region['region_manifest_file']).resolve()
                if not asset.is_relative_to(candidate.parent):raise ValueError('清单越界')
                manifests.append(json.loads(asset.read_text(encoding='utf-8-sig')))
        for manifest in manifests:
            for name,expected_asset in manifest.get('file_sha256',manifest.get('files',{})).items():
                asset=(candidate.parent/name).resolve()
                if not asset.is_relative_to(candidate.parent) or not asset.is_file():raise ValueError('重定位目录缺少资产：'+name)
                if digest(asset,check)!=expected_asset:raise ValueError('重定位资产指纹不一致：'+name)
    data = read_index(root)
    data['records'].setdefault(row['key'], {}).update(location=str(candidate), sha256=expected)
    check();write_index(root, data,commit)


def rebuild(root: Any, check: Callable[..., Any]=lambda:None,commit: Any=None) -> Any:
    """Discovery cache is optional; preserve user notes/hides and relocation."""
    from femto.planning.export_history import scan_exports
    data = read_index(root)
    rows = scan_exports(root, include_hidden=True,check=check)
    for row in rows:
        entry = data['records'].setdefault(row['key'], {})
        target = row.get('target')
        if target and target.is_file() and row['file_status'] != '损坏':
            entry.setdefault('sha256', digest(target,check))
        entry.update(region_key=row['region_key'],version_key=row['version_key'],details=row['details'],
            snapshot={k:str(v) if isinstance(v,Path) else v for k,v in row.items() if k in {'time','stamp','kind','name','folder','source','asset','diagram','archive','region_index'}})
    check(); write_index(root, data,commit)
    return rows


def discover_projects(root: Any, directory: str | Path, check: Callable[..., Any]=lambda:None) -> dict[str, Any]:
    """Explicit, bounded directory discovery rebuilds external project references."""
    from femto.planning.export_history import register_project
    found=[]; skipped=[]; count=0
    for path in Path(directory).rglob('*'):
        check();count+=1
        if count>20000:raise ValueError('目录扫描超过 20000 项预算，请选择更小范围')
        if path.is_file() and path.suffix.lower()=='.zip':
            try:project_metadata(path,check,verify=True)
            except TimeoutError:raise
            except (ValueError,KeyError,zipfile.BadZipFile,OSError) as error:
                skipped.append(dict(path=str(path),reason=str(error)));continue
            check();register_project(root,path,check=check);found.append(str(path))
    return dict(found=found,skipped=skipped)


def repair_index(root: str | Path) -> Any:
    """Keep unreadable bytes as a separate backup before rediscovery."""
    import uuid
    path=Path(root)/'region_library.json'
    try:read_index(root);return None
    except (ValueError,TypeError):pass
    backup=path.with_name('region_library.corrupt.'+uuid.uuid4().hex+'.bak')
    path.rename(backup)
    return backup
