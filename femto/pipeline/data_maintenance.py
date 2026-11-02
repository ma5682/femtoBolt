"""Clinical-workflow data roots and active reference coverage for cold backups.

Read existing records only. Historical lineage in a self-contained portable ZIP
is not treated as an active external dependency. No paths or proofs are rebased.
"""
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from femto import data_backup
from femto.history_store import HistoryStore,_unpack

PATH_FIELDS={'key','source','target','project','project_file','project_location','asset','archive',
             'mesh','source_mesh','source_dataset','tracking_report','region_manifest','manifest_path',
             'path','report','calibration_file','rgb_path','depth_path','image_path'}
PATH_LISTS={'fixed_files','comparison_files','missing_source_files'}
WORKSPACE_PATHS={'source_dataset','source_mesh','tracking_report','project_file','project_location','region_manifest','manifest_path'}
PATTERNS=('complete_demo_*/reconstruction/reconstruction_report.json',
          'complete_demo_*/reconstruction/local_geometry.json',
          'complete_demo_*/reconstruction/region_project_history/*.json',
          'complete_demo_*/reconstruction/surface_paths/*/trajectory.json',
          'complete_demo_*/reconstruction/annotations/*/annotations.json',
          'saved_projects/**/*.json','region_workspaces/**/region_project_history/*.json',
          'region_workspaces/**/surface_paths/*/trajectory.json',
          'hardware_configuration/**/*.json','installation_calibrations/**/*.json')


def default_roots(workspace):
    workspace=Path(workspace).resolve()
    return {name:workspace/name for name in ('datasets','outputs','photos','models') if (workspace/name).is_dir()}


def inspect_references(workspace,*,check=lambda:None,exclude_owner=None):
    """Collect dependencies of active history, published artifacts and device evidence.

    Unregistered arbitrary JSON and historical lineage in portable archives are
    not executable references. Actual backup calls this while DB writers are
    reserved; preview calls are informational and rechecked during capture.
    """
    workspace=Path(workspace).resolve();outputs=workspace/'outputs';datasets=workspace/'datasets'
    exclude_owner = Path(exclude_owner).resolve() if exclude_owner else None
    references=set();pending=[];visited=set();db_rows=0;publications=[]
    def add(value,base):
        if not isinstance(value,(str,Path)) or not str(value):return
        value=Path(value)
        path=value.resolve() if value.is_absolute() else (base/value).resolve()
        references.add(str(path))
        if path.is_dir():
            if (path/'manifest.json').is_file():pending.append(path/'manifest.json')
            if (path/'reconstruction_report.json').is_file():pending.append(path/'reconstruction_report.json')
        elif path.suffix.lower()=='.json' and path.is_file():pending.append(path)
    def walk(value,base,field=''):
        check()
        if isinstance(value,Path):add(value,base)
        elif isinstance(value,dict):
            if value.get('root_id')=='workspace' and isinstance(value.get('relative_path'),str):
                add(data_backup.relative(workspace,value['relative_path']),workspace)
            for key,item in value.items():
                if key=='source_lineage':continue
                walk(item,base,key)
        elif isinstance(value,(tuple,list)):
            for item in value:walk(item,base,'path' if field in PATH_LISTS else field)
        elif isinstance(value,str) and field in PATH_FIELDS:
            # Source labels such as simulation/recorded are not file paths.
            if field in WORKSPACE_PATHS or Path(value).is_absolute() or '/' in value or '\\' in value or Path(value).suffix:
                add(value,workspace if field in WORKSPACE_PATHS else base)
    store=HistoryStore(outputs)
    if store.path.is_file():
        with store.connection(check=check) as connection:
            for row in connection.execute('SELECT DISTINCT r.payload,r.root_id,r.relative_path FROM records r JOIN membership m ON r.id=m.record_id'):
                check();db_rows+=1
                if exclude_owner and store.locate(row['root_id'],row['relative_path']).resolve().is_relative_to(exclude_owner):
                    continue
                add(store.locate(row['root_id'],row['relative_path']),workspace)
                walk(_unpack(json.loads(row['payload'])),workspace)
    execution=outputs/'history_metadata/execution.sqlite3'
    if execution.is_file():
        with closing(sqlite3.connect(execution.as_uri()+'?mode=ro',uri=True,timeout=0)) as connection:
            for snapshot, in connection.execute('SELECT snapshot FROM runs'):
                check();db_rows+=1;document=json.loads(snapshot)
                for asset in document.get('assets',[]):add(asset['path'],workspace)
    for pattern in PATTERNS:pending.extend(outputs.glob(pattern))
    pending.extend(datasets.glob('scene_*_local_*/manifest.json'))
    while pending:
        check();path=pending.pop().resolve()
        if exclude_owner and path.is_relative_to(exclude_owner):continue
        if path in visited:continue
        visited.add(path)
        data_backup.no_links(path)
        if path.stat().st_size>100*1024**2:
            raise ValueError('关联元数据超过100 MiB读取预算：'+str(path))
        document=json.loads(path.read_text('utf-8-sig'))
        if not isinstance(document,dict):raise ValueError('关联元数据不是对象：'+str(path))
        if document.get('schema')=='femto-metadata-publication-v1':
            # Publication targets belong to the directory owning .publications,
            # not the receipt directory. A published stage normally no longer
            # exists; it is not a missing source asset.
            from femto.asset_storage import inspect_publication
            publication=inspect_publication(path,check)
            publications.append(dict(journal=str(path),status=publication['status']))
            if publication['receipt']['state']=='published':add(publication['target'],path.parent.parent)
            if Path(publication['stage']).is_file():add(publication['stage'],path.parent)
            continue
        # Relative paths in capture/region manifests are document-local. Historic
        # absolute source links retain their original binding automatically.
        walk(document,path.parent)
    paths=[Path(value) for value in sorted(references)]
    return dict(scope='active indexed records, published workspace artifacts, dataset manifest chains and device evidence',
                references=[str(p) for p in paths],missing=[str(p) for p in paths if not p.exists()],
                metadata_files=len(visited),database_rows=db_rows,
                publication_receipts=publications,
                portable_lineage_is_not_external_dependency=True)


def backup_workspace(workspace,destination,*,roots=None,check=lambda:None,progress=lambda message:None,
                     allow_existing_missing=False):
    workspace=Path(workspace).resolve();roots=default_roots(workspace) if roots is None else roots
    for path in roots.values():
        root=Path(path).resolve()
        if workspace.is_relative_to(root) or root==Path(root.anchor):
            raise ValueError('请选择采集数据、模型与方案等数据目录，不备份整个项目或磁盘根目录。')
    return data_backup.backup(roots,destination,workspace=workspace,check=check,progress=progress,
        allow_existing_missing=allow_existing_missing,
        database_files=[('outputs','history_metadata/'+name) for name in ('history.sqlite3','execution.sqlite3')
                        if 'outputs' in roots],
        reference_scan=lambda check:inspect_references(workspace,check=check))
