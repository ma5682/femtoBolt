"""Cold data-root backup/restore, with no overwrite or provenance rewriting.

Callers must stop asset writers. SQLite databases are write-locked together;
full inventories check other file changes but are not a filesystem transaction.
Bundles do not rebase absolute provenance paths or overwrite existing roots.
"""
from contextlib import closing, contextmanager
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import sqlite3
import time
import uuid

from femto.data_access import DataLease

SCHEMA = 'femto-data-roots-backup-v1'
RESERVE = 64*1024*1024


def physical(path):
    """Use extended Win32 file paths without changing manifest/provenance paths."""
    path=Path(path)
    value=str(path.absolute())
    if os.name=='nt' and not value.startswith('\\\\?\\') and len(value)>=240:
        return Path('\\\\?\\UNC\\'+value[2:] if value.startswith('\\\\') else '\\\\?\\'+value)
    return path


def location_key(path):
    value=str(Path(path).absolute())
    if os.name=='nt':
        if value.startswith('\\\\?\\UNC\\'):value='\\\\'+value[8:]
        elif value.startswith('\\\\?\\'):value=value[4:]
        value=os.path.normcase(value)
    return value


def walk_files(root,check=lambda:None):
    """Enumerate deep trees explicitly; os.walk can silently skip Win32 errors."""
    pending=[Path(root)]
    while pending:
        folder=pending.pop();check();no_links(folder)
        with os.scandir(physical(folder)) as entries:
            for entry in entries:
                check();path=folder/entry.name;no_links(path)
                if entry.is_dir(follow_symlinks=False):pending.append(path)
                elif entry.is_file(follow_symlinks=False):yield path
                else:raise ValueError('Unsupported filesystem asset: '+str(path))


def digest(path, check=lambda: None):
    result = hashlib.sha256()
    with physical(path).open('rb') as stream:
        while chunk := stream.read(1024*1024):
            check()
            result.update(chunk)
    return result.hexdigest()


def relative(root, name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('Invalid bundle path')
    parts = name.split('/')
    windows = PureWindowsPath(name)
    if windows.drive or windows.root or any(p in ('', '.', '..') or ':' in p or p[-1:] in (' ', '.') for p in parts):
        raise ValueError('Bundle path escapes its root')
    if any(re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])', p.split('.')[0], re.I) for p in parts):
        raise ValueError('Reserved bundle path')
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Bundle path escapes through a link')
    return physical(path)


def no_links(path):
    path=physical(path)
    value = path.lstat()
    if path.is_symlink() or getattr(value, 'st_file_attributes', 0) & 0x400:
        raise ValueError('Links and Windows reparse points are not backup assets: '+str(path))


def inventory(root, *, excluded=(), check=lambda: None):
    result = {}
    excluded_keys={location_key(path) for path in excluded}
    for path in walk_files(root,check):
        if location_key(path) in excluded_keys:
            continue
        key = path.relative_to(root).as_posix()
        result[key] = dict(bytes=physical(path).stat().st_size, sha256=digest(path, check))
    return result


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def space(parent, size, reserve=RESERVE):
    free = shutil.disk_usage(parent).free
    if free < size+reserve:
        raise OSError('磁盘空间不足：需要约%d字节（含预留），可用%d字节' % (size+reserve, free))


def check_database(path):
    # Only detached snapshots are passed here. immutable prevents validation
    # creating WAL/SHM files or consulting stale journals alongside a bundle.
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True)) as database:
        if database.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
            raise ValueError('Database integrity check failed: '+str(path))
        if database.execute('PRAGMA foreign_key_check').fetchone() is not None:
            raise ValueError('Database foreign key check failed: '+str(path))


@contextmanager
def locked_databases(paths):
    """Acquire all SQLite write reservations before reading either snapshot."""
    connections = []
    try:
        for path in sorted(paths):
            connection = sqlite3.connect(path.as_uri()+'?mode=rw', uri=True, timeout=.1)
            connections.append(connection)
            connection.execute('BEGIN IMMEDIATE')
        yield
    finally:
        for connection in reversed(connections):
            try:
                connection.rollback()
            finally:
                connection.close()


def normalized_roots(roots):
    for path in roots.values():
        raw=Path(path)
        if raw.exists():
            no_links(raw)
            if not raw.is_dir():raise ValueError('Data roots must be directories: '+str(raw))
    values = {name: Path(path).resolve() for name, path in roots.items()}
    if not values or any(not re.fullmatch('[a-z][a-z0-9_]{0,31}', name) for name in values):
        raise ValueError('Invalid data-root identifiers')
    for name, root in values.items():
        if any(root.is_relative_to(other) or other.is_relative_to(root)
               for key, other in values.items() if key != name):
            raise ValueError('Data roots overlap')
    return values


def backup(roots, destination, *, workspace, **options):
    with DataLease(workspace,maintenance=True):
        assert_no_pending_restore(workspace)
        return _backup(roots,destination,**options)


def _backup(roots, destination, *, database_files=(), references=(), check=lambda: None, reserve=RESERVE,
            reference_scan=None,progress=lambda message:None,allow_existing_missing=False):
    """Capture all explicitly configured roots, refusing missing external references.

    database_files names are (root_id, relative_path), including absent known
    databases so creation during capture is detected. Asset writers must be
    stopped; this function alone is not an online filesystem transaction.
    """
    roots = normalized_roots(roots)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if any(destination.is_relative_to(root) for root in roots.values()):
        raise ValueError('Backup destination cannot be inside a data root')
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    for reference in references:
        reference = Path(reference).resolve()
        if not reference.exists():
            raise FileNotFoundError('关联资产已缺失：'+str(reference))
        if not any(reference.is_relative_to(root) for root in roots.values()):
            raise ValueError('关联资产不在备份数据根内：'+str(reference))
    declarations=set(database_files)
    estimated_bytes=0
    # Include every SQLite main file, not only the two default metadata stores.
    # Old isolated/catalogue databases must not be copied raw with a live WAL.
    for name,root in roots.items():
        no_links(root)
        for path in walk_files(root,check):
            estimated_bytes+=physical(path).stat().st_size
            with physical(path).open('rb') as stream:
                if stream.read(16)==b'SQLite format 3\0':
                    declarations.add((name,path.relative_to(root).as_posix()))
    databases = [(name, key, relative(roots[name], key)) for name, key in sorted(declarations)]
    if len({path for _, _, path in databases}) != len(databases):
        raise ValueError('Duplicate database declarations')
    present = [(name, key, path) for name, key, path in databases if path.is_file()]
    for _,_,path in databases:
        if not path.is_file() and any(Path(str(path)+suffix).exists() for suffix in ('-wal','-shm')):
            raise ValueError('Database main file missing while journal exists: '+str(path))
    space(destination.parent,estimated_bytes,reserve)
    excluded = {p for _, _, database in databases for p in
                (database, Path(str(database)+'-wal'), Path(str(database)+'-shm'))}
    stage = destination.with_name('.'+destination.name+'.'+uuid.uuid4().hex+'.partial')
    started = time.time()
    stage.mkdir()
    try:
        with locked_databases([p for _, _, p in present]):
            progress('检查关联记录与源资产')
            association=reference_scan(check) if reference_scan is not None else dict(references=[])
            association['bindings']=[]
            association['unavailable']=[]
            for item in association['references']:
                reference=Path(item).resolve()
                if not reference.exists():
                    if not allow_existing_missing:raise FileNotFoundError('关联资产已缺失：'+str(reference))
                    association['unavailable'].append(dict(path=str(reference),reason='missing_before_backup'))
                    continue
                matches=[(name,root) for name,root in roots.items() if reference.is_relative_to(root)]
                if not matches:
                    raise ValueError('关联资产不在备份数据根内，请添加其数据目录：'+str(reference))
                name,root=matches[0]
                association['bindings'].append(dict(root=name,relative_path=reference.relative_to(root).as_posix(),
                                                    kind='directory' if reference.is_dir() else 'file'))
            association['source_assets_complete']=not association['unavailable']
            progress('计算完整数据清单与摘要')
            before = {name: inventory(root, excluded=excluded, check=check) for name, root in roots.items()}
            required = sum(item['bytes'] for rows in before.values() for item in rows.values())
            required += sum(path.stat().st_size+Path(str(path)+'-wal').stat().st_size
                            if Path(str(path)+'-wal').exists() else path.stat().st_size for _, _, path in present)
            space(stage, required, reserve)
            contents = stage/'contents'
            contents.mkdir()
            for name, rows in before.items():
                progress('复制数据：'+name)
                root_target = contents/name
                root_target.mkdir()
                for key, record in rows.items():
                    check()
                    target = relative(root_target, key)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with relative(roots[name], key).open('rb') as source, target.open('xb') as sink:
                        while chunk := source.read(1024*1024):
                            check()
                            space(stage, len(chunk), reserve)
                            sink.write(chunk)
                        sink.flush()
                        os.fsync(sink.fileno())
                    if digest(target, check) != record['sha256']:
                        raise ValueError('Source changed while copying: '+key)
            for name, key, path in present:
                check()
                progress('备份数据库：'+name+'/'+key)
                target = relative(contents/name, key)
                target.parent.mkdir(parents=True, exist_ok=True)
                # Separate reader works while BEGIN IMMEDIATE holds write reservation.
                with closing(sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)) as source:
                    with closing(sqlite3.connect(target)) as sink:
                        source.backup(sink, pages=128, progress=lambda *_: check())
                check_database(target)
            for binding in association['bindings']:
                if binding['kind']=='directory':
                    base=contents/binding['root'];key=binding['relative_path']
                    (base if key=='.' else relative(base,key)).mkdir(parents=True,exist_ok=True)
            progress('重新核对源数据与备份副本')
            after = {name: inventory(root, excluded=excluded, check=check) for name, root in roots.items()}
            if before != after or [p for _, _, p in databases if p.is_file()] != [p for _, _, p in present]:
                raise ValueError('Data changed during backup; bundle was not published')
            manifest = dict(schema=SCHEMA, state='complete', started_unix=started, finished_unix=time.time(),
                            consistency='cold assets with simultaneous SQLite write reservations',
                            asset_writers_must_be_stopped=True, paths_rebased=False,
                            reference_audit=association,
                            roots={name:dict(origin=str(root), files=inventory(contents/name, check=check))
                                   for name, root in roots.items()},
                            databases=[dict(root=name,path=key,present=path.is_file()) for name,key,path in databases])
            write_json(stage/'manifest.json', manifest)
            (stage/'manifest.sha256').write_text(digest(stage/'manifest.json'), encoding='ascii')
            verify(stage, check=check)
            check()
            if destination.exists():
                raise FileExistsError(destination)
            stage.rename(destination)
        return dict(destination=str(destination), manifest=manifest)
    except BaseException as error:
        if stage.is_dir() and not (stage/'failure.json').exists():
            write_json(stage/'failure.json', dict(error=repr(error), state='incomplete'))
        raise


def verify(bundle, *, check=lambda: None):
    bundle = Path(bundle).resolve()
    no_links(bundle)
    manifest_path = bundle/'manifest.json'
    no_links(manifest_path);no_links(bundle/'manifest.sha256')
    check();manifest_bytes=manifest_path.read_bytes()
    verified_digest=hashlib.sha256(manifest_bytes).hexdigest()
    if verified_digest != (bundle/'manifest.sha256').read_text('ascii').strip():
        raise ValueError('Backup manifest checksum mismatch')
    manifest = json.loads(manifest_bytes)
    if manifest.get('schema') != SCHEMA or manifest.get('state') != 'complete':
        raise ValueError('Incomplete or unsupported data bundle')
    normalized_roots({name:bundle/'contents'/name for name in manifest['roots']})
    expected_top = {'contents', 'manifest.json', 'manifest.sha256'}
    if {p.name for p in bundle.iterdir()} != expected_top:
        raise ValueError('Unexpected/incomplete bundle contents')
    if {p.name for p in (bundle/'contents').iterdir()} != set(manifest['roots']):
        raise ValueError('Unexpected data roots')
    for name, item in manifest['roots'].items():
        check()
        keys = list(item['files'])
        if len({p.casefold() for p in keys}) != len(keys):
            raise ValueError('Case-colliding bundle paths')
        for key in keys:
            relative(bundle/'contents'/name, key)
        if inventory(bundle/'contents'/name, check=check) != item['files']:
            raise ValueError('Backup asset checksum or inventory mismatch: '+name)
    for database in manifest['databases']:
        if database['present']:
            check_database(relative(bundle/'contents'/database['root'],database['path']))
    for binding in manifest.get('reference_audit',{}).get('bindings',[]):
        base=bundle/'contents'/binding['root'];key=binding['relative_path']
        path=base if key=='.' else relative(base,key)
        if not (path.is_dir() if binding['kind']=='directory' else path.is_file()):
            raise ValueError('Backup is missing a declared associated asset: '+str(path))
    manifest['_verified_sha256']=verified_digest
    return manifest


def restore_new(bundle, destinations, *, workspace, expected_manifest_sha=None, **options):
    with DataLease(workspace,maintenance=True):
        assert_no_pending_restore(workspace)
        folder=Path(workspace).resolve()/'.data_restores'
        folder.mkdir(exist_ok=True)
        token=uuid.uuid4().hex
        journal=folder/(token+'.json')
        state=dict(schema='femto-data-restore-v1',token=token,state='preparing',
                   bundle=str(Path(bundle).resolve()),workspace=str(Path(workspace).resolve()),
                   destinations={n:str(p) for n,p in normalized_roots(destinations).items()},stages={})
        # Verification and refusal happen before recording a pending operation.
        manifest=verify(bundle,check=options.get('check',lambda:None))
        if set(state['destinations'])!=set(manifest['roots']):
            raise ValueError('Restore destinations must cover every data root')
        source_bundle=Path(bundle).resolve()
        for target in state['destinations'].values():
            path=Path(target)
            if path.is_relative_to(source_bundle) or source_bundle.is_relative_to(path):
                raise ValueError('Restore destination overlaps bundle')
            if not path.parent.is_dir():raise FileNotFoundError(path.parent)
        if expected_manifest_sha is not None and manifest['_verified_sha256']!=expected_manifest_sha:
            raise ValueError('备份清单在预览后已改变，请重新读取并核对。')
        state['manifest_sha256']=manifest['_verified_sha256']
        manifest_bytes=(Path(bundle).resolve()/'manifest.json').read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest()!=state['manifest_sha256']:
            raise ValueError('备份清单在恢复准备前已改变，请重新读取。')
        for target in state['destinations'].values():
            if Path(target).exists():raise FileExistsError(target)
        space(folder,len(manifest_bytes))
        with (folder/(token+'.manifest')).open('xb') as stream:
            stream.write(manifest_bytes);stream.flush();os.fsync(stream.fileno())
        write_json(journal,state)
        return _restore_new(bundle,destinations,journal=journal,state=state,**options)


def pending_restores(workspace):
    result=[]
    for path in (Path(workspace).resolve()/'.data_restores').glob('*.json'):
        try:
            no_links(path);state=json.loads(path.read_text('utf-8'))
            terminal=(isinstance(state,dict) and state.get('schema')=='femto-data-restore-v1'
                      and state.get('token')==path.stem and state.get('state') in ('restored','aborted'))
        except (OSError,ValueError,TypeError):terminal=False
        if not terminal:
            result.append(path)
    return result


def assert_no_pending_restore(workspace):
    pending=pending_restores(workspace)
    if pending:
        raise RuntimeError('上次数据恢复未完成，请先继续或中止恢复：'+str(pending[0]))


def update_json(path,value):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.update')
    write_json(temporary,value)
    os.replace(temporary,path)


def restore_marker(state,name,ready):
    return dict(schema='femto-data-restore-owner-v1',token=state['token'],root=name,
                manifest_sha256=state['manifest_sha256'],ready=ready)


def owned_tree(path,state,name,manifest,*,require_ready=True,check=lambda:None):
    marker=path/'.femto_restore.json'
    no_links(path);no_links(marker)
    value=json.loads(marker.read_text('utf-8'))
    if (type(value.get('ready')) is not bool or value!=restore_marker(state,name,value.get('ready'))
            or require_ready and not value['ready']):
        raise ValueError('恢复暂存或目标不属于本次完整恢复：'+str(path))
    if inventory(path,excluded={marker},check=check)!=manifest['roots'][name]['files']:
        raise ValueError('恢复暂存或目标内容已改变：'+str(path))
    return marker


def load_restore(journal,workspace,destinations,*,allow_partial=False):
    journal=Path(journal).resolve();workspace=Path(workspace).resolve()
    if journal.parent!=workspace/'.data_restores':
        raise ValueError('Restore journal is outside this workspace')
    no_links(journal)
    state=json.loads(journal.read_text('utf-8'))
    expected={n:str(p) for n,p in normalized_roots(destinations).items()}
    if (state.get('schema')!='femto-data-restore-v1' or journal.stem!=state.get('token')
            or state.get('workspace')!=str(workspace) or state.get('destinations')!=expected):
        raise ValueError('Restore journal binding mismatch')
    if state['state'] in ('restored','aborted'):
        raise ValueError('Restore operation is already terminal')
    bundle=Path(state['bundle']).resolve()
    snapshot=journal.with_suffix('.manifest');no_links(snapshot);raw=snapshot.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=state.get('manifest_sha256'):
        raise ValueError('恢复事务保留的原始清单已改变')
    manifest=json.loads(raw)
    if manifest.get('schema')!=SCHEMA or manifest.get('state')!='complete' or set(manifest['roots'])!=set(expected):
        raise ValueError('恢复事务保留的清单不匹配')
    if not set(state['stages']).issubset(expected) or not allow_partial and set(state['stages'])!=set(expected):
        raise ValueError('暂存尚未完整准备，请中止本次恢复后重新开始。')
    for name,target in normalized_roots(destinations).items():
        expected_stage=target.with_name('.'+target.name+'.restore.'+state['token']+'.partial')
        if name in state['stages'] and state['stages'][name]!=str(expected_stage):raise ValueError('Restore staging binding mismatch')
    return journal,state,bundle,manifest


def recover_restore(journal,destinations,*,workspace,check=lambda:None):
    with DataLease(workspace,maintenance=True):
        journal,state,bundle,manifest=load_restore(journal,workspace,destinations)
        for name,target in normalized_roots(destinations).items():
            path=target if target.exists() else Path(state['stages'][name])
            if target.exists():owned_published(target,state,name,manifest,check)
            else:owned_tree(path,state,name,manifest,check=check)
        return publish_restore(journal,state,manifest,check)


def abort_restore(journal,destinations,*,workspace):
    """Move only verified owned roots back to retained staging; delete no data."""
    with DataLease(workspace,maintenance=True):
        journal,state,bundle,manifest=load_restore(journal,workspace,destinations,allow_partial=True)
        for name,target in normalized_roots(destinations).items():
            if target.exists():
                owned_published(target,state,name,manifest,lambda:None)
                if Path(state['stages'][name]).exists():raise FileExistsError(state['stages'][name])
        for name,target in normalized_roots(destinations).items():
            if target.exists():
                marker=target/'.femto_restore.json'
                if not marker.exists():write_json(marker,restore_marker(state,name,True))
                target.rename(state['stages'][name])
        state['state']='aborted';update_json(journal,state)
        return dict(state='aborted',retained_stages=state['stages'])


def owned_published(path,state,name,manifest,check):
    marker=path/'.femto_restore.json'
    if marker.exists():return owned_tree(path,state,name,manifest,check=check)
    no_links(path)
    stat=path.stat()
    if (state['state']!='finalizing' or state.get('published_identities',{}).get(name)!=[stat.st_dev,stat.st_ino]
            or inventory(path,check=check)!=manifest['roots'][name]['files']):
        raise ValueError('恢复目标缺少本次发布归属证据：'+str(path))
    return marker


def publish_restore(journal,state,manifest,check):
    if state['state']!='finalizing':
        for name,text in state['destinations'].items():
            check();target=Path(text);stage=Path(state['stages'][name])
            if target.exists():
                owned_tree(target,state,name,manifest,check=check)
            else:
                owned_tree(stage,state,name,manifest,check=check)
                stage.rename(target)
            state['state']='publishing';update_json(journal,state)
        state['published_identities']={n:[Path(p).stat().st_dev,Path(p).stat().st_ino] for n,p in state['destinations'].items()}
        state['state']='finalizing';update_json(journal,state)
    # Every root is present and verified before any incomplete marker is cleared.
    for name,text in state['destinations'].items():
        target=Path(text);marker=target/'.femto_restore.json'
        owned_published(target,state,name,manifest,check)
        if marker.exists():marker.unlink()
    state['state']='restored';update_json(journal,state)
    return dict(state='restored',roots=state['destinations'],journal=str(journal),paths_rebased=False,
                source_assets_complete=manifest.get('reference_audit',{}).get('source_assets_complete',True),
                unavailable_reference_count=len(manifest.get('reference_audit',{}).get('unavailable',[])),
                original_locations=all(state['destinations'][n]==v['origin'] for n,v in manifest['roots'].items()))


def _restore_new(bundle, destinations, *, journal, state, check=lambda: None, reserve=RESERVE):
    """Stage and verify every root before publishing; never replace existing data.

    Explicit destinations avoid trusting paths in an imported manifest. Absolute
    provenance is preserved, so relocation is not a functional migration.
    """
    bundle = Path(bundle).resolve()
    manifest = verify(bundle, check=check)
    destinations = normalized_roots(destinations)
    if set(destinations) != set(manifest['roots']):
        raise ValueError('Restore destinations must cover every data root')
    token = state['token']
    if manifest['_verified_sha256']!=state['manifest_sha256']:
        raise ValueError('备份在恢复准备期间已改变，未发布任何数据根。')
    staged = {}
    try:
        for name, target in destinations.items():
            if target.exists():
                raise FileExistsError('恢复目标已存在，未覆盖：'+str(target))
            if target.is_relative_to(bundle) or bundle.is_relative_to(target):
                raise ValueError('Restore destination overlaps bundle')
            if not target.parent.is_dir():
                raise FileNotFoundError(target.parent)
        volumes={}
        for name,target in destinations.items():
            device=target.parent.stat().st_dev
            parent,total=volumes.get(device,(target.parent,0))
            volumes[device]=(parent,total+sum(item['bytes'] for item in manifest['roots'][name]['files'].values()))
        for parent,total in volumes.values():space(parent,total,reserve)
        for name, target in destinations.items():
            check()
            space(target.parent, sum(r['bytes'] for r in manifest['roots'][name]['files'].values()), reserve)
            stage = target.with_name('.'+target.name+'.restore.'+token+'.partial')
            stage.mkdir()
            staged[name] = stage
            state['stages'][name]=str(stage);update_json(journal,state)
            write_json(stage/'.femto_restore.json',restore_marker(state,name,False))
            for binding in manifest.get('reference_audit',{}).get('bindings',[]):
                if binding['root']==name and binding['kind']=='directory':
                    key=binding['relative_path']
                    (stage if key=='.' else relative(stage,key)).mkdir(parents=True,exist_ok=True)
            for key in manifest['roots'][name]['files']:
                check()
                source = relative(bundle/'contents'/name,key)
                destination = relative(stage,key)
                destination.parent.mkdir(parents=True,exist_ok=True)
                with source.open('rb') as incoming, destination.open('xb') as outgoing:
                    while chunk := incoming.read(1024*1024):
                        check()
                        space(stage,len(chunk),reserve)
                        outgoing.write(chunk)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
            if inventory(stage,excluded={stage/'.femto_restore.json'},check=check) != manifest['roots'][name]['files']:
                raise ValueError('Restored staging checksum mismatch')
            update_json(stage/'.femto_restore.json',restore_marker(state,name,True))
        return publish_restore(journal,state,manifest,check)
    except BaseException as error:
        if state['state']=='finalizing':
            state['error']=repr(error);update_json(journal,state)
            raise
        # Roll back only exact trees we just published. Retain partial staging.
        for name in reversed(list(staged)):
            target=destinations[name]
            if target.exists() and not staged[name].exists():
                owned_tree(target,state,name,manifest)
                target.rename(staged[name])
        state.update(state='failed',error=repr(error));update_json(journal,state)
        raise
