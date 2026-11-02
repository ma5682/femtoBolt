"""Delete one hidden capture directory under an exclusive maintenance lease."""
import hashlib
import json
import os
from pathlib import Path
import shutil

from femto.data_access import maintenance_session
from femto.data_backup import no_links, assert_no_pending_restore
from femto.history_store import HistoryStore
from femto.pipeline.data_maintenance import inspect_references


def _target(workspace, target):
    workspace = Path(workspace).resolve()
    raw = Path(target).absolute()
    for part in (raw,*raw.parents):
        if part.exists() or part.is_symlink():no_links(part)
    target = raw.resolve()
    allowed = ((workspace/'datasets', 'scene_*_local_*'),
               (workspace/'outputs', 'complete_demo_*'))
    if not any(target.parent == root and target.match(pattern) for root,pattern in allowed):
        raise ValueError('只允许删除本工作区采集历史中的单条数据目录。')
    if str(target) not in HistoryStore(workspace/'outputs').deletions():
        raise ValueError('此记录尚未移除，或已被恢复；请先从历史列表移除。')
    return workspace, target


def _inventory(target, check):
    entries=[]
    def fail(error):raise error
    if target.exists():
        if not target.is_dir():raise ValueError('历史目录已变成其他文件类型。')
        for directory, dirs, files in os.walk(target, followlinks=False, onerror=fail):
            check();no_links(Path(directory))
            for name in sorted(dirs+files):
                check();path=Path(directory)/name;no_links(path)
                stat=path.stat()
                entries.append((str(path.relative_to(target)),path.is_dir(),stat.st_size,stat.st_mtime_ns))
    entries.sort()
    digest=hashlib.sha256(json.dumps(entries,ensure_ascii=False).encode()).hexdigest()
    return dict(files=sum(not e[1] for e in entries),bytes=sum(e[2] for e in entries if not e[1]),digest=digest)


def preview_purge(workspace, target, *, check=lambda:None):
    workspace,target=_target(workspace,target)
    inventory=_inventory(target,check)
    evidence=inspect_references(workspace,check=check,exclude_owner=target)
    blockers=[]
    for name in evidence['references']:
        path=Path(name)
        if path==target or path.is_relative_to(target):blockers.append(name)
    return dict(target=str(target),**inventory,blockers=sorted(set(blockers)),
                exists=target.exists())


def execute_purge(workspace, target, expected_digest=None, *, check=lambda:None, workstation_lease=None):
    # Never remove files while any cooperating workstation is holding data.
    with maintenance_session(workspace,workstation_lease):
        assert_no_pending_restore(workspace)
        plan=preview_purge(workspace,target,check=check)
        if plan['blockers']:raise ValueError('其他记录仍引用此目录，请先处理关联数据。')
        if expected_digest is not None and plan['digest']!=expected_digest:raise ValueError('目录内容已变化，请重新检查后确认。')
        check()
        directory=Path(plan['target'])
        if directory.exists():shutil.rmtree(directory)
        # Keep the hidden tombstone until physical removal fully succeeds.
        store=HistoryStore(Path(workspace)/'outputs')
        with store.connection(write=True) as connection:
            records=connection.execute('SELECT id,root_id,relative_path FROM records').fetchall()
            for row in records:
                if store.locate(row['root_id'],row['relative_path']).resolve().is_relative_to(directory):
                    connection.execute('DELETE FROM membership WHERE record_id=?',(row['id'],))
                    connection.execute('DELETE FROM records WHERE id=?',(row['id'],))
            for row in connection.execute('SELECT identity,root_id,relative_path FROM hidden').fetchall():
                if store.locate(row['root_id'],row['relative_path']).resolve().is_relative_to(directory):
                    connection.execute('DELETE FROM hidden WHERE identity=?',(row['identity'],))
        return plan
