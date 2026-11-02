"""Read-only capacity estimates and conservative cleanup review.

The report is a changing-filesystem estimate, not a backup or authorization to
delete. Names alone cannot prove that draft/publication/restore files are orphaned.
"""
import os
from pathlib import Path
import shutil
import stat
import time

from femto.data_backup import physical


def _kind(relative):
    parts=relative.parts
    if ('.publications' in parts or '.data_restores' in parts or
            any(part.endswith('.partial') for part in parts) or
            relative.name.startswith('.draft_') or relative.suffix in ('.tmp','.stage','.rollback')):
        return 'recovery_review'
    if '__pycache__' in parts and relative.suffix=='.pyc':
        return 'regenerable_cache'
    return 'retained_data'


def inspect_disk(roots, *, check=lambda:None, progress=lambda message:None,
                 sample_limit=100, low_bytes=2*1024**3, low_fraction=.1):
    """Scan explicit nonoverlapping roots without following links or deleting.

    Scan errors make the size incomplete and are surfaced. Sparse files, hard
    links and compression mean logical file bytes differ from allocated space.
    Free space is queried per filesystem device and may change after observation.
    """
    if not isinstance(sample_limit,int) or sample_limit<0:
        raise ValueError('sample_limit must be a nonnegative integer')
    if low_bytes<0 or not 0<=low_fraction<=1:
        raise ValueError('Invalid low-space threshold')
    resolved={str(name):Path(path).absolute() for name,path in roots.items()}
    if not resolved:raise ValueError('请先选择需要统计的数据目录')
    # Never resolve a selected link into an external scan root.
    for root in resolved.values():
        for parent in (root,*root.parents):
            if parent.exists() and (parent.is_symlink() or
                getattr(physical(parent).lstat(),'st_file_attributes',0)&0x400):
                raise ValueError('数据目录不能经符号链接或重解析点扫描：'+str(parent))
    resolved={name:Path(os.path.abspath(path)) for name,path in resolved.items()}
    values=list(resolved.values())
    for i,root in enumerate(values):
        if any(root.is_relative_to(other) or other.is_relative_to(root) for other in values[:i]):
            raise ValueError('数据目录重复或相互包含，无法准确汇总')
    result=dict(schema='femto-disk-preview-v1',started_unix=time.time(),read_only=True,
        estimate_only=True,scan_complete=True,roots={},volumes=[],errors=[],skipped_links=[],
        cleanup=dict(regenerable_cache=dict(files=0,bytes=0,samples=[]),
                     recovery_review=dict(files=0,bytes=0,samples=[])),
        policy='仅预览。疑似Python缓存需确认用途；草稿、发布、恢复暂存需核对，不按名称或后缀删除。')
    devices={}
    def error(path,exc):
        result['scan_complete']=False
        if len(result['errors'])<sample_limit:
            result['errors'].append(dict(path=str(path),error=str(exc)))
        result['error_count']=result.get('error_count',0)+1
    for name,root in resolved.items():
        check();progress('统计磁盘与文件：'+name)
        summary=dict(path=str(root),files=0,bytes=0)
        result['roots'][name]=summary
        try:
            info=physical(root).stat()
            if not stat.S_ISDIR(info.st_mode):raise ValueError('数据根不是目录')
            device=info.st_dev
            if device not in devices:
                usage=shutil.disk_usage(physical(root))
                volume=dict(device=device,path=str(root),total_bytes=usage.total,
                    free_bytes=usage.free,used_bytes=usage.used,
                    low_space=usage.free<max(low_bytes,int(usage.total*low_fraction)))
                devices[device]=volume;result['volumes'].append(volume)
            summary['device']=device
        except (OSError,ValueError) as exc:
            error(root,exc);continue
        pending=[root]
        while pending:
            check();directory=pending.pop()
            try:
                entries=os.scandir(physical(directory))
            except OSError as exc:
                error(directory,exc);continue
            try:
                while True:
                    # Cancellation exceptions must not be mistaken for I/O errors.
                    check()
                    try:entry=next(entries)
                    except StopIteration:break
                    except OSError as exc:
                        error(directory,exc);break
                    path=directory/entry.name
                    try:info=entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        error(path,exc);continue
                    if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
                        result['skipped_link_count']=result.get('skipped_link_count',0)+1
                        if len(result['skipped_links'])<sample_limit:result['skipped_links'].append(str(path))
                        continue
                    if stat.S_ISDIR(info.st_mode):pending.append(path);continue
                    if not stat.S_ISREG(info.st_mode):continue
                    summary['files']+=1;summary['bytes']+=info.st_size
                    kind=_kind(path.relative_to(root.parent))
                    if kind in result['cleanup']:
                        group=result['cleanup'][kind];group['files']+=1;group['bytes']+=info.st_size
                        if len(group['samples'])<sample_limit:
                            group['samples'].append(dict(path=str(path),bytes=info.st_size,modified_unix=info.st_mtime))
            finally:entries.close()
    result['logical_bytes']=sum(value['bytes'] for value in result['roots'].values())
    result['finished_unix']=time.time()
    return result
