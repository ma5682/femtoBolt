"""Local asset locations and recoverable metadata publication.

No geometry policy lives here. Existing archives keep their validation contract.
Publication serializes cooperating writers; external edits are checked before
replace. Recovery never deletes assets or overwrites an unexpected target.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path, PureWindowsPath
import time
import uuid

from femto.io_utils import load_json_object, sha256_file, write_json_strict


def resolve_relative(root, name):
    """Resolve an untrusted relative name within a configured storage root."""
    if not isinstance(name, str) or not name or '\0' in name:
        raise ValueError('Asset location must be a nonempty relative path')
    windows = PureWindowsPath(name)
    if windows.drive or windows.root or Path(name).is_absolute():
        raise ValueError('Absolute asset locations are not allowed')
    if '..' in windows.parts or '..' in Path(name).parts:
        raise ValueError('Asset location traverses its storage root')
    base = Path(root).resolve()
    path = (base / name).resolve()
    if not path.is_relative_to(base) or path == base:
        raise ValueError('Asset location is outside its storage root')
    return path


class LocalAssets:
    def __init__(self, roots):
        self.roots = {str(key): Path(value).resolve() for key, value in roots.items()}

    def locate(self, location, *, require_file=True):
        root = self.roots[location['root_id']]
        path = resolve_relative(root, location['relative_path'])
        if require_file and not path.is_file():
            raise FileNotFoundError(path)
        return path

    def describe(self, path, check=lambda: None):
        path = Path(path).resolve()
        for key, root in sorted(self.roots.items(), key=lambda item: len(str(item[1])), reverse=True):
            if path.is_relative_to(root) and path != root:
                return dict(root_id=key, relative_path=path.relative_to(root).as_posix(),
                            bytes=path.stat().st_size, sha256=sha256_file(path, check))
        raise ValueError('Asset is outside configured storage roots')


class PublicationError(OSError):
    def __init__(self, message, journal, *, published=False):
        super().__init__(message)
        self.journal = Path(journal)
        self.published = published


@contextmanager
def publication_lock(parent, check=lambda: None, seconds=5):
    """Cross-process lock for cooperating publishers; released by OS on crash."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    with (parent/'.publication.lock').open('a+b') as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b'\0'); stream.flush()
        deadline = time.monotonic()+seconds
        while True:
            check()
            stream.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Another metadata publication is still running')
                time.sleep(.025)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _fingerprint(path, check=lambda: None):
    return sha256_file(path, check) if path.exists() else None


def _write_receipt(path, data):
    write_json_strict(path, data, durable=True)


def publish_json(target, payload, check=lambda: None, *, commit=None, create_only=False):
    """Publish metadata; an optional commit callback owns only the final replace."""
    target = Path(target).absolute()
    parent = target.parent.resolve()
    target = resolve_relative(parent, target.name)
    with publication_lock(parent, check):
        if create_only and target.exists():
            raise FileExistsError(target)
        token = uuid.uuid4().hex
        receipts = resolve_relative(parent, '.publications')
        receipts.mkdir(exist_ok=True)
        journal = receipts/(token+'.json')
        stage = receipts/(token+'.stage')
        data = dict(schema='femto-metadata-publication-v1', operation_id=token,
                    target=target.name, stage=stage.name, state='preparing',
                    before_sha256=_fingerprint(target, check), sha256=None)
        published = False
        try:
            check()
            _write_receipt(journal, data)
            write_json_strict(stage, payload, durable=True, check=check)
            data.update(sha256=sha256_file(stage, check), state='ready')
            _write_receipt(journal, data)
            check()
            if _fingerprint(target, check) != data['before_sha256']:
                raise ValueError('Metadata target changed during publication')
            def replace():
                nonlocal published
                os.replace(stage, target)
                # A callback may fail after invoking its action. Preserve the
                # actual filesystem outcome independently of callback success.
                published = True
            if commit is not None:
                commit(replace)
            else:
                replace()
            data['state'] = 'published'
            _write_receipt(journal, data)
            return target
        except BaseException as error:
            # A ready receipt remains sufficient to inspect a crash between
            # file publication and receipt completion. No source is removed.
            if not published:
                data.update(state='failed', error=str(error)[:2048])
                try:
                    _write_receipt(journal, data)
                except OSError:
                    pass
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise PublicationError(str(error), journal, published=published) from error


def inspect_publication(journal, check=lambda: None):
    """Read-only inspection. Locations come from the journal's parent, not data."""
    journal = Path(journal).resolve()
    if journal.parent.name != '.publications':
        raise ValueError('Not a metadata publication journal')
    data = load_json_object(journal)
    token = data.get('operation_id')
    if (data.get('schema') != 'femto-metadata-publication-v1' or
            not isinstance(token, str) or len(token) != 32 or
            any(c not in '0123456789abcdef' for c in token) or journal.stem != token or
            data.get('stage') != token+'.stage'):
        raise ValueError('Invalid metadata publication identity')
    state = data.get('state')
    if not isinstance(state, str) or state not in ('preparing', 'ready', 'published', 'failed'):
        raise ValueError('Invalid metadata publication state')
    parent = journal.parent.parent
    target = resolve_relative(parent, data['target'])
    if target.parent != parent:
        raise ValueError('Publication target must be in the owning directory')
    stage = resolve_relative(journal.parent, data['stage'])
    current = _fingerprint(target, check)
    expected = data.get('sha256')
    if state == 'published':
        status = 'published' if expected and current == expected else 'superseded_or_missing'
    elif state == 'preparing':
        status = 'incomplete'
    elif current != data.get('before_sha256'):
        status = ('published_unrecorded' if state == 'ready' and expected and current == expected
                  else 'conflict')
    elif expected and _fingerprint(stage, check) == expected:
        status = 'recoverable' if state == 'ready' else 'failed_retained'
    elif state == 'ready' and expected and current == expected:
        status = 'published_unrecorded'
    else:
        status = 'incomplete'
    return dict(status=status, journal=str(journal), target=str(target), stage=str(stage), receipt=data)


def recover_publication(journal, check=lambda: None):
    """Finalize an interrupted ready operation. Failed/cancelled work stays failed."""
    journal = Path(journal).resolve()
    inspect_publication(journal, check)  # Reject invalid requests before creating a lock file.
    with publication_lock(journal.parent.parent, check):
        item = inspect_publication(journal, check)
        status = item['status']
        if status == 'published':
            return item
        if item['receipt']['state'] != 'ready' or status not in ('recoverable', 'published_unrecorded'):
            raise ValueError('Publication cannot be recovered: '+status)
        check()
        if status == 'recoverable':
            os.replace(item['stage'], item['target'])
        data = item['receipt']
        data['state'] = 'published'
        _write_receipt(journal, data)
        return inspect_publication(journal, check)
