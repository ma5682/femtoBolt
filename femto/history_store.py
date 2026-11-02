"""SQLite history metadata. Files and their existing validators remain authoritative."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
import uuid

from femto.asset_storage import resolve_relative
from femto.io_utils import load_json_object

SCHEMA_VERSION = 1


def _pack(value):
    if isinstance(value, Path):
        return ['path', str(value)]
    if isinstance(value, dict):
        return ['dict', [[key, _pack(item)] for key, item in value.items()]]
    if isinstance(value, tuple):
        return ['tuple', [_pack(item) for item in value]]
    if isinstance(value, list):
        return ['list', [_pack(item) for item in value]]
    return ['value', value]


def _unpack(value):
    kind, data = value
    if kind == 'path':
        return Path(data)
    if kind == 'dict':
        return {key: _unpack(item) for key, item in data}
    if kind in ('tuple', 'list'):
        items = [_unpack(item) for item in data]
        return tuple(items) if kind == 'tuple' else items
    if kind == 'value':
        return data
    raise ValueError('Invalid history payload')


class HistoryStore:
    """Connections are short-lived and owned by the calling thread."""
    def __init__(self, outputs):
        self.outputs = Path(outputs).resolve()
        self.workspace = self.outputs.parent
        self.path = self.outputs/'history_metadata'/'history.sqlite3'

    def location(self, path):
        path = Path(path).resolve()
        if path.is_relative_to(self.workspace):
            return 'workspace', path.relative_to(self.workspace).as_posix()
        return 'external_legacy', str(path)

    def locate(self, root_id, relative):
        if root_id == 'workspace':
            return resolve_relative(self.workspace, relative)
        if root_id == 'external_legacy' and Path(relative).is_absolute():
            return Path(relative)
        raise ValueError('Unknown history storage location')

    def identity(self, row):
        capture = 'kind' not in row
        path = (row.get('key') if capture else
                row.get('project_location') or row.get('project') or row.get('target') or row.get('source'))
        if not path:
            raise ValueError('History record has no source identity')
        root, relative = self.location(path)
        # Paths are case-insensitive on this Windows product, but user-facing
        # names, UUIDs and stored location spelling are preserved.
        return json.dumps(['capture' if capture else row['kind'], root, relative.casefold(),
                           row.get('region_index'), row.get('revision_id')],
                          ensure_ascii=False, separators=(',', ':'))

    def _legacy_hidden(self):
        path = self.outputs/'local_history_deleted.json'
        if not path.exists():
            return {}
        data = load_json_object(path)
        if data.get('schema') != 'local-history-deleted-v1' or not isinstance(data.get('records'), dict):
            raise ValueError('Invalid history deletion index')
        return data['records']

    @contextmanager
    def connection(self, *, write=False, check=lambda: None):
        check()
        if write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=.1)
        else:
            connection = sqlite3.connect(self.path.as_uri()+'?mode=ro', uri=True, timeout=0)
        connection.row_factory = sqlite3.Row
        # Query cancellation does not invoke any Qt API from worker threads.
        stopped = []
        def progress():
            try:
                check()
                return 0
            except Exception as error:
                stopped.append(error)
                return 1
        connection.set_progress_handler(progress, 500)
        try:
            connection.execute('PRAGMA foreign_keys=ON')
            if write:
                version = connection.execute('PRAGMA user_version').fetchone()[0]
                if version not in (0, SCHEMA_VERSION):
                    raise ValueError('Unsupported history database version; database was not modified')
                connection.execute('PRAGMA journal_mode=WAL')
                connection.execute('PRAGMA synchronous=FULL')
                connection.execute('BEGIN IMMEDIATE')
                version = connection.execute('PRAGMA user_version').fetchone()[0]
                if version == 0:
                    legacy = self._legacy_hidden()
                    for sql in (
                        'CREATE TABLE records (id TEXT PRIMARY KEY, identity TEXT NOT NULL UNIQUE, '
                        'payload TEXT NOT NULL, search_text TEXT NOT NULL, stamp REAL NOT NULL, '
                        'project_id TEXT, revision_id TEXT, parent_revision_id TEXT, '
                        'root_id TEXT NOT NULL, relative_path TEXT NOT NULL)',
                        'CREATE TABLE membership (collection TEXT NOT NULL, record_id TEXT NOT NULL '
                        'REFERENCES records(id), ordinal INTEGER NOT NULL, PRIMARY KEY(collection, record_id))',
                        'CREATE TABLE hidden (identity TEXT PRIMARY KEY, root_id TEXT NOT NULL, '
                        'relative_path TEXT NOT NULL, name TEXT NOT NULL, changed_at REAL NOT NULL)',
                        'CREATE INDEX version_lookup ON records(project_id,revision_id)',
                        'CREATE INDEX parent_lookup ON records(parent_revision_id)',
                        'CREATE TABLE migrations (version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)',
                    ):
                        connection.execute(sql)
                    for path, item in legacy.items():
                        if not isinstance(item, dict):
                            raise ValueError('Invalid legacy hidden record')
                        location = self.location(path)
                        connection.execute('INSERT INTO hidden VALUES (?,?,?,?,?)',
                            (self.identity({'key': path}), *location, str(item.get('name', '')),
                             float(item.get('deleted_at', 0))))
                    connection.execute('INSERT INTO migrations VALUES (?,?)', (1, time.time()))
                    connection.execute('PRAGMA user_version=1')
                elif version != SCHEMA_VERSION:
                    raise ValueError('Unsupported history database version; database was not modified')
            elif connection.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION:
                raise ValueError('History database requires a supported migration')
            yield connection
            if write:
                check()
                connection.commit()
        except sqlite3.OperationalError:
            if stopped:
                raise stopped[0]
            raise
        finally:
            if connection.in_transaction:
                connection.rollback()
            connection.close()

    def replace(self, collection, rows, *, search_text, check=lambda: None):
        """Replace one fully scanned collection atomically, retaining visibility."""
        prepared = []
        seen = set()
        for row in rows:
            check()
            identity = self.identity(row)
            if identity in seen:
                raise ValueError('Duplicate record identity in history snapshot')
            seen.add(identity)
            path = row.get('key') if 'kind' not in row else row.get('project_location') or row.get('project') or row.get('target') or row.get('source')
            location = self.location(path)
            payload = json.dumps(_pack(row), ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            prepared.append((identity, payload, search_text(row).casefold(),
                             float(row.get('stamp', row.get('time', 0))) if 'kind' not in row or 'stamp' in row else 0,
                             row.get('project_id'), row.get('revision_id'), row.get('parent_revision_id'), *location))
        with self.connection(write=True, check=check) as connection:
            connection.execute('DELETE FROM membership WHERE collection=?', (collection,))
            for ordinal, values in enumerate(prepared):
                check()
                connection.execute(
                    'INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(identity) DO UPDATE SET '
                    'payload=excluded.payload,search_text=excluded.search_text,stamp=excluded.stamp,'
                    'project_id=excluded.project_id,revision_id=excluded.revision_id,'
                    'parent_revision_id=excluded.parent_revision_id,root_id=excluded.root_id,relative_path=excluded.relative_path',
                    (uuid.uuid4().hex, *values))
                record_id = connection.execute('SELECT id FROM records WHERE identity=?', (values[0],)).fetchone()[0]
                connection.execute('INSERT INTO membership VALUES (?,?,?)', (collection, record_id, ordinal))

    @staticmethod
    def _query_clause(collection, text, include_hidden, hidden_only):
        clause = (' FROM membership m JOIN records r ON r.id=m.record_id '
                  'LEFT JOIN hidden h ON h.identity=r.identity WHERE m.collection=? '
                  'AND instr(r.search_text,?)>0')
        if hidden_only:
            clause += ' AND h.identity IS NOT NULL'
        elif not include_hidden:
            clause += ' AND h.identity IS NULL'
        return clause, (collection, text.strip().casefold())

    def query(self, collection, text='', *, limit=100000, offset=0, include_hidden=False,
              hidden_only=False, check=lambda: None):
        if not 1 <= limit <= 100000 or offset < 0:
            raise ValueError('Invalid history page bounds')
        with self.connection(check=check) as connection:
            connection.execute('BEGIN')  # Page and total use one consistent snapshot.
            clause, parameters = self._query_clause(collection, text, include_hidden, hidden_only)
            count = connection.execute('SELECT count(*)'+clause, parameters).fetchone()[0]
            rows = connection.execute('SELECT r.identity,r.payload'+clause+
                ' ORDER BY m.ordinal,r.id LIMIT ? OFFSET ?', (*parameters, limit, offset)).fetchall()
            return dict(total=count, identities=[row['identity'] for row in rows],
                        rows=[_unpack(json.loads(row['payload'])) for row in rows])

    def matching_identities(self, collection, text='', *, hidden_only=False, check=lambda: None):
        """Bounded UI lookup: no payload decoding or filesystem inspection."""
        with self.connection(check=check) as connection:
            clause, parameters = self._query_clause(collection, text, False, hidden_only)
            return {row[0] for row in connection.execute('SELECT r.identity'+clause, parameters)}

    def set_hidden(self, rows, hidden=True, check=lambda: None):
        with self.connection(write=True, check=check) as connection:
            for row in rows:
                check()
                identity = self.identity(row)
                if hidden:
                    path = row.get('key') if 'kind' not in row else row.get('project_location') or row.get('project') or row.get('target') or row.get('source')
                    connection.execute('INSERT INTO hidden VALUES (?,?,?,?,?) ON CONFLICT(identity) DO UPDATE SET '
                        'name=excluded.name,changed_at=excluded.changed_at',
                        (identity, *self.location(path), row.get('name', ''), time.time()))
                else:
                    connection.execute('DELETE FROM hidden WHERE identity=?', (identity,))

    def deletions(self):
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM hidden WHERE identity LIKE '[\"capture\",%'").fetchall()
            return {str(self.locate(row['root_id'], row['relative_path'])):
                    dict(name=row['name'], deleted_at=row['changed_at']) for row in rows}

    def versions(self, project_id):
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(
                'SELECT id,revision_id,parent_revision_id FROM records WHERE project_id=? ORDER BY stamp,id',
                (project_id,))]

    def backup(self, destination):
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        with self.connection() as source:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also protects against another backup using the
            # same name between the exists check and connection creation.
            with destination.open('xb'):
                pass
            target = sqlite3.connect(destination)
            try:
                source.backup(target)
            finally:
                target.close()
