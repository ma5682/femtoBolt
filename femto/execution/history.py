"""Append-only simulation runs/events in a database independent of HistoryStore.

    No device waits occur in transactions. Unknown versions are never modified.
    Module import and an empty read do not create files or migrate user data.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3
import time

from femto.execution.models import AssetReference, canonical
from femto.execution.resources import ResourceLease
from femto.execution.state_machine import RunState, TERMINAL
from femto.io_utils import write_json_strict
from femto.paths import execution_database

SCHEMA_VERSION = 2
READABLE_VERSIONS = frozenset({1, 2})


class ExecutionStore:
    def __init__(self, outputs=None, *, timeout_s=.1, migration_check=lambda: None):
        self.path = execution_database(outputs).resolve()
        self.timeout_s = timeout_s
        self.migration_check = migration_check
        self.outputs = self.path.parent.parent

    @contextmanager
    def connection(self, *, write=False):
        if not write and not self.path.is_file():
            raise FileNotFoundError(self.path)
        if write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=self.timeout_s)
        else:
            connection = sqlite3.connect(self.path.as_uri()+'?mode=ro', uri=True, timeout=0)
        connection.row_factory = sqlite3.Row
        try:
            version = connection.execute('PRAGMA user_version').fetchone()[0]
            if version not in ({0, *READABLE_VERSIONS} if write else READABLE_VERSIONS):
                raise ValueError('Unsupported execution database version; no writes performed')
            connection.execute('PRAGMA foreign_keys=ON')
            if write:
                connection.execute('PRAGMA journal_mode=WAL')
                connection.execute('PRAGMA synchronous=FULL')
                connection.execute('BEGIN IMMEDIATE')
                version = connection.execute('PRAGMA user_version').fetchone()[0]
                if version == 0:
                    for sql in (
                        'CREATE TABLE runs (run_id TEXT PRIMARY KEY, snapshot TEXT NOT NULL, '
                        'state TEXT NOT NULL, completed_steps INTEGER NOT NULL, event_sequence INTEGER NOT NULL, '
                        'created_utc REAL NOT NULL, updated_utc REAL NOT NULL, interrupted INTEGER NOT NULL DEFAULT 0)',
                        'CREATE TABLE events (run_id TEXT NOT NULL REFERENCES runs(run_id), sequence INTEGER NOT NULL, '
                        'event_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, utc REAL NOT NULL, '
                        'PRIMARY KEY(run_id,sequence), UNIQUE(run_id,event_id))',
                        'CREATE TABLE migrations (version INTEGER PRIMARY KEY, applied_utc REAL NOT NULL)',
                    ):
                        connection.execute(sql)
                        self.migration_check()
                    connection.execute('INSERT INTO migrations VALUES(1,?)', (time.time(),))
                    connection.execute('PRAGMA user_version=1')
                    version = 1
                if version == 1:
                    # v2 adds the abandoned_unrun stored state. No original
                    # rows/JSON or event sequence are rewritten by migration.
                    connection.execute('INSERT INTO migrations VALUES(2,?)', (time.time(),))
                    self.migration_check()
                    connection.execute('PRAGMA user_version=2')
                elif version != SCHEMA_VERSION:
                    raise ValueError('Execution database version changed during transaction')
            else:
                connection.execute('BEGIN')  # Task snapshot and its ordered events share one read view.
            yield connection
            if write:
                connection.commit()
        finally:
            if connection.in_transaction:
                connection.rollback()
            connection.close()

    def create(self, run_id, snapshot):
        if snapshot.get('source') != 'simulation' or snapshot.get('execution_ready') is not False or snapshot.get('physical_accuracy_verified') is not False:
            raise ValueError('Only explicit development simulation records are accepted')
        serialized = canonical(snapshot)
        with self.connection(write=True) as connection:
            now = time.time()
            connection.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?)',
                               (run_id, serialized, RunState.UNPREPARED.value, 0, 0, now, now, 0))

    def append(self, run_id, event_id, kind, payload, state, completed_steps):
        if not isinstance(state, RunState) or type(completed_steps) is not int or completed_steps < 0:
            raise ValueError('Invalid simulation state or completed step count')
        serialized = canonical(dict(payload, source='simulation', execution_ready=False,
                                    physical_accuracy_verified=False, recorded_state=state.value,
                                    recorded_completed_steps=completed_steps))
        with self.connection(write=True) as connection:
            return self._append(connection, run_id, event_id, kind, serialized, state, completed_steps)

    @staticmethod
    def _append(connection, run_id, event_id, kind, serialized, state, completed_steps):
        duplicate = connection.execute('SELECT sequence,kind,payload FROM events WHERE run_id=? AND event_id=?', (run_id, event_id)).fetchone()
        if duplicate is not None:
            if duplicate['kind'] != kind or duplicate['payload'] != serialized:
                raise ValueError('Duplicate event ID carries contradictory content')
            return duplicate[0]
        row = connection.execute('SELECT event_sequence FROM runs WHERE run_id=?', (run_id,)).fetchone()
        if row is None:
            raise ValueError('Execution run does not exist')
        sequence = row[0]+1
        now = time.time()
        connection.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',
                           (run_id, sequence, event_id, kind, serialized, now))
        connection.execute('UPDATE runs SET state=?,completed_steps=?,event_sequence=?,updated_utc=?, '
                           'interrupted=CASE WHEN ? THEN 1 ELSE interrupted END WHERE run_id=?',
                           (state.value, completed_steps, sequence, now, kind == 'interrupted_recovery', run_id))
        return sequence

    def abandon_unrun(self, run_id, event_id, payload, *, expected_sequence):
        """Prove no request intent exists and commit abandonment atomically."""
        state = RunState.ABANDONED
        serialized = canonical(dict(payload, source='simulation', execution_ready=False,
                                    physical_accuracy_verified=False, recorded_state=state.value,
                                    recorded_completed_steps=0))
        with self.connection(write=True) as connection:
            row = connection.execute('SELECT state,completed_steps,event_sequence FROM runs WHERE run_id=?', (run_id,)).fetchone()
            if (row is None or row['state'] != RunState.UNPREPARED.value
                    or row['completed_steps'] != 0 or row['event_sequence'] != expected_sequence):
                raise ValueError('冻结任务的持久化状态已变化，不能证明尚未下发请求')
            frozen = connection.execute("SELECT 1 FROM events WHERE run_id=? AND kind='task_frozen' LIMIT 1", (run_id,)).fetchone()
            request = connection.execute("SELECT 1 FROM events WHERE run_id=? AND (kind IN "
                "('request_intent','request_sent','request_send_failed','preflight_started','operator_start') "
                "OR json_type(payload,'$.request')='object') LIMIT 1", (run_id,)).fetchone()
            if frozen is None or request is not None:
                raise ValueError('任务记录不能证明尚未下发请求，请使用停止与核对流程')
            return self._append(connection, run_id, event_id, 'task_abandoned_unrun', serialized, state, 0)

    def runs(self, *, limit=200, before=None, run_id=None, with_summary=False):
        if type(limit) is not int or limit < 1:
            raise ValueError('Run list limit must be a positive integer')
        clauses, values = [], []
        if before is not None:
            if (not isinstance(before, (tuple, list)) or len(before) != 2
                    or type(before[0]) not in (int, float) or not math.isfinite(before[0])
                    or not isinstance(before[1], str) or not before[1]):
                raise ValueError('Invalid run-list cursor')
            clauses.append('(created_utc < ? OR (created_utc = ? AND run_id < ?))')
            values.extend((before[0], before[0], before[1]))
        if run_id is not None:
            if not isinstance(run_id, str) or not run_id:
                raise ValueError('Invalid run identity')
            clauses.append('run_id = ?')
            values.append(run_id)
        if not self.path.exists():
            return []
        sql = 'SELECT run_id,state,completed_steps,event_sequence,created_utc,updated_utc,interrupted FROM runs'
        if with_summary:
            sql = sql.replace(' FROM runs', ",json_extract(snapshot,'$.plan.plan.source_path') AS source_path,"
                "json_extract(snapshot,'$.plan.plan.revision_id') AS revision_id,"
                "json_extract(snapshot,'$.plan.plan.task_context') AS task_context,"
                "json_extract(snapshot,'$.plan.plan_source') AS plan_source,"
                "json_array_length(snapshot,'$.plan.steps') AS total_steps FROM runs")
        if clauses:
            sql += ' WHERE '+' AND '.join(clauses)
        sql += ' ORDER BY created_utc DESC, run_id DESC LIMIT ?'
        with self.connection() as connection:
            rows = [dict(row) for row in connection.execute(sql, (*values, limit))]
        if with_summary:
            for row in rows:
                row['task_context'] = json.loads(row['task_context']) if row.get('task_context') else None
        return rows

    def read(self, run_id, *, verify_sources=True, check=lambda: None):
        check()
        with self.connection() as connection:
            row = connection.execute('SELECT * FROM runs WHERE run_id=?', (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            result = dict(row)
            result['snapshot'] = json.loads(result['snapshot'])
            result['events'] = []
            for item in connection.execute('SELECT * FROM events WHERE run_id=? ORDER BY sequence', (run_id,)):
                check()
                result['events'].append(dict(item, payload=json.loads(item['payload'])))
            recorded = RunState(result['state'])
            result['recorded_state'] = recorded.value
            result['display_state'] = (RunState.INTERRUPTED.value if recorded not in TERMINAL else recorded.value)
            result.update(read_only=True, source='simulation', execution_ready=False, physical_accuracy_verified=False)
        failures = []
        assets = result['snapshot'].get('plan', {}).get('assets', [])
        if verify_sources:
            for asset in assets:
                check()
                try:
                    AssetReference(asset['path'], asset['sha256']).verify(check)
                except (OSError, ValueError) as error:
                    failures.append(dict(path=asset['path'], reason=str(error)))
        result['source_integrity'] = dict(status='changed_or_missing' if failures else
                                         'verified' if assets and verify_sources else
                                         'not_checked' if assets else 'not_applicable',
                                         failures=failures, rebound=False)
        manifest = self.path.parent/'manifest.json'
        if manifest.is_file():
            try:
                value = json.loads(manifest.read_text(encoding='utf-8-sig'))
                if not isinstance(value, dict) or value.get('schema') != 'femto-metadata-backup-v1':
                    raise ValueError('Unsupported metadata backup manifest')
                window = value.get('start_utc'), value.get('end_utc')
                if (any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in window)
                        or window[0] > window[1] or value.get('cross_database_atomic') is not False):
                    raise ValueError('Unknown or invalid backup interval')
                result['backup_context'] = dict(cross_database_atomic=False,
                    backup_window_utc=list(window),
                    reason='双库分别备份，跨库关联须结合备份时间范围核对')
            except (OSError, ValueError):
                result['backup_context'] = dict(cross_database_atomic=False, backup_window_utc=None,
                    reason='备份清单缺少有效时间范围或无法读取；时间范围未知，记录仍可只读回看')
        check()
        return result

    def recover_interrupted(self, *, exclude=(), check=lambda: None):
        """Append recovery evidence; never reconstruct adapters or resend requests."""
        check()
        if not self.path.exists():
            return []
        recovered = []
        for row in self.runs(limit=100000):
            check()
            if row['run_id'] in exclude or RunState(row['state']) in TERMINAL:
                continue
            snapshot = self.read(row['run_id'], verify_sources=False, check=check)['snapshot']
            owner_key = snapshot.get('owner_resource_key')
            lease = None
            if owner_key:
                try:
                    lease = ResourceLease([owner_key], owner='recovery:'+row['run_id'], outputs=self.outputs).acquire()
                except RuntimeError:
                    continue  # A live owner still holds the process lock.
            try:
              current = self.read(row['run_id'], verify_sources=False, check=check)
              if RunState(current['recorded_state']) in TERMINAL:
                  continue
              check()
              self.append(row['run_id'], 'process-recovery', 'interrupted_recovery',
                        dict(reason='旧进程结束；请求可能已发送，物理结果未核对；只读回看，不自动重发',
                             monotonic_comparison='not_comparable_across_processes'),
                        RunState.INTERRUPTED, current['completed_steps'])
              recovered.append(row['run_id'])
            finally:
              if lease: lease.release()
        return recovered

    def export(self, run_id, destination, *, check=lambda: None):
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        write_json_strict(destination, self.read(run_id, check=check), check=check)
        return destination

    def backup(self, destination):
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        with self.connection() as source:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb'):
                pass
            target = sqlite3.connect(destination)
            try:
                source.backup(target)
            finally:
                target.close()
