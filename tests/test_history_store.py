"""Transactional history indexing without changing capture/planning evidence."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from femto.history_store import HistoryStore


class HistoryStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = HistoryStore(self.root / 'outputs')
        self.row = dict(key=str(self.root/'outputs/complete_demo_a'), name='肩部 Alpha', time=1,
                        fixed_files=['a'], fixed_file_versions={'a': (12, 34)},
                        mesh=self.root/'mesh.ply')

    def replace(self, rows, **kwargs):
        self.store.replace('captures', rows, search_text=lambda row: row['name'], **kwargs)

    def test_read_does_not_create_database_and_round_trip_preserves_types(self):
        with self.assertRaises(sqlite3.OperationalError):
            self.store.query('captures')
        self.assertFalse(self.store.path.parent.exists())
        self.replace([self.row])
        self.assertEqual(self.store.query('captures')['rows'], [self.row])
        self.assertEqual(self.store.query('captures', '肩部 alpha')['total'], 1)
        self.assertEqual(self.store.query('captures', "%' OR 1=1 --")['total'], 0)

    def test_paging_stable_ids_and_version_parent_links(self):
        rows = [dict(self.row, key=str(self.root/f'outputs/complete_demo_{i}'),
                     project_id='project', revision_id=str(i), parent_revision_id=str(i-1)) for i in range(3)]
        self.replace(rows)
        ids = [row['id'] for row in self.store.versions('project')]
        self.replace(rows)
        self.assertEqual([row['id'] for row in self.store.versions('project')], ids)
        page = self.store.query('captures', limit=1, offset=1)
        self.assertEqual(page['total'], 3)
        self.assertEqual(page['rows'], [rows[1]])
        self.assertEqual({row['parent_revision_id'] for row in self.store.versions('project')}, {'-1','0','1'})
        with self.assertRaises(ValueError):
            self.store.query('captures', limit=0)

    def test_legacy_hidden_imported_once_restore_survives_restart(self):
        self.store.outputs.mkdir()
        legacy = self.store.outputs/'local_history_deleted.json'
        original = json.dumps(dict(schema='local-history-deleted-v1',
            records={self.row['key']: dict(name=self.row['name'], deleted_at=123)})).encode()
        legacy.write_bytes(original)
        self.replace([self.row])
        self.assertEqual(self.store.query('captures')['rows'], [])
        self.assertEqual(self.store.query('captures', hidden_only=True)['rows'], [self.row])
        self.assertEqual(self.store.deletions()[self.row['key']]['deleted_at'], 123)
        self.store.set_hidden([self.row], False)
        reopened = HistoryStore(self.store.outputs)
        reopened.replace('captures', [self.row], search_text=lambda row: row['name'])
        self.assertEqual(reopened.query('captures')['rows'], [self.row])
        self.assertEqual(legacy.read_bytes(), original)

    def test_failed_migration_is_atomic_and_can_retry(self):
        self.store.outputs.mkdir()
        legacy = self.store.outputs/'local_history_deleted.json'
        legacy.write_text('broken')
        with self.assertRaises(ValueError):
            self.replace([self.row])
        with closing(sqlite3.connect(self.store.path)) as connection:
            self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0], 0)
            self.assertEqual(connection.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall(), [])
        legacy.write_text(json.dumps(dict(schema='local-history-deleted-v1', records={})))
        self.replace([self.row])
        self.assertEqual(self.store.query('captures')['rows'], [self.row])

    def test_cancelled_refresh_rolls_back_membership_and_payload(self):
        self.replace([self.row])
        # The check immediately after DELETE membership runs inside the write
        # transaction, so this exercises rollback rather than preflight only.
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            self.replace([dict(self.row, name='changed')], check=cancel)
        self.assertEqual(self.store.query('captures')['rows'], [self.row])

    def test_worker_crash_rolls_back_uncommitted_change(self):
        self.replace([self.row])
        script = ('import os,sqlite3,sys; c=sqlite3.connect(sys.argv[1]); '
                  'c.execute("BEGIN IMMEDIATE"); c.execute("DELETE FROM membership"); os._exit(23)')
        result = subprocess.run([sys.executable, '-c', script, str(self.store.path)], check=False)
        self.assertEqual(result.returncode, 23)
        self.assertEqual(self.store.query('captures')['rows'], [self.row])
        self.replace([self.row])

    def test_sql_failure_rolls_back_refresh_and_cancel_rolls_back_hidden_batch(self):
        self.replace([self.row])
        with self.store.connection(write=True) as connection:
            connection.execute("CREATE TRIGGER reject_refresh BEFORE UPDATE ON records "
                               "BEGIN SELECT RAISE(ABORT, 'simulated storage failure'); END")
        with self.assertRaisesRegex(sqlite3.IntegrityError, 'storage failure'):
            self.replace([dict(self.row, name='changed')])
        self.assertEqual(self.store.query('captures')['rows'], [self.row])
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError('cancelled')
        with self.assertRaises(RuntimeError):
            self.store.set_hidden([self.row, dict(self.row, key=str(self.root/'other'))], check=cancel)
        self.assertEqual(self.store.deletions(), {})

    def test_reader_uses_committed_snapshot_while_writer_is_active(self):
        self.replace([self.row])
        with self.store.connection(write=True) as writer:
            writer.execute('DELETE FROM membership')
            self.assertEqual(self.store.query('captures')['rows'], [self.row])
            with self.assertRaises(sqlite3.OperationalError):
                self.replace([])
        self.assertEqual(self.store.query('captures')['total'], 0)

    def test_duplicates_rejected_before_mutation_and_future_schema_refused(self):
        self.replace([self.row])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.replace([self.row, self.row])
        self.assertEqual(self.store.query('captures')['rows'], [self.row])
        with closing(sqlite3.connect(self.store.path)) as connection:
            connection.execute('PRAGMA user_version=999')
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            self.replace([])
        with closing(sqlite3.connect(self.store.path)) as connection:
            self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0], 999)
            self.assertEqual(connection.execute('SELECT count(*) FROM membership').fetchone()[0], 1)

    def test_backup_contains_hidden_state_and_does_not_overwrite(self):
        self.replace([self.row])
        self.store.set_hidden([self.row])
        target = self.root/'backup/history.sqlite3'
        self.store.backup(target)
        with closing(sqlite3.connect(target)) as connection:
            self.assertEqual(connection.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertEqual(connection.execute('SELECT count(*) FROM hidden').fetchone()[0], 1)
        before = target.read_bytes()
        with self.assertRaises(FileExistsError):
            self.store.backup(target)
        self.assertEqual(target.read_bytes(), before)

    def test_paths_and_missing_project_identity_are_stable(self):
        project = self.root/'saved.region.zip'
        row = dict(kind='区域工程 / 方案', project_location=project, project=project, revision_id='r')
        self.assertEqual(self.store.identity(row), self.store.identity(dict(row, project=None)))
        other_workspace = HistoryStore(self.root/'moved/outputs')
        relative = self.store.location(self.row['key'])[1]
        self.assertEqual(self.store.identity(self.row),
                         other_workspace.identity(dict(self.row, key=other_workspace.workspace/relative)))
        with self.assertRaises(ValueError):
            self.store.locate('workspace', '../outside')


if __name__ == '__main__':
    unittest.main()
