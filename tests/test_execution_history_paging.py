"""Run-list paging stays deterministic and does not mutate history schemas."""
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from femto.execution.history import ExecutionStore, SCHEMA_VERSION
from femto.execution.state_machine import RunState


class ExecutionHistoryPagingTests(unittest.TestCase):
    def add_run(self, store, run_id, timestamp):
        with patch('femto.execution.history.time.time', return_value=timestamp):
            store.create(run_id, dict(source='simulation', execution_ready=False,
                                      physical_accuracy_verified=False))
            store.append(run_id, 'fixture', 'test_fixture', {}, RunState.SUCCEEDED, 0)

    def test_tied_times_and_insertions_do_not_skip_or_repeat_older_records(self):
        with TemporaryDirectory() as root:
            store = ExecutionStore(root)
            for number in range(6):
                self.add_run(store, f'run-{number}', 100.)
            first = store.runs(limit=2)
            self.assertEqual([r['run_id'] for r in first], ['run-5', 'run-4'])
            self.add_run(store, 'newer', 101.)
            second = store.runs(limit=2, before=(first[-1]['created_utc'], first[-1]['run_id']))
            third = store.runs(limit=2, before=(second[-1]['created_utc'], second[-1]['run_id']))
            self.assertEqual([r['run_id'] for r in second+third], ['run-3', 'run-2', 'run-1', 'run-0'])
            self.assertEqual(store.runs(limit=2, before=(100., 'run-0')), [])
            self.assertEqual(store.runs(limit=1, run_id='run-0')[0]['run_id'], 'run-0')
            self.assertEqual(store.runs(run_id='missing'), [])
            with store.connection() as connection:
                self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0], SCHEMA_VERSION)
                self.assertEqual(connection.execute('SELECT COUNT(*) FROM events').fetchone()[0], 7)

    def test_invalid_cursor_is_rejected_without_creating_a_database(self):
        with TemporaryDirectory() as root:
            store = ExecutionStore(root)
            for cursor in ((float('nan'), 'id'), (0., ''), ('0', 'id'), (0.,), 'id'):
                with self.subTest(cursor=cursor), self.assertRaises(ValueError):
                    store.runs(before=cursor)
            self.assertEqual(store.runs(before=(0., 'id')), [])
            self.assertFalse(store.path.exists())


if __name__ == '__main__':
    unittest.main()
