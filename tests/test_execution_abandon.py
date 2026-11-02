"""Abandonment requires positive no-request evidence, never a stop shortcut."""
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sqlite3
import unittest
from unittest.mock import patch

from femto.execution.application import create_coordinator
from femto.execution.history import ExecutionStore, SCHEMA_VERSION
from femto.execution.models import synthetic_plan
from femto.execution.resources import ResourceLease
from femto.execution.state_machine import RunState
from femto.hardware.simulation.scenario import ManualClock


class AbandonmentTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.clock=ManualClock()
        self.coordinator=create_coordinator(outputs=self.temp.name,clock=self.clock)
        self.addCleanup(self.coordinator.close_instances)
        self.run=self.coordinator.freeze(synthetic_plan())
        self.store=ExecutionStore(self.temp.name)

    def test_abandon_preserves_record_and_devices_and_releases_owner_for_new_task(self):
        c=self.coordinator
        c.connect()
        bindings=[a.binding for a in c.adapters]
        with patch.object(c.adapters[0],'submit') as first, patch.object(c.adapters[1],'submit') as second:
            self.assertTrue(c.can_abandon_unrun)
            self.assertTrue(c.abandon_unrun())
            self.assertFalse(c.abandon_unrun(), 'Repeated abandonment cannot create another event')
            first.assert_not_called();second.assert_not_called()
        record=self.store.read(self.run)
        self.assertEqual(record['display_state'],'abandoned_unrun')
        self.assertEqual(record['completed_steps'],0)
        self.assertEqual(record['events'][-1]['kind'],'task_abandoned_unrun')
        self.assertTrue(record['events'][-1]['payload']['no_device_request_dispatched'])
        self.assertNotIn('operator_stop',[e['kind'] for e in record['events']])
        self.assertFalse(record['execution_ready'])
        self.assertFalse(record['physical_accuracy_verified'])
        self.assertFalse(c._owner_lease.held)
        self.assertIsNone(c.plan, 'Large frozen arrays are released after abandonment')
        new=c.freeze(synthetic_plan())
        self.assertNotEqual(new,self.run)
        self.assertEqual(bindings,[a.binding for a in c.adapters])
        self.assertEqual(self.store.read(self.run),record)

    def test_preparing_or_running_task_cannot_abandon(self):
        c=self.coordinator;c.connect();self.assertTrue(c.check())
        self.assertTrue(c.pending)
        self.assertFalse(c.can_abandon_unrun)
        self.assertFalse(c.abandon_unrun())
        for _ in range(20):self.clock.advance(.02);c.poll()
        self.assertIs(c.state,RunState.READY)
        self.assertFalse(c.abandon_unrun())
        c.start();self.assertFalse(c.abandon_unrun())
        c.stop()
        for _ in range(20):self.clock.advance(.02);c.poll()
        self.assertIs(c.state,RunState.STOPPED)
        self.assertNotIn('task_abandoned_unrun',[e['kind'] for e in self.store.read(self.run)['events']])

    def test_database_evidence_rejects_intervening_request_even_if_ui_state_is_unprepared(self):
        self.store.append(self.run,'outside-request','request_intent',dict(request=dict(intent='prepare')),
                          RunState.UNPREPARED,0)
        self.coordinator.event_count+=1
        with self.assertRaisesRegex(ValueError,'请求'):
            self.coordinator.abandon_unrun()
        self.assertIs(self.coordinator.state,RunState.UNPREPARED)
        self.assertTrue(self.coordinator._owner_lease.held)
        self.assertEqual(self.store.read(self.run)['display_state'],'interrupted_needs_review')

    def test_disk_failure_keeps_unprepared_task_owner_and_does_not_create_abandonment(self):
        c=self.coordinator
        with closing(sqlite3.connect(self.store.path)) as blocker:
            blocker.execute('BEGIN IMMEDIATE')
            with self.assertRaisesRegex(RuntimeError,'原任务'):
                c.abandon_unrun()
            self.assertIs(c.state,RunState.UNPREPARED)
            self.assertTrue(c.persistence_gap)
            self.assertTrue(c._owner_lease.held)
            self.assertFalse(c.can_abandon_unrun)
            blocker.rollback()
        self.assertNotIn('task_abandoned_unrun',[e['kind'] for e in self.store.read(self.run)['events']])
        self.assertTrue(c.recover_persistence())

    def test_normal_close_of_unstarted_task_records_abandonment_and_restart_stays_read_only(self):
        self.coordinator.close_instances()
        self.assertEqual(self.store.read(self.run)['display_state'],'abandoned_unrun')
        self.assertEqual(ExecutionStore(self.temp.name).recover_interrupted(),[])
        self.assertEqual(ExecutionStore(self.temp.name).read(self.run)['recorded_state'],'abandoned_unrun')
        key='simulation:task:'+self.run
        lease=ResourceLease([key],owner='restart-read-check',outputs=self.temp.name).acquire()
        lease.release()
        with self.store.connection() as connection:
            self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0],SCHEMA_VERSION)

    def test_v1_read_and_transactional_v2_migration_preserve_original_events(self):
        c=self.coordinator;c.abandon_unrun()
        # Build an old-version copy without the newly added state.
        old=Path(self.temp.name)/'old-data'
        target=old/'history_metadata/execution.sqlite3';target.parent.mkdir(parents=True)
        self.store.backup(target)
        with closing(sqlite3.connect(target)) as connection:
            connection.execute("DELETE FROM events WHERE kind='task_abandoned_unrun'")
            connection.execute("UPDATE runs SET state='unprepared',event_sequence=1")
            connection.execute('DELETE FROM migrations WHERE version=2')
            connection.execute('PRAGMA user_version=1');connection.commit()
        store=ExecutionStore(old)
        before=store.read(self.run,verify_sources=False)
        with store.connection() as connection:self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0],1)
        failing=ExecutionStore(old,migration_check=lambda:(_ for _ in ()).throw(RuntimeError('migration interrupted')))
        with self.assertRaises(RuntimeError):
            failing.create('rejected',dict(source='simulation',execution_ready=False,physical_accuracy_verified=False))
        with store.connection() as connection:self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0],1)
        self.assertEqual(store.read(self.run,verify_sources=False),before)
        store.create('new',dict(source='simulation',execution_ready=False,physical_accuracy_verified=False))
        with store.connection() as connection:
            self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0],2)
        self.assertEqual(store.read(self.run,verify_sources=False),before)

    def test_database_rollback_during_abandon_does_not_claim_success(self):
        c=self.coordinator
        with patch.object(self.store.__class__,'_append',side_effect=OSError('disk full')):
            with self.assertRaises(RuntimeError):c.abandon_unrun()
        self.assertIs(c.state,RunState.UNPREPARED)
        with self.store.connection() as connection:
            self.assertEqual(connection.execute('SELECT state FROM runs WHERE run_id=?',(self.run,)).fetchone()[0],'unprepared')
        self.assertTrue(c._owner_lease.held)

    def test_v2_backup_restores_abandonment_and_v1_reader_rejects_without_changes(self):
        self.coordinator.abandon_unrun()
        before=self.store.read(self.run)
        restored=Path(self.temp.name)/'restored'
        target=restored/'history_metadata/execution.sqlite3';target.parent.mkdir(parents=True)
        self.store.backup(target)
        store=ExecutionStore(restored)
        self.assertEqual(store.read(self.run),before)
        self.assertEqual(store.recover_interrupted(),[])
        with store.connection() as connection:
            self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0],2)
            saved_rows=[tuple(row) for row in connection.execute('SELECT * FROM runs')]
            saved_events=[tuple(row) for row in connection.execute('SELECT * FROM events')]
        with patch('femto.execution.history.READABLE_VERSIONS',frozenset({1})), \
                patch('femto.execution.history.SCHEMA_VERSION',1):
            with self.assertRaisesRegex(ValueError,'Unsupported execution database version'):
                store.read(self.run)
            with self.assertRaisesRegex(ValueError,'Unsupported execution database version'):
                store.create('downgrade',dict(source='simulation',execution_ready=False,physical_accuracy_verified=False))
        with store.connection() as connection:
            self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0],2)
            self.assertEqual([tuple(row) for row in connection.execute('SELECT * FROM runs')],saved_rows)
            self.assertEqual([tuple(row) for row in connection.execute('SELECT * FROM events')],saved_events)


if __name__=='__main__':unittest.main()
