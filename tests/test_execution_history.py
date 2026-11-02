"""Disk/lock/crash boundaries use disposable databases and real repositories."""
from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from femto.execution.application import create_coordinator
from femto.execution.history import ExecutionStore
from femto.execution.models import AssetReference, FrozenPlan, synthetic_plan
from femto.execution.resources import ResourceLease
from femto.execution.state_machine import RunState
from femto.hardware.interface_contract import Intent
from femto.hardware.simulation.scenario import ManualClock
from femto.history_store import HistoryStore
from femto.io_utils import sha256_file
from tools.history_metadata import backup_metadata
from test_execution_logic import advance, prepare, runtime


class ExecutionHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary=TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)

    def task(self):
        coordinator,clock,store=runtime(self.root)
        self.addCleanup(coordinator.close_instances)
        prepare(coordinator,clock)
        return coordinator,clock,store

    def test_two_databases_do_not_block_or_modify_each_other(self):
        old=HistoryStore(self.root)
        row=dict(key=str(self.root/'complete_demo_a'),name='source',time=1)
        old.replace('captures',[row],search_text=lambda r:r['name'])
        with closing(sqlite3.connect(old.path)) as connection:connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        before=sha256_file(old.path)
        with old.connection(write=True):
            coordinator,clock,store=self.task()
            coordinator.start();advance(coordinator,clock,2)
            self.assertIs(coordinator.state,RunState.SUCCEEDED)
        self.assertEqual(sha256_file(old.path),before)
        self.assertEqual(old.query('captures')['total'],1)
        old.set_hidden([row]);self.assertEqual(old.query('captures')['total'],0)
        old.set_hidden([row],False);self.assertEqual(old.query('captures')['total'],1)
        run=store.read(coordinator.run_id)
        old.replace('captures',[],search_text=lambda r:r['name'])
        self.assertEqual(store.read(coordinator.run_id),run)
        with old.connection() as connection:self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0],1)

    def test_execution_lock_timeout_keeps_stop_and_reports_persistence_gap(self):
        coordinator,clock,store=self.task()
        with closing(sqlite3.connect(store.path)) as blocker:
            blocker.execute('BEGIN IMMEDIATE')
            started=time.monotonic()
            self.assertFalse(coordinator.start())
            self.assertLess(time.monotonic()-started,2)
            self.assertTrue(coordinator.persistence_gap)
            advance(coordinator,clock,.4)
            self.assertIn(coordinator.state,(RunState.FAILED,RunState.UNKNOWN))
            self.assertTrue(coordinator.by_id['head'].status().output_off)
            blocker.rollback()
        self.assertTrue(coordinator.recover_persistence())
        events=store.read(coordinator.run_id)['events']
        self.assertEqual(events[-1]['kind'],'persistence_gap_recovered')
        self.assertGreater(events[-1]['payload']['missing_count'],0)
        self.assertFalse(coordinator.start())

    def test_disk_error_before_intent_prevents_step_but_not_stop(self):
        coordinator,clock,store=self.task()
        original=store.append
        def fail(run_id,event_id,kind,*args):
            if kind=='request_intent' and args[0]['request']['intent']=='simulation_step':
                raise OSError('simulated disk full')
            return original(run_id,event_id,kind,*args)
        with patch.object(store,'append',side_effect=fail):
            coordinator.start();advance(coordinator,clock,.5)
        self.assertTrue(coordinator.persistence_gap)
        self.assertIs(coordinator.state,RunState.FAILED)
        self.assertEqual(coordinator.completed_steps,0)
        self.assertTrue(coordinator.by_id['arm'].status().motion_stopped)

    def test_sent_record_failure_prevents_next_participant_dispatch_but_sends_stop(self):
        coordinator,clock,store=self.task()
        original=store.append
        def fail(run_id,event_id,kind,*args):
            if kind=='request_sent':
                pending=coordinator.pending[args[0]['request_id']]
                if pending.request.intent is Intent.SIMULATION_STEP:
                    raise OSError('confirmation log unavailable')
            return original(run_id,event_id,kind,*args)
        with patch.object(store,'append',side_effect=fail):
            coordinator.start();advance(coordinator,clock,.5)
        head_steps=[p for p in coordinator.pending.values() if p.request.binding.device_id=='head'
                    and p.request.intent is Intent.SIMULATION_STEP]
        self.assertEqual(head_steps,[])
        self.assertTrue(all(any(p.request.intent is Intent.REQUEST_STOP and p.request.binding.device_id==a.binding.device_id
                                for p in coordinator.pending.values()) for a in coordinator.controllable))
        self.assertTrue(coordinator.persistence_gap)
        self.assertIs(coordinator.state,RunState.FAILED)
        self.assertEqual(coordinator.completed_steps,0)

    def test_initial_database_failure_blocks_new_task(self):
        coordinator,clock,store=runtime(self.root)
        with patch.object(store,'create',side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):coordinator.freeze(synthetic_plan())
        self.assertIsNone(coordinator.run_id)
        self.assertFalse(any(adapter.submitted for adapter in coordinator.adapters))
        coordinator.close_instances()

    def test_event_deduplication_and_conflicting_duplicate(self):
        coordinator,clock,store=self.task()
        count=store.read(coordinator.run_id)['event_sequence']
        first=store.append(coordinator.run_id,'same-event','diagnostic',{'value':1},RunState.READY,0)
        second=store.append(coordinator.run_id,'same-event','diagnostic',{'value':1},RunState.READY,0)
        self.assertEqual(first,second);self.assertEqual(first,count+1)
        with self.assertRaises(ValueError):store.append(coordinator.run_id,'same-event','diagnostic',{'value':2},RunState.READY,0)
        with self.assertRaises(ValueError):store.append(coordinator.run_id,'same-event','diagnostic',{'value':1},RunState.FAILED,0)
        self.assertEqual(store.read(coordinator.run_id)['event_sequence'],count+1)

    def test_migration_failure_rolls_back_and_unknown_schema_is_unchanged(self):
        calls=[]
        def fail():
            calls.append(1)
            if len(calls)==2:raise RuntimeError('migration interrupted')
        store=ExecutionStore(self.root,migration_check=fail)
        with self.assertRaises(RuntimeError):
            store.create('task',dict(source='simulation',execution_ready=False,physical_accuracy_verified=False))
        with closing(sqlite3.connect(store.path)) as connection:
            self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0],0)
            self.assertFalse(connection.execute("SELECT name FROM sqlite_master WHERE name='runs'").fetchall())
            connection.execute('PRAGMA user_version=99');connection.commit()
            connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        before=sha256_file(store.path)
        with self.assertRaises(ValueError):
            store.create('task',dict(source='simulation',execution_ready=False,physical_accuracy_verified=False))
        self.assertEqual(sha256_file(store.path),before)

    def test_atomic_event_transaction_rolls_back_when_snapshot_update_fails(self):
        coordinator,clock,store=self.task()
        before=store.read(coordinator.run_id)
        with store.connection(write=True) as connection:
            connection.execute("CREATE TRIGGER deny_update BEFORE UPDATE ON runs BEGIN SELECT RAISE(ABORT,'snapshot blocked'); END")
        with self.assertRaises(sqlite3.IntegrityError):store.append(coordinator.run_id,'rollback','test',{},RunState.RUNNING,1)
        self.assertEqual(store.read(coordinator.run_id),before)

    def test_live_owner_is_not_recovered_by_another_window(self):
        coordinator,clock,store=self.task()
        self.assertEqual(ExecutionStore(self.root).recover_interrupted(),[])
        self.assertEqual(store.read(coordinator.run_id)['recorded_state'],'ready_for_simulation')
        coordinator._owner_lease.release()  # Simulates OS releasing the terminated process.
        self.assertEqual(store.recover_interrupted(),[coordinator.run_id])
        self.assertEqual(store.read(coordinator.run_id)['display_state'],'interrupted_needs_review')

    def test_crash_boundaries_do_not_automatically_resend(self):
        for boundary in ('intent_recorded_before_dispatch','dispatched_before_confirmation_record',
                         'feedback_received_before_record','feedback_recorded'):
            with self.subTest(boundary=boundary):
                root=self.root/boundary
                coordinator,clock,store=runtime(root)
                prepare(coordinator,clock)
                before=sum(len(a.submitted) for a in coordinator.adapters)
                class Crash(BaseException):pass
                def crash(name):
                    if name==boundary:raise Crash()
                coordinator.crash_hook=crash
                with self.assertRaises(Crash):
                    coordinator.start();advance(coordinator,clock,.6)
                submitted=sum(len(a.submitted) for a in coordinator.adapters)
                self.assertGreaterEqual(submitted,before)
                coordinator._owner_lease.release();coordinator._lease.release()
                recovered=ExecutionStore(root)
                self.assertEqual(recovered.recover_interrupted(),[coordinator.run_id])
                record=recovered.read(coordinator.run_id)
                self.assertEqual(record['display_state'],'interrupted_needs_review')
                self.assertEqual(sum(len(a.submitted) for a in coordinator.adapters),submitted)
                coordinator.crash_hook=lambda name:None
                for adapter in coordinator.adapters:adapter.close()

    def test_process_resource_contention_rejected_and_os_releases(self):
        key='simulation:shared-arm'
        lease=ResourceLease([key],owner='first',outputs=self.root).acquire()
        self.addCleanup(lease.release)
        code="from femto.execution.resources import ResourceLease; import sys; ResourceLease(['simulation:shared-arm'],owner='second',outputs=sys.argv[1]).acquire()"
        result=subprocess.run([sys.executable,'-X','utf8','-c',code,str(self.root)],capture_output=True,text=True,encoding='utf-8')
        self.assertNotEqual(result.returncode,0);self.assertIn('占用',result.stderr)
        lease.release()
        second=ResourceLease([key],owner='second',outputs=self.root).acquire();second.release()
        result=subprocess.run([sys.executable,'-c',code,str(self.root)],capture_output=True)
        self.assertEqual(result.returncode,0)
        third=ResourceLease([key],owner='third',outputs=self.root).acquire();third.release()

    def test_bundle_backup_restores_both_databases_and_marks_interval(self):
        old=HistoryStore(self.root)
        row=dict(key=str(self.root/'complete_demo_a'),name='example',time=1)
        old.replace('captures',[row],search_text=lambda r:r['name']);old.set_hidden([row])
        coordinator,clock,store=self.task();coordinator.start();advance(coordinator,clock,2)
        manifest=backup_metadata(self.root,self.root/'backup')
        self.assertFalse(manifest['cross_database_atomic'])
        self.assertTrue(all(row['present'] for row in manifest['databases']))
        target=self.root/'restored/history_metadata';target.mkdir(parents=True)
        shutil.copy2(self.root/'backup/history.sqlite3',target/'history.sqlite3')
        shutil.copy2(self.root/'backup/execution.sqlite3',target/'execution.sqlite3')
        shutil.copy2(self.root/'backup/manifest.json',target/'manifest.json')
        self.assertEqual(HistoryStore(self.root/'restored').query('captures',hidden_only=True)['total'],1)
        self.assertEqual(ExecutionStore(self.root/'restored').read(coordinator.run_id)['display_state'],'succeeded')
        self.assertFalse(ExecutionStore(self.root/'restored').read(coordinator.run_id)['backup_context']['cross_database_atomic'])
        with self.assertRaises(FileExistsError):backup_metadata(self.root,self.root/'backup')

    def test_asset_edit_stops_but_unrelated_draft_change_does_not(self):
        source=self.root/'source.json';source.write_text('{"saved":true}',encoding='utf8')
        base=synthetic_plan()
        plan=FrozenPlan(base.data_json,base.steps,(AssetReference(str(source),sha256_file(source)),),base.source_kind)
        coordinator,clock,store=runtime(self.root)
        self.addCleanup(coordinator.close_instances)
        coordinator.connect();coordinator.freeze(plan);coordinator.check();advance(coordinator,clock)
        coordinator.start()
        (self.root/'unrelated_draft.json').write_text('{"changed":true}',encoding='utf8')
        advance(coordinator,clock,.2);self.assertIs(coordinator.state,RunState.RUNNING)
        source.write_text('{"saved":false}',encoding='utf8')
        advance(coordinator,clock,.5);self.assertIs(coordinator.state,RunState.FAILED)
        self.assertEqual(json.loads(coordinator.plan.data_json),base.public_plan)

    def test_replay_detects_source_change_and_never_rebinds_saved_task(self):
        source=self.root/'source.json';source.write_text('{}',encoding='utf8')
        base=synthetic_plan();plan=FrozenPlan(base.data_json,base.steps,(AssetReference(str(source),sha256_file(source)),),base.source_kind)
        coordinator,clock,store=runtime(self.root);self.addCleanup(coordinator.close_instances)
        coordinator.connect();coordinator.freeze(plan);coordinator.check();advance(coordinator,clock)
        coordinator.start();advance(coordinator,clock,2)
        before=store.read(coordinator.run_id)
        source.write_text('{"modified":true}',encoding='utf8')
        after=store.read(coordinator.run_id)
        self.assertEqual(after['source_integrity']['status'],'changed_or_missing')
        self.assertFalse(after['source_integrity']['rebound'])
        self.assertEqual(after['snapshot'],before['snapshot'])
        self.assertEqual(after['events'],before['events'])

    def test_recovery_event_and_interrupted_flag_rollback_together(self):
        coordinator,clock,store=self.task()
        coordinator._owner_lease.release()
        before=store.read(coordinator.run_id)
        with store.connection(write=True) as connection:
            connection.execute("CREATE TRIGGER deny_recovery BEFORE UPDATE ON runs WHEN NEW.interrupted=1 BEGIN SELECT RAISE(ABORT,'recovery blocked'); END")
        with self.assertRaises(sqlite3.IntegrityError):store.recover_interrupted()
        after=store.read(coordinator.run_id)
        self.assertEqual(after['events'],before['events'])
        self.assertEqual(after['interrupted'],0)

    def test_read_snapshot_and_event_list_use_one_sqlite_view(self):
        coordinator,clock,store=self.task()
        before=store.read(coordinator.run_id)['event_sequence']
        original=store.connection
        class ConnectionProbe:
            def __init__(self,connection):self.connection=connection
            def execute(self,sql,*args):
                cursor=self.connection.execute(sql,*args)
                if sql=='SELECT * FROM runs WHERE run_id=?':
                    store.append(coordinator.run_id,'concurrent-read','diagnostic',{},RunState.READY,0)
                return cursor
        @contextmanager
        def concurrent(*,write=False):
            with original(write=write) as connection:
                yield connection if write else ConnectionProbe(connection)
        with patch.object(store,'connection',side_effect=concurrent):
            record=store.read(coordinator.run_id)
        self.assertEqual(record['event_sequence'],before)
        self.assertEqual(len(record['events']),before)
        self.assertEqual(store.read(coordinator.run_id)['event_sequence'],before+1)

    def test_replay_and_export_cancellation_leave_record_and_target_unchanged(self):
        coordinator,clock,store=self.task()
        coordinator.start();advance(coordinator,clock,2)
        before=store.read(coordinator.run_id)
        def cancelled():raise RuntimeError('read cancelled')
        with self.assertRaises(RuntimeError):store.read(coordinator.run_id,check=cancelled)
        target=self.root/'cancelled-export.json'
        with self.assertRaises(RuntimeError):store.export(coordinator.run_id,target,check=cancelled)
        self.assertFalse(target.exists())
        self.assertEqual(store.read(coordinator.run_id),before)

    def test_cancelled_recovery_preserves_record_and_releases_owner_lease(self):
        coordinator, clock, store = self.task()
        coordinator._owner_lease.release()
        before = store.read(coordinator.run_id)
        acquired = []
        original = ResourceLease.acquire
        def acquire(lease):
            result = original(lease)
            if lease.owner.startswith('recovery:'): acquired.append(lease)
            return result
        def cancelled():
            if acquired: raise RuntimeError('recovery cancelled after acquiring owner lock')
        with patch.object(ResourceLease, 'acquire', acquire):
            with self.assertRaises(RuntimeError): store.recover_interrupted(check=cancelled)
        self.assertTrue(acquired)
        self.assertFalse(acquired[0].held)
        self.assertEqual(store.read(coordinator.run_id), before)
        lease = ResourceLease(['simulation:task:'+coordinator.run_id], owner='probe', outputs=self.root).acquire()
        lease.release()

    def test_invalid_backup_manifest_does_not_block_read_only_replay(self):
        coordinator, clock, store = self.task()
        before = store.read(coordinator.run_id)
        manifest = store.path.parent/'manifest.json'
        for value in ([], None, {}, {'schema':'femto-metadata-backup-v1', 'start_utc':1, 'end_utc':0},
                      {'schema':'femto-metadata-backup-v1', 'start_utc':'unknown', 'end_utc':2}):
            with self.subTest(value=value):
                manifest.write_text(json.dumps(value), encoding='utf8')
                result = store.read(coordinator.run_id)
                self.assertEqual(result['events'], before['events'])
                self.assertEqual(result['snapshot'], before['snapshot'])
                self.assertFalse(result['backup_context']['cross_database_atomic'])
                self.assertIsNone(result['backup_context']['backup_window_utc'])


if __name__=='__main__':unittest.main()
