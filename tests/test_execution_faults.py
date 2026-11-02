"""Fault, timing and immutable-binding behavior with a manually advanced clock."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from femto.execution.models import synthetic_plan
from femto.execution.state_machine import RunState
from femto.hardware.interface_contract import DeviceFeedback, DeviceRequest, FeedbackPhase, Intent, Readiness, SourceKind
from femto.hardware.simulation.scenario import Fault, Scenario, load_scenario, save_scenario
from test_execution_logic import advance, prepare, runtime


class ExecutionFaultTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def task(self, scenario=None):
        coordinator, clock, store = runtime(self.root, scenario)
        self.addCleanup(coordinator.close_instances)
        prepare(coordinator, clock)
        return coordinator, clock, store

    def test_saved_scenario_roundtrip_preserves_faults_and_capabilities(self):
        scenario=Scenario(confirm_stop=False,support_pause=False,base_mode='fixed',
                          faults=(Fault('base_move','base',at_s=.5),))
        path=save_scenario(scenario,outputs=self.root)
        self.assertEqual(load_scenario(path),scenario)
        with self.assertRaises(FileExistsError):save_scenario(scenario,outputs=self.root)
        payload=json.loads(path.read_text(encoding='utf-8-sig'));payload['source']='real'
        path.write_text(json.dumps(payload),encoding='utf8')
        with self.assertRaises(ValueError):load_scenario(path)

    def test_receive_and_completion_delays_count_only_confirmation(self):
        scenario = Scenario(faults=(Fault('receive_delay', 'arm', 'simulation_step', delay_s=.5),
                                    Fault('complete_delay', 'arm', 'simulation_step', delay_s=.7)))
        coordinator, clock, _ = self.task(scenario)
        coordinator.start(); advance(coordinator, clock, .4)
        self.assertEqual(coordinator.completed_steps, 0)
        self.assertIs(coordinator.state, RunState.RUNNING)
        advance(coordinator, clock, 3)
        self.assertIs(coordinator.state, RunState.SUCCEEDED)

    def test_no_response_stops_with_explicit_timeout_and_no_fabricated_progress(self):
        coordinator, clock, store = self.task(Scenario(faults=(Fault('no_response', 'arm', 'simulation_step'),)))
        coordinator.start(); advance(coordinator, clock, 3)
        self.assertIs(coordinator.state, RunState.FAILED)
        self.assertEqual(coordinator.completed_steps, 0)
        self.assertIn('request_timeout', [e['kind'] for e in store.read(coordinator.run_id)['events']])

    def test_reconnect_invalidates_task_without_replaying_old_steps(self):
        coordinator, clock, store = self.task()
        coordinator.start(); arm = coordinator.by_id['arm']
        submitted = tuple(arm.submitted)
        old = arm.binding
        arm.inject('reconnect'); coordinator.poll(); advance(coordinator, clock, 3)
        self.assertNotEqual(old, arm.binding)
        self.assertIs(coordinator.state, RunState.UNKNOWN)
        self.assertEqual(sum(p.request.intent is Intent.SIMULATION_STEP for p in coordinator.pending.values()), 2)
        self.assertEqual(tuple(arm.submitted[:len(submitted)]), submitted)
        self.assertTrue(any(e['kind']=='feedback_rejected' for e in store.read(coordinator.run_id)['events']))

    def test_disconnect_needs_unknown_stop_and_explicit_new_instance(self):
        coordinator, clock, _ = self.task()
        coordinator.start(); coordinator.by_id['arm'].inject('disconnect')
        coordinator.poll(); advance(coordinator, clock, 3)
        self.assertIs(coordinator.state, RunState.UNKNOWN)
        self.assertTrue(json.loads(coordinator.snapshot())['resource_held'])
        with self.assertRaises(RuntimeError): coordinator.freeze(synthetic_plan())

    def test_current_terminal_contradiction_retains_resource_old_binding_does_not(self):
        coordinator, clock, _ = self.task()
        coordinator.start(); advance(coordinator, clock, 2)
        self.assertIs(coordinator.state, RunState.SUCCEEDED)
        pending = next(p for p in reversed(tuple(coordinator.pending.values()))
                       if p.request.binding.device_id=='arm' and p.request.intent is Intent.REQUEST_STOP)
        feedback = DeviceFeedback(pending.request.request_id, pending.request.binding, Intent.REQUEST_STOP,
                                  3, FeedbackPhase.COMPLETED, motion_stopped=False)
        coordinator._feedback(replace(feedback, binding=replace(feedback.binding, source=SourceKind.REPLAY)))
        self.assertIs(coordinator.state, RunState.SUCCEEDED)
        coordinator._feedback(replace(feedback, request_id='another-request'))
        self.assertIs(coordinator.state, RunState.SUCCEEDED)
        coordinator._feedback(feedback)
        self.assertIs(coordinator.state, RunState.UNKNOWN)
        self.assertTrue(json.loads(coordinator.snapshot())['resource_held'])
        self.assertFalse(coordinator.start())

    def test_late_valid_fault_revokes_ready_and_old_epoch_fault_is_rejected(self):
        coordinator, clock, _ = self.task()
        coordinator.start(); advance(coordinator, clock, 2)
        pending = next(p for p in reversed(tuple(coordinator.pending.values())) if p.request.binding.device_id=='arm')
        feedback = DeviceFeedback(pending.request.request_id, pending.request.binding, pending.request.intent,
                                  5, FeedbackPhase.FAILED, motion_stopped=True, error_code='new-fault')
        coordinator._feedback(replace(feedback, binding=replace(feedback.binding, connection_epoch=0)))
        self.assertIs(coordinator.state, RunState.SUCCEEDED)
        coordinator._feedback(feedback)
        self.assertIs(coordinator.state, RunState.FAILED)
        with self.assertRaises(RuntimeError): coordinator.freeze(synthetic_plan())

    def test_late_status_unknown_does_not_keep_a_successful_stop_claim(self):
        coordinator, clock, _ = self.task()
        coordinator.start(); advance(coordinator, clock, 2)
        head = coordinator.by_id['head']
        head.output_off = None; head._status_event('status_unknown'); coordinator.poll()
        self.assertIs(coordinator.state, RunState.UNKNOWN)
        self.assertTrue(json.loads(coordinator.snapshot())['resource_held'])

    def test_new_terminal_feedback_unknown_revokes_only_its_relevant_physical_confirmation(self):
        for outcome in (RunState.SUCCEEDED, RunState.STOPPED, RunState.CANCELLED, RunState.FAILED):
            for device in ('arm', 'head'):
                with self.subTest(outcome=outcome, device=device):
                    coordinator, clock, store = runtime(self.root/outcome.value/device)
                    self.addCleanup(coordinator.close_instances)
                    prepare(coordinator, clock); coordinator.start()
                    if outcome is not RunState.SUCCEEDED:
                        coordinator.stop(outcome=outcome)
                    advance(coordinator, clock, 2)
                    self.assertIs(coordinator.state, outcome)
                    completed = coordinator.completed_steps
                    pending = next(p for p in coordinator.pending.values()
                                   if p.request.binding.device_id == device
                                   and p.request.intent is Intent.REQUEST_STOP)
                    confirmed = DeviceFeedback(pending.request.request_id, pending.request.binding,
                        Intent.REQUEST_STOP, 3, FeedbackPhase.COMPLETED, Readiness.READY,
                        motion_stopped=True if device == 'arm' else None,
                        output_off=True if device == 'head' else None)
                    coordinator._feedback(confirmed)
                    self.assertIs(coordinator.state, outcome)  # Unrelated unknown field is not an alarm.
                    unknown = replace(confirmed, motion_stopped=None, output_off=None)
                    coordinator._feedback(unknown)  # Same sequence cannot revoke a newer confirmation.
                    self.assertIs(coordinator.state, outcome)
                    newer = replace(unknown, sequence=4)
                    coordinator._feedback(replace(newer, binding=replace(newer.binding, source=SourceKind.REPLAY)))
                    coordinator._feedback(replace(newer, binding=replace(newer.binding,
                        connection_epoch=newer.binding.connection_epoch-1)))
                    self.assertIs(coordinator.state, outcome)
                    coordinator._feedback(newer)
                    self.assertIs(coordinator.state, RunState.UNKNOWN)
                    self.assertEqual(coordinator.completed_steps, completed)
                    self.assertTrue(json.loads(coordinator.snapshot())['resource_held'])
                    self.assertEqual(store.read(coordinator.run_id)['recorded_state'], RunState.UNKNOWN.value)

    def test_cancel_request_preserves_unconfirmed_physical_state(self):
        coordinator, clock, _ = runtime(self.root, Scenario(
            faults=(Fault('complete_delay', 'arm', 'simulation_step', delay_s=.75),)))
        self.addCleanup(coordinator.close_instances)
        arm = coordinator.by_id['arm']; arm.connect(); arm.poll()
        active = DeviceRequest('individual-step', arm.binding, Intent.SIMULATION_STEP)
        arm.submit(active, synthetic_plan().steps[0]); arm.poll()
        request = DeviceRequest('cancel-individual', arm.binding, Intent.CANCEL_REQUEST, active.request_id)
        arm.submit(request); clock.advance(.3)
        events = arm.poll()
        # A cancellation never supplies the global stop/output confirmation.
        cancelled = [e for e in events if isinstance(e, DeviceFeedback) and e.phase is FeedbackPhase.CANCELLED]
        self.assertEqual(len(cancelled), 1)
        self.assertIs(cancelled[0].motion_stopped, False)
        self.assertIsNone(arm._active)
        stop = DeviceRequest('global-stop', arm.binding, Intent.REQUEST_STOP)
        arm.submit(stop); clock.advance(.3); arm.poll()
        self.assertIs(arm.motion_stopped, True)

    def test_config_replacement_blocks_resume_without_changing_frozen_snapshot(self):
        coordinator, clock, store = self.task()
        coordinator.start(); coordinator.pause(); advance(coordinator, clock, .4)
        before = store.read(coordinator.run_id)['snapshot']
        arm = coordinator.by_id['arm']; arm.configuration = replace(arm.configuration, revision='2')
        self.assertFalse(coordinator.resume()); advance(coordinator, clock, .4)
        self.assertIs(coordinator.state, RunState.FAILED)
        self.assertEqual(store.read(coordinator.run_id)['snapshot'], before)

    def test_slow_log_batch_repolls_before_dispatch_and_missing_heartbeat_still_fails(self):
        coordinator, clock, store = self.task()
        original = store.append
        delayed = []
        def slow(*args):
            result = original(*args)
            if args[2]=='device_feedback' and not delayed:
                delayed.append(True); clock.advance(.6)
            return result
        coordinator.start()
        with patch.object(store, 'append', side_effect=slow): advance(coordinator, clock, .1)
        self.assertIs(coordinator.state, RunState.RUNNING)
        advance(coordinator, clock, 2)
        self.assertIs(coordinator.state, RunState.SUCCEEDED)
        coordinator.close_instances()
        second, other_clock, _ = runtime(self.root/'stale')
        self.addCleanup(second.close_instances); prepare(second, other_clock)
        with patch.object(second.by_id['arm'], 'poll', return_value=()):
            advance(second, other_clock, .7)
        self.assertIs(second.state, RunState.STOP_REQUESTED)

    def test_persistence_uses_up_remaining_heartbeat_age_without_false_disconnect(self):
        coordinator, clock, store = self.task(Scenario(
            faults=(Fault('complete_delay', 'head', 'simulation_step', delay_s=.1),)))
        coordinator.start()
        clock.advance(.21); coordinator.poll()  # Fresh heartbeat, before arm completion.
        original = store.append
        delayed = []
        def slow(*args):
            result = original(*args)
            if (args[2] == 'device_feedback' and args[3]['feedback']['phase'] == 'completed'
                    and args[3]['feedback']['binding']['device_id'] == 'arm' and not delayed):
                delayed.append(True); clock.advance(.47)
            return result
        clock.advance(.05)
        with patch.object(store, 'append', side_effect=slow): coordinator.poll()
        self.assertTrue(delayed)
        self.assertGreater(clock.now()-coordinator._statuses['head'].timestamp_s,
                           coordinator.scenario.status_max_age_s)
        self.assertIs(coordinator.state, RunState.RUNNING)
        self.assertEqual(coordinator.completed_steps, 0)
        self.assertEqual(sum(p.request.intent is Intent.SIMULATION_STEP
                             for p in coordinator.pending.values()), 2)
        coordinator.poll(); advance(coordinator, clock, 2)
        self.assertIs(coordinator.state, RunState.SUCCEEDED)
        self.assertNotIn('operator_stop', [e['kind'] for e in store.read(coordinator.run_id)['events']
                                         if e['payload'].get('requested_outcome') == 'failed'])

    def test_slow_plan_hash_refreshes_status_before_prepare(self):
        coordinator, clock, _ = runtime(self.root)
        self.addCleanup(coordinator.close_instances)
        coordinator.connect(); plan = synthetic_plan(); coordinator.freeze(plan)
        original = type(plan).verify_assets
        def slow(value, *args):
            original(value, *args); clock.advance(.75)
        with patch.object(type(plan), 'verify_assets', slow):
            self.assertTrue(coordinator.check())
        advance(coordinator, clock, .4)
        self.assertIs(coordinator.state, RunState.READY)
        self.assertTrue(coordinator.preflight['simulation_ready'])

    def test_slow_received_record_repolls_completion_before_timeout(self):
        coordinator, clock, store = self.task()
        coordinator.start()
        original=store.append;delayed=[]
        def slow(*args):
            result=original(*args)
            if args[2]=='device_feedback' and args[3]['feedback']['phase']=='received' and not delayed:
                delayed.append(True);clock.advance(2.5)
            return result
        clock.advance(.04)
        with patch.object(store,'append',side_effect=slow):coordinator.poll()
        self.assertIs(coordinator.state,RunState.RUNNING)
        coordinator.poll();advance(coordinator,clock,2)
        self.assertIs(coordinator.state,RunState.SUCCEEDED)
        self.assertNotIn('request_timeout',[e['kind'] for e in store.read(coordinator.run_id)['events']])

    def test_waiting_stop_feedback_is_polled_after_slow_received_record(self):
        coordinator,clock,store=self.task()
        coordinator.start();coordinator.stop()
        original=store.append;delayed=[]
        def slow(*args):
            result=original(*args)
            if args[2]=='device_feedback' and args[3]['feedback']['phase']=='received' and not delayed:
                delayed.append(True);clock.advance(2.5)
            return result
        clock.advance(.04)
        with patch.object(store,'append',side_effect=slow):coordinator.poll()
        self.assertIs(coordinator.state,RunState.STOP_REQUESTED)
        coordinator.poll()
        self.assertIs(coordinator.state,RunState.STOPPED)

    def test_completed_feedback_with_fault_or_not_ready_never_counts_step(self):
        for status in ('error', 'not_ready', 'unknown'):
            with self.subTest(status=status):
                coordinator, clock, _ = runtime(self.root/status, Scenario(complete_delay_s=.5))
                self.addCleanup(coordinator.close_instances); prepare(coordinator, clock)
                advance(coordinator, clock, .4)
                coordinator.start()
                pending = next(p for p in coordinator.pending.values()
                               if p.request.intent is Intent.SIMULATION_STEP and p.request.binding.device_id == 'arm')
                feedback = DeviceFeedback(pending.request.request_id, pending.request.binding, pending.request.intent,
                    3, FeedbackPhase.COMPLETED,
                    Readiness.NOT_READY if status == 'not_ready' else Readiness.UNKNOWN if status == 'unknown' else Readiness.READY,
                    True, 'reported-error' if status == 'error' else None)
                coordinator._feedback(feedback); advance(coordinator, clock, 1)
                self.assertIs(coordinator.state, RunState.FAILED)
                self.assertEqual(coordinator.completed_steps, 0)

    def test_close_attempts_every_device_and_retains_resources_on_failure(self):
        coordinator, clock, _ = self.task()
        coordinator.start()
        with patch.object(coordinator.by_id['arm'], 'close', side_effect=OSError('close failed')):
            with self.assertRaises(RuntimeError): coordinator.close_instances()
        self.assertFalse(coordinator.by_id['head'].connected)
        self.assertIs(coordinator.state, RunState.UNKNOWN)
        self.assertTrue(json.loads(coordinator.snapshot())['resource_held'])
        with self.assertRaises(RuntimeError): coordinator.connect()
        coordinator.close_instances()
        self.assertFalse(json.loads(coordinator.snapshot())['resource_held'])

    def test_critical_queue_overflow_does_not_start_or_succeed(self):
        coordinator, clock, store = self.task(Scenario(max_events=8))
        coordinator.start()
        with patch.object(coordinator.by_id['arm'], 'poll', side_effect=OverflowError('bounded queue')):
            coordinator.poll(); advance(coordinator, clock, 3)
        self.assertIs(coordinator.state, RunState.UNKNOWN)
        self.assertEqual(coordinator.completed_steps, 0)
        self.assertTrue(any(e['kind']=='critical_queue_gap' for e in store.read(coordinator.run_id)['events']))

    def test_event_gap_during_stop_never_records_a_normal_terminal_result(self):
        for outcome in (RunState.SUCCEEDED, RunState.STOPPED, RunState.CANCELLED):
            with self.subTest(outcome=outcome):
                coordinator, clock, store = runtime(self.root/outcome.value)
                self.addCleanup(coordinator.close_instances)
                prepare(coordinator, clock); coordinator.start()
                if outcome is RunState.SUCCEEDED:
                    for _ in range(200):
                        clock.advance(.02); coordinator.poll()
                        if coordinator.state is RunState.STOP_REQUESTED:
                            break
                else:
                    coordinator.stop(outcome=outcome)
                self.assertIs(coordinator.state, RunState.STOP_REQUESTED)
                stops = tuple(coordinator._group)
                coordinator.critical_queue_gap('lost critical event during stop')
                self.assertEqual(tuple(coordinator._group), stops)
                advance(coordinator, clock, .5)
                self.assertIs(coordinator.state, RunState.FAILED)
                record = store.read(coordinator.run_id)
                normal = {RunState.SUCCEEDED.value, RunState.STOPPED.value, RunState.CANCELLED.value}
                self.assertFalse(any(e['payload'].get('to_state') in normal for e in record['events']))
                self.assertFalse(json.loads(coordinator.snapshot())['resource_held'])
                with self.assertRaises(RuntimeError): coordinator.freeze(synthetic_plan())

    def test_base_invalidation_during_final_stop_never_transiently_succeeds(self):
        coordinator, clock, store = self.task(Scenario(base_mode='fixed'))
        coordinator.start()
        for _ in range(200):
            clock.advance(.02); coordinator.poll()
            if coordinator.state is RunState.STOP_REQUESTED:
                break
        self.assertIs(coordinator.state, RunState.STOP_REQUESTED)
        coordinator.by_id['base'].inject('base_unknown')
        coordinator.poll(); advance(coordinator, clock, .5)
        self.assertIs(coordinator.state, RunState.FAILED)
        self.assertFalse(any(e['payload'].get('to_state') == RunState.SUCCEEDED.value
                             for e in store.read(coordinator.run_id)['events']))

    def test_device_poll_error_during_final_stop_is_recorded_and_degrades_result(self):
        coordinator, clock, store = self.task()
        coordinator.start()
        for _ in range(200):
            clock.advance(.02); coordinator.poll()
            if coordinator.state is RunState.STOP_REQUESTED:
                break
        with patch.object(coordinator.by_id['arm'], 'poll', side_effect=OSError('feedback transport failed')):
            coordinator.poll()
        advance(coordinator, clock, .5)
        self.assertIs(coordinator.state, RunState.FAILED)
        gaps = [e for e in store.read(coordinator.run_id)['events'] if e['kind'] == 'critical_queue_gap']
        self.assertTrue(gaps)
        self.assertIn('feedback transport failed', gaps[0]['payload']['reason'])

    def test_contradictory_stop_confirmation_before_other_devices_finish_is_unknown(self):
        scenario = Scenario(faults=(Fault('complete_delay', 'head', 'request_stop', delay_s=.25),))
        for kind in ('status', 'feedback'):
            with self.subTest(kind=kind):
                coordinator, clock, store = runtime(self.root/kind, scenario)
                self.addCleanup(coordinator.close_instances)
                prepare(coordinator, clock); coordinator.start()
                for _ in range(200):
                    clock.advance(.02); coordinator.poll()
                    if coordinator.state is RunState.STOP_REQUESTED:
                        break
                advance(coordinator, clock, .3)
                self.assertIs(coordinator.state, RunState.STOP_REQUESTED)
                arm = coordinator.by_id['arm']
                if kind == 'status':
                    arm.motion_stopped = None
                    arm._status_event('stop_state_unknown'); coordinator.poll()
                else:
                    pending = next(p for p in coordinator.pending.values()
                                   if p.request.binding.device_id == 'arm' and p.request.intent is Intent.REQUEST_STOP)
                    self.assertTrue(pending.done)
                    coordinator._feedback(DeviceFeedback(pending.request.request_id, arm.binding, Intent.REQUEST_STOP,
                                                          3, FeedbackPhase.COMPLETED, Readiness.READY, False))
                advance(coordinator, clock, .5)
                self.assertIs(coordinator.state, RunState.UNKNOWN)
                self.assertTrue(json.loads(coordinator.snapshot())['resource_held'])
                self.assertFalse(any(e['payload'].get('to_state') == RunState.SUCCEEDED.value
                                     for e in store.read(coordinator.run_id)['events']))

    def test_success_requires_every_step_even_when_stop_outcome_requests_success(self):
        coordinator, clock, store = self.task()
        coordinator.start(); coordinator.stop(outcome=RunState.SUCCEEDED)
        advance(coordinator, clock, .4)
        self.assertIs(coordinator.state, RunState.FAILED)
        self.assertEqual(coordinator.completed_steps, 0)
        self.assertFalse(any(e['payload'].get('to_state') == RunState.SUCCEEDED.value
                             for e in store.read(coordinator.run_id)['events']))

    def test_paused_physical_state_becoming_unknown_revokes_resume(self):
        coordinator, clock, _ = self.task()
        coordinator.start(); coordinator.pause(); advance(coordinator, clock, .4)
        self.assertIs(coordinator.state, RunState.PAUSED)
        arm = coordinator.by_id['arm']
        arm.motion_stopped = None; arm._status_event('pause_state_unknown')
        coordinator.poll()
        self.assertFalse(coordinator.resume())
        advance(coordinator, clock, .4)
        self.assertIs(coordinator.state, RunState.FAILED)

    def test_new_frozen_task_discards_previous_preflight_and_stop_context(self):
        coordinator, clock, _ = self.task()
        coordinator.start(); advance(coordinator, clock, 2)
        self.assertIs(coordinator.state, RunState.SUCCEEDED)
        previous_run = coordinator.run_id
        self.assertTrue(coordinator.preflight['simulation_ready'])
        coordinator.freeze(synthetic_plan())
        self.assertNotEqual(coordinator.run_id, previous_run)
        self.assertIsNone(coordinator.preflight)
        self.assertIsNone(coordinator._group_intent)
        self.assertIsNone(coordinator._stop_deadline)
        self.assertEqual(coordinator.completed_steps, 0)
        self.assertFalse(json.loads(coordinator.snapshot())['can_start'])

    def test_request_faults_cannot_be_scheduled_as_device_state_events(self):
        for kind in ('receive_delay', 'complete_delay', 'reject', 'no_response',
                     'stop_unconfirmed', 'duplicate', 'out_of_order', 'old_epoch'):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                Fault(kind, 'arm', at_s=.5)

    def test_shutdown_deadline_keeps_unknown_and_refuses_new_work(self):
        coordinator, clock, _ = self.task(Scenario(faults=(Fault('no_response', 'head', 'request_stop'),)))
        coordinator.start(); coordinator.request_shutdown(); advance(coordinator, clock, .4)
        coordinator.shutdown_expired()
        self.assertIs(coordinator.state, RunState.UNKNOWN)
        self.assertFalse(coordinator.start())
        with self.assertRaises(RuntimeError): coordinator.freeze(synthetic_plan())


if __name__=='__main__': unittest.main()
