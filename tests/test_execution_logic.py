"""Deterministic contract, geometry and simulation checks without devices/Qt."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from femto.dataset.installation_calibration import RDF_AXES, TransformRecord, load_record, save_record
from femto.execution.coordinator import Coordinator
from femto.execution.history import ExecutionStore
from femto.execution.models import synthetic_plan
from femto.execution.state_machine import RunState, transition
from femto.geometry.axial_gap import SurfaceEvidence, axial_gap
from femto.geometry.frame_transforms import SpatialContext, camera_to_tool, compose
from femto.hardware.configuration import DeviceConfiguration, create_adapter
from femto.hardware.interface_contract import (
    Capabilities, DeviceBinding, DeviceFeedback, DeviceKind, DeviceRequest, FeedbackPhase,
    Intent, Readiness, SourceKind, validate_request,
)
from femto.hardware.simulation.scenario import Fault, ManualClock, Scenario


def transform(target, source, xyz=(0, 0, 0), *, dynamic=False, timestamp=0):
    matrix = np.eye(4); matrix[:3, 3] = xyz
    axes = lambda name: RDF_AXES if name in ('C', 'W0') else 'synthetic right-handed XYZ'
    return TransformRecord(target, source, matrix, target+source, '1', 'synthetic',
                           ('synthetic-arm', 'synthetic-camera'), ('synthetic-mount',),
                           json.dumps(dict(kind='synthetic_test')), axes(target), axes(source),
                           dynamic, timestamp if dynamic else None, 'test-clock' if dynamic else None)


def runtime(root, scenario=None, *, instance='test-instance', spatial=None):
    scenario = scenario or Scenario()
    clock = ManualClock(domain='test-clock')
    devices = [DeviceConfiguration('arm', DeviceKind.ARM, 'simulation', 'arm', '1'),
               DeviceConfiguration('head', DeviceKind.TREATMENT_HEAD, 'simulation', 'head', '1')]
    if scenario.base_mode != 'absent':
        devices.append(DeviceConfiguration('base', DeviceKind.BASE, 'simulation', 'base', '1'))
    adapters = [create_adapter(c, scenario, clock, instance_id=instance, epoch_seed=1) for c in devices]
    store = ExecutionStore(root)
    coordinator = Coordinator(adapters, clock, scenario, store, outputs=root, spatial_context=spatial)
    return coordinator, clock, store


def advance(coordinator, clock, seconds=.4):
    for _ in range(round(seconds/.02)):
        clock.advance(.02); coordinator.poll()


def prepare(coordinator, clock):
    coordinator.connect(); coordinator.freeze(synthetic_plan()); coordinator.check()
    advance(coordinator, clock)


class ContractAndGeometryTests(unittest.TestCase):
    def test_new_control_intents_still_require_capabilities_and_simulation_protocol(self):
        binding = DeviceBinding('a', DeviceKind.ARM, 1, SourceKind.SIMULATION)
        for intent in (Intent.REQUEST_PAUSE, Intent.REQUEST_RESUME, Intent.SIMULATION_STEP):
            request = DeviceRequest('r', binding, intent)
            with self.assertRaises(ValueError): validate_request(request, Capabilities(binding))
            self.assertEqual(validate_request(request, Capabilities(binding, 'femto-simulation-v1', {intent})), request)
        real = replace(binding, source=SourceKind.REAL)
        with self.assertRaises(ValueError):
            validate_request(DeviceRequest('s', real, Intent.SIMULATION_STEP), Capabilities(real, 'femto-simulation-v1', {Intent.SIMULATION_STEP}))

    def test_unconfigured_and_real_adapters_do_not_fallback(self):
        for adapter in ('unconfigured', 'real', 'replay'):
            config = DeviceConfiguration('a', DeviceKind.ARM, adapter, 'a', '1')
            with self.assertRaises(NotImplementedError):
                create_adapter(config, Scenario(), ManualClock(), instance_id='x')
        with self.assertRaises(ValueError):
            DeviceConfiguration('a', DeviceKind.ARM, 'simulation', 'a', '1', connection_json='{"capabilities":["request_stop"]}')

    def test_known_transform_and_axis_direction(self):
        bf, fc, fu = transform('B', 'F', (1, 0, 0)), transform('F', 'C', (0, 2, 0)), transform('F', 'U', (0, 1, 0))
        np.testing.assert_allclose(compose(bf, fc).apply([0, 0, 0]), [1, 2, 0])
        np.testing.assert_allclose(camera_to_tool(fc, fu, rigid_installation_verified=True).apply([0, 0, 0]), [0, 1, 0])
        with self.assertRaises(ValueError): compose(fc, bf)
        with self.assertRaises(ValueError): compose(None)
        with self.assertRaises(ValueError): camera_to_tool(fc, fu, rigid_installation_verified=False)
        matrix = np.eye(4); matrix[0, 0] = 1.001
        with self.assertRaises(ValueError): replace(fc, matrix=matrix)
        with self.assertRaises(ValueError): replace(fc, unit='millimetre')
        with self.assertRaises(ValueError): replace(fc, source_axes='RUB')

    def test_installation_records_are_versioned_and_cannot_self_certify(self):
        with TemporaryDirectory() as root:
            record = transform('F', 'C')
            path = save_record(record, outputs=root)
            self.assertEqual(load_record(path), record)
            self.assertFalse(path.is_relative_to(Path(root)/'datasets'))
            with self.assertRaises(FileExistsError): save_record(record, outputs=root)
            payload = record.to_dict(); payload['physical_accuracy_verified'] = True
            with self.assertRaises(ValueError): TransformRecord.from_dict(payload)

    def test_spatial_reference_invalidates_without_rewriting_history(self):
        context = SpatialContext(session_id='s', hand_eye=transform('F', 'C', (0, .1, 0)),
                                 tool_mount=transform('F', 'U', (0, 0, .1)), synchronization_tolerance_s=.01,
                                 max_pose_age_s=.5, clock_domain='test-clock', budget_source='synthetic:scenario-v1')
        pose = transform('B', 'F', (1, 0, 0), dynamic=True)
        chain = context.establish_initial_reference(pose, frame_id=0, reference_frame_id=0,
                                                   camera_timestamp_s=0, base_state='locked', installation_evidence='test')
        before = chain.to_dict()
        np.testing.assert_allclose(context.current(pose, now_s=.1, clock_domain='test-clock', camera_timestamp_s=0)['camera'].matrix, np.eye(4), atol=1e-10)
        with self.assertRaises(ValueError): context.current(pose, now_s=1, clock_domain='test-clock')
        context.invalidate('base_move')
        with self.assertRaises(ValueError): context.current(pose, now_s=.1, clock_domain='test-clock')
        with self.assertRaises(ValueError):
            context.establish_initial_reference(pose, frame_id=1, reference_frame_id=0, camera_timestamp_s=0, base_state='locked', installation_evidence='test')
        self.assertEqual(chain.to_dict(), before)

    def test_ray_gap_uses_supported_nearest_front_surface(self):
        surface = SurfaceEvidence(((-1, -1, .2), (0, 1, .2), (1, -1, .2)), ((0, 1, 2),),
                                  (1,), (('original-view-1',),), 'synthetic_test', 'C', known_occlusion=False)
        result = axial_gap(surface, [0, 0, 0], [0, 0, 2], frame='C', mode='synthetic_test', calibration_applicable=True)
        self.assertAlmostEqual(result['distance_m'], .2)
        self.assertFalse(result['execution_ready'])
        for modified, expected in ((replace(surface, confidence=(.3,)), 'nearest_surface_not_measured_supported'),
                                    (replace(surface, view_support=((),)), 'nearest_surface_not_measured_supported'),
                                    (replace(surface, known_occlusion=True), 'known_occlusion'),
                                    (replace(surface, faces=((0, 2, 1),)), 'backface_hit')):
            self.assertEqual(axial_gap(modified, [0, 0, 0], [0, 0, 1], frame='C', calibration_applicable=True)['reason'], expected)
        self.assertIsNone(axial_gap(surface, [0, 0, 0], [0, 0, 1], frame='C')['distance_m'])
        self.assertIsNone(axial_gap(surface, [0, 0, 0], [0, 0, 1], frame='C', mode='realtime', calibration_applicable=True)['distance_m'])
        self.assertEqual(axial_gap(surface, [4, 0, 0], [0, 0, 1], frame='C', calibration_applicable=True)['reason'], 'no_forward_intersection')


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def create(self, scenario=None):
        coordinator, clock, store = runtime(self.root, scenario)
        self.addCleanup(coordinator.close_instances)
        return coordinator, clock, store

    def test_normal_closed_loop_persists_intent_before_feedback(self):
        coordinator, clock, store = self.create()
        self.assertFalse(store.path.exists())
        self.assertEqual(store.runs(), [])
        self.assertFalse(store.path.exists())
        prepare(coordinator, clock)
        self.assertIs(coordinator.state, RunState.READY)
        self.assertFalse(any(c['status']=='passed' for c in coordinator.preflight['checks'] if c['scope']=='real_execution'))
        coordinator.start(); advance(coordinator, clock, 2)
        self.assertIs(coordinator.state, RunState.SUCCEEDED)
        self.assertEqual(coordinator.completed_steps, 4)
        self.assertFalse(json.loads(coordinator.snapshot())['resource_held'])
        record = store.read(coordinator.run_id)
        kinds = [e['kind'] for e in record['events']]
        self.assertLess(kinds.index('request_intent'), kinds.index('request_sent'))
        self.assertLess(kinds.index('request_sent'), kinds.index('device_feedback'))
        self.assertEqual(record['display_state'], 'succeeded')
        self.assertEqual(record['snapshot']['plan']['plan_source'], 'synthetic_test')
        self.assertTrue(record['read_only'])

    def test_pause_is_confirmed_and_resume_rechecks_without_duplicate_requests(self):
        coordinator, clock, _ = self.create(Scenario(complete_delay_s=.5))
        coordinator.connect(); coordinator.freeze(synthetic_plan()); coordinator.check(); advance(coordinator, clock, .6)
        coordinator.start(); self.assertTrue(coordinator.pause()); self.assertFalse(coordinator.pause())
        self.assertIs(coordinator.state, RunState.PAUSE_REQUESTED)
        advance(coordinator, clock, .6)
        self.assertIs(coordinator.state, RunState.PAUSED)
        count = coordinator.completed_steps
        advance(coordinator, clock, 2)
        self.assertEqual(coordinator.completed_steps, count)
        self.assertTrue(coordinator.resume()); advance(coordinator, clock, 4)
        self.assertIs(coordinator.state, RunState.SUCCEEDED)

    def test_unconfirmed_stop_is_unknown_and_keeps_resources(self):
        coordinator, clock, _ = self.create(Scenario(confirm_stop=False))
        prepare(coordinator, clock); coordinator.start(); coordinator.stop()
        self.assertIs(coordinator.state, RunState.STOP_REQUESTED)
        advance(coordinator, clock)
        self.assertIs(coordinator.state, RunState.UNKNOWN)
        with self.assertRaises(RuntimeError): coordinator.freeze(synthetic_plan())
        self.assertTrue(json.loads(coordinator.snapshot())['resource_held'])
        coordinator.close_instances()
        self.assertFalse(json.loads(coordinator.snapshot())['resource_held'])

    def test_missing_head_stop_cannot_be_confirmed_by_arm(self):
        coordinator, clock, _ = self.create(Scenario(faults=(Fault('no_response', 'head', 'request_stop'),)))
        prepare(coordinator, clock); coordinator.start(); coordinator.stop()
        advance(coordinator, clock, .4)
        self.assertIs(coordinator.state, RunState.STOP_REQUESTED)
        advance(coordinator, clock, 2)
        self.assertIs(coordinator.state, RunState.UNKNOWN)

    def test_duplicate_old_epoch_and_out_of_order_feedback_do_not_add_steps(self):
        faults = tuple(Fault(kind, 'arm', 'simulation_step') for kind in ('duplicate', 'old_epoch', 'out_of_order'))
        coordinator, clock, store = self.create(Scenario(faults=faults))
        prepare(coordinator, clock); coordinator.start(); advance(coordinator, clock, 2)
        self.assertIs(coordinator.state, RunState.SUCCEEDED)
        self.assertEqual(coordinator.completed_steps, 4)
        self.assertGreaterEqual(sum(e['kind']=='feedback_rejected' for e in store.read(coordinator.run_id)['events']), 3)

    def test_capability_differences_and_fixed_base(self):
        coordinator, clock, _ = self.create(Scenario(support_pause=False, base_mode='fixed'))
        prepare(coordinator, clock); coordinator.start()
        self.assertFalse(coordinator.pause())
        base = coordinator.by_id['base']
        self.assertNotIn(Intent.PREPARE, base.capabilities().supported_intents)
        advance(coordinator, clock, 2)
        self.assertIs(coordinator.state, RunState.SUCCEEDED)

    def test_only_fixed_base_and_empty_scenes_cannot_prepare_or_run(self):
        for fixed in (True,False):
            with self.subTest(fixed=fixed):
                scenario=Scenario(base_mode='fixed')
                coordinator,clock,store=runtime(self.root/str(fixed),scenario)
                base=coordinator.by_id['base']
                selected=(base,) if fixed else ()
                coordinator=Coordinator(selected,clock,scenario,store,outputs=self.root/str(fixed))
                self.addCleanup(coordinator.close_instances)
                coordinator.connect();coordinator.freeze(synthetic_plan())
                self.assertFalse(coordinator.check())
                self.assertIs(coordinator.state,RunState.FAILED)
                self.assertFalse(coordinator.preflight['simulation_ready'])
                self.assertFalse(coordinator.start())
                self.assertEqual(base.submitted,[])

    def test_rejection_stops_all_devices_before_failure(self):
        coordinator, clock, _ = self.create(Scenario(faults=(Fault('reject', 'arm', 'simulation_step'),)))
        prepare(coordinator, clock); coordinator.start(); advance(coordinator, clock, 1)
        self.assertIs(coordinator.state, RunState.FAILED)
        self.assertEqual(coordinator.completed_steps, 0)
        self.assertTrue(coordinator.by_id['head'].status().output_off)

    def test_base_motion_revokes_task_and_spatial_reference(self):
        spatial = SpatialContext(session_id='s', hand_eye=transform('F', 'C'), tool_mount=transform('F', 'U'),
                                 synchronization_tolerance_s=.01, max_pose_age_s=.5,
                                 clock_domain='test-clock', budget_source='synthetic:1')
        pose = transform('B', 'F', dynamic=True)
        spatial.establish_initial_reference(pose, frame_id=0, reference_frame_id=0,
                                            camera_timestamp_s=0, base_state='locked', installation_evidence='test')
        coordinator, clock, _ = runtime(self.root, Scenario(base_mode='fixed'), spatial=spatial)
        self.addCleanup(coordinator.close_instances)
        prepare(coordinator, clock); coordinator.start(); coordinator.by_id['base'].inject('base_move')
        advance(coordinator, clock, .6)
        self.assertIs(coordinator.state, RunState.FAILED)
        self.assertIsNone(spatial.base_to_w0)
        self.assertTrue(spatial.requires_new_session)

    def test_base_unknown_and_mount_unknown_revoke_current_context(self):
        for kind in ('base_unknown','mount_unknown'):
            with self.subTest(kind=kind):
                spatial=SpatialContext(session_id=kind,hand_eye=transform('F','C'),tool_mount=transform('F','U'),
                    synchronization_tolerance_s=.01,max_pose_age_s=.5,clock_domain='test-clock',budget_source='synthetic:1')
                spatial.establish_initial_reference(transform('B','F',dynamic=True),frame_id=0,reference_frame_id=0,
                    camera_timestamp_s=0,base_state='locked',installation_evidence='synthetic')
                coordinator,clock,store=runtime(self.root/kind,Scenario(base_mode='fixed'),spatial=spatial)
                self.addCleanup(coordinator.close_instances);prepare(coordinator,clock)
                before=store.read(coordinator.run_id)['snapshot']['spatial_context']
                coordinator.start();coordinator.by_id['base' if kind=='base_unknown' else 'arm'].inject(kind)
                advance(coordinator,clock,.6)
                self.assertIsNone(spatial.base_to_w0)
                self.assertTrue(spatial.requires_new_session)
                self.assertIn(coordinator.state,(RunState.FAILED,RunState.UNKNOWN))
                self.assertEqual(store.read(coordinator.run_id)['snapshot']['spatial_context'],before)

    def test_illegal_state_change_cannot_skip_physical_stop(self):
        with self.assertRaises(ValueError): transition(RunState.RUNNING, RunState.SUCCEEDED)


if __name__ == '__main__':
    unittest.main()
