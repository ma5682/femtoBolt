"""Deterministic low-level simulated devices. No execution/planning/Qt imports."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import replace
import heapq
import uuid

from femto.hardware.interface_contract import (
    Capabilities, DeviceBinding, DeviceFeedback, DeviceKind, DeviceRequest, DeviceStatus,
    FeedbackPhase, Intent, Readiness, SimulationStep, SourceKind, SpatialState, validate_request,
)


class SimulatedDevice:
    def __init__(self, configuration, scenario, clock, *, instance_id, epoch_seed=None):
        if configuration.adapter != 'simulation' or configuration.kind is DeviceKind.CAMERA:
            raise ValueError('Simulation cannot take ownership of a real camera or non-simulation binding')
        self.configuration, self.scenario, self.clock = configuration, scenario, clock
        self.instance_id = instance_id
        self.resource_key = 'simulation:'+instance_id+':'+configuration.resource_id
        self.binding = DeviceBinding(configuration.device_id, configuration.kind,
                                     epoch_seed if epoch_seed is not None else uuid.uuid4().int >> 64,
                                     SourceKind.SIMULATION)
        self.connected = False
        self.readiness = Readiness.UNKNOWN
        self.motion_stopped = self.output_off = None
        self.spatial_state = SpatialState.UNKNOWN if configuration.kind is DeviceKind.BASE else SpatialState.NOT_APPLICABLE
        self.fault = None
        self._sequence = 0
        self._status_timestamp = 0.
        self._serial = 0
        self._scheduled = []
        self._events = deque()
        self._active = None
        self._step_generation = 0
        self._paused = False
        self._last_heartbeat = 0.
        self._scenario_started = None
        self._counts = Counter()
        self._used_faults = set()
        self.submitted = []  # Bounded diagnostic IDs, not a task interpreter.

    @property
    def fixed_base(self):
        return self.binding.kind is DeviceKind.BASE and self.scenario.base_mode == 'fixed'

    def capabilities(self):
        if not self.connected:
            return Capabilities(self.binding)
        intents = {Intent.QUERY_STATUS, Intent.QUERY_CAPABILITIES}
        if not self.fixed_base:
            intents |= {Intent.PREPARE, Intent.SIMULATION_STEP, Intent.REQUEST_STOP, Intent.CANCEL_REQUEST}
            if self.scenario.support_pause:
                intents |= {Intent.REQUEST_PAUSE, Intent.REQUEST_RESUME}
        return Capabilities(self.binding, 'femto-simulation-v1', frozenset(intents))

    def _physical(self, stopped):
        if self.binding.kind in (DeviceKind.ARM, DeviceKind.BASE) and not self.fixed_base:
            self.motion_stopped = stopped
        elif self.binding.kind in (DeviceKind.TREATMENT_HEAD, DeviceKind.ULTRASOUND_GENERATOR):
            self.output_off = stopped

    def status(self):
        return DeviceStatus(self.binding, self._sequence, self.clock.domain, self._status_timestamp,
                            self.connected, self.readiness, self.motion_stopped, self.output_off,
                            self.spatial_state, self.fault, 'status', self.configuration.revision)

    def _emit(self, value):
        if len(self._events) >= self.scenario.max_events:
            raise OverflowError('Simulated device critical event queue overflow')
        self._events.append(value)

    def _status_event(self, event_type='status'):
        self._sequence += 1
        self._status_timestamp = self.clock.now()
        self._emit(replace(self.status(), event_type=event_type))

    def connect(self):
        if self.connected:
            return self.binding
        self.binding = replace(self.binding, connection_epoch=self.binding.connection_epoch+1)
        if self._scenario_started is None:
            self._scenario_started = self.clock.now()
        self._sequence = 0
        self.connected = True
        self.readiness = Readiness.READY
        self._physical(True)
        self.fault = None
        if self.binding.kind is DeviceKind.BASE:
            self.spatial_state = SpatialState.FIXED_WITH_EVIDENCE if self.fixed_base else SpatialState.LOCKED
        self._status_event('connected')
        self._last_heartbeat = self.clock.now()
        return self.binding

    def close(self):
        self.connected = False
        self._active = None
        self._scheduled.clear()
        self._events.clear()
        self.readiness = Readiness.UNKNOWN
        self.motion_stopped = self.output_off = None

    def _schedule(self, when, action, value):
        if len(self._scheduled) >= self.scenario.max_events:
            raise OverflowError('Simulation scheduled event queue overflow')
        self._serial += 1
        heapq.heappush(self._scheduled, (when, self._serial, action, value))

    def _feedback(self, request, sequence, phase, *, error=None):
        return DeviceFeedback(request.request_id, request.binding, request.intent, sequence, phase,
                              self.readiness, self.motion_stopped, error, self.output_off)

    def _request_faults(self, request):
        self._counts[request.intent.value] += 1
        result = []
        for index, fault in enumerate(self.scenario.faults):
            if (index not in self._used_faults and fault.at_s is None
                    and fault.device_id in (None, self.binding.device_id)
                    and fault.intent in (None, request.intent.value)
                    and self._counts[request.intent.value] == fault.occurrence):
                self._used_faults.add(index)
                result.append(fault)
        return result

    def submit(self, request: DeviceRequest, step: SimulationStep | None = None):
        if not self.connected:
            raise ConnectionError('Simulated device disconnected')
        validate_request(request, self.capabilities())
        if request.binding != self.binding:
            raise ValueError('Request targets an obsolete simulated connection')
        if request.intent is Intent.SIMULATION_STEP:
            if not isinstance(step, SimulationStep) or self._active is not None or self._paused:
                raise ValueError('Simulation step is invalid or another step is in flight')
        elif step is not None:
            raise ValueError('Step data cannot be placed inside another intent')
        self.submitted.append(request.request_id)
        self.submitted[:] = self.submitted[-256:]
        faults = self._request_faults(request)
        kinds = {fault.kind for fault in faults}
        if 'disconnect' in kinds:
            self.inject('disconnect')
            return
        if 'no_response' in kinds:
            return
        receive = self.scenario.receive_delay_s+sum(f.delay_s for f in faults if f.kind == 'receive_delay')
        complete = max(receive, self.scenario.complete_delay_s+sum(f.delay_s for f in faults if f.kind == 'complete_delay'))
        now = self.clock.now()
        self._schedule(now+receive, 'feedback', self._feedback(request, 1, FeedbackPhase.RECEIVED))
        if 'reject' in kinds or self.fault:
            self._schedule(now+complete, 'feedback', self._feedback(request, 2, FeedbackPhase.FAILED, error='scripted_rejection'))
            return
        if request.intent is Intent.SIMULATION_STEP:
            self._step_generation += 1
            self._active = (request, complete, now, self._step_generation)
            if self.binding.kind in (DeviceKind.TREATMENT_HEAD, DeviceKind.ULTRASOUND_GENERATOR):
                self.output_off = not self.scenario.step_output_active
            else:
                self._physical(False)
            self._status_event('step_started')
            self._schedule(now+complete, 'complete_step', (request, self._step_generation))
        else:
            unconfirmed = 'stop_unconfirmed' in kinds or not self.scenario.confirm_stop
            self._schedule(now+complete, 'complete_control', (request, unconfirmed))
        for fault in faults:
            if fault.kind == 'duplicate':
                self._schedule(now+complete+.01, 'duplicate', request)
            elif fault.kind == 'out_of_order':
                self._schedule(now+complete+.02, 'feedback', self._feedback(request, 1, FeedbackPhase.RECEIVED))
            elif fault.kind == 'old_epoch':
                stale = replace(request, binding=replace(request.binding, connection_epoch=request.binding.connection_epoch-1))
                self._schedule(now+complete+.03, 'feedback', self._feedback(stale, 9, FeedbackPhase.COMPLETED))
            elif fault.kind in ('fault', 'base_unlock', 'base_move', 'base_unknown', 'mount_unknown', 'reconnect'):
                self._schedule(now+fault.delay_s, 'inject', fault.kind)

    def inject(self, kind):
        if kind == 'disconnect':
            self.connected = False
            self.readiness = Readiness.UNKNOWN
            self.motion_stopped = self.output_off = None
            self._status_event('disconnected')
        elif kind == 'reconnect':
            self.connected = False
            self._active = None
            self._paused = False
            self.connect()  # Old scheduled feedback is retained for epoch rejection tests.
        elif kind in ('base_unlock', 'base_move', 'base_unknown'):
            self.spatial_state = {'base_unlock': SpatialState.UNLOCKED, 'base_move': SpatialState.MOVED,
                                  'base_unknown': SpatialState.UNKNOWN}[kind]
            self.readiness = Readiness.NOT_READY
            self._status_event(kind)
        elif kind in ('fault', 'mount_unknown'):
            self.fault = 'scripted_device_fault' if kind == 'fault' else 'installation_unknown'
            self.readiness = Readiness.NOT_READY
            self._status_event(kind)
        else:
            raise ValueError('Fault is not a device-state injection')

    def poll(self):
        now = self.clock.now()
        for index, fault in enumerate(self.scenario.faults):
            if (self._scenario_started is not None and fault.at_s is not None
                    and fault.at_s <= now-self._scenario_started and index not in self._used_faults
                    and fault.device_id in (None, self.binding.device_id)):
                self._used_faults.add(index)
                self.inject(fault.kind)
        count = 0
        while self._scheduled and self._scheduled[0][0] <= now:
            count += 1
            if count > self.scenario.max_events:
                raise OverflowError('Simulation poll critical event budget exceeded')
            _, _, action, value = heapq.heappop(self._scheduled)
            if action == 'inject':
                self.inject(value)
            elif action == 'feedback':
                self._emit(value)
            elif action == 'duplicate':
                self._emit(self._feedback(value, 2, FeedbackPhase.COMPLETED))
            elif action == 'complete_step':
                request, generation = value
                if (self._active is not None and self._active[0] == request and self._active[3] == generation
                        and not self._paused and self.connected):
                    self._active = None
                    self._physical(True)
                    self._emit(self._feedback(request, 2, FeedbackPhase.COMPLETED))
                    self._status_event('step_completed')
            elif action == 'complete_control':
                request, unconfirmed = value
                if request.binding != self.binding or not self.connected:
                    self._emit(self._feedback(request, 2, FeedbackPhase.COMPLETED))
                    continue
                if request.intent is Intent.REQUEST_PAUSE:
                    self._paused = True
                    if self._active:
                        original, duration, started, _ = self._active
                        self._step_generation += 1
                        self._active = (original, max(.01, duration-(now-started)), now, self._step_generation)
                    self._physical(True if not unconfirmed else None)
                elif request.intent is Intent.REQUEST_RESUME:
                    self._paused = False
                    if self._active:
                        original, duration, _, _ = self._active
                        self._step_generation += 1
                        self._active = (original, duration, now, self._step_generation)
                        if self.binding.kind in (DeviceKind.TREATMENT_HEAD, DeviceKind.ULTRASOUND_GENERATOR):
                            self.output_off = not self.scenario.step_output_active
                        else:
                            self._physical(False)
                        self._schedule(now+duration, 'complete_step', (original, self._step_generation))
                elif request.intent is Intent.REQUEST_STOP:
                    if not unconfirmed:
                        self._active = None
                        self._paused = False
                    self._physical(True if not unconfirmed else None)
                elif request.intent is Intent.CANCEL_REQUEST:
                    # Cancel the referenced software operation, retaining the
                    # last physical state. A separate stop confirms still/off.
                    if self._active and self._active[0].request_id == request.related_request_id:
                        original = self._active[0]
                        self._step_generation += 1
                        self._active = None
                        self._emit(self._feedback(original, 2, FeedbackPhase.CANCELLED))
                        self._emit(self._feedback(request, 2, FeedbackPhase.COMPLETED))
                    else:
                        self._emit(self._feedback(request, 2, FeedbackPhase.FAILED, error='request_not_in_flight'))
                    continue
                self._emit(self._feedback(request, 2, FeedbackPhase.COMPLETED))
                self._status_event('control_completed')
        if self.connected and now-self._last_heartbeat >= self.scenario.heartbeat_s:
            self._last_heartbeat = now
            self._status_event('heartbeat')
        events = tuple(self._events)
        self._events.clear()
        return events
