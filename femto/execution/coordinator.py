"""Serial simulation coordinator with durable intent-before-dispatch ordering.

    All methods belong to one owner thread. Display snapshots contain only JSON,
    never live model/adapter aliases. Device messages, not timers, count steps.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import uuid

from femto.dataset.provenance import value_hash
from femto.execution.models import FrozenPlan, canonical, plan_summary
from femto.execution.preflight import inspect
from femto.execution.resources import ResourceLease
from femto.execution.state_machine import RunState, TERMINAL, transition
from femto.hardware.interface_contract import (
    DeviceFeedback, DeviceKind, DeviceRequest, DeviceStatus, FeedbackPhase, Intent,
    Readiness, SourceKind, SpatialState, validate_feedback, validate_status,
)


@dataclass
class Pending:
    request: DeviceRequest
    deadline_s: float
    last_sequence: int | None = None
    done: bool = False
    step_index: int | None = None


class Coordinator:
    def __init__(self, adapters, clock, scenario, store, *, outputs=None, spatial_context=None,
                 crash_hook=lambda boundary: None):
        self.adapters = tuple(adapters)
        if any(a.binding.source is not SourceKind.SIMULATION or a.binding.kind is DeviceKind.CAMERA for a in self.adapters):
            raise ValueError('Simulator accepts only owned simulation devices; no real camera or replay control')
        ids = [a.binding.device_id for a in self.adapters]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate simulated device identity')
        self.by_id = dict(zip(ids, self.adapters))
        self.clock, self.scenario, self.store = clock, scenario, store
        self.outputs, self.spatial_context, self.crash_hook = outputs, spatial_context, crash_hook
        self.run_id = None
        self.state = RunState.UNPREPARED
        self.plan = None
        self._plan_summary = None
        self.completed_steps = 0
        self.events = deque(maxlen=400)
        self.event_count = 0
        self.pending = {}
        self._statuses = {}
        self._status_sequences = {}
        self.status_revision = 0
        self._group = set()
        self._group_intent = None
        self._step_requests = set()
        self._finished_request_ids = deque(maxlen=256)
        self._bound = {}
        self._config_hashes = {}
        self._asset_signatures = {}
        self._spatial_binding = None
        self._lease = None
        self._owner_lease = None
        self._desired_terminal = None
        self._stop_deadline = None
        self._closing = False
        self._instances_closed = False
        self.persistence_gap = False
        self.persistence_error = None
        self._missing_count = 0
        self._gap_first = self._gap_last = None
        self._halted = False
        self._recovery_required = False
        self.preflight = None
        self.last_error = None
        self._request_attempts = 0

    @property
    def controllable(self):
        return tuple(a for a in self.adapters if not (a.binding.kind is DeviceKind.BASE and self.scenario.base_mode == 'fixed'))

    def connect(self):
        if self._closing:
            raise RuntimeError('旧模拟实例已关闭，不能重新连接；请建立独立实例')
        if self.run_id and self._recovery_required:
            raise RuntimeError('旧模拟实例仍需核对，请显式关闭后建立独立实例')
        if self.run_id is not None and (self.state not in TERMINAL and self.state is not RunState.UNPREPARED
                                        or self._lease and self._lease.held):
            raise RuntimeError('请先停止当前任务或关闭旧模拟实例，再重新连接')
        for adapter in self.adapters:
            adapter.connect()
            for message in adapter.poll():
                if isinstance(message, DeviceStatus):
                    self._statuses[message.binding.device_id] = message
                    self._status_sequences[message.binding.device_id] = message.sequence
                    self.status_revision += 1
        self._recovery_required = False
        self.last_error = None
        if self.run_id is not None and self.state is RunState.UNPREPARED:
            self._bound = {a.binding.device_id: a.binding for a in self.adapters}
            self._event('task_connections_bound', dict(bindings=[asdict(a.binding) for a in self.adapters]))

    def _gap(self, error):
        self.persistence_gap = True
        self.persistence_error = str(error)
        self._missing_count += 1
        self._gap_first = self._gap_first if self._gap_first is not None else self.event_count
        self._gap_last = self.event_count
        self._halted = True

    def _event(self, kind, payload=None, *, target=None, completed_steps=None):
        self.event_count += 1
        target = self.state if target is None else transition(self.state, target)
        completed = self.completed_steps if completed_steps is None else completed_steps
        body = dict(local_sequence=self.event_count, clock_domain=self.clock.domain,
                    monotonic_s=self.clock.now(), from_state=self.state.value, to_state=target.value,
                    completed_steps=completed, **(payload or {}))
        event_id = uuid.uuid4().hex
        durable = False
        if self.run_id is not None:
            try:
                self.store.append(self.run_id, event_id, kind, body, target, completed)
                durable = True
            except (OSError, ValueError, RuntimeError) as error:
                self._gap(error)
            except Exception as error:
                # Includes SQLite errors. Logging failure cannot disable stop.
                self._gap(error)
        self.state, self.completed_steps = target, completed
        self.events.append(dict(kind=kind, payload=body, persisted=durable))
        return durable

    def freeze(self, plan: FrozenPlan, *, check=lambda: None):
        check()
        if self._closing:
            raise RuntimeError('应用正在关闭，不能建立新任务')
        if self._lease and self._lease.held or self.run_id is not None and self.state not in TERMINAL:
            raise RuntimeError('旧任务仍持有资源或停止结果未知')
        if self._recovery_required or self.run_id and self.state in (RunState.UNKNOWN, RunState.INTERRUPTED):
            raise RuntimeError('旧实例状态需核对；请显式关闭旧模拟实例后建立独立实例')
        if not isinstance(plan, FrozenPlan):
            raise ValueError('Expected verified immutable simulation plan')
        asset_signatures = {a.path: a.signature() for a in plan.assets}
        plan.verify_assets(check)
        if any(a.signature() != asset_signatures[a.path] for a in plan.assets):
            raise ValueError('引用资产在冻结期间发生变化')
        self._bound = {a.binding.device_id: a.binding for a in self.adapters}
        self._config_hashes = {a.binding.device_id: value_hash(a.configuration.to_dict()) for a in self.adapters}
        self._recovery_required = False
        self.run_id = uuid.uuid4().hex
        if self._owner_lease:
            self._owner_lease.release()
        self._owner_lease = ResourceLease(['simulation:task:'+self.run_id], owner=self.run_id, outputs=self.outputs).acquire()
        self.state = RunState.UNPREPARED
        self.completed_steps = 0
        self.pending.clear(); self._group.clear(); self._step_requests.clear()
        self._group_intent = None
        self._finished_request_ids.clear()
        self.events.clear(); self.event_count = 0
        self.persistence_gap = False; self.persistence_error = None
        self._missing_count = 0; self._gap_first = self._gap_last = None
        self._halted = False; self._desired_terminal = None
        self._stop_deadline = None
        self.preflight = None
        self.last_error = None
        self._request_attempts = 0
        self.plan = plan
        self._asset_signatures = asset_signatures
        self._spatial_binding = self._spatial_signature()
        self._plan_summary = plan_summary(plan)
        snapshot = dict(schema='femto-simulation-task-v1', version=1, run_id=self.run_id,
                        source='simulation', plan=plan.to_dict(), scenario=self.scenario.to_dict(),
                        resource_keys=[a.resource_key for a in self.adapters],
                        owner_resource_key='simulation:task:'+self.run_id,
                        devices=[dict(configuration=a.configuration.to_dict(), binding=asdict(a.binding),
                                      capabilities=sorted(i.value for i in a.capabilities().supported_intents),
                                      protocol_id=a.capabilities().protocol_id) for a in self.adapters],
                        spatial_context=(dict(session_id=self.spatial_context.session_id,
                                              generation=self.spatial_context.generation,
                                              chain=self.spatial_context.base_to_w0.to_dict() if self.spatial_context.base_to_w0 else None)
                                         if self.spatial_context else None),
                        used_calibrations=([dict(id=r.calibration_id, revision=r.revision, sha256=r.digest)
                                            for r in (self.spatial_context.hand_eye, self.spatial_context.tool_mount)]
                                           if self.spatial_context else []), clock_domain=self.clock.domain,
                        execution_ready=False, physical_accuracy_verified=False)
        try:
            self.store.create(self.run_id, snapshot)
        except Exception:
            self._owner_lease.release()
            self.run_id = None
            self.plan = None
            self._plan_summary = None
            raise
        self._event('task_frozen', dict(source='simulation'))
        self.crash_hook('task_recorded')
        return self.run_id

    @property
    def can_abandon_unrun(self):
        return (self.run_id is not None and self.state is RunState.UNPREPARED
                and self.plan is not None and not self._closing and not self._halted
                and not self.persistence_gap and not self._recovery_required
                and self._request_attempts == 0 and not self.pending and not self._group
                and not self._step_requests and self.completed_steps == 0
                and not (self._lease and self._lease.held))

    def abandon_unrun(self, *, reason='操作员撤销冻结'):
        if not self.can_abandon_unrun:
            return False
        sequence = self.event_count+1
        body = dict(local_sequence=sequence, clock_domain=self.clock.domain,
                    monotonic_s=self.clock.now(), from_state=self.state.value,
                    to_state=RunState.ABANDONED.value, completed_steps=0, reason=reason,
                    no_device_request_dispatched=True, request_attempts=0,
                    real_stop_certification=False)
        event_id = uuid.uuid4().hex
        try:
            self.store.abandon_unrun(self.run_id, event_id, body, expected_sequence=self.event_count)
        except ValueError as error:
            self.last_error = str(error)
            self._halted = self._recovery_required = True
            raise
        except Exception as error:
            self._gap(error)
            self.last_error = '撤销冻结记录未保存，原任务与所有者保留：'+str(error)
            raise RuntimeError(self.last_error) from error
        self.event_count = sequence
        self.state = transition(self.state, RunState.ABANDONED)
        self.events.append(dict(kind='task_abandoned_unrun', payload=body, persisted=True))
        self._halted = True
        self.plan = None
        self._asset_signatures.clear()
        if self._owner_lease:
            self._owner_lease.release()
        return True

    def check(self, *, check=lambda: None):
        if (self.state is not RunState.UNPREPARED or self.plan is None
                or self._closing or self._halted or self.persistence_gap):
            return False
        if not self._event('preflight_started', target=RunState.CHECKING):
            self._event('preflight_failed', dict(reason='关键记录不可写'), target=RunState.FAILED)
            return False
        self.preflight = inspect(self.plan, self.adapters, clock=self.clock, scenario=self.scenario,
                                 storage_writable=not self.persistence_gap, spatial_context=self.spatial_context,
                                 refresh_status=self.poll, verify_check=check)
        if self.state is not RunState.CHECKING:
            return False
        if not self._event('preflight_result', self.preflight):
            self._event('preflight_failed', dict(reason='检查结果记录失败'), target=RunState.FAILED)
            return False
        if not self.preflight['simulation_ready']:
            self._event('preflight_failed', dict(reason='模拟前置条件未满足'), target=RunState.FAILED)
            return False
        try:
            self.poll()  # Refresh again after writing the preflight result.
            if self.state is not RunState.CHECKING or self._halted or self.persistence_gap:
                return False
            self._dependencies()
            self._lease = ResourceLease([a.resource_key for a in self.adapters], owner=self.run_id, outputs=self.outputs).acquire()
        except (RuntimeError, ValueError) as error:
            self._event('resource_busy', dict(reason=str(error)), target=RunState.FAILED)
            return False
        self._start_group(Intent.PREPARE)
        return True

    def _request(self, adapter, intent, *, step=None, safety=False):
        if not safety and (self._halted or self.persistence_gap or self._closing):
            return None
        self._request_attempts += 1
        request = DeviceRequest(uuid.uuid4().hex, adapter.binding, intent)
        pending = Pending(request, self.clock.now()+self.scenario.request_timeout_s,
                          step_index=step.index if step else None)
        self.pending[request.request_id] = pending
        durable = self._event('request_intent', dict(request=asdict(request), step=asdict(step) if step else None))
        self.crash_hook('intent_recorded_before_dispatch')
        if not durable and not safety:
            pending.done = True
            return None
        try:
            pending.deadline_s = self.clock.now()+self.scenario.request_timeout_s
            adapter.submit(request, step)
            self.crash_hook('dispatched_before_confirmation_record')
            self._event('request_sent', dict(request_id=request.request_id, binding=asdict(request.binding)))
        except Exception as error:
            pending.done = True
            self._event('request_send_failed', dict(request_id=request.request_id, reason=str(error)))
            self.last_error = str(error)
            return None
        return request.request_id

    def _start_group(self, intent):
        self._group = set()
        self._group_intent = intent
        for adapter in self.controllable:
            request_id = self._request(adapter, intent, safety=intent is Intent.REQUEST_STOP)
            if request_id:
                self._group.add(request_id)
            if intent is not Intent.REQUEST_STOP and (request_id is None or self._halted):
                break
        if (len(self._group) != len(self.controllable)
                or intent is not Intent.REQUEST_STOP and self._halted):
            if intent is Intent.REQUEST_STOP:
                self._unknown('部分停止请求无法发送')
            else:
                self.stop(reason='请求发送或持久化失败', outcome=RunState.FAILED)

    def _status_fresh(self, status, at_s):
        return (status is not None and status.clock_domain == self.clock.domain
                and 0 <= at_s-status.timestamp_s <= self.scenario.status_max_age_s)

    def _dependencies(self, *, verify_content=False, status_time_s=None):
        if verify_content:
            self.plan.verify_assets()
        status_time_s = self.clock.now() if status_time_s is None else status_time_s
        for asset in self.plan.assets:
            if asset.signature() != self._asset_signatures[asset.path]:
                raise ValueError('当前冻结任务引用的资产已改变或缺失：'+asset.path)
        for adapter in self.adapters:
            device = adapter.binding.device_id
            if adapter.binding != self._bound[device] or not adapter.connected:
                raise ValueError('设备断线或连接代次改变')
            if value_hash(adapter.configuration.to_dict()) != self._config_hashes[device]:
                raise ValueError('当前设备配置被替换')
            status = self._statuses.get(device)
            if status is None or status.fault or status.readiness is not Readiness.READY:
                raise ValueError('设备报告故障或状态不就绪')
            if not self._status_fresh(status, status_time_s):
                raise ValueError('设备状态已过期或时钟域不一致')
            if self.state in (RunState.READY, RunState.PAUSED) and self._physical_unconfirmed(status):
                raise ValueError('已确认的机械静止/输出关闭状态已失效')
            if adapter.binding.kind is DeviceKind.BASE and status.spatial_state not in (SpatialState.LOCKED, SpatialState.FIXED_WITH_EVIDENCE):
                raise ValueError('底座解锁、移动或空间状态未知')
        if self.spatial_context and self.spatial_context.invalid_reason:
            raise ValueError(self.spatial_context.invalid_reason)
        if self._spatial_signature() != self._spatial_binding:
            raise ValueError('当前任务的空间参考或安装标定版本已改变')

    def _spatial_signature(self):
        context = self.spatial_context
        return (context.session_id, context.generation, context.hand_eye.digest, context.tool_mount.digest,
                value_hash(context.base_to_w0.to_dict()) if context.base_to_w0 else None) if context else None

    def start(self, *, check=lambda: None):
        if self.state is not RunState.READY or self._closing or self.persistence_gap or self._halted:
            return False
        try:
            self.plan.verify_assets(check)
            self.poll()  # Receive fresh status after potentially slow file verification.
            if self.state is not RunState.READY or self.persistence_gap or self._halted:
                return False
            self._dependencies()
        except (OSError, ValueError) as error:
            self.stop(reason=str(error), outcome=RunState.FAILED)
            return False
        if not self._event('operator_start', target=RunState.RUNNING):
            self.stop(reason='开始记录失败', outcome=RunState.FAILED)
            return False
        self.poll()  # Operator-intent persistence may also consume the status age.
        return True

    def _next_step(self):
        if self.state is not RunState.RUNNING or self._halted or self._step_requests:
            return
        if self.completed_steps == len(self.plan.steps):
            self.stop(reason='全部模拟步骤已确认，核对结束状态', outcome=RunState.SUCCEEDED)
            return
        try:
            self._dependencies()
        except (OSError, ValueError) as error:
            self.stop(reason=str(error), outcome=RunState.FAILED)
            return
        step = self.plan.steps[self.completed_steps]
        for adapter in self.controllable:
            request_id = self._request(adapter, Intent.SIMULATION_STEP, step=step)
            if request_id:
                self._step_requests.add(request_id)
            if request_id is None or self._halted:
                break
        if len(self._step_requests) != len(self.controllable) or self._halted:
            self.stop(reason='模拟步骤未全部发送', outcome=RunState.FAILED)

    def pause(self):
        if (self.state is not RunState.RUNNING or self._halted or self.persistence_gap
                or self._closing or not self.supports(Intent.REQUEST_PAUSE)):
            return False
        if not self._event('operator_pause', target=RunState.PAUSE_REQUESTED):
            self.stop(reason='暂停意图记录失败', outcome=RunState.FAILED)
            return False
        self._start_group(Intent.REQUEST_PAUSE)
        return True

    def resume(self, *, check=lambda: None):
        if (self.state is not RunState.PAUSED or self.persistence_gap or self._halted
                or self._closing or not self.supports(Intent.REQUEST_RESUME)):
            return False
        try:
            self.plan.verify_assets(check)
            self.poll()
            if self.state is not RunState.PAUSED or self.persistence_gap or self._halted:
                return False
            self._dependencies()
        except (OSError, ValueError) as error:
            self.stop(reason=str(error), outcome=RunState.FAILED)
            return False
        if not self._event('operator_resume', target=RunState.RESUME_REQUESTED):
            self.stop(reason='继续意图记录失败', outcome=RunState.FAILED)
            return False
        self._start_group(Intent.REQUEST_RESUME)
        return True

    def supports(self, intent):
        return bool(self.controllable) and all(intent in a.capabilities().supported_intents for a in self.controllable)

    def _confirmed_still(self, feedback):
        if feedback.binding.kind in (DeviceKind.ARM, DeviceKind.BASE):
            return feedback.motion_stopped is True
        if feedback.binding.kind in (DeviceKind.TREATMENT_HEAD, DeviceKind.ULTRASOUND_GENERATOR):
            return feedback.output_off is True
        return False

    def _physical_unconfirmed(self, message):
        if message.binding.kind is DeviceKind.BASE and self.scenario.base_mode == 'fixed':
            return False  # Structural participant; no motion interface or stop claim.
        return not self._confirmed_still(message)

    def _stop_confirmation_changed(self, message):
        return (self.state is RunState.STOP_REQUESTED and self._physical_unconfirmed(message)
                and any(self.pending[request_id].done
                        and self.pending[request_id].request.binding == message.binding
                        for request_id in self._group))

    def stop(self, *, reason='操作员请求停止', outcome=RunState.STOPPED):
        if self.state in TERMINAL:
            return False
        if self.state is RunState.STOP_REQUESTED:
            # A new failure must downgrade the in-flight stop, without sending
            # duplicate requests or discarding confirmations already received.
            if outcome is RunState.FAILED and self._desired_terminal is not RunState.FAILED:
                self._desired_terminal = RunState.FAILED
                self.last_error = reason
                self._event('stop_outcome_invalidated', dict(reason=reason, requested_outcome=outcome.value))
            return False
        if self.state is RunState.UNPREPARED:
            self._event('operator_cancel_unprepared', dict(reason=reason), target=RunState.FAILED)
            return True
        self._desired_terminal = outcome
        self._stop_deadline = self.clock.now()+self.scenario.request_timeout_s
        self._event('operator_stop', dict(reason=reason, requested_outcome=outcome.value), target=RunState.STOP_REQUESTED)
        self._halted = True
        self._start_group(Intent.REQUEST_STOP)
        return True

    def _unknown(self, reason):
        if self.state is RunState.UNKNOWN:
            return
        self.last_error = reason
        self._event('stop_result_unknown', dict(reason=reason, resource_release=False), target=RunState.UNKNOWN)
        self._halted = True
        self._recovery_required = True
        self._retain_resources()
        # Lease retained until the old simulated instance is explicitly closed.

    def _retain_resources(self):
        if self.run_id is None or self._lease and self._lease.held:
            return
        try:
            self._lease = ResourceLease([a.resource_key for a in self.adapters], owner=self.run_id, outputs=self.outputs).acquire()
        except RuntimeError as error:
            self.last_error = str(error)
            self._event('resource_reacquire_failed', dict(reason=str(error), old_result_unknown=True))

    def _status(self, status):
        adapter = self.by_id.get(status.binding.device_id)
        try:
            if adapter is None:
                raise ValueError('Unknown device')
            validate_status(status, current_binding=adapter.binding,
                            last_sequence=self._status_sequences.get(status.binding.device_id)
                            if self._statuses.get(status.binding.device_id, status).binding == adapter.binding else None)
        except ValueError as error:
            self._event('status_rejected', dict(reason=str(error), status=asdict(status)))
            return
        self._status_sequences[status.binding.device_id] = status.sequence
        self._statuses[status.binding.device_id] = status
        self.status_revision += 1
        meaningful = status.event_type != 'heartbeat'
        if meaningful:
            self._event('device_status', dict(status=asdict(status)))
        invalid = (not status.connected or status.fault or status.readiness is not Readiness.READY
                   or self.run_id is not None and self._bound.get(status.binding.device_id) != status.binding)
        base_invalid = status.binding.kind is DeviceKind.BASE and status.spatial_state not in (SpatialState.LOCKED, SpatialState.FIXED_WITH_EVIDENCE)
        if self.spatial_context and (base_invalid or status.event_type == 'mount_unknown'):
            self.spatial_context.invalidate(status.event_type, mount=status.event_type == 'mount_unknown')
        if self.run_id is None:
            return
        if self._stop_confirmation_changed(status):
            self._unknown('等待其他设备停止时，已确认设备的新状态未证明静止/输出关闭')
            return
        if invalid or base_invalid:
            self._recovery_required = True
            if self.state not in TERMINAL:
                self.stop(reason=status.fault or status.event_type, outcome=RunState.FAILED)
            elif self.state in (RunState.SUCCEEDED, RunState.STOPPED, RunState.CANCELLED):
                self._event('late_device_alarm', dict(status=asdict(status)), target=RunState.FAILED)
                if self._physical_unconfirmed(status):
                    self._unknown('结束后设备故障或断线，停止/输出关闭状态未知')
            elif self.state is RunState.FAILED and self._physical_unconfirmed(status):
                self._unknown('故障任务的当前停止/输出关闭状态未知')
        elif (self.state in (RunState.SUCCEEDED, RunState.STOPPED, RunState.CANCELLED, RunState.FAILED)
              and self._physical_unconfirmed(status)):
            # Do not discard a current-epoch contradiction just because task ended.
            self._recovery_required = True
            self._event('late_physical_state_alarm', dict(status=asdict(status)), target=RunState.UNKNOWN)
            self._halted = True
            self._retain_resources()

    def _feedback_alarm(self, feedback, reason, *, fault=True):
        self._recovery_required = True
        self._halted = True
        self.last_error = reason
        if self.state in (RunState.SUCCEEDED, RunState.STOPPED, RunState.CANCELLED):
            self._event('late_feedback_alarm', dict(feedback=asdict(feedback), reason=reason),
                        target=RunState.FAILED if fault else RunState.UNKNOWN)
            if not self._confirmed_still(feedback):
                self._unknown('结束后的矛盾反馈未证明静止/输出关闭')
                self._retain_resources()
        elif self.state is RunState.FAILED and not self._confirmed_still(feedback):
            self._unknown('故障任务的新反馈未证明静止/输出关闭')
        elif self.state is RunState.STOP_REQUESTED:
            # An alarm can coexist with a genuine physical-stop confirmation.
            # Keep collecting all participants, but never call this success.
            self._desired_terminal = RunState.FAILED
            self._event('device_feedback_alarm', dict(feedback=asdict(feedback), reason=reason))
        elif self.state not in TERMINAL:
            self.stop(reason=reason, outcome=RunState.FAILED)

    def _feedback(self, feedback):
        pending = self.pending.get(feedback.request_id)
        adapter = self.by_id.get(feedback.binding.device_id)
        identity_valid = False
        try:
            if pending is None or adapter is None:
                raise ValueError('Unknown or expired request')
            validate_feedback(pending.request, feedback, current_binding=adapter.binding,
                              last_sequence=pending.last_sequence)
            if self._bound.get(feedback.binding.device_id) != feedback.binding:
                raise ValueError('Feedback is from another task connection')
            identity_valid = True
            if pending.done:
                pending.last_sequence = feedback.sequence
                raise ValueError('Request already ended')
        except ValueError as error:
            self._event('feedback_rejected', dict(reason=str(error), feedback=asdict(feedback)))
            # A new fault on an otherwise valid current binding is still an alarm.
            fault = bool(feedback.error_code or feedback.readiness is Readiness.NOT_READY)
            contradiction = (self.state in (RunState.SUCCEEDED, RunState.STOPPED, RunState.CANCELLED, RunState.FAILED)
                             and self._physical_unconfirmed(feedback))
            if identity_valid and (fault or contradiction):
                self._feedback_alarm(feedback, feedback.error_code or '设备反馈不就绪或结束物理状态矛盾', fault=fault)
            if identity_valid and self._stop_confirmation_changed(feedback):
                self._unknown('等待其他设备停止时，新的请求反馈撤销了静止/输出关闭确认')
            return
        pending.last_sequence = feedback.sequence
        self.crash_hook('feedback_received_before_record')
        durable = self._event('device_feedback', dict(feedback=asdict(feedback),
                                                     sequence_scope='request'))
        self.crash_hook('feedback_recorded')
        if (feedback.error_code or feedback.readiness is Readiness.NOT_READY
                or feedback.phase is FeedbackPhase.COMPLETED and feedback.intent is not Intent.REQUEST_STOP
                and feedback.readiness is not Readiness.READY):
            self._feedback_alarm(feedback, feedback.error_code or '设备完成反馈未证明就绪')
        if feedback.phase is FeedbackPhase.RECEIVED:
            return
        pending.done = True
        self._finished_request_ids.append(feedback.request_id)
        if feedback.phase is not FeedbackPhase.COMPLETED:
            if self.state is RunState.STOP_REQUESTED and feedback.request_id in self._group:
                self._unknown('停止被拒绝或取消，未证明设备静止和输出关闭')
            elif self.state not in TERMINAL and self.state is not RunState.STOP_REQUESTED:
                self.stop(reason=feedback.error_code or feedback.phase.value, outcome=RunState.FAILED)
            return
        if feedback.request_id in self._group:
            if self._group_intent in (Intent.REQUEST_STOP, Intent.REQUEST_PAUSE) and not self._confirmed_still(feedback):
                if self._group_intent is Intent.REQUEST_STOP:
                    self._unknown('收到停止完成反馈，但物理静止/模拟输出关闭仍未知')
                else:
                    self.stop(reason='暂停未确认物理状态', outcome=RunState.FAILED)
                return
        if not durable and self.state is not RunState.STOP_REQUESTED and self.state not in TERMINAL:
            self.stop(reason='反馈记录失败', outcome=RunState.FAILED)

    def _finish_groups(self):
        if self._group and all(self.pending[i].done for i in self._group):
            intent = self._group_intent
            self._group.clear(); self._group_intent = None
            if self.state in TERMINAL:
                return
            if intent is Intent.PREPARE and self.state is RunState.CHECKING:
                self._event('prepare_confirmed', target=RunState.READY)
            elif intent is Intent.REQUEST_PAUSE and self.state is RunState.PAUSE_REQUESTED:
                for request_id in self._step_requests:
                    self.pending[request_id].deadline_s = float('inf')
                self._event('pause_all_confirmed', target=RunState.PAUSED)
            elif intent is Intent.REQUEST_RESUME and self.state is RunState.RESUME_REQUESTED:
                for request_id in self._step_requests:
                    self.pending[request_id].deadline_s = self.clock.now()+self.scenario.request_timeout_s
                self._event('resume_all_confirmed', target=RunState.RUNNING)
            elif intent is Intent.REQUEST_STOP and self.state is RunState.STOP_REQUESTED:
                outcome = self._desired_terminal or RunState.STOPPED
                if outcome is RunState.SUCCEEDED and self.completed_steps != len(self.plan.steps):
                    self.last_error = '模拟步骤尚未全部确认，不能记录成功'
                    self._event('completion_requirements_failed', dict(reason=self.last_error))
                    outcome = RunState.FAILED
                if self.persistence_gap or self._recovery_required:
                    outcome = RunState.FAILED
                durable = self._event('stop_all_confirmed', dict(physical_confirmed=True), target=outcome)
                if not durable and outcome is not RunState.FAILED:
                    self._event('completion_not_persisted', dict(reason='完成状态未落盘'), target=RunState.FAILED)
                if self._lease:
                    self._lease.release()
        if self._step_requests and all(self.pending[i].done for i in self._step_requests):
            self._step_requests.clear()
            if self.state in (RunState.RUNNING, RunState.PAUSE_REQUESTED, RunState.PAUSED, RunState.RESUME_REQUESTED) and not self.persistence_gap:
                self._event('simulation_step_confirmed', dict(step=asdict(self.plan.steps[self.completed_steps])),
                            completed_steps=self.completed_steps+1)
        # Bound memory without deleting any durable event. Keep recent completed
        # requests so late contradictory feedback can still be diagnosed.
        retained = set(self._finished_request_ids) | self._group | self._step_requests
        self.pending = {key: value for key, value in self.pending.items() if not value.done or key in retained}

    def poll(self):
        batch_started = self.clock.now()
        messages = []
        try:
            for adapter in self.adapters:
                for message in adapter.poll():
                    if not isinstance(message, (DeviceStatus, DeviceFeedback)):
                        raise ValueError('Adapter returned an unsupported device event')
                    messages.append(message)
                    if len(messages) > self.scenario.max_events:
                        raise OverflowError('Coordinator critical event queue exceeded')
        except Exception as error:
            # Transport/decoding errors are event gaps too. Stop confirmation
            # remains bounded even when an adapter fails during final stopping.
            self.critical_queue_gap(str(error))
            if self.state is RunState.STOP_REQUESTED and self.clock.now() >= self._stop_deadline:
                self._unknown('停止确认超时；包括事件队列缺口，结果保持未知')
            return
        status_observed_at = self.clock.now()
        for message in messages:
            if isinstance(message, DeviceStatus):
                self._status(message)
            else:
                self._feedback(message)
        if self.run_id is None:
            return
        self._finish_groups()
        if self.state not in TERMINAL:
            if self.state is RunState.STOP_REQUESTED and batch_started >= self._stop_deadline:
                self._unknown('停止确认超时；请求收到不代表已停止')
            else:
                timed_out = [p for key, p in self.pending.items() if not p.done and batch_started >= p.deadline_s
                             and (self.state is not RunState.STOP_REQUESTED or key in self._group)]
                if timed_out:
                    self._event('request_timeout', dict(request_ids=[p.request.request_id for p in timed_out]))
                    if self.state is RunState.STOP_REQUESTED:
                        self._unknown('停止反馈超时')
                    else:
                        self.stop(reason='设备请求超时', outcome=RunState.FAILED)
                elif self.state in (RunState.RUNNING, RunState.READY, RunState.PAUSED):
                    try:
                        # Audit freshness at receipt, before our own event
                        # writes consume the remaining heartbeat age.
                        self._dependencies(status_time_s=status_observed_at)
                    except (OSError, ValueError) as error:
                        self.stop(reason=str(error), outcome=RunState.FAILED)
        if self.persistence_gap and self.state not in TERMINAL and self.state is not RunState.STOP_REQUESTED:
            self.stop(reason='事件持久化缺口', outcome=RunState.FAILED)
        # Persistence may take longer than the status-age budget. Do not send
        # a new step using that old batch, or declare the unpolled simulator
        # disconnected. The next bounded poll receives actual status events;
        # an adapter that stops emitting heartbeats still fails freshness.
        dispatch_time = self.clock.now()
        if (dispatch_time-status_observed_at < self.scenario.status_max_age_s
                and all(self._status_fresh(self._statuses.get(a.binding.device_id), dispatch_time)
                        for a in self.adapters)):
            self._next_step()

    def recover_persistence(self):
        if not self.persistence_gap or self.run_id is None:
            return False
        payload = dict(reason=self.persistence_error, missing_count=self._missing_count,
                       local_sequence_start=self._gap_first, local_sequence_end=self._gap_last,
                       unreceived_device_state_not_reconstructed=True, requires_new_task=True)
        try:
            self.store.append(self.run_id, uuid.uuid4().hex, 'persistence_gap_recovered', payload,
                              self.state, self.completed_steps)
        except Exception as error:
            self.persistence_error = str(error)
            return False
        self.persistence_gap = False
        self.persistence_error = None
        self._missing_count = 0
        self._gap_first = self._gap_last = None
        return True

    def snapshot(self):
        return canonical(dict(source='simulation', run_id=self.run_id, state=self.state.value,
                              completed_steps=self.completed_steps,
                              total_steps=(self._plan_summary or {}).get('software_steps', 0),
                              progress_semantics='confirmed_software_steps_only', persistence_gap=self.persistence_gap,
                              plan_summary=self._plan_summary,
                              persistence_error=self.persistence_error, last_error=self.last_error,
                              preflight=self.preflight, events=list(self.events),
                              devices=[dict(configuration=a.configuration.to_dict(), status=asdict(self._statuses.get(a.binding.device_id, a.status())),
                                            capabilities=sorted(i.value for i in a.capabilities().supported_intents),
                                            motion_applicable=a.binding.kind is DeviceKind.ARM or a.binding.kind is DeviceKind.BASE and a in self.controllable,
                                            output_applicable=a.binding.kind in (DeviceKind.TREATMENT_HEAD, DeviceKind.ULTRASOUND_GENERATOR),
                                            protocol_id=a.capabilities().protocol_id) for a in self.adapters],
                              can_start=self.state is RunState.READY and not self.persistence_gap and not self._closing and not self._halted,
                              can_pause=self.state is RunState.RUNNING and not self.persistence_gap and not self._halted and not self._closing and self.supports(Intent.REQUEST_PAUSE),
                              can_resume=self.state is RunState.PAUSED and not self.persistence_gap and not self._halted and not self._closing and self.supports(Intent.REQUEST_RESUME),
                              can_stop=self.state not in TERMINAL and self.state not in (RunState.UNPREPARED, RunState.STOP_REQUESTED),
                              can_abandon_unrun=self.can_abandon_unrun,
                              resource_held=bool(self._lease and self._lease.held),
                              execution_ready=False, physical_accuracy_verified=False))

    def request_shutdown(self):
        if self.can_abandon_unrun:
            self.abandon_unrun(reason='应用关闭，冻结任务尚未下发请求')
        self._closing = True
        if self.run_id is not None and self.state not in TERMINAL:
            self.stop(reason='应用关闭', outcome=RunState.CANCELLED)

    def shutdown_expired(self):
        if self.run_id is not None and self.state not in TERMINAL:
            self._unknown('应用退出等待期限结束；停止结果未知')

    def critical_queue_gap(self, reason):
        if self.run_id is None:
            self.last_error = reason
            return
        self._event('critical_queue_gap', dict(reason=reason, missing_event_count_unknown=True))
        self._halted = True
        self._recovery_required = True
        if self.state not in TERMINAL:
            self.stop(reason=reason, outcome=RunState.FAILED)
        elif self.state not in (RunState.UNKNOWN, RunState.ABANDONED):
            self._unknown('关键事件缺口；结束状态无法继续确认：'+reason)

    def close_instances(self):
        """Explicitly destroy simulation only; never certify a real stop."""
        if self.can_abandon_unrun:
            self.abandon_unrun(reason='关闭模拟实例，冻结任务尚未下发请求')
        self._closing = True
        if self._instances_closed:
            return
        if self.run_id is not None and self.state not in TERMINAL:
            self.stop(reason='关闭模拟实例', outcome=RunState.CANCELLED)
            self._unknown('模拟实例关闭前未获得停止确认；旧结果保持未知')
        errors = []
        for adapter in self.adapters:
            try:
                adapter.close()
            except Exception as error:
                errors.append(adapter.binding.device_id+': '+str(error))
        if errors:
            self.last_error = '模拟实例关闭失败：'+'; '.join(errors)
            self._event('simulation_instances_close_failed', dict(reason=self.last_error, resource_release=False))
            self._halted = self._recovery_required = True
            if self.run_id is not None and self.state is not RunState.ABANDONED:
                self._unknown(self.last_error)
            elif self.run_id is not None:
                self._retain_resources()
            raise RuntimeError(self.last_error)
        if self.run_id is not None:
            self._event('simulation_instances_closed', dict(old_result_preserved=True, real_stop_certification=False))
        if self._lease:
            self._lease.release()
        if self._owner_lease:
            self._owner_lease.release()
        self._halted = True
        self._instances_closed = True
