"""Transport-free device messages; validation never grants execution authority.

These records do not open a device, perform motion, emit energy, or admit frames.
Vendor protocols must define operation semantics before an adapter is connected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DeviceKind(str, Enum):
    CAMERA = 'camera'
    ARM = 'arm'
    BASE = 'base'
    TREATMENT_HEAD = 'treatment_head'
    ULTRASOUND_GENERATOR = 'ultrasound_generator'


class SourceKind(str, Enum):
    REAL = 'real'
    REPLAY = 'replay'
    SIMULATION = 'simulation'


class Intent(str, Enum):
    QUERY_STATUS = 'query_status'
    QUERY_CAPABILITIES = 'query_capabilities'
    PREPARE = 'prepare'
    CANCEL_REQUEST = 'cancel_request'
    REQUEST_STOP = 'request_stop'
    REQUEST_PAUSE = 'request_pause'
    REQUEST_RESUME = 'request_resume'
    SIMULATION_STEP = 'simulation_step'


class FeedbackPhase(str, Enum):
    RECEIVED = 'received'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class Readiness(str, Enum):
    UNKNOWN = 'unknown'
    NOT_READY = 'not_ready'
    READY = 'ready'


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ' must be a nonempty string')


def _natural(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + ' must be a nonnegative integer')


@dataclass(frozen=True)
class DeviceBinding:
    device_id: str
    kind: DeviceKind
    connection_epoch: int
    source: SourceKind

    def __post_init__(self):
        _text(self.device_id, 'device_id')
        _natural(self.connection_epoch, 'connection_epoch')
        if not isinstance(self.kind, DeviceKind) or not isinstance(self.source, SourceKind):
            raise ValueError('Device kind and source must be explicit contract values')


@dataclass(frozen=True)
class Capabilities:
    binding: DeviceBinding
    protocol_id: str | None = None
    supported_intents: frozenset[Intent] = field(default_factory=frozenset)

    def __post_init__(self):
        if not isinstance(self.binding, DeviceBinding):
            raise ValueError('Capabilities require a device binding')
        if self.protocol_id is not None:
            _text(self.protocol_id, 'protocol_id')
        intents = frozenset(self.supported_intents)
        if any(not isinstance(intent, Intent) for intent in intents):
            raise ValueError('Unsupported capability intent')
        object.__setattr__(self, 'supported_intents', intents)


@dataclass(frozen=True)
class DeviceRequest:
    request_id: str
    binding: DeviceBinding
    intent: Intent
    related_request_id: str | None = None

    def __post_init__(self):
        _text(self.request_id, 'request_id')
        if not isinstance(self.binding, DeviceBinding) or not isinstance(self.intent, Intent):
            raise ValueError('Request requires a device binding and known intent')
        if self.related_request_id is not None:
            _text(self.related_request_id, 'related_request_id')
            if self.related_request_id == self.request_id:
                raise ValueError('A request cannot refer to itself')
        if self.intent is Intent.CANCEL_REQUEST and self.related_request_id is None:
            raise ValueError('Cancellation must identify the request being cancelled')


@dataclass(frozen=True)
class DeviceFeedback:
    request_id: str
    binding: DeviceBinding
    intent: Intent
    sequence: int
    phase: FeedbackPhase
    readiness: Readiness = Readiness.UNKNOWN
    motion_stopped: bool | None = None
    error_code: str | None = None
    output_off: bool | None = None

    def __post_init__(self):
        _text(self.request_id, 'request_id')
        _natural(self.sequence, 'sequence')
        if (not isinstance(self.binding, DeviceBinding) or not isinstance(self.intent, Intent)
                or not isinstance(self.phase, FeedbackPhase) or not isinstance(self.readiness, Readiness)):
            raise ValueError('Feedback requires explicit binding, intent, phase and readiness')
        if self.motion_stopped is not None and type(self.motion_stopped) is not bool:
            raise ValueError('Unknown motion state must remain None')
        if self.error_code is not None:
            _text(self.error_code, 'error_code')
        if self.output_off is not None and type(self.output_off) is not bool:
            raise ValueError('Unknown output state must remain None')

    @property
    def request_completed(self):
        """The provider reports completion; this does not verify physical outcome."""
        return self.phase is FeedbackPhase.COMPLETED


def validate_request(request: DeviceRequest, capabilities: Capabilities | None = None):
    """Check message suitability only. No transport or execution follows."""
    if not isinstance(request, DeviceRequest):
        raise ValueError('Expected DeviceRequest')
    if capabilities is not None:
        if not isinstance(capabilities, Capabilities) or capabilities.binding != request.binding:
            raise ValueError('Capabilities belong to another device connection or source')
    if request.intent in (Intent.QUERY_STATUS, Intent.QUERY_CAPABILITIES):
        return request
    if request.intent is Intent.SIMULATION_STEP and (
            request.binding.source is not SourceKind.SIMULATION
            or capabilities is None or capabilities.protocol_id != 'femto-simulation-v1'):
        raise ValueError('Simulation steps require the explicit software simulation protocol')
    if (capabilities is None or capabilities.protocol_id is None
            or request.intent not in capabilities.supported_intents):
        raise ValueError('This intent requires an identified protocol and declared capability')
    return request


def validate_feedback(request: DeviceRequest, feedback: DeviceFeedback, *,
                      current_binding: DeviceBinding, last_sequence: int | None = None):
    """Reject stale or mismatched messages without changing capture/device state."""
    if not isinstance(request, DeviceRequest) or not isinstance(feedback, DeviceFeedback):
        raise ValueError('Expected request and feedback contract records')
    if (request.binding != current_binding or feedback.binding != current_binding
            or request.request_id != feedback.request_id or request.intent is not feedback.intent):
        raise ValueError('Feedback device, connection, source or request does not match')
    if last_sequence is not None:
        _natural(last_sequence, 'last_sequence')
        if feedback.sequence <= last_sequence:
            raise ValueError('Feedback is duplicate or out of order')
    return feedback


class SpatialState(str, Enum):
    UNKNOWN = 'unknown'
    LOCKED = 'locked'
    FIXED_WITH_EVIDENCE = 'fixed_with_evidence'
    UNLOCKED = 'unlocked'
    MOVED = 'moved'
    NOT_APPLICABLE = 'not_applicable'


@dataclass(frozen=True)
class DeviceStatus:
    """Continuous status sequence is scoped to one binding, not one request.

    Times can only be compared in the same clock domain. A source label alone
    is not an authenticated connection; the adapter owner constructs bindings.
    """
    binding: DeviceBinding
    sequence: int
    clock_domain: str
    timestamp_s: float
    connected: bool
    readiness: Readiness = Readiness.UNKNOWN
    motion_stopped: bool | None = None
    output_off: bool | None = None
    spatial_state: SpatialState = SpatialState.NOT_APPLICABLE
    fault: str | None = None
    event_type: str = 'status'
    parameter_revision: str | None = None

    def __post_init__(self):
        import math
        if not isinstance(self.binding, DeviceBinding):
            raise ValueError('Status requires a device binding')
        _natural(self.sequence, 'status.sequence')
        _text(self.clock_domain, 'clock_domain')
        if (type(self.timestamp_s) not in (int, float) or not math.isfinite(self.timestamp_s)
                or self.timestamp_s < 0 or type(self.connected) is not bool):
            raise ValueError('Invalid status timestamp or connection state')
        if not isinstance(self.readiness, Readiness) or not isinstance(self.spatial_state, SpatialState):
            raise ValueError('Status requires explicit readiness and spatial state')
        for value in (self.motion_stopped, self.output_off):
            if value is not None and type(value) is not bool:
                raise ValueError('Unknown physical states must remain None')
        _text(self.event_type, 'event_type')
        for name in ('fault', 'parameter_revision'):
            if getattr(self, name) is not None:
                _text(getattr(self, name), name)


def validate_status(status: DeviceStatus, *, current_binding: DeviceBinding,
                    last_sequence: int | None = None):
    if not isinstance(status, DeviceStatus) or status.binding != current_binding:
        raise ValueError('Status belongs to another device, source or connection epoch')
    if last_sequence is not None:
        _natural(last_sequence, 'last_status_sequence')
        if status.sequence <= last_sequence:
            raise ValueError('Status is duplicate or out of order')
    return status


@dataclass(frozen=True)
class SimulationStep:
    """Versioned software step reference; contains no motion or energy payload."""
    index: int
    reference_sha256: str
    role: str
    reason: str | None = None
    schema: str = 'femto-simulation-step-v1'

    def __post_init__(self):
        _natural(self.index, 'step.index')
        if (self.schema != 'femto-simulation-step-v1' or self.role not in ('scan', 'connector', 'gap')
                or not isinstance(self.reference_sha256, str) or len(self.reference_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.reference_sha256)):
            raise ValueError('Invalid software simulation step contract')
        if self.role == 'gap':
            _text(self.reason, 'gap.reason')
