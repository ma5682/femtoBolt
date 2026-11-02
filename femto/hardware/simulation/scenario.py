"""Saved software timing and fault scripts, unrelated to vendor control periods."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
import uuid

from femto.asset_storage import publish_json, resolve_relative
from femto.hardware.interface_contract import Intent
from femto.io_utils import load_json_object
from femto.paths import hardware_configuration_root

SCHEMA = 'femto-simulation-scenario-v1'
FAULTS = frozenset({'receive_delay', 'complete_delay', 'reject', 'no_response', 'disconnect',
                   'reconnect', 'old_epoch', 'duplicate', 'out_of_order', 'fault',
                   'stop_unconfirmed', 'base_unlock', 'base_move', 'base_unknown', 'mount_unknown'})
STATE_FAULTS = frozenset({'disconnect', 'reconnect', 'fault', 'base_unlock', 'base_move',
                         'base_unknown', 'mount_unknown'})


class ManualClock:
    def __init__(self, *, domain=None):
        self.domain = domain or 'simulation:'+uuid.uuid4().hex
        self._now = 0.

    def now(self):
        return self._now

    def advance(self, seconds):
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Simulation clock must advance by finite nonnegative seconds')
        self._now += seconds


class MonotonicClock:
    def __init__(self):
        self.domain = 'simulation-monotonic:'+uuid.uuid4().hex
        self._origin = time.monotonic()

    def now(self):
        return time.monotonic()-self._origin


@dataclass(frozen=True)
class Fault:
    kind: str
    device_id: str | None = None
    intent: str | None = None
    occurrence: int = 1
    at_s: float | None = None
    delay_s: float = 1.

    def __post_init__(self):
        if (self.kind not in FAULTS or type(self.occurrence) is not int or self.occurrence < 1
                or type(self.delay_s) not in (int, float) or not math.isfinite(self.delay_s) or self.delay_s < 0):
            raise ValueError('Invalid deterministic fault')
        if self.intent is not None:
            Intent(self.intent)
        if self.at_s is not None and (type(self.at_s) not in (int, float) or not math.isfinite(self.at_s) or self.at_s < 0):
            raise ValueError('Invalid fault time')
        if self.at_s is not None and self.kind not in STATE_FAULTS:
            raise ValueError('Timed faults must be device-state events; request faults use intent and occurrence')
        if self.device_id is not None and (not isinstance(self.device_id, str) or not self.device_id):
            raise ValueError('Invalid fault device')


@dataclass(frozen=True)
class Scenario:
    scenario_id: str = 'normal'
    revision: str = '1'
    receive_delay_s: float = .03
    complete_delay_s: float = .25
    heartbeat_s: float = .1
    request_timeout_s: float = 2.
    status_max_age_s: float = .5
    shutdown_timeout_s: float = 2.
    tick_s: float = .02
    support_pause: bool = True
    confirm_stop: bool = True
    step_output_active: bool = False
    base_mode: str = 'absent'
    max_events: int = 256
    faults: tuple[Fault, ...] = ()

    def __post_init__(self):
        if not isinstance(self.scenario_id, str) or not self.scenario_id or not isinstance(self.revision, str) or not self.revision:
            raise ValueError('Scenario identity/version required')
        for name in ('receive_delay_s', 'complete_delay_s', 'heartbeat_s', 'request_timeout_s',
                     'status_max_age_s', 'shutdown_timeout_s', 'tick_s'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid simulation timing '+name)
        if (self.complete_delay_s < self.receive_delay_s or self.tick_s > self.status_max_age_s
                or self.heartbeat_s > self.status_max_age_s or self.base_mode not in ('absent', 'fixed', 'controllable')
                or type(self.support_pause) is not bool or type(self.confirm_stop) is not bool
                or type(self.step_output_active) is not bool
                or type(self.max_events) is not int or not 8 <= self.max_events <= 4096):
            raise ValueError('Invalid simulation capability, timing or bounded queue')
        values = tuple(self.faults)
        if any(not isinstance(v, Fault) for v in values) or len(values) > 100:
            raise ValueError('Invalid or unbounded fault script')
        object.__setattr__(self, 'faults', values)

    def to_dict(self):
        fields = asdict(self)
        fields['faults'] = list(fields['faults'])
        return dict(schema=SCHEMA, version=1, source='simulation', **fields,
                    timing_source='software_test_budget', execution_ready=False, physical_accuracy_verified=False)

    @classmethod
    def from_dict(cls, value):
        if (not isinstance(value, dict) or value.get('schema') != SCHEMA or type(value.get('version')) is not int
                or value['version'] != 1 or value.get('source') != 'simulation'
                or value.get('execution_ready') is not False or value.get('physical_accuracy_verified') is not False
                or value.get('timing_source') != 'software_test_budget'):
            raise ValueError('Unsupported simulation scenario schema')
        fields = {key: value[key] for key in cls.__dataclass_fields__}
        fields['faults'] = tuple(Fault(**v) for v in fields['faults'])
        result = cls(**fields)
        if result.to_dict() != value:
            raise ValueError('Unexpected simulation scenario fields')
        return result


def load_scenario(path):
    from pathlib import Path
    if Path(path).stat().st_size > 1024**2:
        raise ValueError('Simulation scenario exceeds 1 MiB')
    return Scenario.from_dict(load_json_object(path))


def save_scenario(scenario, *, outputs=None):
    root = hardware_configuration_root(outputs)/'simulation_scenarios'
    path = resolve_relative(root, scenario.scenario_id+'/'+scenario.revision+'.json')
    publish_json(path, scenario.to_dict(), create_only=True)
    return path
