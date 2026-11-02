"""Versioned static configuration. Loading never connects or grants capability."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from femto.asset_storage import publish_json, resolve_relative
from femto.hardware.interface_contract import DeviceKind, SourceKind
from femto.io_utils import load_json_object
from femto.paths import hardware_configuration_root

SCHEMA = 'femto-device-configuration-v1'


@dataclass(frozen=True)
class DeviceConfiguration:
    device_id: str
    kind: DeviceKind
    adapter: str
    resource_id: str
    revision: str
    model: str | None = None
    connection_json: str = '{}'

    def __post_init__(self):
        for name in ('device_id', 'resource_id', 'revision'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError('Invalid device configuration '+name)
        if not isinstance(self.kind, DeviceKind) or self.adapter not in ('unconfigured', 'simulation', 'real', 'replay'):
            raise ValueError('Unknown device kind or adapter')
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError('Invalid model description')
        value = json.loads(self.connection_json)
        if not isinstance(value, dict) or any(k in value for k in ('capabilities', 'readiness', 'connected', 'source')):
            raise ValueError('Connection configuration cannot assign capability or runtime state')
        object.__setattr__(self, 'connection_json', json.dumps(value, sort_keys=True, allow_nan=False))

    @property
    def source(self):
        return {'simulation': SourceKind.SIMULATION, 'real': SourceKind.REAL,
                'replay': SourceKind.REPLAY}.get(self.adapter)

    def to_dict(self):
        return dict(schema=SCHEMA, version=1, device_id=self.device_id, kind=self.kind.value,
                    adapter=self.adapter, resource_id=self.resource_id, revision=self.revision,
                    model=self.model, connection=json.loads(self.connection_json),
                    execution_ready=False, physical_accuracy_verified=False)

    @classmethod
    def from_dict(cls, value):
        fields = {'schema', 'version', 'device_id', 'kind', 'adapter', 'resource_id', 'revision',
                  'model', 'connection', 'execution_ready', 'physical_accuracy_verified'}
        if (not isinstance(value, dict) or set(value) != fields or value.get('schema') != SCHEMA
                or type(value.get('version')) is not int or value['version'] != 1
                or value.get('execution_ready') is not False or value.get('physical_accuracy_verified') is not False):
            raise ValueError('Unsupported device configuration schema/version')
        return cls(value['device_id'], DeviceKind(value['kind']), value['adapter'], value['resource_id'],
                   value['revision'], value['model'], json.dumps(value['connection'], allow_nan=False))


def save_configuration(configuration, *, outputs=None):
    root = hardware_configuration_root(outputs)
    path = resolve_relative(root, configuration.device_id+'/'+configuration.revision+'.json')
    publish_json(path, configuration.to_dict(), create_only=True)
    return path


def load_configuration(path):
    path = Path(path)
    if path.stat().st_size > 1024**2:
        raise ValueError('Configuration exceeds 1 MiB')
    return DeviceConfiguration.from_dict(load_json_object(path))


def create_adapter(configuration, scenario, clock, *, instance_id, epoch_seed=None):
    if configuration.adapter != 'simulation':
        raise NotImplementedError('设备未配置或真实/回放控制适配器未实现；不会回退模拟')
    from femto.hardware.simulation.adapter import SimulatedDevice
    return SimulatedDevice(configuration, scenario, clock, instance_id=instance_id, epoch_seed=epoch_seed)
