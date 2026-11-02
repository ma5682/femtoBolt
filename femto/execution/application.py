"""Application composition shared by the Qt owner and headless CLI."""
from __future__ import annotations

import json
import uuid

from femto.execution.coordinator import Coordinator
from femto.execution.history import ExecutionStore
from femto.geometry.axial_gap import SurfaceEvidence, axial_gap
from femto.hardware.configuration import DeviceConfiguration, create_adapter
from femto.hardware.interface_contract import DeviceKind
from femto.hardware.simulation.scenario import ManualClock, Scenario


def configurations(scenario):
    result = [DeviceConfiguration('simulation-arm', DeviceKind.ARM, 'simulation', 'arm', '1', '软件模拟机械臂'),
              DeviceConfiguration('simulation-head', DeviceKind.TREATMENT_HEAD, 'simulation', 'head', '1', '软件模拟治疗头')]
    if scenario.base_mode != 'absent':
        result.append(DeviceConfiguration('simulation-base', DeviceKind.BASE, 'simulation', 'base', '1',
                                          '有安装证据的合成固定底座' if scenario.base_mode == 'fixed' else '软件模拟可控底座'))
    return tuple(result)


def create_coordinator(*, outputs=None, scenario=None, clock=None, instance_id=None,
                       device_configurations=None, crash_hook=lambda boundary: None):
    scenario = scenario or Scenario()
    clock = clock or ManualClock()
    instance_id = instance_id or uuid.uuid4().hex
    devices = configurations(scenario) if device_configurations is None else tuple(device_configurations)
    adapters = tuple(create_adapter(config, scenario, clock, instance_id=instance_id) for config in devices)
    store = ExecutionStore(outputs)
    return Coordinator(adapters, clock, scenario, store, outputs=outputs, crash_hook=crash_hook)


def synthetic_gap_demo():
    surface = SurfaceEvidence(((-.1, -.1, .2), (0, .1, .2), (.1, -.1, .2)), ((0, 1, 2),),
                              (1,), (('synthetic-original-view',),), 'synthetic_test', 'C', known_occlusion=False)
    return axial_gap(surface, [0, 0, 0], [0, 0, 1], frame='C', mode='synthetic_test', calibration_applicable=True)
