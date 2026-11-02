"""Narrow device-owner boundary, independent of planning, execution and Qt."""
from __future__ import annotations

from typing import Protocol

from femto.hardware.configuration import DeviceConfiguration
from femto.hardware.interface_contract import (
    Capabilities, DeviceBinding, DeviceFeedback, DeviceRequest, DeviceStatus, SimulationStep,
)


class DevicePort(Protocol):
    resource_key: str
    binding: DeviceBinding
    configuration: DeviceConfiguration
    connected: bool

    def connect(self) -> DeviceBinding: ...
    def close(self) -> None: ...
    def capabilities(self) -> Capabilities: ...
    def status(self) -> DeviceStatus: ...
    def submit(self, request: DeviceRequest, step: SimulationStep | None = None) -> None: ...
    def poll(self) -> tuple[DeviceFeedback | DeviceStatus, ...]: ...
