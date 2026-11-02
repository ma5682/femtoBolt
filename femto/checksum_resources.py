"""CPU affinity and available-memory policy for independent file checksums."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os

from femto.resource_policy import MemorySnapshot, MiB, preparation_resources


@dataclass(frozen=True)
class CpuCapacity:
    available_logical_cpus: int
    source: str


@dataclass(frozen=True)
class ChecksumResources:
    memory: MemorySnapshot
    cpu: CpuCapacity
    reserve_bytes: int
    workers: int
    file_count: int
    file_bytes: int
    policy_version: str = 'available-memory-checksum-workers-v1'


def cpu_capacity() -> CpuCapacity:
    """Respect process affinity; unknown capacity retains one worker."""
    try:
        if hasattr(os, 'process_cpu_count'):
            count = os.process_cpu_count()
            if count:return CpuCapacity(count, 'process_cpu_count')
        if hasattr(os, 'sched_getaffinity'):
            count = len(os.sched_getaffinity(0))
            if count:return CpuCapacity(count, 'sched_getaffinity')
        if os.name == 'nt':
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            process = kernel.GetCurrentProcess
            process.argtypes, process.restype = [], ctypes.c_void_p
            affinity = kernel.GetProcessAffinityMask
            affinity.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            affinity.restype = ctypes.c_int
            current, system = ctypes.c_size_t(), ctypes.c_size_t()
            if not affinity(process(), ctypes.byref(current), ctypes.byref(system)):
                raise OSError(ctypes.get_last_error())
            count = int(current.value).bit_count()
            if count:return CpuCapacity(count, 'GetProcessAffinityMask')
    except (OSError, ValueError, AttributeError):
        pass
    return CpuCapacity(1, 'unavailable')


def checksum_resources(file_count: int, file_bytes: int, *, memory: MemorySnapshot | None = None,
                       cpu: CpuCapacity | None = None) -> ChecksumResources:
    """Use at most two independent readers only with task, CPU and RAM headroom."""
    if type(file_count) is not int or type(file_bytes) is not int or min(file_count, file_bytes) < 0:
        raise ValueError('Checksum task sizes must be nonnegative integers')
    base = preparation_resources(0, file_bytes=file_bytes, memory=memory)
    measured = cpu_capacity() if cpu is None else cpu
    total, available = base.memory.total_bytes, base.memory.available_bytes
    known = type(total) is int and type(available) is int and total > 0 and 0 <= available <= total
    headroom = max(0, available-base.reserve_bytes) if known else 0
    enough_cpu = type(measured.available_logical_cpus) is int and measured.available_logical_cpus >= 4
    workers = 2 if enough_cpu and headroom >= 512*MiB and file_count >= 4 and file_bytes >= 16*MiB else 1
    return ChecksumResources(base.memory, measured, base.reserve_bytes, workers, file_count, file_bytes)
