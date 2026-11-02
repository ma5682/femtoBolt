"""Bounded preparation resources; never a geometry or publication budget.

Decisions use current available memory and are frozen for one task. They only
size disposable buffers/caches, not numerical batches, proof order or workers.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

MiB = 1024 * 1024
GiB = 1024 * MiB


@dataclass(frozen=True)
class MemorySnapshot:
    total_bytes: int | None
    available_bytes: int | None
    source: str


@dataclass(frozen=True)
class PreparationResources:
    memory: MemorySnapshot
    reserve_bytes: int
    cache_budget_bytes: int
    candidate_cache_entries: int
    hash_buffer_bytes: int
    policy_version: str = 'available-memory-preparation-v1'


class _WindowsMemoryStatus(ctypes.Structure):
    # A per-call Structure/POINTER declaration is retained by ctypes' type cache.
    _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32)] + [
        (name, ctypes.c_uint64) for name in ('total', 'available', 'page_total',
        'page_available', 'virtual_total', 'virtual_available', 'extended')]


@lru_cache(maxsize=1)
def _windows_memory_function():
    """Bind once; each observation still reads a fresh system memory status."""
    function = ctypes.WinDLL('kernel32', use_last_error=True).GlobalMemoryStatusEx
    function.argtypes = [ctypes.POINTER(_WindowsMemoryStatus)]
    function.restype = ctypes.c_int
    return function


def _windows_memory_snapshot() -> MemorySnapshot:
    status = _WindowsMemoryStatus()
    status.length = ctypes.sizeof(status)
    if not _windows_memory_function()(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error())
    return MemorySnapshot(int(status.total), int(status.available), 'GlobalMemoryStatusEx')


def _proc_memory_snapshot() -> MemorySnapshot:
    values = {}
    for line in Path('/proc/meminfo').read_text(encoding='ascii').splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable'):
            values[key] = int(value.split()[0]) * 1024
    return MemorySnapshot(values['MemTotal'], values['MemAvailable'], 'proc-meminfo')


def memory_snapshot() -> MemorySnapshot:
    """Read current available memory with a bounded, dependency-free binding."""
    try:
        return _windows_memory_snapshot() if os.name == 'nt' else _proc_memory_snapshot()
    except (OSError, ValueError, KeyError, AttributeError):
        return MemorySnapshot(None, None, 'unavailable')


def preparation_resources(task_items: int, *, file_bytes: int | None = None,
                          memory: MemorySnapshot | None = None) -> PreparationResources:
    """Reserve OS/UI headroom and bound both memory and useful cache capacity.

Each candidate entry has <=128 int64 IDs and float64 XYZ rows. A conservative
8 KiB allowance includes arrays, keys and cache overhead; at most 16,384 entries
are retained. Multiple operations still own independent caches and must use the
existing single-worker UI gates. Unknown memory keeps the previous 4,096 cap.
    """
    if task_items < 0 or file_bytes is not None and file_bytes < 0:
        raise ValueError('Resource task sizes must be nonnegative')
    measured = memory_snapshot() if memory is None else memory
    total, available = measured.total_bytes, measured.available_bytes
    known = (type(total) is int and type(available) is int
             and total > 0 and 0 <= available <= total)
    useful_entries = max(64, task_items * 4)
    if known:
        reserve = max(2 * GiB, total // 8)
        headroom = max(0, available - reserve)
        cache_bytes = min(128 * MiB, headroom // 32)
        entries = min(16384, useful_entries, cache_bytes // 8192)
        buffer_cap = 64 * 1024 if headroom < 256 * MiB else MiB
    else:
        reserve, cache_bytes = 2 * GiB, 32 * MiB
        entries, buffer_cap = min(4096, useful_entries), MiB
    buffer_bytes = buffer_cap if file_bytes is None else max(1, min(buffer_cap, file_bytes))
    return PreparationResources(measured, reserve, cache_bytes, entries, buffer_bytes)
