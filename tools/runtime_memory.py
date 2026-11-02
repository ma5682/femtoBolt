"""Bounded diagnostic summaries; these are never performance timing evidence."""
from __future__ import annotations

from contextlib import AbstractContextManager
import gc
import os
import time
import tracemalloc

import numpy as np
import psutil


def performance_environment():
    """Read cumulative load counters outside measured work; no tracing enabled."""
    others, errors = [], []
    for process in psutil.process_iter(['pid', 'name', 'cpu_times', 'memory_info']):
        try:
            info = process.info
            if info['pid'] == os.getpid() or 'python' not in (info['name'] or '').lower():
                continue
            cpu = info['cpu_times']
            others.append(dict(pid=info['pid'], name=info['name'],
                cpu_seconds=cpu.user+cpu.system if cpu else None,
                rss_bytes=info['memory_info'].rss if info['memory_info'] else None))
        except psutil.Error as error:
            errors.append(type(error).__name__)
    return dict(sampled_unix=time.time(), system_cpu_times=psutil.cpu_times()._asdict(),
                available_bytes=psutil.virtual_memory().available,
                other_python_processes=others[:16], observed_python_process_count=len(others),
                observation_errors=errors[:8], allocation_tracer_active=tracemalloc.is_tracing())


class MemoryDiagnostics(AbstractContextManager):
    def __init__(self, mode='weak'):
        if mode not in ('minimal', 'weak', 'allocations'):
            raise ValueError('Unknown memory observation mode')
        self.mode = mode
        self.process = psutil.Process()
        self.owned_trace = False
        self.previous = None

    def __enter__(self):
        if self.mode == 'allocations':
            if tracemalloc.is_tracing():
                raise RuntimeError('Refusing to alter an existing allocation tracer')
            tracemalloc.start(6)
            self.owned_trace = True
        return self

    def __exit__(self, *_):
        self.previous = None
        if self.owned_trace:
            tracemalloc.stop()
            self.owned_trace = False

    def snapshot(self, *, allocation_detail=False):
        started = time.perf_counter()
        info = self.process.memory_info()
        result = dict(mode=self.mode, rss_bytes=info.rss,
                      thread_count=self.process.num_threads(), gc_counts=list(gc.get_count()),
                      gc_enabled=gc.isenabled(), pid=os.getpid())
        try:
            full = self.process.memory_full_info()
            result['private_bytes'] = getattr(full, 'private', None)
            result['uss_bytes'] = getattr(full, 'uss', None)
        except (psutil.Error, OSError) as error:
            result['private_memory_error'] = repr(error)
        result['system_cpu_times'] = psutil.cpu_times()._asdict()
        result['system_available_bytes'] = psutil.virtual_memory().available
        result['sampled_unix'] = time.time()
        from femto.geometry import exact_surface
        result['geometry_caches'] = {name: getattr(exact_surface, name).cache_info()._asdict()
                                    for name in ('_dyadic_weights', '_dyadic_coefficients', '_unit_edges')}
        if self.owned_trace:
            current, peak = tracemalloc.get_traced_memory()
            result['traced_current_bytes'], result['traced_peak_bytes'] = current, peak
            result['tracer_metadata_bytes'] = tracemalloc.get_tracemalloc_memory()
            if allocation_detail:
                snapshot = tracemalloc.take_snapshot()
                domain = getattr(np.lib, 'tracemalloc_domain', None)
                result['numpy_domain'] = domain
                result['numpy_domain_bytes'] = (sum(trace.size for trace in snapshot.traces
                    if trace.domain == domain) if domain is not None else None)
                result['largest_live_python_sites'] = [
                    dict(site=str(item.traceback), bytes=item.size, count=item.count)
                    for item in snapshot.statistics('lineno')[:12]]
                if self.previous is not None:
                    result['allocation_changes'] = [dict(site=str(item.traceback),
                        bytes_diff=item.size_diff, count_diff=item.count_diff)
                        for item in snapshot.compare_to(self.previous, 'lineno')[:12]]
                # Retain one allocation description snapshot, never stack frames
                # or live object references. Account for tracer metadata above.
                self.previous = snapshot
        result['observation_seconds'] = time.perf_counter()-started
        result['attribution_limit'] = 'Untraced RSS is not an Open3D allocation measurement.'
        return result
