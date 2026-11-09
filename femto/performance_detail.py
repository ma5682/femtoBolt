"""Opt-in aggregate observations, never geometry evidence or replay inputs.

No file I/O occurs on a hot path. A benchmark (or diagnostic operation) owns
the context and persists its snapshot separately. Normal operation is disabled.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter
import math

_CURRENT = ContextVar('femto_performance_detail', default=None)
MAX_ROWS = 192
MAX_COUNTERS = 48


class Detail:
    def __init__(self):
        self.rows = {}
        self.omitted = 0

    def add(self, name, *, seconds=None, calls=0, **counts):
        if name not in self.rows and len(self.rows) >= MAX_ROWS:
            self.omitted += 1
            return
        row = self.rows.setdefault(name, dict(calls=0, seconds=0., counts={}))
        row['calls'] += calls
        if seconds is not None and math.isfinite(seconds) and seconds >= 0:
            row['seconds'] += seconds
        for key, value in counts.items():
            if key not in row['counts'] and len(row['counts']) >= MAX_COUNTERS:
                self.omitted += 1
                continue
            row['counts'][key] = row['counts'].get(key, 0) + int(value)

    def snapshot(self):
        return dict(schema='femto-performance-detail-v1', observations_not_replay_inputs=True,
                    timings_include_nested_calls=True, omitted=self.omitted,
                    rows={k:dict(calls=v['calls'], seconds=v['seconds'], counts=dict(v['counts']))
                          for k,v in self.rows.items()})


def current():
    return _CURRENT.get()


def _record(detail, name, **values):
    try:
        detail.add(name, **values)
    except Exception:
        # Instrumentation must not replace an original result or exception.
        detail.omitted += 1


@contextmanager
def collect():
    detail = Detail()
    token = _CURRENT.set(detail)
    try:
        yield detail
    finally:
        _CURRENT.reset(token)


@contextmanager
def timed(name):
    detail = current()
    if detail is None:
        yield
        return
    start = perf_counter()
    try:
        yield
    finally:
        _record(detail, name, calls=1, seconds=perf_counter()-start)


def observed(name):
    def decorate(function):
        @wraps(function)
        def run(*args, **kwargs):
            detail = current()
            if detail is None:
                return function(*args, **kwargs)
            start = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                _record(detail, name, calls=1, seconds=perf_counter()-start)
        return run
    return decorate


def count(name, **counts):
    detail = current()
    if detail is not None:
        _record(detail, name, **counts)
