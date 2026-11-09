"""Bounded local timing/error records, separate from reproducible evidence.

Arguments, environment variables and raw frames are never serialized here.
Recording errors cannot change a calculation, publication or original failure.
Each operation owns a new directory; no latest pointer or old record is changed.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import uuid
from typing import Any, Callable, Iterator, ParamSpec, TypeVar

from femto.io_utils import write_json_strict
from femto.paths import OUTPUTS_ROOT

CURRENT = ContextVar('femto_operation_diagnostics', default=None)
MAX_EVENTS = 512
MAX_TRACEBACK_BYTES = 512*1024
MAX_FAILURE_LOG_BYTES = 16*1024*1024
P = ParamSpec('P')
R = TypeVar('R')


@dataclass(frozen=True)
class RetainedArtifact:
    path: str
    kind: str


@dataclass(frozen=True)
class OperationSummary:
    status: str
    stage: str | None
    error_type: str | None
    reason: str | None
    directory: str | None
    retained: tuple[RetainedArtifact, ...]
    recording_error: str | None


def mark_retained(path: str | Path, kind: str) -> None:
    """Register only a successfully published artifact, never inferred paths."""
    active = CURRENT.get()
    if active is not None:
        artifact = RetainedArtifact(str(Path(path).resolve()), str(kind)[:256])
        if artifact not in active.retained and len(active.retained) < 16:
            active.retained.append(artifact)


def diagnostics_root() -> Path:
    value = os.environ.get('FEMTO_DIAGNOSTICS_DIR')
    return Path(value).resolve() if value else OUTPUTS_ROOT/'diagnostics'


def _new_directory(kind: str, root: str | Path | None = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    directory = (diagnostics_root() if root is None else Path(root))/kind/(stamp+'_'+uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    return directory


class OperationRecord:
    def __init__(self, name: str, root: str | Path | None = None) -> None:
        self.name, self.root = str(name), root
        self.directory = None
        self.recording_error = None
        self.started = time.perf_counter()
        from femto.performance_detail import current
        self.detail = current()
        self.created_utc = datetime.now(timezone.utc).isoformat()
        self.events = self.omitted_events = self.next_stage = 0
        self.stack = []
        self.status = 'running'
        self.failed_stage = None
        self._failed_exception = None
        self._failed_depth = None
        self.error = None
        self.retained: list[RetainedArtifact] = []
        self._events_stream = None
        self.totals = defaultdict(lambda: dict(calls=0, inclusive_seconds=0., self_seconds=0., failed_calls=0, stopped_calls=0))

    def summary(self) -> OperationSummary:
        """Immutable thread handoff; no exception, mutable stage or file reads."""
        error = self.error or {}
        return OperationSummary(self.status, self.failed_stage, error.get('type'),
            error.get('message'), str(self.directory) if self.directory else None,
            tuple(self.retained), self.recording_error)

    def _emit(self, value: dict[str, Any]) -> None:
        if self.directory is None or self.recording_error:return
        if self.events >= MAX_EVENTS:
            self.omitted_events += 1; return
        try:
            # One handle per operation: identical lines, each flushed at once,
            # without reopening the file for every stage boundary.
            if self._events_stream is None:
                self._events_stream = (self.directory/'events.jsonl').open('a', encoding='utf-8')
            self._events_stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')
            self._events_stream.flush()
            self.events += 1
        except Exception as error:self.recording_error = str(error)[:4096]

    def _close_events(self) -> None:
        stream, self._events_stream = self._events_stream, None
        if stream is not None:
            try:stream.close()
            except Exception as error:self.recording_error = self.recording_error or str(error)[:4096]

    def _save(self, status: str, error: dict[str, Any] | None = None) -> None:
        if self.directory is None:return
        value = dict(schema='femto-operation-diagnostics-v1', operation=self.name,
            created_utc=self.created_utc, status=status, elapsed_seconds=time.perf_counter()-self.started,
            stages=dict(self.totals), recorded_events=self.events, omitted_events=self.omitted_events,
            error=error, recording_error=self.recording_error,
            failed_stage=self.failed_stage, retained_artifacts=[asdict(item) for item in self.retained],
            timings_are_observations_not_replay_inputs=True)
        if self.detail is not None:
            try:value['performance_detail'] = self.detail.snapshot()
            except Exception as failure:
                self.recording_error = str(failure)[:4096]
                value['recording_error'] = self.recording_error
        try:write_json_strict(self.directory/'operation.json', value, durable=True)
        except Exception as failure:self.recording_error = str(failure)[:4096]

    def __enter__(self) -> OperationRecord:
        try:self.directory = _new_directory('operations', self.root)
        except Exception as error:self.recording_error = str(error)[:4096]
        self.token = CURRENT.set(self)
        self._save('running')
        return self

    def __exit__(self, kind, value, tb) -> bool:
        try:
            error = None
            if value is not None:
                error = dict(type=kind.__name__, message=str(value)[:4096])
                if self.directory is not None:
                    try:
                        data = ''.join(traceback.format_exception(kind, value, tb)).encode('utf-8')
                        error['traceback_truncated'] = len(data) > MAX_TRACEBACK_BYTES
                        with (self.directory/'error.log').open('xb') as stream:stream.write(data[-MAX_TRACEBACK_BYTES:])
                    except Exception as failure:self.recording_error = str(failure)[:4096]
            status = 'complete' if value is None else 'cancelled' if 'Cancel' in kind.__name__ else 'failed'
            if value is None:
                self.failed_stage = None
            elif self._failed_exception is None or (self._failed_exception is not value and value.__cause__ is not self._failed_exception):
                self.failed_stage = self.name
            self.status, self.error = status, error
            self._close_events()
            self._save(status, error)
        finally:
            self._close_events()
            self._failed_exception = None
            self._failed_depth = None
            CURRENT.reset(self.token)
        return False

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        number = self.next_stage; self.next_stage += 1
        frame = dict(number=number, child_seconds=0.)
        parent = self.stack[-1]['number'] if self.stack else None
        started = time.perf_counter(); self.stack.append(frame)
        self._emit(dict(event='start', stage=name, number=number, parent=parent,
                        elapsed_seconds=started-self.started))
        status = 'complete'
        try:yield
        except BaseException as error:
            if self._failed_exception is None or (self._failed_exception is not error and error.__cause__ is not self._failed_exception):
                self.failed_stage = name
                self._failed_depth = len(self.stack)
            self._failed_exception = error
            status = 'budget_stop' if type(error).__name__ == '_BudgetStop' else 'cancelled' if 'Cancel' in type(error).__name__ else 'failed'
            raise
        finally:
            elapsed = time.perf_counter()-started
            if status == 'complete' and self._failed_depth is not None and self._failed_depth > len(self.stack):
                # A successful enclosing stage handled its child exception.
                # Release traceback frames before the next large calculation.
                self._failed_exception = self._failed_depth = self.failed_stage = None
            self.stack.pop()
            if self.stack:self.stack[-1]['child_seconds'] += elapsed
            row = self.totals[name]
            row['calls'] += 1; row['inclusive_seconds'] += elapsed
            row['self_seconds'] += max(0., elapsed-frame['child_seconds'])
            row['failed_calls'] += int(status == 'failed')
            row['stopped_calls'] += int(status in ('budget_stop', 'cancelled'))
            self._emit(dict(event='end', stage=name, number=number, parent=parent,
                            seconds=elapsed, status=status))


@contextmanager
def record_operation(name: str, *, root: str | Path | None = None) -> Iterator[OperationRecord]:
    active = CURRENT.get()
    if active is not None:
        with active.stage(name):yield active
    else:
        with OperationRecord(name, root) as created:
            with created.stage(name):yield created


def diagnosed(name: str, *, operation: bool = False) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Time an existing operation without changing its inputs or return value."""
    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def measured(*args: P.args, **kwargs: P.kwargs) -> R:
            active = CURRENT.get()
            if active is not None:
                with active.stage(name):return function(*args, **kwargs)
            if operation:
                with record_operation(name):return function(*args, **kwargs)
            return function(*args, **kwargs)
        return measured
    return decorate


def retain_subprocess_failure(log_path: str | Path, *, status: str, stats: dict[str, Any],
                              error: BaseException, root: str | Path | None = None) -> Path | None:
    """Preserve a closed worker log before a temporary caller can remove it.

    Full bytes are kept up to 16 MiB. Larger logs retain a clearly marked tail.
    No command arguments, request payload or environment are captured.
    """
    try:
        directory = _new_directory('subprocess_failures', root)
        source = Path(log_path)
        size = source.stat().st_size
        with source.open('rb') as stream:
            stream.seek(max(0, size-MAX_FAILURE_LOG_BYTES))
            data = stream.read(MAX_FAILURE_LOG_BYTES)
        with (directory/'worker.log').open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        allowed = ('elapsed_s', 'returncode', 'peak_job_committed_mb', 'peak_rss_sampled_mb',
                   'memory_limit_mb', 'memory_measurement')
        measured = {key:value for key in allowed if (value := stats.get(key)) is None
                    or type(value) in (str, int, bool) or type(value) is float and math.isfinite(value)}
        digest = hashlib.sha256(data).hexdigest()
        value = dict(schema='femto-subprocess-failure-v1', status=str(status),
            error_type=type(error).__name__, error=str(error)[:4096], source_log=str(source.resolve()),
            source_bytes=size, stored_bytes=len(data), truncated=len(data) != size,
            stored_sha256=digest, source_sha256=digest if len(data) == size else None,
            measurements=measured, command_and_environment_recorded=False)
        write_json_strict(directory/'failure.json', value, durable=True)
        return directory
    except Exception:
        # Preserve the original cancellation/process error even on disk failure.
        return None
