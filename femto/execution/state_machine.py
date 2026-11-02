"""Explicit software task transitions; requests never imply physical outcomes."""
from enum import Enum


class RunState(str, Enum):
    UNPREPARED = 'unprepared'
    CHECKING = 'checking'
    READY = 'ready_for_simulation'
    RUNNING = 'running'
    PAUSE_REQUESTED = 'pause_requested'
    PAUSED = 'pause_confirmed'
    RESUME_REQUESTED = 'resume_requested'
    STOP_REQUESTED = 'stop_requested'
    STOPPED = 'stopped'
    CANCELLED = 'cancelled'
    ABANDONED = 'abandoned_unrun'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'
    UNKNOWN = 'result_unknown'
    INTERRUPTED = 'interrupted_needs_review'


TERMINAL = frozenset({RunState.STOPPED, RunState.CANCELLED, RunState.SUCCEEDED,
                      RunState.ABANDONED, RunState.FAILED, RunState.UNKNOWN, RunState.INTERRUPTED})
TRANSITIONS = {
    RunState.UNPREPARED: {RunState.CHECKING, RunState.FAILED, RunState.ABANDONED},
    RunState.CHECKING: {RunState.READY, RunState.STOP_REQUESTED, RunState.FAILED, RunState.UNKNOWN},
    RunState.READY: {RunState.RUNNING, RunState.STOP_REQUESTED, RunState.CANCELLED, RunState.FAILED, RunState.UNKNOWN},
    RunState.RUNNING: {RunState.PAUSE_REQUESTED, RunState.STOP_REQUESTED, RunState.UNKNOWN},
    RunState.PAUSE_REQUESTED: {RunState.PAUSED, RunState.STOP_REQUESTED, RunState.UNKNOWN},
    RunState.PAUSED: {RunState.RESUME_REQUESTED, RunState.STOP_REQUESTED, RunState.UNKNOWN},
    RunState.RESUME_REQUESTED: {RunState.RUNNING, RunState.STOP_REQUESTED, RunState.UNKNOWN},
    RunState.STOP_REQUESTED: {RunState.STOPPED, RunState.CANCELLED, RunState.SUCCEEDED,
                            RunState.FAILED, RunState.UNKNOWN},
    # A current-generation fault after a normal terminal result still matters.
    RunState.SUCCEEDED: {RunState.FAILED, RunState.UNKNOWN},
    RunState.STOPPED: {RunState.FAILED, RunState.UNKNOWN},
    RunState.CANCELLED: {RunState.FAILED, RunState.UNKNOWN},
    RunState.FAILED: {RunState.UNKNOWN},
    RunState.ABANDONED: set(), RunState.UNKNOWN: set(), RunState.INTERRUPTED: set(),
}


def transition(current, target):
    if not isinstance(current, RunState) or not isinstance(target, RunState):
        raise ValueError('Task states must be explicit')
    if target == current:
        return current
    if target not in TRANSITIONS[current]:
        raise ValueError(f'Illegal task transition: {current.value} → {target.value}')
    return target
