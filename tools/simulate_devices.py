"""Run/replay the same simulation business chain used by the Qt workbench."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from femto.execution.application import create_coordinator, synthetic_gap_demo
from femto.execution.history import ExecutionStore
from femto.execution.models import load_saved_plan, synthetic_plan
from femto.execution.state_machine import RunState, TERMINAL
from femto.hardware.simulation.scenario import Fault, ManualClock, Scenario, load_scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, required=True, help='Product output root or independent test directory')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--synthetic', action='store_true', help='Explicit software test; never an import fallback')
    source.add_argument('--plan', type=Path, help='Saved ZIP or native trajectory, verified read-only')
    source.add_argument('--replay', help='Run ID: read-only, never creates devices or resends requests')
    source.add_argument('--list', action='store_true', help='List persisted simulation runs without creating a database')
    parser.add_argument('--scenario', type=Path)
    parser.add_argument('--fault', choices=('reject', 'no_response', 'stop_unconfirmed', 'duplicate', 'old_epoch', 'disconnect', 'base_move'))
    parser.add_argument('--pause', action='store_true', help='Exercise pause and resume during simulation')
    parser.add_argument('--stop', action='store_true', help='Request stop after first step dispatch')
    parser.add_argument('--export', type=Path)
    parser.add_argument('--gap-demo', action='store_true', help='Independent synthetic 0.2 m axial gap')
    args = parser.parse_args()
    store = ExecutionStore(args.outputs)
    if args.list:
        print(json.dumps(store.runs(), ensure_ascii=False, indent=2)); return 0
    if args.replay:
        record = store.read(args.replay)
        if args.export: store.export(args.replay, args.export)
        print(json.dumps(record, ensure_ascii=False, indent=2)); return 0
    if not (args.synthetic or args.plan):
        if args.gap_demo:
            print(json.dumps(synthetic_gap_demo(), ensure_ascii=False, indent=2)); return 0
        parser.error('Choose --synthetic, --plan, --list or --replay explicitly')
    plan = synthetic_plan() if args.synthetic else load_saved_plan(args.plan, progress=lambda text: print(text, file=sys.stderr))
    scenario = load_scenario(args.scenario) if args.scenario else Scenario()
    if args.fault:
        from dataclasses import replace
        intent = 'request_stop' if args.fault == 'stop_unconfirmed' else 'simulation_step'
        fault = Fault(args.fault, 'simulation-base' if args.fault == 'base_move' else 'simulation-arm',
                      intent, at_s=.65 if args.fault == 'base_move' else None)
        scenario = replace(scenario, faults=scenario.faults+(fault,),
                           base_mode='fixed' if args.fault == 'base_move' else scenario.base_mode)
    clock = ManualClock()
    coordinator = create_coordinator(outputs=args.outputs, scenario=scenario, clock=clock)
    try:
        # Required vertical order: frozen plan → connect → check → run → replay.
        coordinator.freeze(plan)
        coordinator.connect()
        coordinator.check()
        paused = resumed = stopped = started = False
        for _ in range(100000):
            clock.advance(scenario.tick_s)
            coordinator.poll()
            if coordinator.state is RunState.READY and not started:
                coordinator.start(); started = True
            if args.pause and coordinator.state is RunState.RUNNING and not paused:
                coordinator.pause(); paused = True
            if args.pause and coordinator.state is RunState.PAUSED and not resumed:
                coordinator.resume(); resumed = True
            if args.stop and coordinator.state is RunState.RUNNING and not stopped:
                coordinator.stop(); stopped = True
            if coordinator.state in TERMINAL:
                break
        else:
            coordinator.request_shutdown()
            raise RuntimeError('Simulation test budget exhausted')
        report = json.loads(coordinator.snapshot())
        report['database'] = str(store.path)
        report['gap_demo'] = synthetic_gap_demo() if args.gap_demo else None
        if args.export:
            store.export(coordinator.run_id, args.export)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if coordinator.state in (RunState.SUCCEEDED, RunState.STOPPED) else 2
    finally:
        coordinator.close_instances()


if __name__ == '__main__':
    raise SystemExit(main())
