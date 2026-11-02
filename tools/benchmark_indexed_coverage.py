"""Matched real-project legacy/indexed coverage, retention and independent replay."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from femto.planning import coverage_certificate as coverage, coverage_workflow, region_project
from femto.planning.surface_trajectory import SurfaceTrajectory
from femto.dataset.scene_data_kind import classify_dataset_data_kind
from tools.benchmark_surface_coverage import Measurements, source_files, write, proof_summary, comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/'outputs').resolve()) or output == ROOT/'outputs':
        parser.error('Output must be a new subdirectory in workspace outputs')
    if output.exists() or not 1 <= args.repetitions <= 5:parser.error('New output and 1-5 repetitions required')
    output.mkdir(parents=True)
    os.environ['FEMTO_DIAGNOSTICS_DIR'] = str(output/'diagnostics')
    snapshot = source_files()
    write(output/'source_at_start.json', snapshot)
    measurements = Measurements(output)
    source = measurements.run('real', 0, 'open', lambda check:
        region_project.open_project(args.project.resolve(), workspace=output/'workspace',
                                    progress=lambda _:check()))
    error = None
    comparisons = []
    try:
        if source is None or source.saved_plan is None or source.target_ledger is None:
            raise ValueError('A fully verified original-target project is required')
        dataset = Path(source.store.origin_reference['source_dataset'])
        if classify_dataset_data_kind(dataset) != 'real_femto':
            raise ValueError('Real acquisition provenance required')
        trajectory = SurfaceTrajectory.from_dict(source.saved_plan['surface_trajectory'])
        if trajectory.planner['algorithm_version'] not in ('quality-constrained-surface-v3', 'constructed-surface-route-v1'):
            raise ValueError('Use an unrefined fixed-model base route')
        for repetition in range(args.repetitions):
            values = {}
            strategies = [('legacy', coverage.CURRENT_ALGORITHM), ('indexed', coverage.INDEXED_ALGORITHM)]
            if repetition % 2:strategies.reverse()
            for name, algorithm in strategies:
                domain, seeds, model, binding = coverage_workflow.bundle(trajectory, source.target_ledger)
                result = measurements.run(name, repetition, 'coverage', lambda check:
                    coverage.evaluate(domain, seeds, model, algorithm=algorithm, route_binding=binding,
                                      budget=coverage.CoverageBudget(seconds=30), check=check),
                    lambda certificate:proof_summary(output/f'{name}_{repetition}_assets', certificate, domain))
                if result is None:raise RuntimeError('Coverage failed; see retained records')
                values[name] = result
                replay = measurements.run(name, repetition, 'replay', lambda check:
                    coverage.validate(result, domain, seeds, model, route_binding=binding, check=check),
                    lambda certificate:dict(certificate_sha256=certificate.metadata['sha256']))
                if replay is None:raise RuntimeError('Independent proof replay failed')
            compared = comparison(values['legacy'], values['indexed'], domain)
            comparisons.append(compared)
            write(output/f'comparison_{repetition}.json', compared)
            if compared['lost_weight_units']:raise AssertionError('Indexed trial loses legacy confirmed set')
    except Exception as failure:
        error = dict(type=type(failure).__name__, message=str(failure))
    unchanged = source_files() == snapshot
    success = error is None and unchanged and all(row['status']=='complete' for row in measurements.rows)
    write(output/'result.json', dict(scope='real_saved_geometry_not_new_hardware_accuracy',
        project=str(args.project.resolve()), rows=measurements.rows, comparisons=comparisons,
        source_python_unchanged=unchanged, failure=error, all_checks_passed=success))
    print(json.dumps(dict(output=str(output), all_checks_passed=success)), flush=True)
    return 0 if success else 1


if __name__ == '__main__':raise SystemExit(main())
