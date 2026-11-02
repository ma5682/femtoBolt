"""Profile saving a fixed existing plan, retaining full independent readback.

Writes only to a new workspace outputs folder. No planning, geometry checks,
budgets or production algorithms are replaced. cProfile is optional because
its instrumentation affects timing and may exhaust the ordinary operation
budget. GC callbacks observe collections without requesting or disabling GC.
"""
from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import io
import json
from pathlib import Path
import pstats
import statistics
import sys
import time
import traceback
import tracemalloc
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from femto.io_utils import write_json_strict
from tools.benchmark_workstation import Measurements, file_digest
from tools.save_environment import performance_environment
from tools.save_stage_observation import observe_save_stages

STAGES = ('source_open', 'native_save', 'portable_save', 'native_readback', 'portable_readback')


def measurement_errors(rows, repetitions):
    """Require every declared stage and a functioning RSS sampler."""
    errors = []
    expected_repetitions = set(range(repetitions))
    if any(row.get('repetition') not in expected_repetitions for row in rows):
        errors.append('Measured stages contain an undeclared repetition')
    for repetition in range(repetitions):
        measured = [row for row in rows if row.get('repetition') == repetition]
        if [row.get('stage') for row in measured] != list(STAGES):
            errors.append(f'Incomplete or duplicate stages in repetition {repetition}')
        if any(row.get('status') != 'passed' or row.get('sampling_errors') != [] for row in measured):
            errors.append(f'Stage or RSS sampling failure in repetition {repetition}')
    return errors


def finalize_checks(metadata, equivalence, rows, repetitions):
    """A completed workload still fails when its measurement evidence is invalid."""
    metadata['measurement_errors'] = measurement_errors(rows, repetitions)
    metadata['measurement_checks_passed'] = not metadata['measurement_errors']
    metadata['all_checks_passed'] = bool(metadata['status'] == 'passed'
        and metadata['source_unchanged'] and metadata['code_unchanged']
        and metadata['measurement_checks_passed']
        and len(equivalence) == repetitions and all(row['all_checks_passed'] for row in equivalence))
    if metadata['status'] == 'passed' and not metadata['all_checks_passed']:
        metadata.update(status='failed',
                        error='Final input/code, independent readback or measurement checks failed')


class GCObservation:
    """Observe one measured stage; keep no references to collected objects."""
    def __init__(self):
        self.started = {}
        self.rows = {generation: dict(collections=0, seconds=0., collected=0, uncollectable=0)
                     for generation in range(3)}

    def callback(self, phase, information):
        generation = information['generation']
        if phase == 'start':
            self.started[generation] = time.perf_counter()
        elif generation in self.started:
            elapsed = time.perf_counter()-self.started.pop(generation)
            row = self.rows[generation]
            row['collections'] += 1
            row['seconds'] += elapsed
            row['collected'] += information.get('collected', 0)
            row['uncollectable'] += information.get('uncollectable', 0)

    def report(self):
        return dict(total_seconds=sum(row['seconds'] for row in self.rows.values()),
                    generations=self.rows, unfinished_generations=sorted(self.started),
                    policy='observed only; no explicit collections or GC setting changes')


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def evidence_values(read, names):
    """Compare all saved target/proof arrays and headers, including hint history."""
    output = {}
    for name in sorted(names):
        if not any(key in name for key in ('target_surface', 'coverage_certificate')):
            continue
        if name.endswith('.json'):
            output[name] = dict(json_sha256=content_digest(json.loads(read(name))))
        elif name.endswith('.npz'):
            values = {}
            with np.load(io.BytesIO(read(name)), allow_pickle=False) as packed:
                for key in packed.files:
                    array = np.ascontiguousarray(packed[key])
                    values[key] = dict(dtype=array.dtype.str, shape=list(array.shape),
                                       sha256=hashlib.sha256(array.tobytes()).hexdigest())
            output[name] = values
    if 'target_surface.json' not in output or 'target_surface.npz' not in output:
        raise ValueError('Use a fixed saved project with complete target evidence')
    return output


def archive_values(path):
    with zipfile.ZipFile(path) as archive:
        document = json.loads(archive.read('project.json'))
        return document, evidence_values(archive.read, archive.namelist())


def folder_values(path):
    names = [item.name for item in path.iterdir() if item.is_file()]
    return evidence_values(lambda name: (path/name).read_bytes(), names)


def route_digest(trajectory):
    # Production semantic equality ignores only relocated source paths,
    # presentation status and their corresponding data-layer path declaration.
    from femto.planning.refinement_validation import semantic_payload
    return content_digest(semantic_payload(trajectory))


@observe_save_stages
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--profile', choices=('none', 'saves', 'all'), default='none')
    args = parser.parse_args(argv)
    if tracemalloc.is_tracing():
        raise RuntimeError('Disable allocation tracing for save performance measurement')
    output, project = args.output.resolve(), args.project.resolve()
    if not output.is_relative_to(ROOT/'outputs') or args.repetitions < 1:
        raise ValueError('Use a new workspace outputs directory and positive repetitions')
    output.mkdir(parents=True, exist_ok=False)
    document, expected_evidence = archive_values(project)
    plan = document.get('plan')
    if not plan or not plan.get('planning_context'):
        raise ValueError('The input must contain a saved plan and target')
    initial_project = file_digest(project)
    initial_code = {path.relative_to(ROOT).as_posix(): file_digest(path)
                    for path in sorted((ROOT/'femto').rglob('*.py'))}
    metadata = dict(schema='fixed-plan-save-profile-v1', status='running',
        project=str(project), project_sha256=initial_project,
        probe_sha256=file_digest(Path(__file__)), code_sha256=initial_code,
        repetitions=args.repetitions, profile=args.profile,
        target_algorithm=plan['planning_context']['revision']['inputs']['algorithms']['target'],
        route_algorithm=plan['surface_trajectory']['planner']['algorithm_version'],
        target_leaf_count=plan['planning_context']['target']['summary']['leaf_count'],
        scope='fixed existing software plan; serial saves and full independent readback',
        timing_limitations='cProfile affects timing; process RSS includes retained earlier objects')
    write_json_strict(output/'manifest.json', metadata)

    # Import dependencies before timing.
    from femto.annotation.fixed_rgb_geometry import GeometryBudget
    from femto.operation_diagnostics import record_operation
    from femto.planning import region_project, surface_workspace, trajectory_export
    from femto.planning.surface_trajectory import SurfaceTrajectory
    expected_route = route_digest(SurfaceTrajectory.from_dict(plan['surface_trajectory']))
    measurements = Measurements(output)
    instrumentation, equivalence = [], []

    def measured(stage, repetition, function):
        observation = GCObservation()
        profiled = args.profile == 'all' or (args.profile == 'saves' and stage.endswith('_save'))
        profile = cProfile.Profile() if profiled else None
        row = dict(stage=stage, repetition=repetition, profiled=profiled)
        row['environment_before'] = performance_environment()
        gc.callbacks.append(observation.callback)
        try:
            def observed():
                with record_operation(stage, root=output/'diagnostics'):
                    if profile is not None:
                        return profile.runcall(function)
                    return function()
            return measurements.run(stage, repetition, observed)
        finally:
            gc.callbacks.remove(observation.callback)
            row['gc'] = observation.report()
            row['environment_after'] = performance_environment()
            instrumentation.append(row)
            write_json_strict(output/'instrumentation.json', instrumentation)
            if profile is not None:
                prefix = output/f'run_{repetition}_{stage}'
                profile.dump_stats(str(prefix)+'.prof')
                with Path(str(prefix)+'.txt').open('w', encoding='utf-8') as stream:
                    pstats.Stats(profile, stream=stream).sort_stats('cumulative').print_stats(100)

    try:
        for repetition in range(args.repetitions):
            folder = output/f'run_{repetition}'
            folder.mkdir()
            model = measured('source_open', repetition, lambda: region_project.open_project(
                project, workspace=folder/'opened'))
            region, current_plan = model.store.regions[0], model.saved_plan
            if route_digest(SurfaceTrajectory.from_dict(current_plan['surface_trajectory'])) != expected_route:
                raise AssertionError('Input open changed route semantics')
            native = measured('native_save', repetition, lambda: surface_workspace.export(
                model, region, current_plan, GeometryBudget().check))
            portable = measured('portable_save', repetition, lambda: region_project.save_project(
                model, region, folder/'result.region.zip', current_plan, GeometryBudget().check,
                catalog_root=folder/'catalogue', allow_replace=False))
            restored_native = measured('native_readback', repetition, lambda: trajectory_export.load_trajectory(
                native, check=GeometryBudget().check, reproduce=True))
            restored_portable = measured('portable_readback', repetition, lambda: region_project.open_project(
                portable, workspace=folder/'reopened'))
            saved_document, portable_evidence = archive_values(portable)
            comparisons = dict(
                native_route_semantics_equal=route_digest(restored_native) == expected_route,
                portable_route_semantics_equal=route_digest(SurfaceTrajectory.from_dict(
                    restored_portable.saved_plan['surface_trajectory'])) == expected_route,
                portable_saved_plan_equal=saved_document['plan'] == current_plan,
                native_all_evidence_equal=folder_values(native) == expected_evidence,
                portable_all_evidence_equal=portable_evidence == expected_evidence)
            equivalence.append(dict(repetition=repetition, native=str(native), portable=str(portable),
                                     checks=comparisons, all_checks_passed=all(comparisons.values())))
            write_json_strict(output/'equivalence.json', equivalence)
            if not all(comparisons.values()):
                raise AssertionError('Saved or independently reopened route/evidence changed')
        metadata['status'] = 'passed'
    except BaseException as error:
        metadata.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        metadata['source_unchanged'] = file_digest(project) == initial_project
        metadata['code_unchanged'] = initial_code == {
            path.relative_to(ROOT).as_posix(): file_digest(path)
            for path in sorted((ROOT/'femto').rglob('*.py'))}
        summary = {}
        for stage in dict.fromkeys(row['stage'] for row in measurements.rows):
            rows = [row for row in measurements.rows if row['stage'] == stage and row['status'] == 'passed']
            gc_rows = [row for row in instrumentation if row['stage'] == stage]
            if rows:
                summary[stage] = dict(completed=len(rows),
                    median_seconds=statistics.median(row['seconds'] for row in rows),
                    max_sampled_rss_bytes=max(row['sampled_rss_peak_bytes'] for row in rows),
                    total_observed_gc_seconds=sum(row['gc']['total_seconds'] for row in gc_rows))
        finalize_checks(metadata, equivalence, measurements.rows, args.repetitions)
        write_json_strict(output/'summary.json', summary)
        write_json_strict(output/'manifest.json', metadata)
    if not metadata['all_checks_passed']:
        raise AssertionError('Run lacks complete equivalent output or unchanged input/code evidence')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
