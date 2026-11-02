"""Measure fixed offline workstation inputs without publishing user history.

Run from the workspace Python. Each attempt uses a new output directory.
Timing includes production validation and file I/O, not imports or input hashes.
RSS is sampled for this process only; this command does not start camera workers.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import platform
import statistics
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil

from femto.io_utils import write_json_strict


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix(): file_digest(p)
            for p in sorted(root.rglob('*')) if p.is_file()}


class Measurements:
    def __init__(self, output):
        self.output = output
        self.rows = []

    def run(self, stage, repetition, function):
        process = psutil.Process()
        initial = process.memory_info().rss
        peak = [initial]
        errors = []
        stop = threading.Event()

        def sample():
            try:
                while not stop.wait(.05):
                    peak[0] = max(peak[0], process.memory_info().rss)
            except Exception as error:
                errors.append(repr(error))

        watcher = threading.Thread(target=sample, daemon=True)
        started, cpu = time.perf_counter(), time.process_time()
        row = dict(stage=stage, repetition=repetition, status='running')
        print(json.dumps(row), flush=True)
        watcher.start()
        try:
            value = function()
            row['status'] = 'passed'
            return value
        except BaseException as error:
            row.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            row.update(seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu)
            peak[0] = max(peak[0], process.memory_info().rss)
            stop.set()
            watcher.join()
            row.update(rss_start_bytes=initial, sampled_rss_peak_bytes=peak[0],
                       rss_end_bytes=process.memory_info().rss,
                       rss_sampling_seconds=.05, sampling_errors=errors)
            self.rows.append(row)
            write_json_strict(self.output/'stages.json', self.rows)
            print(json.dumps(row, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--reconstruction-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'outputs'):
        raise ValueError('Benchmark output must be a new directory under this workspace outputs')
    if args.repetitions < 1:
        raise ValueError('Repetitions must be positive')
    output.mkdir(parents=True, exist_ok=False)
    project = args.project.resolve()
    reference_report = json.loads(args.reconstruction_report.read_text(encoding='utf-8-sig'))
    dataset = Path(reference_report['source_dataset']).resolve()
    manifest = json.loads((dataset/'manifest.json').read_text(encoding='utf-8-sig'))
    # Hash the declared dataset and its raw source. Hashing is outside stage timing.
    source_roots = {dataset}
    if manifest.get('source_dataset'):
        source_roots.add(Path(manifest['source_dataset']).resolve())
    sources = {str(path): inventory(path) for path in source_roots}
    code = {p.relative_to(ROOT).as_posix(): file_digest(p)
            for p in sorted((ROOT/'femto').rglob('*.py'))}
    metadata = dict(scope='fixed_real_data_offline_software_benchmark',
                    project=str(project), project_sha256=file_digest(project),
                    reconstruction_report=str(args.reconstruction_report.resolve()),
                    reconstruction_report_sha256=file_digest(args.reconstruction_report),
                    input_files=sources, code_sha256=code,
                    python=sys.version, platform=platform.platform(),
                    logical_cpu_count=psutil.cpu_count(),
                    physical_memory_bytes=psutil.virtual_memory().total,
                    measurement='single-process sampled RSS; 50 ms; no cold-cache guarantee',
                    repetitions=args.repetitions, status='running')
    write_json_strict(output/'manifest.json', metadata)

    # Imports precede all measurements.
    from femto.pipeline.scene_reconstruction import reconstruct_scene
    from femto.pipeline.local_history import scan_history
    from femto.planning import region_project, surface_workspace
    from femto.annotation.fixed_rgb_geometry import GeometryBudget
    from femto.planning.planning_revision import digest
    from femto.operation_diagnostics import record_operation
    parameters = {key: value for key, value in reference_report['parameters'].items()
                  if key in inspect.signature(reconstruct_scene).parameters}
    measurements = Measurements(output)
    equivalence = []

    def measured(stage, repetition, function):
        def observed():
            with record_operation(stage, root=output/'diagnostics'):
                return function()
        return measurements.run(stage, repetition, observed)

    try:
        for repetition in range(args.repetitions):
            folder = output/f'run_{repetition}'
            folder.mkdir()
            rebuilt = measured('reconstruct', repetition,
                lambda: reconstruct_scene(dataset, folder/'reconstruction', **parameters))
            history = measured('history_list', repetition,
                lambda: scan_history(ROOT/'datasets', ROOT/'outputs'))
            model = measured('history_open', repetition,
                lambda: region_project.open_project(project, workspace=folder/'opened'))
            region = model.store.regions[0]
            original = deepcopy(model.saved_plan)
            if original is None:
                raise ValueError('Use a project with saved route and parameters')
            plan = measured('planning_generate', repetition,
                lambda: surface_workspace.generate(model, region, original['treatment_head'],
                                                    GeometryBudget().check))
            native = measured('native_save', repetition,
                lambda: surface_workspace.export(model, region, plan, GeometryBudget().check))
            portable = measured('portable_save', repetition,
                lambda: region_project.save_project(model, region, folder/'result.region.zip', plan,
                    GeometryBudget().check, catalog_root=folder/'catalogue', allow_replace=False))
            reopened = measured('saved_history_open', repetition,
                lambda: region_project.open_project(portable, workspace=folder/'reopened'))
            matched = digest(reopened.saved_plan['surface_trajectory']) == digest(plan['surface_trajectory'])
            equivalence.append(dict(repetition=repetition, saved_route_equal=matched,
                                    native=str(native), history_count=len(history),
                                    rebuilt_frame_count=rebuilt['frame_count'],
                                    generated_route_sha256=digest(plan['surface_trajectory'])))
            write_json_strict(output/'equivalence.json', equivalence)
            if not matched:
                raise AssertionError('Saved route did not reproduce')
        metadata['status'] = 'passed'
    except BaseException as error:
        metadata.update(status='failed', error=repr(error))
        raise
    finally:
        metadata['source_unchanged'] = (metadata['project_sha256'] == file_digest(project)
            and sources == {str(path): inventory(path) for path in source_roots}
            and metadata['reconstruction_report_sha256'] == file_digest(args.reconstruction_report))
        metadata['code_unchanged'] = code == {p.relative_to(ROOT).as_posix(): file_digest(p)
            for p in sorted((ROOT/'femto').rglob('*.py'))}
        summaries = {}
        for stage in dict.fromkeys(row['stage'] for row in measurements.rows):
            rows = [row for row in measurements.rows if row['stage'] == stage and row['status'] == 'passed']
            if rows:
                summaries[stage] = dict(completed=len(rows),
                    median_seconds=statistics.median(row['seconds'] for row in rows),
                    max_seconds=max(row['seconds'] for row in rows),
                    max_sampled_rss_bytes=max(row['sampled_rss_peak_bytes'] for row in rows))
        write_json_strict(output/'summary.json', summaries)
        write_json_strict(output/'manifest.json', metadata)
    if not metadata['source_unchanged'] or not metadata['code_unchanged']:
        raise AssertionError('Benchmark inputs or production code changed during measurement')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
