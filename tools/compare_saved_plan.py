"""Alternate complete save/readback runs with archived and current pixel clipping.

The reference is trusted local Python explicitly supplied by the caller.
Both variants use the current remaining backend, including identical checks,
budgets and independent readback. No cProfile instrumentation is enabled.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from femto.geometry import pixel_clip
from femto.io_utils import write_json_strict
from femto.planning import target_surface
from tools.benchmark_workstation import file_digest
from tools.profile_saved_plan import main as profile_plan

TARGET_IMPORTS = ('area2', 'clip_triangle_planes', 'covers_in_front', 'interior_overlap',
                  'pixel_planes', 'project', 'shared_edge_separates')


@contextmanager
def reference_backend(reference):
    """Replace all bound target imports as well as the module's public entry."""
    if reference.EPS != pixel_clip.EPS:
        raise ValueError('Archived pixel clipping uses different error bounds')
    with ExitStack() as stack:
        for name in (*TARGET_IMPORTS, 'clip_triangle'):
            stack.enter_context(patch.object(pixel_clip, name, getattr(reference, name)))
        for name in TARGET_IMPORTS:
            stack.enter_context(patch.object(target_surface, name, getattr(reference, name)))
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=2)
    args = parser.parse_args(argv)
    output, project, reference = args.output.resolve(), args.project.resolve(), args.reference.resolve()
    if not output.is_relative_to(ROOT/'outputs') or args.repetitions < 2:
        raise ValueError('Use a new workspace outputs directory and at least two alternating pairs')
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('pixel_clip_reference', reference)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    initial_code = {path.relative_to(ROOT).as_posix(): file_digest(path)
                    for path in sorted((ROOT/'femto').rglob('*.py'))}
    metadata = dict(schema='saved-plan-comparison-v1', status='running',
        project=str(project), project_sha256=file_digest(project),
        reference=str(reference), reference_sha256=file_digest(reference),
        code_sha256=initial_code, repetitions=args.repetitions,
        policy=('Alternating complete before/after save and independent readback; '
                'ordinary operation budgets; no profiler; explicit gc.collect only '
                'between complete variant runs outside every measured stage.'))
    write_json_strict(output/'manifest.json', metadata)
    runs = []
    try:
        for repetition in range(args.repetitions):
            order = ('before', 'after') if repetition % 2 == 0 else ('after', 'before')
            for variant in order:
                destination = output/f'pair_{repetition}_{variant}'
                gc.collect()
                arguments = ['--project', str(project), '--output', str(destination), '--profile', 'none']
                if variant == 'before':
                    with reference_backend(previous):
                        profile_plan(arguments)
                else:
                    profile_plan(arguments)
                report = json.loads((destination/'manifest.json').read_text(encoding='utf-8'))
                if not report['all_checks_passed']:
                    raise AssertionError(f'{variant} lacks equivalent saved and reopened output')
                rows = json.loads((destination/'stages.json').read_text(encoding='utf-8'))
                runs.append(dict(repetition=repetition, variant=variant,
                    output=str(destination), all_checks_passed=True, stages=rows))
                write_json_strict(output/'runs.json', runs)
        metadata['status'] = 'passed'
    except BaseException as error:
        metadata.update(status='failed', error=repr(error))
        raise
    finally:
        metadata['source_unchanged'] = metadata['project_sha256'] == file_digest(project)
        metadata['reference_unchanged'] = metadata['reference_sha256'] == file_digest(reference)
        metadata['code_unchanged'] = initial_code == {
            path.relative_to(ROOT).as_posix(): file_digest(path)
            for path in sorted((ROOT/'femto').rglob('*.py'))}
        summary = {}
        for variant in ('before', 'after'):
            selected = [run for run in runs if run['variant'] == variant]
            values = {}
            for stage in ('source_open', 'native_save', 'portable_save', 'native_readback', 'portable_readback'):
                rows = [row for run in selected for row in run['stages'] if row['stage'] == stage]
                if rows:
                    values[stage] = dict(seconds=[row['seconds'] for row in rows],
                        median_seconds=statistics.median(row['seconds'] for row in rows),
                        max_sampled_rss_bytes=max(row['sampled_rss_peak_bytes'] for row in rows))
            totals = [sum(row['seconds'] for row in run['stages']
                          if row['stage'] in ('native_save', 'portable_save')) for run in selected]
            if totals:
                values['combined_save'] = dict(seconds=totals, median_seconds=statistics.median(totals))
            summary[variant] = values
        metadata['all_checks_passed'] = bool(metadata['status'] == 'passed'
            and metadata['source_unchanged'] and metadata['reference_unchanged']
            and metadata['code_unchanged'] and len(runs) == args.repetitions*2)
        write_json_strict(output/'summary.json', summary)
        write_json_strict(output/'manifest.json', metadata)
    if not metadata['all_checks_passed']:
        raise AssertionError('Comparison input, output or code evidence is incomplete')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
