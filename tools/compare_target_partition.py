"""A/B replay of fixed saved target pages with an archived exact-partition module.

The reference is trusted local Python source explicitly supplied by the caller.
Only PixelPartition is substituted; source checks, budgets, page frontiers and
array/metadata comparisons run identically for both variants.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import io
import json
from pathlib import Path
import statistics
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from femto.geometry import exact_pixel_partition as exact
from femto.io_utils import write_json_strict
from femto.planning import target_surface as target, target_paging as paging
from tools.benchmark_workstation import Measurements, file_digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'outputs') or args.repetitions < 1:
        raise ValueError('Use a new workspace outputs directory and positive repetitions')
    output.mkdir(parents=True, exist_ok=False)
    project, reference = args.project.resolve(), args.reference.resolve()
    initial_project, initial_reference = file_digest(project), file_digest(reference)
    code = {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sorted((ROOT/'femto').rglob('*.py'))}
    metadata = dict(project=str(project), project_sha256=initial_project,
                    reference=str(reference), reference_sha256=initial_reference,
                    code_sha256=code, repetitions=args.repetitions, status='running',
                    scope='same saved page frontiers; exact array and metadata replay; software only')
    write_json_strict(output/'manifest.json', metadata)
    module_spec = importlib.util.spec_from_file_location('partition_reference', reference)
    previous = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(previous)
    current = exact.PixelPartition
    if previous.ALGORITHM != exact.ALGORITHM:
        raise ValueError('Reference algorithm differs')
    with zipfile.ZipFile(project) as archive:
        document = json.loads(archive.read('project.json'))
        revision = document['plan']['planning_context']['revision']
        header = json.loads(archive.read('target_surface.json'))
        target.validate_metadata(header, revision)
        with np.load(io.BytesIO(archive.read('target_surface.npz')), allow_pickle=False) as packed:
            original = {name: packed[name] for name in packed.files}
    inputs = {name: original[name] for name in paging.SOURCE_NAMES}
    target._check_inputs(inputs, revision)
    for array in original.values():
        array.setflags(write=False)
    headers = header.get('construction_pages') or [header]
    measurement = Measurements(output)
    checks = []

    def replay(variant, repetition):
        exact.PixelPartition = previous.PixelPartition if variant == 'before' else current
        deadline = time.monotonic()+120
        def check():
            if time.monotonic() > deadline:
                raise TimeoutError('Target replay exceeded its 120 s operation budget')
        start = 0
        for page in headers:
            arrays, metrics = target._construct(inputs, revision, target.TargetBudget(**page['budget']),
                                                check, frontier=page['metrics'])
            ledger = target._seal(arrays, revision, target.TargetBudget(**page['budget']), metrics,
                                  page['quality_evidence_sha256'], page['effective_domain_sha256'])
            if ledger.metadata != page:
                raise AssertionError(f'{variant}: saved target page metadata changed')
            end = start+len(arrays['area_m2'])
            for name in target._LEAVES:
                # Byte equality is stricter than equal tolerances or summaries.
                if arrays[name].tobytes() != original[name][start:end].tobytes():
                    raise AssertionError(f'{variant}: saved target leaf bytes changed: {name}')
            start = end
        if start != len(original['area_m2']):
            raise AssertionError('Replay omitted saved leaves')
        checks.append(dict(variant=variant, repetition=repetition, page_count=len(headers),
                           leaf_count=start, exact_saved_metadata=True, exact_saved_leaf_bytes=True))
        write_json_strict(output/'equivalence.json', checks)

    try:
        for repetition in range(args.repetitions):
            # Alternate order to reduce warm-cache/order bias.
            for variant in (('before', 'after') if repetition % 2 == 0 else ('after', 'before')):
                gc.collect()
                measurement.run(variant, repetition, lambda v=variant: replay(v, repetition))
        metadata['status'] = 'passed'
    except BaseException as error:
        metadata.update(status='failed', error=repr(error))
        raise
    finally:
        exact.PixelPartition = current
        metadata['source_unchanged'] = (file_digest(project) == initial_project
                                         and file_digest(reference) == initial_reference)
        metadata['code_unchanged'] = code == {
            p.relative_to(ROOT).as_posix(): file_digest(p) for p in sorted((ROOT/'femto').rglob('*.py'))}
        summary = {}
        for variant in ('before', 'after'):
            rows = [row for row in measurement.rows if row['stage'] == variant and row['status'] == 'passed']
            if rows:
                summary[variant] = dict(completed=len(rows),
                    median_seconds=statistics.median(row['seconds'] for row in rows),
                    max_sampled_rss_bytes=max(row['sampled_rss_peak_bytes'] for row in rows))
        write_json_strict(output/'summary.json', summary)
        write_json_strict(output/'manifest.json', metadata)
    if not metadata['source_unchanged'] or not metadata['code_unchanged']:
        raise AssertionError('Source or production code changed during comparison')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
