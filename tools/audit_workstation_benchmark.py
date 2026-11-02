"""Audit repeated outputs, or compare an optimized run with a fixed baseline.

Single-run use: ``python tools/audit_workstation_benchmark.py RUN``
Cross-run use: ``python tools/audit_workstation_benchmark.py RUN --baseline BASELINE``
The cross-run audit reads every generated output again. Input equality refers
to the hashes recorded during both runs, not the current contents of datasets.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from femto.io_utils import write_json_strict
from tools.benchmark_workstation import file_digest

STAGES = ('reconstruct', 'history_list', 'history_open', 'planning_generate',
          'native_save', 'portable_save', 'saved_history_open')
OUTPUT_FIELDS = ('reconstructed_mesh_file_sha256', 'ordered_route_sha256',
                 'target_arrays', 'target_summary', 'has_coverage')


def plan_values(path):
    with zipfile.ZipFile(path) as archive:
        document = json.loads(archive.read('project.json'))
        trajectory = document['plan']['surface_trajectory']
        ordered = hashlib.sha256(json.dumps(trajectory['ordered_items'], sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        arrays = {}
        with np.load(io.BytesIO(archive.read('target_surface.npz')), allow_pickle=False) as target:
            for name in target.files:
                array = np.ascontiguousarray(target[name])
                arrays[name] = dict(shape=list(array.shape), dtype=str(array.dtype),
                                    sha256=hashlib.sha256(array.tobytes()).hexdigest())
        return dict(ordered_route_sha256=ordered, target_arrays=arrays,
                    target_summary=document['plan']['planning_context']['target']['summary'],
                    has_coverage=bool(document['plan'].get('coverage_evidence')))


def audit(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'))
    rows = json.loads((root/'stages.json').read_text(encoding='utf-8-sig'))
    equivalence = json.loads((root/'equivalence.json').read_text(encoding='utf-8-sig'))
    errors = []
    for key, expected in (('status', 'passed'), ('source_unchanged', True), ('code_unchanged', True)):
        if manifest.get(key) != expected:
            errors.append(f'Manifest {key}: {manifest.get(key)!r}')
    expected_repetitions = set(range(manifest['repetitions']))
    if any(row.get('repetition') not in expected_repetitions for row in rows):
        errors.append('Measured stages contain an undeclared repetition')
    if any(row.get('repetition') not in expected_repetitions for row in equivalence):
        errors.append('Independent replay evidence contains an undeclared repetition')
    for repetition in range(manifest['repetitions']):
        measured = [row for row in rows if row['repetition'] == repetition]
        if [row['stage'] for row in measured] != list(STAGES):
            errors.append(f'Incomplete or duplicate stages in repetition {repetition}')
        if any(row['status'] != 'passed' or row['sampling_errors'] for row in measured):
            errors.append(f'Stage or measurement failure in repetition {repetition}')
        matches = [row for row in equivalence if row['repetition'] == repetition]
        if len(matches) != 1 or not matches[0]['saved_route_equal']:
            errors.append(f'Missing or failed independent replay in repetition {repetition}')
    values = []
    for repetition in range(manifest['repetitions']):
        folder = root/f'run_{repetition}'
        value = plan_values(folder/'result.region.zip')
        value['reconstructed_mesh_file_sha256'] = file_digest(folder/'reconstruction'/'scene_tsdf_mesh_m.ply')
        values.append(value)
    same = all(value == values[0] for value in values)
    if not same:
        errors.append('Repeated reconstruction, route or target values differ; inspect before timing comparison')
    return dict(all_checks_passed=not errors, errors=errors,
                original_project=plan_values(Path(manifest['project'])),
                repetitions=values, repetitions_equivalent=same,
                scope='offline benchmark and exact repeated output comparison; not physical accuracy')


def _sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


def _inventory_valid(value):
    return (isinstance(value, dict) and bool(value)
            and all(isinstance(root, str) and isinstance(files, dict) and bool(files)
                    and all(isinstance(name, str) and _sha256(digest)
                            for name, digest in files.items())
                    for root, files in value.items()))


def _input_comparison(baseline, optimized):
    comparison = {}
    for key in ('project_sha256', 'reconstruction_report_sha256'):
        before, after = baseline.get(key), optimized.get(key)
        comparison[key] = dict(baseline=before, optimized=after,
                               equal=_sha256(before) and _sha256(after) and before == after)
    before, after = baseline.get('input_files'), optimized.get('input_files')
    valid = _inventory_valid(before) and _inventory_valid(after)
    differences = []
    if valid:
        for root in sorted(before.keys() | after.keys()):
            left, right = before.get(root, {}), after.get(root, {})
            for name in sorted(left.keys() | right.keys()):
                if left.get(name) != right.get(name):
                    differences.append(dict(root=root, file=name,
                                            baseline=left.get(name), optimized=right.get(name)))
    comparison['input_files'] = dict(
        equal=bool(valid and before == after), valid=bool(valid),
        baseline_file_count=sum(map(len, before.values())) if _inventory_valid(before) else None,
        optimized_file_count=sum(map(len, after.values())) if _inventory_valid(after) else None,
        differing_files=differences)
    return comparison


def _audited_run(root):
    """Return a failure report even if an output is missing or unreadable."""
    manifest = {}
    try:
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'))
        if not isinstance(manifest, dict):
            raise ValueError('Manifest must be an object')
        repetitions = manifest.get('repetitions')
        if type(repetitions) is not int or repetitions < 1:
            raise ValueError('Manifest repetitions must be a positive integer')
        return manifest, audit(root)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError,
            EOFError, zipfile.BadZipFile) as error:
        return manifest if isinstance(manifest, dict) else {}, dict(
            all_checks_passed=False, errors=[f'{type(error).__name__}: {error}'],
            repetitions=[], repetitions_equivalent=False)


def compare(baseline, optimized):
    """Require both complete runs and exact equality of every measured output."""
    baseline, optimized = Path(baseline).resolve(), Path(optimized).resolve()
    before, left = _audited_run(baseline)
    after, right = _audited_run(optimized)
    errors = [f'{label}: {error}' for label, report in (('baseline', left), ('optimized', right))
              for error in report['errors']]
    inputs = _input_comparison(before, after)
    errors.extend(f'Fixed input mismatch or missing evidence: {key}'
                  for key, check in inputs.items() if not check['equal'])
    left_values, right_values = left['repetitions'], right['repetitions']
    counts_equal = bool(left_values and len(left_values) == len(right_values))
    if not counts_equal:
        errors.append('Baseline and optimized repetition counts are missing or differ')
    rounds = []
    for repetition in range(max(len(left_values), len(right_values))):
        left_value = left_values[repetition] if repetition < len(left_values) else None
        right_value = right_values[repetition] if repetition < len(right_values) else None
        checks = {field: (left_value is not None and right_value is not None
                          and field in left_value and field in right_value
                          and left_value[field] == right_value[field])
                  for field in OUTPUT_FIELDS}
        changed_arrays = []
        if left_value is not None and right_value is not None:
            left_arrays, right_arrays = left_value['target_arrays'], right_value['target_arrays']
            changed_arrays = sorted(name for name in left_arrays.keys() | right_arrays.keys()
                                    if left_arrays.get(name) != right_arrays.get(name))
        rounds.append(dict(repetition=repetition, all_equal=all(checks.values()),
                           checks=checks, differing_target_arrays=changed_arrays))
        errors.extend(f'Repetition {repetition} differs: {field}'
                      for field, equal in checks.items() if not equal)
    original_equal = (left.get('original_project') is not None
                      and left.get('original_project') == right.get('original_project'))
    if not original_equal:
        errors.append('Original project route or target values differ or are unavailable')
    left_code, right_code = before.get('code_sha256', {}), after.get('code_sha256', {})
    code_changes = (sorted(name for name in left_code.keys() | right_code.keys()
                           if left_code.get(name) != right_code.get(name))
                    if isinstance(left_code, dict) and isinstance(right_code, dict) else None)
    return dict(
        schema='workstation-benchmark-comparison-v1',
        all_checks_passed=not errors, errors=errors,
        baseline=str(baseline), optimized=str(optimized),
        fixed_inputs=inputs, original_project_equal=original_equal,
        repetition_counts_equal=counts_equal,
        round_comparisons=rounds, baseline_audit=left, optimized_audit=right,
        changed_production_files=code_changes,
        scope=('Recorded fixed-input hashes and freshly read generated outputs; '
               'code may differ between runs but must remain unchanged within each run. '
               'Generated plans are compared with generated plans, not the original coverage plan. '
               'No timing speedup or physical accuracy claim is made by this audit.'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline', type=Path,
                        help='Compare every repetition against this completed benchmark')
    parser.add_argument('--report', type=Path, help='Override the JSON report destination')
    args = parser.parse_args(argv)
    report = compare(args.baseline, args.output) if args.baseline else audit(args.output)
    destination = args.report or args.output/('comparison.json' if args.baseline else 'audit.json')
    write_json_strict(destination, report)
    print(json.dumps({key: report[key] for key in ('all_checks_passed', 'errors')}, ensure_ascii=False))
    return 0 if report['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
