"""Serial AB/BA/AB complete-save comparisons of two immutable backends."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--v1', type=Path, required=True)
    parser.add_argument('--exact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', choices=('v1', 'exact'), nargs='+', default=['v1', 'exact'])
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'outputs'):
        raise ValueError('Use a new workspace outputs directory')
    output.mkdir(exist_ok=False)
    files = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in (args.before, args.after, args.v1, args.exact,
                          Path(__file__), ROOT/'tools/run_archived_backend.py',
                          ROOT/'tools/profile_saved_plan.py', ROOT/'tools/profile_qt_save.py',
                          ROOT/'tools/save_stage_observation.py', ROOT/'tools/runtime_memory.py')}
    files.update({str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted((ROOT/'tools').rglob('*.py'))})
    manifest = dict(status='running', fixed_files=files, runs=[], cases=args.cases,
        policy='Three alternating pairs per input and tool; serial fresh equal subprocesses; no forced GC; no allocation tracer')
    def save():
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save()
    try:
        for case in args.cases:
            for pair in range(3):
                order = ('before', 'after') if pair % 2 == 0 else ('after', 'before')
                for tool in ('profile_saved_plan', 'profile_qt_save'):
                    for variant in order:
                        name = f'{case}_pair{pair}_{tool}_{variant}'
                        destination = output/name
                        provenance = output/(name+'_backend.json')
                        command = [sys.executable, '-X', 'utf8', str(ROOT/'tools/run_archived_backend.py'),
                            '--archive', str(getattr(args, variant).resolve()), '--tool', tool,
                            '--provenance', str(provenance), '--project', str(getattr(args, case).resolve()),
                            '--output', str(destination)]
                        print(json.dumps(dict(starting=name)), flush=True)
                        started = time.monotonic()
                        with (output/(name+'.log')).open('x', encoding='utf-8') as stream:
                            completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                        result = json.loads((destination/'manifest.json').read_text(encoding='utf-8')) if (destination/'manifest.json').exists() else {}
                        backend = json.loads(provenance.read_text(encoding='utf-8')) if provenance.exists() else {}
                        passed = completed.returncode == 0 and result.get('all_checks_passed') and backend.get('archive_unchanged')
                        row = dict(name=name, case=case, pair=pair, tool=tool, variant=variant,
                            process_seconds=time.monotonic()-started, returncode=completed.returncode,
                            all_checks_passed=bool(passed))
                        manifest['runs'].append(row)
                        save()
                        print(json.dumps(row), flush=True)
                        if not passed:
                            raise RuntimeError('Complete comparison failed: '+name)
        summary = {}
        for case in args.cases:
            result = {}
            for tool, stages in (('profile_saved_plan', ('native_save', 'portable_save')),
                                 ('profile_qt_save', ('qt_complete_save',))):
                for stage in stages:
                    result[stage] = {}
                    for variant in ('before', 'after'):
                        seconds = []
                        for row in manifest['runs']:
                            if (row['case'], row['tool'], row['variant']) != (case, tool, variant):
                                continue
                            measurements = json.loads((output/row['name']/'stages.json').read_text())
                            seconds.extend(m['seconds'] for m in measurements if m['stage'] == stage)
                        if len(seconds) != 3:
                            raise AssertionError('Missing timing evidence')
                        result[stage][variant] = dict(seconds=seconds, median_seconds=statistics.median(seconds),
                            min_seconds=min(seconds), max_seconds=max(seconds))
            result['native_and_complete_medians_improved'] = all(
                result[stage]['after']['median_seconds'] < result[stage]['before']['median_seconds']
                for stage in ('native_save', 'qt_complete_save'))
            summary[case] = result
        (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        manifest['status'] = 'passed'
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        manifest['fixed_files_unchanged'] = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
                                               for path, digest in files.items())
        if not manifest['fixed_files_unchanged']:
            manifest['status'] = 'failed'
        save()


if __name__ == '__main__':
    main()
