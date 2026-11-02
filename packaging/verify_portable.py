"""Test the frozen executable, zip only distributable files, then test extraction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--extract', type=Path, required=True)
    parser.add_argument('--source-test-report', type=Path)
    args = parser.parse_args()
    source_tests = None
    if args.source_test_report:
        source_tests = json.loads(args.source_test_report.read_text(encoding='utf-8'))
        assert source_tests['status'] == 'passed' and source_tests['source_unchanged']
    environment = dict(os.environ)
    win = Path(os.environ.get('SystemRoot', r'C:\Windows'))
    environment['PATH'] = os.pathsep.join(map(str, [win / 'System32', win]))
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
        environment.pop(key, None)
    environment['QT_QPA_PLATFORM'] = 'offscreen'

    def smoke(bundle):
        subprocess.run([str(bundle / 'FemtoWorkstation.exe'), '--package-smoke-test'],
            cwd=bundle.parent, env=environment, timeout=90, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW)
        result = json.loads((bundle / 'Log' / 'package-smoke.json').read_text(encoding='utf-8'))
        assert result['status'] == 'passed' and result['frozen']
        assert Path(result['workspace']) == bundle.resolve()
        assert 'comparison_worker_entry' in result['checks']
        assert 'reduced_planning_reference' in result['checks']
        return result

    original = smoke(args.bundle)
    excluded = {'Log', 'outputs', 'datasets', 'photos', '.data_maintenance.lock'}
    paths = [p for p in args.bundle.rglob('*') if p.is_file()
             and p.relative_to(args.bundle).parts[0] not in excluded]
    assert not any('gitee_credentials' in p.name or '.venv' in p.parts or '.git' in p.parts for p in paths)
    with zipfile.ZipFile(args.archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, Path(args.bundle.name) / path.relative_to(args.bundle))
    args.extract.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.archive) as archive:
        assert archive.testzip() is None
        archive.extractall(args.extract)
    relocated = smoke(args.extract / args.bundle.name)
    digest = hashlib.file_digest(args.archive.open('rb'), 'sha256').hexdigest()
    report = {'status': 'passed', 'original': original, 'relocated': relocated,
              'archive': str(args.archive), 'sha256': digest,
              'zip_bytes': args.archive.stat().st_size,
              'uncompressed_bytes': sum(p.stat().st_size for p in paths), 'files': len(paths),
              'clean_path': environment['PATH'],
              'physical_camera_tested': False, 'second_computer_tested': False}
    if source_tests is not None:
        report['source_tests_passed'] = source_tests['tests_run']
        report['source_test_report'] = str(args.source_test_report.resolve())
    report_path = args.archive.with_suffix('.verification.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    args.archive.with_suffix('.sha256.txt').write_text(digest + '  ' + args.archive.name + '\n', encoding='utf-8')
    shutil.copy2(args.extract / args.bundle.name / 'Log' / 'package-preview.png', args.archive.with_suffix('.preview.png'))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
