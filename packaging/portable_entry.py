"""Frozen workstation entry, including the existing maintenance subprocess."""
import multiprocessing
import runpy
import sys
from pathlib import Path


def main():
    multiprocessing.freeze_support()
    root = Path(sys.executable).resolve().parent
    (root / 'Log').mkdir(exist_ok=True)
    if sys.stdout is None:
        sys.stdout = (root / 'Log' / 'launcher.log').open('a', encoding='utf-8', buffering=1)
    if sys.stderr is None:
        sys.stderr = sys.stdout
    args = sys.argv[1:]
    if args[:2] == ['-X', 'utf8']:
        args = args[2:]
    worker_modules = {'femto.app.qt_data_maintenance',
                      'femto.pipeline.local_offline', 'femto.pipeline.local_comparison'}
    if len(args) >= 2 and args[0] == '-m':
        if args[1] not in worker_modules:
            raise ValueError('Unsupported bundled worker: ' + args[1])
        module = args[1]
        sys.argv = [sys.argv[0], *args[2:]]
        runpy.run_module(module, run_name='__main__')
    elif sys.argv[1:2] == ['--package-smoke-test']:
        from portable_smoke import main as smoke
        smoke()
    else:
        from femto.app.qt_main import main as workstation
        raise SystemExit(workstation())


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
