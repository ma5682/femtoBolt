"""Build an explicitly allowlisted portable test distribution."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import importlib.util
import os

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--assemble-only', action='store_true')
    args = parser.parse_args()
    if not args.assemble_only:
        environment = dict(os.environ)
        win = Path(os.environ.get('SystemRoot', r'C:\Windows'))
        environment['PATH'] = os.pathsep.join(map(str, [win / 'System32', win,
            Path(sys.executable).parent, Path(sys.base_prefix), Path(sys.base_prefix) / 'DLLs']))
        environment.pop('PYTHONPATH', None)
        environment.pop('PYTHONHOME', None)
        subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
                        '--distpath', str(args.output), '--workpath', str(args.work),
                        str(ROOT / 'packaging' / 'FemtoWorkstation.spec')], cwd=ROOT, env=environment, check=True)
    bundle = args.output / 'FemtoWorkstation'
    if not (bundle / 'FemtoWorkstation.exe').is_file():
        raise FileNotFoundError('Build must succeed before assembly')
    # Qt ships a newer MSVC runtime than Python. The loader searches the bundle
    # root first; use Qt's complete matching runtime there to avoid old exports.
    qt = Path(importlib.util.find_spec('PySide6').origin).parent
    for pattern in ('msvcp140*.dll', 'vcruntime140*.dll', 'concrt140.dll'):
        for dll in qt.glob(pattern):
            shutil.copy2(dll, bundle / '_internal' / dll.name)
    # MediaPipe loads this library via importlib.resources, not an extension import.
    mp = Path(importlib.util.find_spec('mediapipe').origin).parent
    shutil.copytree(mp / 'tasks' / 'c', bundle / '_internal' / 'mediapipe' / 'tasks' / 'c', dirs_exist_ok=True)
    o3d = Path(importlib.util.find_spec('open3d').origin).parent
    if (o3d / 'resources').is_dir():
        shutil.copytree(o3d / 'resources', bundle / '_internal' / 'open3d' / 'resources', dirs_exist_ok=True)
    for name in ('licenses', 'models'):
        shutil.copytree(ROOT / name, bundle / name, dirs_exist_ok=True)
    for name in ('THIRD_PARTY_NOTICES.md', 'SOURCE_PROVENANCE.md'):
        shutil.copy2(ROOT / name, bundle / name)
    shutil.copy2(ROOT / 'packaging' / '使用说明.txt', bundle / '使用说明.txt')
    (bundle / '无相机预览.cmd').write_text('@echo off\r\ncd /d "%~dp0"\r\nstart "" "%~dp0FemtoWorkstation.exe" --ui-preview\r\n', encoding='ascii')
    print(bundle)


if __name__ == '__main__':
    main()
