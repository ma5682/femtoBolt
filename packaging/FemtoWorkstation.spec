from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
datas = collect_data_files('femto') + collect_data_files('pyorbbecsdk')
for name in ('PySide6', 'numpy', 'open3d', 'opencv-contrib-python', 'mediapipe', 'vtk', 'pyorbbecsdk2'):
    datas += copy_metadata(name)
hidden = collect_submodules('femto') + collect_submodules('pyorbbecsdk')
a = Analysis([str(root / 'packaging' / 'portable_entry.py')],
    pathex=[str(root), str(root / 'packaging')],
    binaries=collect_dynamic_libs('pyorbbecsdk'), datas=datas,
    hiddenimports=hidden, hookspath=[], runtime_hooks=[],
    excludes=['torch', 'tensorflow', 'IPython', 'jupyter', 'pytest'], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='FemtoWorkstation',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='FemtoWorkstation')
