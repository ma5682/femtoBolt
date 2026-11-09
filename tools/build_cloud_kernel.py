"""Explicit offline build of this checkout's optional Windows cloud kernel.

Requires the current Python's headers, setuptools and an installed MSVC x64
toolchain. Does not install packages, change other environments or build an EXE.
Build products and the exact input/toolchain/command hashes are recorded under
outputs/. The compiled extension is the only generated file placed in femto/.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import sysconfig


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path)
    args = parser.parse_args()
    if sys.platform != 'win32' or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise RuntimeError('This build recipe is validated only for Windows x64')
    import setuptools
    from setuptools import Distribution, Extension
    from setuptools.command.build_ext import build_ext

    source = ROOT / 'native/cloud_predicate.c'
    directory = args.build_dir or ROOT / 'outputs' / ('cloud_kernel_build_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    calls = []

    class RecordedBuild(build_ext):
        def build_extensions(self):
            if self.compiler.compiler_type != 'msvc':
                raise RuntimeError('A strict MSVC build is required')
            call = self.compiler.call

            def recorded(command, **kwargs):
                calls.append([str(part) for part in command])
                return call(command, **kwargs)

            self.compiler.call = recorded
            super().build_extensions()

    source_sha = sha256(source)
    # /fp:strict forbids reassociation and contraction into fused multiply-add.
    flags = ['/O2', '/fp:strict', '/W4', '/WX']
    extension = Extension('femto.geometry._cloud_predicate', [str(source)],
        define_macros=[('FEMTO_STRICT_FP', '1'), ('FEMTO_SOURCE_SHA256', '"' + source_sha + '"')],
        extra_compile_args=flags)
    dist = Distribution(dict(name='femto-cloud-predicate', version='1.0', ext_modules=[extension],
                             cmdclass={'build_ext': RecordedBuild}))
    command = dist.get_command_obj('build_ext')
    command.build_lib = str(directory / 'stage')
    command.build_temp = str(directory / 'objects')
    command.force = True
    command.ensure_finalized()
    command.run()
    staged = Path(command.get_ext_fullpath(extension.name))
    if sha256(source) != source_sha:
        raise RuntimeError('Native source changed during the build; discard this result')
    binary = ROOT / 'femto/geometry' / staged.name
    record = dict(schema='femto-cloud-kernel-build-v1',
        source=str(source), source_sha256=source_sha,
        build_script_sha256=sha256(__file__), binary=str(binary), binary_sha256=sha256(staged),
        compiler=command.compiler.cc, compiler_sha256=sha256(command.compiler.cc),
        commands=calls, flags=flags, python=sys.version, python_abi=sysconfig.get_config_var('SOABI'),
        platform=platform.platform(), setuptools=setuptools.__version__)
    manifest = directory / 'build.json'
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    # Publish only a complete build. A loaded/locked older extension is retained
    # if Windows refuses the atomic replacement; no process is stopped here.
    temporary = binary.with_name(binary.name + '.' + directory.name + '.partial')
    try:
        shutil.copyfile(staged, temporary)
        os.replace(temporary, binary)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(dict(binary=str(binary), binary_sha256=record['binary_sha256'],
                         source_sha256=source_sha, manifest=str(manifest)), ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
