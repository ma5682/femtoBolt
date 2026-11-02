"""Run current measurement tools with an explicitly archived femto backend.

Only Python implementation bytes are redirected. Logical filenames remain at
the project root so existing resource paths retain their normal meaning.
The archive and each module digest are recorded separately from disk hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class ArchivedBackend(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, archive):
        self.sources = {}
        with zipfile.ZipFile(archive) as source:
            for name in source.namelist():
                if name.startswith('femto/') and name.endswith('.py'):
                    self.sources[name] = source.read(name)
        if 'femto/__init__.py' not in self.sources:
            raise ValueError('Archive has no complete femto package')
        self.loaded = {}

    def entry(self, fullname):
        base = fullname.replace('.', '/')
        package = base+'/__init__.py'
        return package if package in self.sources else base+'.py'

    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'femto' and not fullname.startswith('femto.'):
            return None
        name = self.entry(fullname)
        if name not in self.sources:
            raise ImportError('Archived backend is missing '+fullname)
        return importlib.util.spec_from_loader(fullname, self,
            origin=str(ROOT/name), is_package=name.endswith('/__init__.py'))

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = self.entry(module.__name__)
        module.__file__ = str(ROOT/name)
        if name.endswith('/__init__.py'):
            module.__path__ = [str((ROOT/name).parent)]
        self.loaded[name] = hashlib.sha256(self.sources[name]).hexdigest()
        exec(compile(self.sources[name], module.__file__, 'exec'), module.__dict__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--tool', choices=('profile_saved_plan', 'profile_qt_save'), required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.provenance.exists():
        raise FileExistsError(args.provenance)
    archive = args.archive.resolve()
    before = hashlib.sha256(archive.read_bytes()).hexdigest()
    loader = ArchivedBackend(archive)
    sys.meta_path.insert(0, loader)
    sys.path.insert(0, str(ROOT))
    sys.argv = [str(ROOT/'tools'/f'{args.tool}.py'), *remaining]
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        result = dict(archive=str(archive), archive_sha256=before,
            archive_unchanged=hashlib.sha256(archive.read_bytes()).hexdigest() == before,
            loaded_modules=loader.loaded,
            policy='Archived implementation; measurement manifests separately hash current disk files')
        args.provenance.parent.mkdir(parents=True, exist_ok=True)
        args.provenance.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
