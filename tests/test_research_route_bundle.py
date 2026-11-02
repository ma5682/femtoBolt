"""Integrity and non-overwrite checks for the standalone research bundle."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('research_route_bundle', ROOT/'femto/planning/research_route_bundle.py')
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)

class BundleTests(unittest.TestCase):
    def test_reject_unexpected_and_duplicate_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('../outside', b'x')
            with self.assertRaisesRegex(ValueError, '清单'):
                bundle.verify(path)
            with zipfile.ZipFile(path, 'w') as archive:
                for name in bundle.FILES:
                    archive.writestr(name, b'')
                archive.writestr('manifest.json', b'{}')
                with self.assertWarns(UserWarning):
                    archive.writestr('manifest.json', b'{}')
            with self.assertRaisesRegex(ValueError, '清单'):
                bundle.verify(path)

    def test_reject_duplicate_json_fields(self):
        with self.assertRaises(ValueError):
            bundle.read_json(b'{"schema":1,"schema":2}')

    def test_reject_changed_asset_before_opening_projects(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'tampered.zip'
            records = {name: {'size': 1, 'sha256': bundle.digest(b'x')} for name in bundle.FILES}
            manifest = dict(schema=bundle.SCHEMA, files=records, **bundle.FLAGS)
            with zipfile.ZipFile(path, 'w') as archive:
                for name in bundle.FILES:
                    archive.writestr(name, b'y' if name == 'recipe.json' else b'x')
                archive.writestr('manifest.json', json.dumps(manifest))
            with patch.object(bundle, 'open_project') as open_project:
                with self.assertRaisesRegex(ValueError, '摘要不匹配'):
                    bundle.verify(path)
                open_project.assert_not_called()

    def test_never_overwrite_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)/'existing.zip'
            destination.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                bundle.pack(directory, destination)
            self.assertEqual(destination.read_bytes(), b'keep')

    def test_failed_replay_never_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            for name in bundle.FILES:
                path = source/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'x')
            destination = root/'result.zip'
            with patch.object(bundle, 'verify', side_effect=ValueError('replay failed')):
                with self.assertRaisesRegex(ValueError, 'replay failed'):
                    bundle.pack(source, destination)
            self.assertFalse(destination.exists())

if __name__ == '__main__': unittest.main()
