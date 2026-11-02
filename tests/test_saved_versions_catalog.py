"""Global saved history includes legacy catalogs without duplicate projects."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from femto.planning.catalog import register_project, scan_saved_versions
from femto.dataset.task_context import new_task
from femto.history_store import HistoryStore


class SavedVersionsCatalogTests(unittest.TestCase):
    def project(self, path, revision, stamp):
        path.parent.mkdir(parents=True, exist_ok=True)
        assets = {name: b'test asset' for name in (
            'scene_tsdf_mesh_m.ply', 'rgb.png', 'depth.npy', 'image_valid.npy')}
        metadata = dict(schema='femto-region-project-v1', project_id='project',
                        revision_id=revision, region={'name': revision},
                        task_context=dict(new_task(), name='Catalog task'),
                        files={name: hashlib.sha256(data).hexdigest()
                               for name, data in assets.items()})
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('project.json', json.dumps(metadata))
            for name, data in assets.items():
                archive.writestr(name, data)
        os.utime(path, (stamp, stamp))
        return path.resolve()

    def test_merges_global_capture_and_workspace_indexes_and_deduplicates(self):
        with TemporaryDirectory() as folder:
            outputs = Path(folder) / 'outputs'
            global_root = outputs / 'saved_projects'
            capture = outputs / 'complete_demo_old' / 'reconstruction'
            workspace = outputs / 'region_workspaces' / 'imported'
            shared = self.project(outputs / 'archives' / 'shared.zip', 'shared', 1000)
            legacy = self.project(capture / 'older.zip', 'capture-only', 2000)
            imported = self.project(workspace / 'imported.zip', 'workspace-only', 3000)
            register_project(global_root, shared)
            register_project(capture, shared)
            register_project(capture, legacy)
            register_project(workspace, imported)

            # An indexed portable project replaces its adjacent native route in
            # the global list, so saving once does not produce two choices.
            paired = self.project(capture / 'surface_paths' / 'paired' /
                                  'planning.region.zip', 'paired', 4000)
            register_project(capture, paired)
            (paired.parent / 'trajectory.json').write_text(
                json.dumps({'schema': 'femto-surface-trajectory-v1'}), encoding='utf-8')

            before = {path: path.read_bytes() for path in (shared, legacy, imported, paired)}
            rows = scan_saved_versions(outputs)
            self.assertEqual([row['project'] for row in rows],
                             [paired, imported, legacy, shared])
            self.assertEqual([row['revision_id'] for row in rows],
                             ['paired', 'workspace-only', 'capture-only', 'shared'])
            self.assertTrue(all(row['file_status'] == '工程资产指纹一致' for row in rows))
            self.assertTrue(all(row['name'].startswith('Catalog task · ') for row in rows))
            self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_scan_honors_cancellation(self):
        with TemporaryDirectory() as folder:
            def cancel():
                raise TimeoutError('cancelled')
            with self.assertRaisesRegex(TimeoutError, 'cancelled'):
                scan_saved_versions(folder, check=cancel)

    def test_missing_or_corrupt_shared_project_does_not_break_sqlite_refresh(self):
        for state in ('文件丢失', '损坏'):
            with self.subTest(state=state), TemporaryDirectory() as folder:
                outputs = Path(folder) / 'outputs'
                global_root = outputs / 'saved_projects'
                capture = outputs / 'complete_demo_old' / 'reconstruction'
                shared = self.project(outputs / 'archives' / 'shared.zip', 'shared', 1000)
                intact = self.project(capture / 'intact.zip', 'intact', 2000)
                register_project(global_root, shared)
                register_project(capture, shared)
                register_project(capture, intact)
                if state == '文件丢失':
                    shared.unlink()
                else:
                    shared.write_bytes(b'corrupt archive')
                before = {path: path.read_bytes() for path in outputs.rglob('*') if path.is_file()}

                rows = scan_saved_versions(outputs)
                self.assertEqual(len(rows), 2)
                damaged = next(row for row in rows if row['revision_id'] == 'shared')
                self.assertEqual(damaged['file_status'], state)
                self.assertIsNone(damaged['project'])
                self.assertEqual(damaged['project_location'], shared)
                self.assertEqual(next(row for row in rows if row['revision_id'] == 'intact')['project'],
                                 intact)
                self.assertEqual({path: path.read_bytes() for path in before}, before)

                store = HistoryStore(outputs)
                store.replace('exports:all', rows, search_text=lambda row: row['name'])
                self.assertEqual(store.query('exports:all')['rows'], rows)


if __name__ == '__main__':
    unittest.main()
