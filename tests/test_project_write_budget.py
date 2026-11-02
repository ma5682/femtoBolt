"""Project manifests obey cancellation/size limits without losing saved evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import open3d as o3d
import test_surface_workspace as fixture
from femto.app.qt_planning_io import clone_model
from femto.annotation.fixed_rgb_geometry import GeometryTaskCancelled
from femto.operation_diagnostics import record_operation
from femto.planning import region_project, surface_workspace


class ProjectWriteBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=fixture.SurfaceWorkspaceTests();cls.fixture.setUp()
        cls.model=cls.fixture.model;cls.region=cls.fixture.region()
        cls.plan=surface_workspace.generate(cls.model,cls.region,cls.fixture.head)
        o3d.io.write_triangle_mesh(cls.model.store.source,cls.model.mesh)
        cls.base=Path(cls.fixture.temp.name)/'original.region.zip'
        region_project.save_project(cls.model,cls.region,cls.base,cls.plan)
        cls.original=cls.base.read_bytes()

    @classmethod
    def tearDownClass(cls):cls.fixture.doCleanups()

    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.folder=Path(temporary.name)
        self.destination=self.folder/'original.region.zip';self.destination.write_bytes(self.original)
        self.copy=clone_model(self.model);self.revision=self.copy.revision_id

    def preserved(self):
        self.assertEqual(self.destination.read_bytes(),self.original)
        self.assertEqual(self.copy.revision_id,self.revision)
        self.assertEqual(list(self.folder.iterdir()),[self.destination])

    def test_manifest_size_limit_preserves_old_archive_and_revision(self):
        with patch.object(region_project,'MAX_PROJECT_JSON_BYTES',100):
            with self.assertRaisesRegex(ValueError,'完整工程清单超过'):
                region_project.save_project(self.copy,self.region,self.destination,self.plan)
        self.preserved()

    def test_mid_manifest_cancel_preserves_old_archive_and_revision(self):
        def check():
            if any(p.stat().st_size for p in self.folder.rglob('.project.json-*.tmp')):
                raise GeometryTaskCancelled('manifest cancelled')
        with self.assertRaisesRegex(GeometryTaskCancelled,'manifest cancelled'):
            region_project.save_project(self.copy,self.region,self.destination,self.plan,check)
        self.preserved()

    def test_open_manifest_cancel_removes_incomplete_workspace_and_keeps_source(self):
        root=self.folder/'opened'
        class Cancel:
            def is_set(self):return any(p.stat().st_size for p in root.rglob('.project.json-*.tmp'))
        with self.assertRaises(GeometryTaskCancelled):
            region_project.open_project(self.base,workspace=root,cancel=Cancel())
        self.assertEqual(list(root.iterdir()),[])
        self.assertEqual(self.base.read_bytes(),self.original)

    def test_manifest_write_error_never_updates_existing_archive(self):
        from femto import io_utils
        original=io_utils.os.replace
        def fail(source,destination):
            if Path(source).name.startswith('.project.json-'):raise OSError('manifest publication failed')
            return original(source,destination)
        with patch('femto.io_utils.os.replace',side_effect=fail):
            with self.assertRaisesRegex(OSError,'manifest publication failed'):
                region_project.save_project(self.copy,self.region,self.destination,self.plan)
        self.preserved()

    def test_open_validation_plan_cancel_keeps_source_and_cleans_workspace(self):
        root=self.folder/'opened'
        class Cancel:
            def is_set(self):return any(p.stat().st_size for p in root.rglob('.candidate_path.json-*.tmp'))
        with self.assertRaises(GeometryTaskCancelled):
            region_project.open_project(self.base,workspace=root,cancel=Cancel())
        self.assertEqual(list(root.iterdir()),[])
        self.assertEqual(self.base.read_bytes(),self.original)

    def test_saved_manifest_bytes_and_full_round_trip_with_diagnostic_stages(self):
        destination=self.folder/'new.region.zip';frozen=deepcopy(self.plan)
        with record_operation('save_test',root=self.folder/'diagnostics') as operation:
            region_project.save_project(self.copy,self.region,destination,self.plan)
        meta=json.loads((operation.directory/'operation.json').read_text(encoding='utf-8'))
        self.assertEqual(meta['status'],'complete')
        for stage in ('portable.manifest','portable.archive'):
            self.assertEqual(meta['stages'][stage]['calls'],1)
        with zipfile.ZipFile(destination) as archive:
            raw=archive.read('project.json');payload=json.loads(raw)
        self.assertEqual(raw,json.dumps(payload,ensure_ascii=False,indent=None,allow_nan=False).encode('utf-8'))
        restored=region_project.open_project(destination,workspace=self.folder/'opened')
        self.assertEqual(restored.saved_plan['surface_trajectory'],frozen['surface_trajectory'])
        self.assertEqual(restored.target_ledger.metadata,frozen['planning_context']['target'])
        self.assertEqual(self.plan,frozen)


if __name__=='__main__':unittest.main()
