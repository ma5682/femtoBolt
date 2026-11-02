"""P1 assets participate in the existing atomic project/version transaction."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import open3d as o3d

import test_surface_workspace as workspace_fixture
from femto.app.qt_planning_io import clone_model
from femto.planning import planning_revision as revision, planning_state, target_surface as target
from femto.planning import surface_workspace as workspace
from femto.planning.region_project import save_project, open_project


class PlanningPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = workspace_fixture.SurfaceWorkspaceTests(); cls.fixture.setUp()
        cls.model = cls.fixture.model; cls.region = cls.fixture.region()
        cls.plan = workspace.generate(cls.model, cls.region, cls.fixture.head)
        o3d.io.write_triangle_mesh(cls.model.store.source, cls.model.mesh)
        cls.base = Path(cls.fixture.temp.name)/'baseline.region.zip'
        save_project(cls.model, cls.region, cls.base, cls.plan)
        cls.original = cls.base.read_bytes()

    @classmethod
    def tearDownClass(cls):cls.fixture.doCleanups()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def rewrite(self, change):
        with zipfile.ZipFile(self.base) as archive:
            assets = {name: archive.read(name) for name in archive.namelist()}
        data = json.loads(assets['project.json'])
        change(data, assets)
        assets['project.json'] = json.dumps(data, ensure_ascii=False).encode('utf-8')
        destination = self.folder/'changed.region.zip'
        with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, value in assets.items():archive.writestr(name, value)
        return destination

    def test_zip_restores_target_inputs_and_keeps_save_uuid_separate(self):
        restored = open_project(self.base, workspace=self.folder/'opened')
        self.assertEqual(restored.target_ledger.metadata, self.plan['planning_context']['target'])
        self.assertEqual(restored.saved_plan['planning_context'], self.plan['planning_context'])
        old_uuid = restored.revision_id
        path = self.folder/'next.region.zip'
        save_project(restored, restored.store.regions[0], path, restored.saved_plan)
        with zipfile.ZipFile(path) as archive:data = json.loads(archive.read('project.json'))
        self.assertNotEqual(data['revision_id'], old_uuid)
        self.assertEqual(data['parent_revision_id'], old_uuid)
        self.assertEqual(data['plan']['planning_context']['revision'], self.plan['planning_context']['revision'])
        self.assertEqual(self.base.read_bytes(), self.original)

    def test_rehashed_outer_archive_still_rejects_corrupt_target_arrays(self):
        def corrupt(data, assets):
            assets[target.ASSET] = b'not an array archive'
            data['files'][target.ASSET] = hashlib.sha256(assets[target.ASSET]).hexdigest()
        path = self.rewrite(corrupt)
        with self.assertRaises((ValueError, zipfile.BadZipFile)):
            open_project(path, workspace=self.folder/'opened')
        self.assertEqual(list((self.folder/'opened').iterdir()), [])
        self.assertEqual(self.base.read_bytes(), self.original)

    def test_mixed_revision_target_cannot_be_rehashed_into_valid_plan(self):
        def corrupt(data, assets):
            plan = data['plan']
            context = deepcopy(plan['planning_context'])
            inputs = deepcopy(context['revision']['inputs'])
            inputs['quality']['fake_new_threshold'] = True
            context['revision'] = revision.create(inputs)
            plan['planning_context'] = context
            plan['surface_trajectory']['planning_context'] = deepcopy(context)
            plan['plan_binding'] = planning_state.binding(self.model.mesh, self.region, plan)
        path = self.rewrite(corrupt)
        with self.assertRaises(ValueError):open_project(path, workspace=self.folder/'opened')

    def test_asset_write_failure_preserves_previous_archive_and_save_version(self):
        destination = self.folder/'previous.region.zip'; destination.write_bytes(self.original)
        model = clone_model(self.model)
        before = model.revision_id
        with patch('femto.planning.target_surface.write', side_effect=OSError('synthetic disk failure')):
            with self.assertRaisesRegex(OSError, 'synthetic disk failure'):
                save_project(model, self.region, destination, self.plan)
        self.assertEqual(destination.read_bytes(), self.original)
        self.assertEqual(model.revision_id, before)
        self.assertFalse(any(p.name.startswith('.planning') for p in self.folder.iterdir()))

    def test_legacy_v3_remains_legacy_without_silent_inventory_upgrade(self):
        old = deepcopy(self.plan)
        del old['planning_context']; del old['surface_trajectory']['planning_context']
        old['plan_binding'] = planning_state.binding(self.model.mesh, self.region, old)
        destination = self.folder/'legacy.region.zip'
        save_project(clone_model(self.model), self.region, destination, old)
        restored = open_project(destination, workspace=self.folder/'opened')
        self.assertIsNone(restored.target_ledger)
        self.assertNotIn('planning_context', restored.saved_plan)
        self.assertIn('旧记录', restored.plan_warning)
        self.assertEqual(restored.saved_plan['segments'], old['segments'])


if __name__ == '__main__':unittest.main()
