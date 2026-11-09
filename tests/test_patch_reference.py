"""Full reference/intent preservation and self-contained source verification."""
from pathlib import Path
import unittest
import numpy as np
import test_qt_planning as fixture
from femto.planning.patch_reference import create,load,SOURCE_ASSET
from femto.planning.region_project import save_project,open_project
from femto.app.qt_planning_io import extract_region,masks_from_region


class PatchReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixture.QtPlanningTests.setUpClass()

    def setUp(self):
        self.case=fixture.QtPlanningTests();self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.source=self.case.model;self.root=Path(self.case.temp.name)

    def test_full_reference_region_and_original_mesh_survive_portable_reopen(self):
        full_bytes=Path(self.source.store.source).read_bytes()
        model=create(self.source,self.root/'patch')
        np.testing.assert_array_equal(model.rgb,self.source.rgb)
        np.testing.assert_array_equal(model.projection.depth,self.source.projection.depth)
        selected=np.zeros(model.rgb.shape[:2],np.uint8);selected[25:50,20:42]=1;selected[25:50,60:82]=1
        model,region=extract_region(model,selected,np.zeros_like(selected),'split','',None)
        save_project(model,region,self.root/'test.region.zip')
        reopened=open_project(self.root/'test.region.zip',workspace=self.root/'opened')
        np.testing.assert_array_equal(masks_from_region(reopened.store.regions[0],selected.shape)[0],selected)
        self.assertEqual((reopened.reconstruction/SOURCE_ASSET).read_bytes(),full_bytes)
        self.assertEqual(Path(self.source.store.source).read_bytes(),full_bytes)
        self.assertEqual(reopened.source_surface_check,model.source_surface_check)

    def test_derived_reload_checks_source_asset_and_reference_arrays(self):
        model=create(self.source,self.root/'patch')
        restored=load(model.reconstruction)
        np.testing.assert_array_equal(restored.projection.valid,model.projection.valid)
        source=model.reconstruction/SOURCE_ASSET
        content=source.read_bytes();source.write_bytes(content+b'changed')
        with self.assertRaisesRegex(ValueError,'完整原始表面'):load(model.reconstruction)
        source.write_bytes(content)
        depth=model.reconstruction/'reference_depth.npy';values=np.load(depth);values[30,30]+=1;np.save(depth,values)
        with self.assertRaisesRegex(ValueError,'绑定不一致'):load(model.reconstruction)

    def test_raw_missing_depth_cannot_become_supported(self):
        self.source.projection.depth[35,35]=0
        model=create(self.source,self.root/'patch')
        self.assertFalse(model.projection.valid[35,35])
        self.assertFalse(model.projection.new_support[35,35])

    def test_native_save_rejects_changed_full_source_before_route_export(self):
        from femto.planning.surface_workspace import export
        model=create(self.source,self.root/'patch')
        source=model.reconstruction/SOURCE_ASSET;source.write_bytes(source.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'完整原表面已改变'):export(model,{}, {})


if __name__=='__main__':unittest.main()
