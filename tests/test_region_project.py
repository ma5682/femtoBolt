import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
import open3d as o3d
from types import SimpleNamespace
from scene_annotations import AnnotationStore
from fixed_rgb_geometry import FixedProjection
from region_project import save_project,open_project
from region_coverage import coverage_plan


class ProjectTests(unittest.TestCase):
    def test_relocated_project_without_original_and_saved_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'original.ply'
            mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector([[0,0,.5],[.04,0,.5],[.04,.04,.5],[0,.04,.5]]),o3d.utility.Vector3iVector([[0,1,2],[0,2,3]]))
            o3d.io.write_triangle_mesh(str(source),mesh)
            store=AnnotationStore(mesh,source)
            view={'reference_frame_id':0,'session_id':'test','rgb_sha256':'original-reference'}
            region=store.add_region(dict(triangle_indices=[0,1],screen_polygon=[[0,0],[40,0],[40,40],[0,40]],view_reference=view))
            projection=FixedProjection(mesh,np.diag([500.,500.,1.]),np.full((50,50),500,np.uint16))
            model=SimpleNamespace(store=store,projection=projection,rgb=np.full((50,50,3),100,np.uint8),view_reference=view,max_radius_mm=0.)
            plan=coverage_plan(projection,mesh,region,4,20,boundary_margin_mm=0)
            bundle=save_project(model,region,root/'a.zip',plan)
            from export_history import scan_exports,register_project
            self.assertEqual(scan_exports(root)[0]['project'],bundle.resolve())
            register_project(root,bundle)
            self.assertEqual(len([r for r in scan_exports(root) if r['kind']=='区域工程 / 方案']),1)
            source.unlink()
            moved=root/'moved.zip';bundle.rename(moved)
            self.assertIsNone(scan_exports(root)[0]['project'])
            loaded=open_project(moved,root/'work')
            np.testing.assert_allclose(np.asarray(loaded.mesh.vertices),np.asarray(mesh.vertices))
            self.assertEqual(loaded.saved_plan['segments'],plan['segments'])
            self.assertEqual(loaded.store.regions[0]['annotation_id'],region['annotation_id'])
            self.assertTrue(loaded.projection.valid.any())
            save_project(loaded,loaded.store.regions[0],root/'second.zip',loaded.saved_plan)
            reopened=open_project(root/'second.zip',root/'second-work')
            self.assertEqual(reopened.project_id,loaded.project_id)

    def test_archive_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.zip'
            with zipfile.ZipFile(path,'w') as z:z.writestr('../outside','bad');z.writestr('project.json','{}')
            with self.assertRaises(ValueError):open_project(path,Path(tmp)/'work')
            self.assertFalse((Path(tmp)/'outside').exists())


if __name__=='__main__':unittest.main()
