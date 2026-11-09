"""Native staging keeps Unicode destinations, failures and same-volume moves."""
import errno
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase,main
from unittest.mock import patch
import numpy as np
import open3d as o3d
from femto.geometry import native_io,fusion


class NativeIOStagingTests(TestCase):
    def test_unique_ascii_names_and_writable_volume_fallback(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);fallback=root/'fallback';same=root/'same';same.mkdir()
            target=root/'中文目录'/'点云.ply'
            with patch.object(native_io,'_volume_cache',return_value=same):
                a=native_io.temporary_path(target,fallback)
                b=native_io.temporary_path(target,fallback)
            self.assertNotEqual(a,b)
            self.assertTrue(str(a).isascii());self.assertEqual(a.parent,same)
            for error in (None,PermissionError('volume root unavailable')):
                with patch.object(native_io,'_volume_cache',return_value=None,side_effect=error):
                    self.assertEqual(native_io.temporary_path(target,fallback).parent,fallback)
            reserve=native_io.tempfile.mkstemp
            def readonly(*args,**kwargs):
                if Path(kwargs['dir'])==same:raise PermissionError('staging is readonly')
                return reserve(*args,**kwargs)
            with patch.object(native_io,'_volume_cache',return_value=same),patch.object(native_io.tempfile,'mkstemp',side_effect=readonly):
                self.assertEqual(native_io.temporary_path(target,fallback).parent,fallback)

    def test_same_volume_replace_and_cross_volume_move_fallback_preserve_bytes(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);source=root/'stage.ply';target=root/'结果.ply'
            source.write_bytes(b'new');target.write_bytes(b'old')
            with patch.object(native_io.shutil,'move',side_effect=AssertionError('must replace')):
                native_io.publish(source,target)
            self.assertEqual(target.read_bytes(),b'new');self.assertFalse(source.exists())
            source.write_bytes(b'fallback')
            with patch.object(native_io.os,'replace',side_effect=OSError(errno.EXDEV,'different volume')):
                native_io.publish(source,target)
            self.assertEqual(target.read_bytes(),b'fallback');self.assertFalse(source.exists())

    def test_cloud_roundtrip_and_failed_write_cleanup_preserve_existing_target(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);stage=root/'stage';stage.mkdir()
            cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector([[0,0,1],[.01,0,1]]))
            target=root/'中文'/'点云.ply'
            with patch.object(native_io,'_volume_cache',return_value=stage):
                self.assertTrue(fusion.write_point_cloud_safe(target,cloud))
                old=target.read_bytes()
                read=fusion.read_geometry_safe(target,mesh=False)
                np.testing.assert_array_equal(np.asarray(read.points),np.asarray(cloud.points))
                def failed(path,*args,**kwargs):
                    Path(path).write_bytes(b'partial');return False
                with patch.object(fusion.o3d.io,'write_point_cloud',side_effect=failed):
                    self.assertFalse(fusion.write_point_cloud_safe(target,cloud))
                self.assertEqual(target.read_bytes(),old)
                self.assertFalse(list(stage.iterdir()))

                def raised(path,*args,**kwargs):
                    Path(path).write_bytes(b'partial');raise OSError('native writer failed')
                with patch.object(fusion.o3d.io,'write_point_cloud',side_effect=raised):
                    with self.assertRaisesRegex(OSError,'native writer failed'):
                        fusion.write_point_cloud_safe(target,cloud)
                self.assertEqual(target.read_bytes(),old)
                self.assertFalse(list(stage.iterdir()))

                def copy_failed(source,destination):
                    Path(destination).write_bytes(b'partial');raise OSError('staging copy failed')
                with patch.object(fusion.shutil,'copy2',side_effect=copy_failed):
                    with self.assertRaisesRegex(OSError,'staging copy failed'):
                        fusion.read_geometry_safe(target,mesh=False)
                self.assertEqual(target.read_bytes(),old)
                self.assertFalse(list(stage.iterdir()))

    def test_optional_mesh_formats_preserve_geometry_and_clean_staging(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);stage=root/'stage';stage.mkdir()
            mesh=o3d.geometry.TriangleMesh.create_box()
            mesh.compute_vertex_normals();mesh.paint_uniform_color([.1,.3,.7])
            with patch.object(native_io,'_volume_cache',return_value=stage):
                result=fusion.export_mesh_formats(mesh,root/'中文结果',write_obj_glb=True)
                self.assertTrue(result['intermediate_obj_glb_written'])
                for kind in ('ply','obj'):
                    restored=fusion.read_geometry_safe(Path(result[kind]),mesh=True)
                    # The OBJ reader may renumber vertices. The oriented
                    # triangle sequence must still contain the same geometry.
                    np.testing.assert_array_equal(
                        np.asarray(restored.vertices)[np.asarray(restored.triangles)],
                        np.asarray(mesh.vertices)[np.asarray(mesh.triangles)])
                restored=fusion.trimesh.load(result['glb'],force='mesh',process=False)
                np.testing.assert_array_equal(restored.vertices,np.asarray(mesh.vertices))
                np.testing.assert_array_equal(restored.faces,np.asarray(mesh.triangles))
            self.assertFalse(list(stage.iterdir()))

if __name__=='__main__':main()
