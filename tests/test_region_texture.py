import tempfile
import unittest
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from region_texture import textured_region,source_vertex_normals
from export_preview import load_preview


class TextureTests(unittest.TestCase):
    def test_uv_geometry_colors_survive_glb_and_preview(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'scene_tsdf_mesh_m.ply'
            mesh=o3d.geometry.TriangleMesh()
            mesh.vertices=o3d.utility.Vector3dVector([[0,0,.5],[1,0,.5],[1,1,.5],[0,1,.5]])
            mesh.triangles=o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
            uv=np.array([[[0,0],[1,0],[1,1]],[[.25,.25],[.75,.75],[0,1]]])
            np.save(root/'mesh_triangle_uv.npy',uv)
            pixels=np.array([[[255,0,0],[0,255,0]],[[0,0,255],[255,255,0]]],np.uint8)
            Image.fromarray(pixels).save(root/'texture_multiview.png')
            result,metadata=textured_region(mesh,[1],source)
            np.testing.assert_allclose(result.vertices,np.asarray(mesh.vertices)[[0,2,3]])
            scene=trimesh.Scene(result);path=root/'region.glb';path.write_bytes(scene.export(file_type='glb'))
            loaded=load_preview(path)[0]
            np.testing.assert_allclose(np.asarray(loaded.triangle_uvs),uv[1],atol=1e-7)
            np.testing.assert_array_equal(np.asarray(loaded.textures[0]),pixels[::-1])
            np.testing.assert_allclose(np.asarray(loaded.vertices),result.vertices)
            self.assertTrue(metadata['texture_embedded'])
            np.save(root/'mesh_triangle_uv.npy',uv[:1])
            with self.assertRaises(ValueError):textured_region(mesh,[0],source)

    def test_missing_texture_falls_back(self):
        with tempfile.TemporaryDirectory() as folder:
            result,metadata=textured_region(None,[],Path(folder)/'scene_tsdf_mesh_m.ply')
            self.assertIsNone(result)
            self.assertEqual(metadata['appearance'],'vertex_colors_without_photo_texture')

    def test_smooth_normals_survive_uv_seams_glb_and_preview(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            vertices=np.array([[0.,0,.5],[1,0,.5],[1,1,.7],[0,1,.5]])
            faces=np.array([[0,1,2],[0,2,3]])
            mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces))
            np.save(root/'mesh_triangle_uv.npy',np.array([[[0,0],[1,0],[1,1]],[[.1,.1],[.8,.8],[0,1]]]))
            Image.new('RGB',(8,8),'red').save(root/'texture_multiview.png')
            textured,_=textured_region(mesh,[0,1],root/'scene_tsdf_mesh_m.ply')
            expected=source_vertex_normals(vertices,faces)[faces].reshape(-1,3)
            np.testing.assert_allclose(textured.vertex_normals,expected)
            np.testing.assert_allclose(textured.vertex_normals[0],textured.vertex_normals[3])
            scene=trimesh.Scene();transform=np.diag([1.,-1.,-1.,1.])
            scene.add_geometry(textured,transform=transform)
            path=root/'region.glb';path.write_bytes(scene.export(file_type='glb'))
            restored=trimesh.load(path,force='scene',process=False)
            exported=next(iter(restored.geometry.values()))
            np.testing.assert_allclose(exported.vertex_normals,expected,atol=1e-6)
            np.testing.assert_allclose(exported.vertices,vertices[faces].reshape(-1,3),atol=1e-7)
            preview=load_preview(path)[0]
            np.testing.assert_allclose(np.asarray(preview.vertex_normals),expected*[1,-1,-1],atol=1e-6)


if __name__=='__main__':unittest.main()
