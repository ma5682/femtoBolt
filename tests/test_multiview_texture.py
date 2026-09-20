import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import open3d as o3d
from PIL import Image
import trimesh
from femto.geometry.multiview_texture import project_visible, export_multiview_texture


class MultiviewTextureTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[-.1,-.1,1],[.1,-.1,1],[-.1,.1,1],[-.1,-.1,1.2],[.1,-.1,1.2],[-.1,.1,1.2]])
        self.faces = np.array([[0,2,1],[3,5,4]])
        self.K = np.array([[100,0,40],[0,100,30],[0,0,1]],float)
        self.depth = np.full((60,80),1000,np.uint16)

    def test_depth_occlusion_and_backfaces_do_not_receive_photo(self):
        xy,score = project_visible(self.vertices,self.faces,np.eye(4),self.K,np.zeros(8),self.depth)
        np.testing.assert_allclose(xy[0],[[30,20],[30,40],[50,20]])
        self.assertGreater(score[0],0)
        self.assertLess(score[1],0)
        _,reverse = project_visible(self.vertices,self.faces[:,::-1],np.eye(4),self.K,np.zeros(8),self.depth)
        self.assertTrue((reverse<0).all())
        _,unknown = project_visible(self.vertices,self.faces,np.eye(4),self.K,np.zeros(8),np.zeros_like(self.depth))
        self.assertTrue((unknown<0).all())

    def test_atlas_is_original_pixels_and_exports_keep_corner_uvs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'中文纹理'
            (root/'calibration').mkdir(parents=True)
            intr = dict(width=80,height=60,fx=100,fy=100,cx=40,cy=30)
            dist = dict(model='BROWN_CONRADY_K6',**dict.fromkeys(('k1','k2','p1','p2','k3','k4','k5','k6'),0))
            (root/'calibration/color_intrinsic.json').write_text(json.dumps(intr))
            (root/'calibration/distortion.json').write_text(json.dumps(dict(rgb=dist)))
            pixels = np.arange(60*80*3,dtype=np.uint8).reshape(60,80,3)
            Image.fromarray(pixels).save(root/'rgb.png')
            Image.fromarray(self.depth).save(root/'depth.png')
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(self.vertices),o3d.utility.Vector3iVector(self.faces))
            mesh.paint_uniform_color([.2,.3,.4])
            exports = dict(ply=str(root/'mesh.ply'),obj=str(root/'mesh.obj'),glb=str(root/'mesh.glb'))
            result = export_multiview_texture(mesh,root,[dict(rgb='rgb.png',depth='depth.png')],[np.eye(4)],root,exports)
            atlas = np.asarray(Image.open(result['texture']))
            np.testing.assert_array_equal(atlas[:60,:80],pixels)
            report = json.loads(Path(result['texture_report']).read_text(encoding='utf-8'))
            self.assertEqual(report['textured_triangles'],1)
            self.assertEqual(report['fallback_triangles'],1)
            np.testing.assert_array_equal(np.asarray(mesh.triangles),self.faces)
            uv = np.load(result['triangle_uv'])
            np.testing.assert_allclose(uv[0,0],[30.5/80,1-20.5/atlas.shape[0]])
            exported = trimesh.load(result['glb'],force='mesh',process=False)
            np.testing.assert_allclose(exported.vertices[exported.faces],self.vertices[self.faces],atol=1e-6)
            np.testing.assert_allclose(exported.visual.uv[exported.faces],uv,atol=1e-6)
