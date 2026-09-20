import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import trimesh
from femto.planning.export_history import scan_exports,scan_region_exports,local_asset
from femto.viewer.export_preview import load_preview


class ExportHistoryTests(unittest.TestCase):
    def test_windowless_output_streams_support_viewer_logging(self):
        import sys
        from unittest.mock import patch
        from femto.viewer.export_preview import ensure_output_streams
        with tempfile.TemporaryDirectory() as folder:
            with patch('femto.viewer.export_preview.LOG_ROOT',Path(folder)/'Log'):
                with patch.object(sys,'stdout',None),patch.object(sys,'stderr',None):
                    stream=ensure_output_streams()
                    try:
                        print('viewer callback output')
                        sys.stderr.write('viewer diagnostic\n')
                    finally:stream.close()
            text=(Path(folder)/'Log'/'export_preview.log').read_text(encoding='utf-8')
            self.assertIn('viewer callback output',text)
            self.assertIn('viewer diagnostic',text)

    def test_old_exports_missing_files_and_corrupt_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'annotations'/'old';folder.mkdir(parents=True)
            (folder/'mesh.ply').write_text('placeholder')
            (folder/'annotations.json').write_text(json.dumps({'regions':[
                {'name':'区域A','mesh_file':'mesh.ply'}, {'name':'缺失','glb_file':'lost.glb'}]}))
            path=root/'region_paths'/'p';path.mkdir(parents=True)
            (path/'path.json').write_text(json.dumps({'region_id':'r','segments':[]}))
            broken=root/'region_paths'/'broken';broken.mkdir()
            (broken/'path.json').write_text('{')
            rows=scan_exports(root)
            self.assertEqual(len(rows),4)
            self.assertEqual(sum(row['asset'] is not None for row in rows),1)
            self.assertTrue(any(row['kind']=='异常记录' for row in rows))
            self.assertEqual(len(scan_exports(root/'missing')),0)
            self.assertIsNone(local_asset(path,'../../annotations/old/mesh.ply'))

    def test_product_history_lists_only_region_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'annotations'/'export';folder.mkdir(parents=True)
            (folder/'region.glb').write_bytes(b'fixture')
            (folder/'annotations.json').write_text(json.dumps({'points':[{'name':'point'}],'regions':[
                {'name':'kept area','glb_file':'region.glb'}, {'name':'missing area','mesh_file':'missing.ply'}]}))
            route=root/'region_paths'/'route';route.mkdir(parents=True)
            (route/'path.json').write_text(json.dumps({'region_id':'route','segments':[]}))
            rows=scan_region_exports(root)
            self.assertEqual([row['name'] for row in rows],['kept area','missing area'])
            self.assertEqual(sum(row['asset'] is not None for row in rows),1)
            self.assertTrue(all(row['kind']=='区域' for row in rows))

    def test_glb_preview_preserves_transform_and_colors(self):
        with tempfile.TemporaryDirectory() as temp:
            mesh=trimesh.creation.box(extents=[.01,.02,.03]);mesh.visual.vertex_colors=[255,215,30,255]
            scene=trimesh.Scene();transform=np.diag([1.,-1.,-1.,1.]);transform[2,3]=-.5
            scene.add_geometry(mesh,transform=transform)
            path=Path(temp)/'scene.glb';path.write_bytes(scene.export(file_type='glb'))
            geometry=load_preview(path)[0]
            np.testing.assert_allclose(np.asarray(geometry.vertices).mean(axis=0),[0,0,-.5],atol=1e-6)
            np.testing.assert_allclose(np.asarray(geometry.vertex_colors)[0],[1,215/255,30/255])
            self.assertEqual(len(geometry.triangles),len(mesh.faces))


if __name__=='__main__':unittest.main()
