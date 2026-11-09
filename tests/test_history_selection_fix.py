"""Capture/derivative association and fixed-reference selection UI regressions."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import zipfile

import numpy as np
from PySide6.QtCore import Qt, QPoint, QEventLoop, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from femto.app.qt_region_capture import CaptureRegionEditor
from femto.annotation.round_brush import stroke
from femto.planning.catalog import register_project, scan_capture_exports


class CaptureCatalogTests(unittest.TestCase):
    def project(self, outputs, name, capture, stamp):
        root = outputs/'planning_models'/name
        root.mkdir(parents=True)
        path = root/'planning.region.zip'
        assets = {key:b'fixture' for key in ('scene_tsdf_mesh_m.ply','rgb.png','depth.npy','image_valid.npy')}
        record = dict(schema='femto-region-project-v1',project_id=name,revision_id=name,
                      region={'name':'同名工作区域'},
                      source_lineage={'source_mesh':str(capture/'scene_tsdf_mesh_m.ply')},
                      files={key:hashlib.sha256(value).hexdigest() for key,value in assets.items()})
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('project.json',json.dumps(record))
            for key,value in assets.items():archive.writestr(key,value)
        os.utime(path,(stamp,stamp));register_project(root,path)
        return path

    def test_derivative_only_and_global_indexes_join_exact_capture_and_keep_versions(self):
        with TemporaryDirectory() as tmp:
            outputs=Path(tmp)/'outputs';capture=outputs/'complete_demo_a/reconstruction'
            other=outputs/'complete_demo_b/reconstruction'
            capture.mkdir(parents=True);other.mkdir(parents=True)
            first=self.project(outputs,'first',capture,1000)
            second=self.project(outputs,'second',capture,2000)
            unrelated=self.project(outputs,'unrelated',other,3000)
            register_project(outputs/'saved_projects',second)
            before={path:path.read_bytes() for path in outputs.rglob('*') if path.is_file()}
            rows=scan_capture_exports(capture)
            self.assertEqual([row['project'] for row in rows],[second,first])
            self.assertNotIn(unrelated,[row['project'] for row in rows])
            self.assertEqual(before,{path:path.read_bytes() for path in before})

    def test_corrupt_derivative_is_not_automatically_openable(self):
        with TemporaryDirectory() as tmp:
            outputs=Path(tmp)/'outputs';capture=outputs/'complete_demo_a/reconstruction'
            capture.mkdir(parents=True)
            project=self.project(outputs,'saved',capture,1000)
            project.write_bytes(b'corrupt')
            self.assertFalse(any(row.get('project') for row in scan_capture_exports(capture)))

    def test_cancel_during_derived_lookup_propagates(self):
        with TemporaryDirectory() as tmp:
            outputs=Path(tmp)/'outputs';capture=outputs/'complete_demo_a/reconstruction'
            capture.mkdir(parents=True);self.project(outputs,'saved',capture,1000)
            def cancel():raise TimeoutError('cancelled')
            with self.assertRaisesRegex(TimeoutError,'cancelled'):
                scan_capture_exports(capture,check=cancel)


class RegionStyleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.editor=CaptureRegionEditor();self.editor.resize(1100,700)
        self.reference=SimpleNamespace(rgb=np.zeros((240,320,3),np.uint8),binding={'id':1},frame_count=1)
        self.editor.install(self.reference);self.editor.show();self.app.processEvents()
        self.addCleanup(self.editor.close)

    def test_slider_keyboard_erase_and_undo_keep_binary_original_size(self):
        editor=self.editor;canvas=editor.canvas
        editor.brush_slider.setValue(64)
        self.assertEqual(editor.brush.value(),64);self.assertEqual(canvas.brush_size,64)
        editor.brush.setValue(40);self.assertEqual(editor.brush_slider.value(),40)
        center=canvas.rect().center()
        QTest.mousePress(canvas,Qt.LeftButton,pos=center)
        QTest.mouseMove(canvas,center+QPoint(140,20))
        QTest.mouseRelease(canvas,Qt.LeftButton,pos=center+QPoint(140,20))
        original=canvas.work.copy();self.assertTrue(original.any())
        self.assertEqual(original.shape,(240,320));self.assertTrue(np.isin(original,[0,1]).all())
        editor.erase.click();QTest.mouseClick(canvas,Qt.LeftButton,pos=center)
        self.assertLess(canvas.work.sum(),original.sum())
        canvas.setFocus();QTest.keyClick(canvas,Qt.Key_Z,Qt.ControlModifier)
        np.testing.assert_array_equal(canvas.work,original)
        editor.clear.click();self.assertFalse(canvas.work.any())
        editor.undo.click();np.testing.assert_array_equal(canvas.work,original)
        editor.install(self.reference,error='重建失败')
        np.testing.assert_array_equal(canvas.work,original)
        self.assertTrue(editor.advanced_toggle.isVisible())
        self.assertTrue(editor.undo.isEnabled())

    def test_round_sweep_preserves_holes_and_matches_disk_radius(self):
        mask=np.zeros((81,101),np.uint8)
        stroke(mask,(30.,40.),(70.,40.),20.)
        yy,xx=np.indices(mask.shape)
        nearest_x=np.clip(xx,30,70)
        np.testing.assert_array_equal(mask,((xx-nearest_x)**2+(yy-40)**2<=100).astype(np.uint8))
        stroke(mask,(50.,40.),(50.,40.),8.,erase=True)
        self.assertFalse(mask[40,50]);self.assertTrue(mask[32,50])

    def test_repaint_outline_and_zoom_do_not_change_intent(self):
        editor=self.editor;canvas=editor.canvas
        canvas._before=(canvas.work.copy(),canvas.forbidden.copy())
        canvas._paint((0.,0.));canvas._paint((200.,160.));canvas.finish_stroke()
        saved=canvas.work.copy()
        self.assertFalse(canvas.grab().isNull())
        self.assertIsNotNone(canvas._outline_path)
        canvas.fit_selection();self.assertEqual(editor.zoom_label.text(),f'{canvas.zoom*100:.0f}%')
        editor.reset.click();self.assertEqual(editor.zoom_label.text(),'100%')
        np.testing.assert_array_equal(canvas.work,saved)
        canvas.undo();canvas.redo();np.testing.assert_array_equal(canvas.work,saved)


class CaptureMaskReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_unsaved_capture_restores_mask_and_visible_3d_without_writing_source(self):
        from test_region_first_reconstruction import dataset_fixture
        from femto.geometry.local_region import load_reference, save_request
        from femto.pipeline.scene_reconstruction import reconstruct_scene
        from femto.app.qt_review import ReviewPage
        from femto.viewer.vtk_scene import scene_snapshot
        with TemporaryDirectory() as tmp:
            root=Path(tmp);_,dataset=dataset_fixture(root,count=1)
            reference=load_reference(dataset);mask=np.zeros(reference.rgb.shape[:2],np.uint8)
            mask[35:85,50:110]=1;mask[50:60,70:80]=0
            request=save_request(reference,mask,root/'request')
            rec=root/'outputs/complete_demo_fixture/reconstruction'
            reconstruct_scene(dataset,rec,depth_min_m=.25,depth_max_m=2.21,region_request=request)
            before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
            page=ReviewPage()
            try:
                page.load_source(rec)
                loop=QEventLoop();deadline=QTimer();deadline.setSingleShot(True)
                deadline.timeout.connect(loop.quit);page.runner.finished.connect(loop.quit)
                deadline.start(90000)
                if page.busy:loop.exec()
                deadline.stop();self.assertFalse(page.busy)
                self.assertIsNotNone(page.region,page.status.text())
                np.testing.assert_array_equal(page.canvas.work,mask)
                self.assertFalse(page.canvas.editable)
                self.assertIsNone(page.plan)
                scene=scene_snapshot(page.model,page.region,selection_only=True)
                self.assertGreater(len(scene.triangles),0)
                self.assertEqual(before,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before})
            finally:page.shutdown();page.close()


if __name__=='__main__':unittest.main()
