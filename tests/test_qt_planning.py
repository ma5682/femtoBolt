"""Qt planning uses real masks/geometry and preserves the published route."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

from copy import deepcopy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d
from PySide6.QtCore import QEventLoop, QTimer, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from test_fixed_rgb_experience import make_model, K
from test_mesh_slice_planner import grid_mesh
from femto.app.qt_planning import PlanningPage, HeadForm, review_findings
from femto.app.qt_planning_io import extract_region
from femto.planning.planning_drafts import Journal, read
from femto.planning.region_project import open_project
from femto.planning.trajectory_export import load_trajectory
from femto.planning.surface_trajectory import to_player_plan


class QtPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        output_root=Path(self.temp.name)/'outputs'
        for module in ('femto.app.qt_planning','femto.planning.region_project'):
            scoped=patch(module+'.OUTPUTS_ROOT',output_root)
            scoped.start();self.addCleanup(scoped.stop)
        mesh=grid_mesh(width=.12,height=.08,cell=.005)
        legacy=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces))
        self.model=make_model(self.temp.name,legacy)
        self.model.view_reference['undistorted_intrinsic']=K.tolist()
        self.model.store.origin_reference={'kind':'initial_color_camera','provenance':'synthetic_test'}
        o3d.io.write_triangle_mesh(self.model.store.source,self.model.mesh)
        self.page=PlanningPage();self.page.resize(1280,850);self.page.show()
        self.journal=Journal(self.model.reconstruction,Path(self.temp.name)/'drafts')
        self.page.install_model(self.model,self.model.reconstruction,self.journal)
        self.app.processEvents()
        self.addCleanup(self.close_page)

    def close_page(self):
        deadline=time.monotonic()+10
        while not self.page.shutdown() and time.monotonic()<deadline:QTest.qWait(50)
        self.page.close();self.app.processEvents()

    def wait_task(self):
        # A native event loop releases the GIL while the Python worker runs.
        # Repeated PySide QTest.qWait calls can starve that worker, unlike the
        # application's normal QApplication.exec loop. Keep the same deadline.
        loop=QEventLoop();deadline=QTimer()
        deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit)
        self.page.runner.finished.connect(loop.quit)
        try:
            deadline.start(60000)
            if self.page.busy:loop.exec()
        finally:
            deadline.stop()
            self.page.runner.finished.disconnect(loop.quit)
        self.assertFalse(self.page.busy,'Qt task did not finish')
        self.assertFalse(self.page._error,self.page.status_label.text())

    def mark(self,split=False):
        self.page.canvas.work[:]=self.model.projection.valid.astype(np.uint8)
        if split:self.page.canvas.forbidden[:,48:53]=1
        self.page.canvas.rebuild_layers();self.page._selection_changed()
        self.page.commit_region();self.wait_task()

    def test_opt_in_local_layout_is_generated_and_marks_old_choice_stale(self):
        from femto.planning.fixed_width_layout import OPTIMIZED_PATCH_VERSION
        self.mark()
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(OPTIMIZED_PATCH_VERSION))
        self.page.generate();self.wait_task()
        self.assertEqual(self.page.plan['planning_context']['revision']['inputs']['algorithms']['layout'],OPTIMIZED_PATCH_VERSION)
        self.assertTrue(self.page.usable())
        self.assertIn('局部方向适配（试验）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True)
        self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'v4-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),OPTIMIZED_PATCH_VERSION)
        self.assertTrue(self.page.usable())
        self.assertTrue(self.page._variable_source_current())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(None));self.assertFalse(self.page.usable())
        self.assertFalse(self.page._variable_source_current())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(OPTIMIZED_PATCH_VERSION));self.assertTrue(self.page.usable())
        self.assertTrue(self.page._variable_source_current())
        self.page.checkpoint();self.journal.wait()
        draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],OPTIMIZED_PATCH_VERSION)
        self.page.restore_draft(draft)
        self.assertEqual(self.page.head.layout_algorithm(),OPTIMIZED_PATCH_VERSION)
        self.assertIsNone(self.page.plan)

    def test_local_layout_choice_never_changes_device_parameter_status(self):
        from femto.planning.fixed_width_layout import OPTIMIZED_PATCH_VERSION
        form=HeadForm();self.addCleanup(form.close)
        before=form.record()
        form.layout_choice.setCurrentIndex(form.layout_choice.findData(OPTIMIZED_PATCH_VERSION))
        self.assertEqual(form.record(),before)
        form.mode.setCurrentIndex(form.mode.findData('direct_row_spacing'))
        self.assertIsNone(form.layout_algorithm())

    def test_improved_route_choice_survives_archive_and_invalidates_on_switch(self):
        from femto.planning.continuous_quality import IMPROVED_ROUTING
        from femto.planning.fixed_width_layout import OPTIMIZED_PATCH_VERSION
        self.mark()
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(IMPROVED_ROUTING))
        self.page.generate();self.wait_task()
        self.assertEqual(self.page.plan['planning_context']['revision']['inputs']['algorithms']['routing'],IMPROVED_ROUTING)
        self.assertTrue(self.page.usable())
        self.assertIn('局部方向与连接优化（试验）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True)
        self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'two-opt-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),IMPROVED_ROUTING);self.assertTrue(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(OPTIMIZED_PATCH_VERSION))
        self.assertFalse(self.page.usable());self.assertFalse(self.page._variable_source_current())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(IMPROVED_ROUTING))
        self.assertTrue(self.page.usable());self.page.checkpoint();self.journal.wait()
        draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],IMPROVED_ROUTING)
        self.page.restore_draft(draft)
        self.assertEqual(self.page.head.layout_algorithm(),IMPROVED_ROUTING)

    def test_region_only_save_restores_improved_route_choice(self):
        from femto.planning.continuous_quality import IMPROVED_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(IMPROVED_ROUTING))
        path=Path(self.temp.name)/'region-only-two-opt.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-two-opt-reopened')
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],IMPROVED_ROUTING)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),IMPROVED_ROUTING)

    def test_support_route_choice_survives_archive_and_invalidates_on_switch(self):
        from femto.planning.continuous_quality import IMPROVED_ROUTING,SUPPORT_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SUPPORT_ROUTING))
        self.page.generate();self.wait_task()
        self.assertEqual(self.page.plan['planning_context']['revision']['inputs']['algorithms']['routing'],SUPPORT_ROUTING)
        self.assertTrue(self.page.usable());self.assertIn('局部方向与连接优化（试验新版）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True);self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'support-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),SUPPORT_ROUTING);self.assertTrue(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(IMPROVED_ROUTING))
        self.assertFalse(self.page.usable());self.assertFalse(self.page._variable_source_current())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SUPPORT_ROUTING))
        self.page.checkpoint();self.journal.wait();draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],SUPPORT_ROUTING)
        self.page.restore_draft(draft);self.assertEqual(self.page.head.layout_algorithm(),SUPPORT_ROUTING)

    def test_region_only_save_restores_support_route_choice(self):
        from femto.planning.continuous_quality import SUPPORT_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SUPPORT_ROUTING))
        path=Path(self.temp.name)/'region-only-support.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-support-reopened')
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],SUPPORT_ROUTING)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),SUPPORT_ROUTING)

    def test_balanced_route_choice_survives_archive_and_draft(self):
        from femto.planning.continuous_quality import SUPPORT_ROUTING,BALANCED_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(BALANCED_ROUTING))
        self.page.generate();self.wait_task()
        self.assertEqual(self.page.plan['planning_context']['revision']['inputs']['algorithms']['routing'],BALANCED_ROUTING)
        self.assertTrue(self.page.usable());self.assertIn('局部方向与绕行优化（试验）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True);self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'balanced-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),BALANCED_ROUTING);self.assertTrue(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SUPPORT_ROUTING))
        self.assertFalse(self.page.usable());self.assertFalse(self.page._variable_source_current())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(BALANCED_ROUTING))
        self.page.checkpoint();self.journal.wait();draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],BALANCED_ROUTING)
        self.page.restore_draft(draft);self.assertEqual(self.page.head.layout_algorithm(),BALANCED_ROUTING)

    def test_region_only_save_restores_balanced_route_choice(self):
        from femto.planning.continuous_quality import BALANCED_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(BALANCED_ROUTING))
        path=Path(self.temp.name)/'region-only-balanced.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-balanced-reopened')
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],BALANCED_ROUTING)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),BALANCED_ROUTING)

    def test_relocation_choice_survives_archive_and_draft(self):
        from femto.planning.continuous_quality import RELOCATION_ROUTING,BRIDGE_REORDER_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(RELOCATION_ROUTING))
        self.page.generate();self.wait_task()
        self.assertEqual(self.page.plan['planning_context']['revision']['inputs']['algorithms']['routing'],RELOCATION_ROUTING)
        self.assertTrue(self.page.usable());self.assertIn('局部方向与扫描段移动（试验）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True);self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'relocation-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),RELOCATION_ROUTING);self.assertTrue(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(BRIDGE_REORDER_ROUTING))
        self.assertFalse(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(RELOCATION_ROUTING))
        self.page.checkpoint();self.journal.wait();draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],RELOCATION_ROUTING)
        self.page.restore_draft(draft);self.assertEqual(self.page.head.layout_algorithm(),RELOCATION_ROUTING)

    def test_deep_relocation_choice_survives_archive_and_draft(self):
        from femto.planning.continuous_quality import DEEP_RELOCATION_ROUTING, RELOCATION_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(DEEP_RELOCATION_ROUTING))
        self.page.generate();self.wait_task()
        self.assertEqual(self.page.plan['planning_context']['revision']['inputs']['algorithms']['routing'],DEEP_RELOCATION_ROUTING)
        self.assertTrue(self.page.usable());self.assertIn('局部方向与深度换序（试验）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True);self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'deep-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),DEEP_RELOCATION_ROUTING);self.assertTrue(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(RELOCATION_ROUTING))
        self.assertFalse(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(DEEP_RELOCATION_ROUTING))
        self.page.checkpoint();self.journal.wait();draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],DEEP_RELOCATION_ROUTING)
        self.page.restore_draft(draft);self.assertEqual(self.page.head.layout_algorithm(),DEEP_RELOCATION_ROUTING)

    def test_global_floor_choice_survives_archive_and_draft(self):
        from femto.planning.fixed_width_layout import COORDINATED_FLOOR_PATCH_VERSION
        from femto.planning.continuous_quality import GLOBAL_FLOOR_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(GLOBAL_FLOOR_ROUTING))
        self.page.generate();self.wait_task()
        self.assertIsNotNone(self.page.plan,self.page._error)
        algorithms=self.page.plan['planning_context']['revision']['inputs']['algorithms']
        self.assertEqual(algorithms['layout'],COORDINATED_FLOOR_PATCH_VERSION)
        self.assertEqual(algorithms['routing'],GLOBAL_FLOOR_ROUTING)
        self.assertTrue(self.page.usable());self.assertIn('自适应分区与统一换序预算（试验）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True);self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'global-floor-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),GLOBAL_FLOOR_ROUTING);self.assertTrue(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(COORDINATED_FLOOR_PATCH_VERSION))
        self.assertFalse(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(GLOBAL_FLOOR_ROUTING))
        self.page.checkpoint();self.journal.wait();draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],GLOBAL_FLOOR_ROUTING)
        self.page.restore_draft(draft);self.assertEqual(self.page.head.layout_algorithm(),GLOBAL_FLOOR_ROUTING)

    def test_region_only_save_restores_global_floor_choice(self):
        from femto.planning.continuous_quality import GLOBAL_FLOOR_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(GLOBAL_FLOOR_ROUTING))
        path=Path(self.temp.name)/'region-only-global-floor.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-global-floor-reopened')
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],GLOBAL_FLOOR_ROUTING)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),GLOBAL_FLOOR_ROUTING)

    def test_refined_connector_choice_survives_archive_and_draft(self):
        from femto.planning.fixed_width_layout import COORDINATED_FLOOR_PATCH_VERSION
        from femto.planning.continuous_quality import REFINED_CONNECTOR_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(REFINED_CONNECTOR_ROUTING))
        self.page.generate();self.wait_task()
        self.assertIsNotNone(self.page.plan,self.page._error)
        algorithms=self.page.plan['planning_context']['revision']['inputs']['algorithms']
        self.assertEqual(algorithms['layout'],COORDINATED_FLOOR_PATCH_VERSION)
        self.assertEqual(algorithms['routing'],REFINED_CONNECTOR_ROUTING)
        self.assertTrue(self.page.usable());self.assertIn('贴面连接优化（试验）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True);self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'refined-connector-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),REFINED_CONNECTOR_ROUTING);self.assertTrue(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(COORDINATED_FLOOR_PATCH_VERSION))
        self.assertFalse(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(REFINED_CONNECTOR_ROUTING))
        self.page.checkpoint();self.journal.wait();draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],REFINED_CONNECTOR_ROUTING)
        self.page.restore_draft(draft);self.assertEqual(self.page.head.layout_algorithm(),REFINED_CONNECTOR_ROUTING)


    def test_region_only_save_restores_refined_connector_choice(self):
        from femto.planning.continuous_quality import REFINED_CONNECTOR_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(REFINED_CONNECTOR_ROUTING))
        path=Path(self.temp.name)/'region-only-refined-connector.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-refined-connector-reopened')
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],REFINED_CONNECTOR_ROUTING)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),REFINED_CONNECTOR_ROUTING)


    def test_coordinated_floor_choice_survives_archive_and_draft(self):
        from femto.planning.fixed_width_layout import COORDINATED_FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION
        from femto.planning.continuous_quality import UNIFIED_FLOOR_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(COORDINATED_FLOOR_PATCH_VERSION))
        self.page.generate();self.wait_task()
        self.assertIsNotNone(self.page.plan,self.page._error)
        algorithms=self.page.plan['planning_context']['revision']['inputs']['algorithms']
        self.assertEqual(algorithms['layout'],COORDINATED_FLOOR_PATCH_VERSION);self.assertEqual(algorithms['routing'],UNIFIED_FLOOR_ROUTING)
        self.assertTrue(self.page.usable());self.assertIn('自适应分区与接缝协调（试验）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True);self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'coordinated-floor-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),COORDINATED_FLOOR_PATCH_VERSION);self.assertTrue(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(ADAPTIVE_FLOOR_PATCH_VERSION))
        self.assertFalse(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(COORDINATED_FLOOR_PATCH_VERSION))
        self.page.checkpoint();self.journal.wait();draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],COORDINATED_FLOOR_PATCH_VERSION)
        self.page.restore_draft(draft);self.assertEqual(self.page.head.layout_algorithm(),COORDINATED_FLOOR_PATCH_VERSION)

    def test_region_only_save_restores_coordinated_floor_choice(self):
        from femto.planning.fixed_width_layout import COORDINATED_FLOOR_PATCH_VERSION
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(COORDINATED_FLOOR_PATCH_VERSION))
        path=Path(self.temp.name)/'region-only-coordinated-floor.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-coordinated-floor-reopened')
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],COORDINATED_FLOOR_PATCH_VERSION)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),COORDINATED_FLOOR_PATCH_VERSION)

    def test_adaptive_floor_choice_survives_archive_and_draft(self):
        from femto.planning.fixed_width_layout import ADAPTIVE_FLOOR_PATCH_VERSION
        from femto.planning.continuous_quality import UNIFIED_FLOOR_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(ADAPTIVE_FLOOR_PATCH_VERSION))
        self.page.generate();self.wait_task()
        self.assertIsNotNone(self.page.plan,self.page._error)
        algorithms=self.page.plan['planning_context']['revision']['inputs']['algorithms']
        self.assertEqual(algorithms['layout'],ADAPTIVE_FLOOR_PATCH_VERSION);self.assertEqual(algorithms['routing'],UNIFIED_FLOOR_ROUTING)
        self.assertTrue(self.page.usable());self.assertIn('自适应分区与快速换序（试验）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True);self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'adaptive-floor-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),ADAPTIVE_FLOOR_PATCH_VERSION);self.assertTrue(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(UNIFIED_FLOOR_ROUTING))
        self.assertFalse(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(ADAPTIVE_FLOOR_PATCH_VERSION))
        self.page.checkpoint();self.journal.wait();draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],ADAPTIVE_FLOOR_PATCH_VERSION)
        self.page.restore_draft(draft);self.assertEqual(self.page.head.layout_algorithm(),ADAPTIVE_FLOOR_PATCH_VERSION)

    def test_region_only_save_restores_adaptive_floor_choice(self):
        from femto.planning.fixed_width_layout import ADAPTIVE_FLOOR_PATCH_VERSION
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(ADAPTIVE_FLOOR_PATCH_VERSION))
        path=Path(self.temp.name)/'region-only-adaptive-floor.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-adaptive-floor-reopened')
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],ADAPTIVE_FLOOR_PATCH_VERSION)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),ADAPTIVE_FLOOR_PATCH_VERSION)

    def test_unified_floor_choice_survives_archive_and_draft(self):
        from femto.planning.continuous_quality import UNIFIED_FLOOR_ROUTING,FLOOR_DETOUR_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(UNIFIED_FLOOR_ROUTING))
        self.page.generate();self.wait_task()
        self.assertIsNotNone(self.page.plan,self.page._error)
        algorithms=self.page.plan['planning_context']['revision']['inputs']['algorithms']
        self.assertEqual(algorithms['routing'],UNIFIED_FLOOR_ROUTING);self.assertEqual(algorithms['minimum_metric_fraction'],.65)
        self.assertTrue(self.page.usable());self.assertIn('局部疏密与快速换序（试验）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True);self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'unified-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),UNIFIED_FLOOR_ROUTING);self.assertTrue(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(FLOOR_DETOUR_ROUTING))
        self.assertFalse(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(UNIFIED_FLOOR_ROUTING))
        self.page.checkpoint();self.journal.wait();draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],UNIFIED_FLOOR_ROUTING)
        self.page.restore_draft(draft);self.assertEqual(self.page.head.layout_algorithm(),UNIFIED_FLOOR_ROUTING)

    def test_region_only_save_restores_unified_floor_choice(self):
        from femto.planning.continuous_quality import UNIFIED_FLOOR_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(UNIFIED_FLOOR_ROUTING))
        path=Path(self.temp.name)/'region-only-unified.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-unified-reopened')
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],UNIFIED_FLOOR_ROUTING)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),UNIFIED_FLOOR_ROUTING)

    def test_floor_detour_choice_survives_archive_and_draft(self):
        from femto.planning.continuous_quality import FLOOR_DETOUR_ROUTING,DEEP_RELOCATION_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(FLOOR_DETOUR_ROUTING))
        self.page.generate();self.wait_task()
        self.assertIsNotNone(self.page.plan,self.page._error)
        algorithms=self.page.plan['planning_context']['revision']['inputs']['algorithms']
        self.assertEqual(algorithms['routing'],FLOOR_DETOUR_ROUTING);self.assertEqual(algorithms['minimum_metric_fraction'],.65)
        self.assertTrue(self.page.usable());self.assertIn('局部疏密与绕行优化（试验）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True);self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'floor-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),FLOOR_DETOUR_ROUTING);self.assertTrue(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(DEEP_RELOCATION_ROUTING))
        self.assertFalse(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(FLOOR_DETOUR_ROUTING))
        self.page.checkpoint();self.journal.wait();draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],FLOOR_DETOUR_ROUTING)
        self.page.restore_draft(draft);self.assertEqual(self.page.head.layout_algorithm(),FLOOR_DETOUR_ROUTING)

    def test_region_only_save_restores_floor_detour_choice(self):
        from femto.planning.continuous_quality import FLOOR_DETOUR_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(FLOOR_DETOUR_ROUTING))
        path=Path(self.temp.name)/'region-only-floor.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-floor-reopened')
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],FLOOR_DETOUR_ROUTING)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),FLOOR_DETOUR_ROUTING)

    def test_region_only_save_restores_deep_relocation_choice(self):
        from femto.planning.continuous_quality import DEEP_RELOCATION_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(DEEP_RELOCATION_ROUTING))
        path=Path(self.temp.name)/'region-only-deep.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-deep-reopened')
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],DEEP_RELOCATION_ROUTING)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),DEEP_RELOCATION_ROUTING)

    def test_bridge_reorder_choice_survives_archive_and_draft(self):
        from femto.planning.continuous_quality import BRIDGE_REORDER_ROUTING,JOINT_REORDER_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(BRIDGE_REORDER_ROUTING))
        self.page.generate();self.wait_task()
        self.assertEqual(self.page.plan['planning_context']['revision']['inputs']['algorithms']['routing'],BRIDGE_REORDER_ROUTING)
        self.assertTrue(self.page.usable());self.assertIn('局部方向与两步换序（试验）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True);self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'bridge-reorder-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),BRIDGE_REORDER_ROUTING);self.assertTrue(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(JOINT_REORDER_ROUTING))
        self.assertFalse(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(BRIDGE_REORDER_ROUTING))
        self.page.checkpoint();self.journal.wait();draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],BRIDGE_REORDER_ROUTING)
        self.page.restore_draft(draft);self.assertEqual(self.page.head.layout_algorithm(),BRIDGE_REORDER_ROUTING)

    def test_region_only_save_restores_bridge_reorder_choice(self):
        from femto.planning.continuous_quality import BRIDGE_REORDER_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(BRIDGE_REORDER_ROUTING))
        path=Path(self.temp.name)/'region-only-bridge-reorder.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-bridge-reopened')
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],BRIDGE_REORDER_ROUTING)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),BRIDGE_REORDER_ROUTING)

    def test_region_only_save_restores_relocation_choice(self):
        from femto.planning.continuous_quality import RELOCATION_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(RELOCATION_ROUTING))
        path=Path(self.temp.name)/'region-only-relocation.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-relocation-reopened')
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],RELOCATION_ROUTING)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),RELOCATION_ROUTING)

    def test_bridge_reorder_budget_fallback_is_visible(self):
        from femto.planning.continuous_quality import BRIDGE_REORDER_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(BRIDGE_REORDER_ROUTING))
        with patch('femto.planning.bridge_joint_incremental._connect_reusing',
                   side_effect=ValueError('连续路线整边搜索超过预算，请缩小选区')):
            self.page.generate();self.wait_task()
        self.assertTrue(self.page.usable())
        self.assertIn('连接优化因计算预算回退',self.page.parameter_summary.text())

    def test_joint_reorder_choice_survives_archive_and_draft(self):
        from femto.planning.continuous_quality import JOINT_REORDER_ROUTING,SHORT_DETOUR_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(JOINT_REORDER_ROUTING))
        self.page.generate();self.wait_task()
        self.assertEqual(self.page.plan['planning_context']['revision']['inputs']['algorithms']['routing'],JOINT_REORDER_ROUTING)
        self.assertTrue(self.page.usable());self.assertIn('局部方向与联合换序（试验）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True);self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'joint-reorder-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),JOINT_REORDER_ROUTING);self.assertTrue(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SHORT_DETOUR_ROUTING))
        self.assertFalse(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(JOINT_REORDER_ROUTING))
        self.page.checkpoint();self.journal.wait();draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],JOINT_REORDER_ROUTING)
        self.page.restore_draft(draft);self.assertEqual(self.page.head.layout_algorithm(),JOINT_REORDER_ROUTING)

    def test_region_only_save_restores_joint_reorder_choice(self):
        from femto.planning.continuous_quality import JOINT_REORDER_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(JOINT_REORDER_ROUTING))
        path=Path(self.temp.name)/'region-only-joint-reorder.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-joint-reopened')
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],JOINT_REORDER_ROUTING)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),JOINT_REORDER_ROUTING)

    def test_joint_reorder_budget_fallback_is_visible(self):
        from femto.planning.continuous_quality import JOINT_REORDER_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(JOINT_REORDER_ROUTING))
        with patch('femto.planning.bounded_joint_reorder.connect',
                   side_effect=ValueError('连续路线整边搜索超过预算，请缩小选区')):
            self.page.generate();self.wait_task()
        self.assertTrue(self.page.usable())
        self.assertIn('连接优化因计算预算回退',self.page.parameter_summary.text())

    def test_short_detour_choice_survives_archive_and_draft(self):
        from femto.planning.continuous_quality import SHORT_DETOUR_ROUTING,PRESERVED_DETOUR_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SHORT_DETOUR_ROUTING))
        self.page.generate();self.wait_task()
        self.assertEqual(self.page.plan['planning_context']['revision']['inputs']['algorithms']['routing'],SHORT_DETOUR_ROUTING)
        self.assertTrue(self.page.usable());self.assertIn('局部方向与短距换序（试验）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True);self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'short-detour-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),SHORT_DETOUR_ROUTING);self.assertTrue(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(PRESERVED_DETOUR_ROUTING))
        self.assertFalse(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SHORT_DETOUR_ROUTING))
        self.page.checkpoint();self.journal.wait();draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],SHORT_DETOUR_ROUTING)
        self.page.restore_draft(draft);self.assertEqual(self.page.head.layout_algorithm(),SHORT_DETOUR_ROUTING)

    def test_region_only_save_restores_short_detour_choice(self):
        from femto.planning.continuous_quality import SHORT_DETOUR_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SHORT_DETOUR_ROUTING))
        path=Path(self.temp.name)/'region-only-short-detour.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-short-reopened')
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],SHORT_DETOUR_ROUTING)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),SHORT_DETOUR_ROUTING)

    def test_short_detour_budget_fallback_is_visible(self):
        from femto.planning.continuous_quality import SHORT_DETOUR_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SHORT_DETOUR_ROUTING))
        with patch('femto.planning.endpoint_preserving_detour.connect',
                   side_effect=ValueError('连续路线整边搜索超过预算，请缩小选区')):
            self.page.generate();self.wait_task()
        self.assertTrue(self.page.usable())
        self.assertIn('连接优化因计算预算回退',self.page.parameter_summary.text())

    def test_preserved_detour_choice_survives_archive_and_draft(self):
        from femto.planning.continuous_quality import PRESERVED_DETOUR_ROUTING,SEAM_BALANCED_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(PRESERVED_DETOUR_ROUTING))
        self.page.generate();self.wait_task()
        self.assertEqual(self.page.plan['planning_context']['revision']['inputs']['algorithms']['routing'],PRESERVED_DETOUR_ROUTING)
        self.assertTrue(self.page.usable());self.assertIn('局部方向与减少绕行（试验）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True);self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'preserved-detour-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),PRESERVED_DETOUR_ROUTING);self.assertTrue(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SEAM_BALANCED_ROUTING))
        self.assertFalse(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(PRESERVED_DETOUR_ROUTING))
        self.page.checkpoint();self.journal.wait();draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],PRESERVED_DETOUR_ROUTING)
        self.page.restore_draft(draft);self.assertEqual(self.page.head.layout_algorithm(),PRESERVED_DETOUR_ROUTING)

    def test_region_only_save_restores_preserved_detour_choice(self):
        from femto.planning.continuous_quality import PRESERVED_DETOUR_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(PRESERVED_DETOUR_ROUTING))
        path=Path(self.temp.name)/'region-only-preserved-detour.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-preserved-detour-reopened')
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],PRESERVED_DETOUR_ROUTING)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),PRESERVED_DETOUR_ROUTING)

    def test_preserved_detour_budget_fallback_is_visible(self):
        from femto.planning.continuous_quality import PRESERVED_DETOUR_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(PRESERVED_DETOUR_ROUTING))
        with patch('femto.planning.endpoint_preserving_detour.connect',
                   side_effect=ValueError('连续路线整边搜索超过预算，请缩小选区')):
            self.page.generate();self.wait_task()
        self.assertTrue(self.page.usable())
        self.assertIn('连接优化因计算预算回退',self.page.parameter_summary.text())

    def test_foldback_choice_survives_archive_and_draft(self):
        from femto.planning.fixed_width_layout import FOLDBACK_PATCH_VERSION
        from femto.planning.continuous_quality import SEAM_BALANCED_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(FOLDBACK_PATCH_VERSION))
        self.page.generate();self.wait_task()
        algorithms=self.page.plan['planning_context']['revision']['inputs']['algorithms']
        self.assertEqual(algorithms['layout'],FOLDBACK_PATCH_VERSION);self.assertEqual(algorithms['routing'],SEAM_BALANCED_ROUTING)
        self.assertTrue(self.page.usable());self.assertIn('局部方向与折返识别（试验）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True);self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'foldback-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),FOLDBACK_PATCH_VERSION);self.assertTrue(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SEAM_BALANCED_ROUTING))
        self.assertFalse(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(FOLDBACK_PATCH_VERSION))
        self.page.checkpoint();self.journal.wait();draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],FOLDBACK_PATCH_VERSION)
        self.page.restore_draft(draft);self.assertEqual(self.page.head.layout_algorithm(),FOLDBACK_PATCH_VERSION)

    def test_region_only_save_restores_foldback_choice(self):
        from femto.planning.fixed_width_layout import FOLDBACK_PATCH_VERSION
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(FOLDBACK_PATCH_VERSION))
        path=Path(self.temp.name)/'region-only-foldback.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-foldback-reopened')
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],FOLDBACK_PATCH_VERSION)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),FOLDBACK_PATCH_VERSION)

    def test_seam_balanced_choice_survives_archive_draft_and_budget_fallback(self):
        from femto.planning.continuous_quality import SEAM_BALANCED_ROUTING,SEAM_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SEAM_BALANCED_ROUTING))
        self.page.generate();self.wait_task()
        self.assertEqual(self.page.plan['planning_context']['revision']['inputs']['algorithms']['routing'],SEAM_BALANCED_ROUTING)
        self.assertTrue(self.page.usable());self.assertIn('局部方向与接缝联合优化（试验）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True);self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'seam-balanced-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),SEAM_BALANCED_ROUTING);self.assertTrue(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SEAM_ROUTING))
        self.assertFalse(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SEAM_BALANCED_ROUTING))
        self.page.checkpoint();self.journal.wait();draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],SEAM_BALANCED_ROUTING)
        self.page.restore_draft(draft);self.assertEqual(self.page.head.layout_algorithm(),SEAM_BALANCED_ROUTING)

    def test_seam_balanced_budget_fallback_is_visible(self):
        from femto.planning.continuous_quality import SEAM_BALANCED_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SEAM_BALANCED_ROUTING))
        with patch('femto.planning.balanced_visit_improvement.propose',
                   side_effect=ValueError('连续路线整边搜索超过预算，请缩小选区')):
            self.page.generate();self.wait_task()
        self.assertTrue(self.page.usable())
        self.assertIn('连接优化因计算预算回退',self.page.parameter_summary.text())

    def test_region_only_save_restores_seam_balanced_choice(self):
        from femto.planning.continuous_quality import SEAM_BALANCED_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SEAM_BALANCED_ROUTING))
        path=Path(self.temp.name)/'region-only-seam-balanced.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-seam-balanced-reopened')
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],SEAM_BALANCED_ROUTING)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),SEAM_BALANCED_ROUTING)

    def test_seam_route_choice_survives_archive_and_draft(self):
        from femto.planning.continuous_quality import SEAM_ROUTING,BALANCED_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SEAM_ROUTING))
        self.page.generate();self.wait_task()
        self.assertEqual(self.page.plan['planning_context']['revision']['inputs']['algorithms']['routing'],SEAM_ROUTING)
        self.assertTrue(self.page.usable());self.assertIn('局部方向与接缝顺接（试验）',self.page.parameter_summary.text())
        self.page.set_stage(2);self.page.reviewed.setChecked(True);self.page.save_plan();self.wait_task()
        archive=self.page.archive
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'seam-reopened')
        self.page.install_model(reopened,archive/'planning.region.zip',self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),SEAM_ROUTING);self.assertTrue(self.page.usable())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(BALANCED_ROUTING))
        self.assertFalse(self.page.usable());self.assertFalse(self.page._variable_source_current())
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SEAM_ROUTING))
        self.page.checkpoint();self.journal.wait();draft=read(self.journal.path,self.model)
        self.assertEqual(draft['qt_layout_algorithm'],SEAM_ROUTING)
        self.page.restore_draft(draft);self.assertEqual(self.page.head.layout_algorithm(),SEAM_ROUTING)

    def test_region_only_save_restores_seam_route_choice(self):
        from femto.planning.continuous_quality import SEAM_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SEAM_ROUTING))
        path=Path(self.temp.name)/'region-only-seam.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-seam-reopened')
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],SEAM_ROUTING)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),SEAM_ROUTING)

    def test_seam_route_budget_fallback_is_visible(self):
        from femto.planning.continuous_quality import SEAM_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(SEAM_ROUTING))
        with patch('femto.planning.seam_visit_proposal.propose',
                   side_effect=ValueError('连续路线整边搜索超过预算，请缩小选区')):
            self.page.generate();self.wait_task()
        self.assertTrue(self.page.usable())
        self.assertIn('连接优化因计算预算回退',self.page.parameter_summary.text())

    def test_improved_route_budget_fallback_is_visible(self):
        from femto.planning.continuous_quality import IMPROVED_ROUTING
        self.mark();self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(IMPROVED_ROUTING))
        with patch('femto.planning.verified_visit_improvement.improve',
                   side_effect=ValueError('连续路线整边搜索超过预算，请缩小选区')):
            self.page.generate();self.wait_task()
        self.assertTrue(self.page.usable())
        self.assertIn('连接优化因计算预算回退',self.page.parameter_summary.text())

    def test_region_only_save_restores_opt_in_layout(self):
        from femto.planning.fixed_width_layout import OPTIMIZED_PATCH_VERSION
        self.mark()
        self.page.head.layout_choice.setCurrentIndex(self.page.head.layout_choice.findData(OPTIMIZED_PATCH_VERSION))
        path=Path(self.temp.name)/'region-only-v4.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(path),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(path,workspace=Path(self.temp.name)/'region-only-reopened')
        self.assertIsNone(reopened.saved_plan)
        self.assertEqual(reopened.planning_parameters['values']['layout_algorithm'],OPTIMIZED_PATCH_VERSION)
        self.page.install_model(reopened,path,self.journal)
        self.assertEqual(self.page.head.layout_algorithm(),OPTIMIZED_PATCH_VERSION)

    def test_real_region_route_archive_reopens_same_motion(self):
        self.mark(split=True)
        self.assertEqual(self.page._stage,1)
        self.assertEqual(len(self.page.region['selection_intent']['components']),2)
        self.page.head.fields['row_spacing_mm'].setText('7')
        self.page.generate();self.wait_task()
        self.assertGreaterEqual(len(self.page.plan['segments']),2)
        self.assertEqual(sum(gap['inter_block'] for gap in self.page.plan['chain_breaks']),1)
        self.assertTrue(all(not gap['transition_planned'] for gap in self.page.plan['chain_breaks']))
        self.assertIn('工作块间移动未规划', self.page.warning.text())
        self.assertTrue(any(role=='scan' for role,_ in self.page.canvas._runs))
        self.assertIn('质量与覆盖宽度约束',self.page.warning.text())
        plan=deepcopy(self.page.plan)
        self.page.set_stage(2);self.page.reviewed.setChecked(True)
        with patch('femto.planning.mesh_slice_planner.plan_mesh_slice',side_effect=AssertionError('save replanned')):
            self.page.save_plan();self.wait_task()
        archive=self.page.archive
        self.assertEqual(to_player_plan(load_trajectory(archive,reproduce=False))['segments'],plan['segments'])
        reopened=open_project(archive/'planning.region.zip',workspace=Path(self.temp.name)/'reopened')
        self.assertEqual(reopened.saved_plan['segments'],plan['segments'])
        self.assertFalse(self.page.dirty)

    def test_reopened_region_starts_with_region_review(self):
        self.mark()
        self.page.install_model(self.page.model, self.model.reconstruction, self.journal)
        self.assertEqual(self.page._stage, 0)
        self.assertFalse(self.page.canvas.show_plan)
        self.assertTrue(self.page.canvas.support)
        self.assertIn('当前来源', self.page.source_label.text())
        self.assertIn('请先核对', self.page.status_label.text())
        self.page.primary_action()
        self.assertEqual(self.page._stage, 1)
        self.assertTrue(self.page.canvas.show_plan)
        self.assertFalse(self.page.canvas.support)

    def test_partial_route_findings_are_visible_before_save(self):
        plan = {
            'surface_trajectory': {'statistics': {
                'geometry_unconfirmed_pixel_count': 686,
                'coverage_status': 'partial',
                'block_count': 3,
                'unplannable_block_count': 0,
            }},
            'chain_breaks': [
                {'inter_block': True}, {'inter_block': True},
                {'inter_block': False},
            ],
            'spacing_report': {'status': 'not_fully_verified'},
        }
        findings = review_findings(plan, None)
        self.assertTrue(any('686' in item for item in findings))
        self.assertTrue(any('覆盖未完整' in item for item in findings))
        self.assertTrue(any('工作块内路线断开' in item for item in findings))
        self.assertTrue(any('工作块间移动未规划' in item for item in findings))

    def test_exact_draft_masks_restore_and_failed_load_preserves_state(self):
        self.page.head.mode.setCurrentIndex(self.page.head.mode.findData('direct_row_spacing'))
        self.page.canvas.work[30:70,32:68]=1
        self.page.canvas.work[40:44,42:45]=0
        self.page.canvas.forbidden[50:54,50:54]=1
        self.page._selection_changed();self.page.head.fields['row_spacing_mm'].setText('2.4')
        self.page.checkpoint();self.journal.wait()
        draft=read(self.journal.path,self.model)
        original=self.page.canvas.work.copy()
        self.page.canvas.work[:]=0
        self.page.restore_draft(draft)
        np.testing.assert_array_equal(self.page.canvas.work,original)
        self.assertEqual(self.page.head.record()['row_spacing_mm'],2.4)
        previous=self.page.model
        from PySide6.QtWidgets import QMessageBox
        with patch.object(QMessageBox,'question',return_value=QMessageBox.Yes):
            self.page.load_source(Path(self.temp.name)/'missing.region.zip')
            deadline=time.monotonic()+10
            while self.page.busy and time.monotonic()<deadline:QTest.qWait(20)
        self.assertIs(self.page.model,previous)
        np.testing.assert_array_equal(self.page.canvas.work,original)
        self.assertTrue(self.page._error)

    def test_parameter_limits_invalidate_previous_route(self):
        self.mark()
        self.page.generate();self.wait_task()
        self.assertTrue(self.page.usable())
        self.page.head.mode.setCurrentIndex(1)
        self.page.head.fields['effective_treatment_width_mm'].setText('0.5')
        self.page.head.fields['overlap_percent'].setText('90')
        self.assertIsNotNone(self.page.plan)
        self.assertFalse(self.page.usable())
        with self.assertRaises(ValueError):self.page.head.record()
        self.assertFalse(self.page.primary.isEnabled())
        self.page.head.fields['effective_treatment_width_mm'].setText('3')
        self.page.head.fields['overlap_percent'].setText('20')
        self.assertEqual(self.page.head.record()['row_spacing_mm'],2.4)
        self.page.head.fields['overlap_percent'].setText('100')
        with self.assertRaises(ValueError):self.page.head.record()

    def test_failed_recalculation_keeps_last_reviewable_route(self):
        self.mark();self.page.generate();self.wait_task();previous=deepcopy(self.page.plan)
        self.page.reviewed.setChecked(True)
        with patch('femto.app.qt_planning.surface_workspace.generate_checked',side_effect=RuntimeError('synthetic planner failure')):
            self.page.generate()
            deadline=time.monotonic()+10
            while self.page.busy and time.monotonic()<deadline:QTest.qWait(20)
        self.assertEqual(self.page.plan,previous)
        self.assertFalse(self.page.reviewed.isChecked())
        self.assertIn('synthetic planner failure',self.page.status_label.text())

    def test_cancelled_recalculation_keeps_last_reviewable_route(self):
        self.mark();self.page.generate();self.wait_task();previous=deepcopy(self.page.plan)
        def blocked(model,region,head,check,progress):
            progress('等待取消')
            while True:check()

        with patch('femto.app.qt_planning.surface_workspace.generate_checked',side_effect=blocked):
            self.page.generate()
            deadline=time.monotonic()+5
            while not self.page.busy and time.monotonic()<deadline:QTest.qWait(10)
            self.assertTrue(self.page.busy)
            self.page.runner.cancel();self.wait_task()
        self.assertEqual(self.page.plan,previous)
        self.assertIn('任务已取消',self.page.status_label.text())

    def test_save_as_rebinds_source_and_draft_journal(self):
        self.mark()
        target=Path(self.temp.name)/'saved.region.zip'
        old_source=self.page.source
        old_journal=self.page.journal
        loaded=[];self.page.loaded.connect(loaded.append)
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(target),'')), \
             patch('femto.planning.region_project.save_project',return_value=target):
            self.page.save_as();self.wait_task()
        self.assertEqual(self.page.source,target.resolve())
        self.assertEqual(loaded,[str(target.resolve())])
        self.assertNotEqual(self.page.journal.path,old_journal.path)
        self.assertEqual(self.page.archive,target.resolve())
        self.assertFalse(self.page.dirty)
        self.assertFalse(old_journal.path.exists())
        self.page.name.setText('另存后的区域')
        self.page._metadata_changed()
        self.page.checkpoint();self.page.journal.wait()
        self.assertTrue(self.page.journal.path.exists())

    def test_region_only_save_as_reopens_saved_parameters(self):
        self.mark()
        self.page.head.mode.setCurrentIndex(self.page.head.mode.findData('direct_row_spacing'))
        self.page.head.fields['row_spacing_mm'].setText('7')
        expected=self.page.request()
        target=Path(self.temp.name)/'parameters.region.zip'
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(target),'')):
            self.page.save_as();self.wait_task()
        reopened=open_project(target,workspace=Path(self.temp.name)/'reopened_parameters')
        self.assertIsNone(reopened.saved_plan)
        self.assertEqual(reopened.planning_parameters,expected)
        self.page.install_model(reopened,target)
        self.assertEqual(self.page.request(),expected)
        self.assertEqual(self.page.head.record()['row_spacing_mm'],7)
        self.assertFalse(self.page.dirty)

    def test_cancelled_region_extraction_does_not_mutate_original(self):
        import threading
        from femto.annotation.fixed_rgb_geometry import GeometryTaskCancelled
        cancel=threading.Event();cancel.set()
        with self.assertRaises(GeometryTaskCancelled):
            extract_region(self.model,self.model.projection.valid,np.zeros((100,100),bool),'区域','',cancel)
        self.assertEqual(self.model.store.regions,[])

    def test_qt_stroke_undo_and_busy_gate(self):
        canvas=self.page.canvas
        rectangle=canvas.image_rect();center=rectangle.center().toPoint()
        QTest.mousePress(canvas,Qt.LeftButton,pos=center)
        QTest.mouseRelease(canvas,Qt.LeftButton,pos=center)
        self.assertTrue(canvas.work.any())
        self.assertTrue(self.page.selection_dirty)
        canvas.undo();self.assertFalse(canvas.work.any())
        self.page.runner.busy=True;self.page.refresh()
        QTest.mouseClick(canvas,Qt.LeftButton,pos=center)
        self.assertFalse(canvas.work.any())
        self.assertFalse(self.page.can_leave())
        self.page.runner.busy=False;self.page.refresh()


if __name__=='__main__':unittest.main()
