"""All calculation dependencies, UI-only metadata, immutable worker input and races."""
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from test_fixed_rgb_experience import make_model, K
from femto.app.qt_common import TaskRunner, TaskResultDiscarded
from femto.app.qt_planning import PlanningPage
from femto.app.qt_planning_io import clone_model
from femto.planning import planning_revision as revision
from femto.planning.treatment_head import head_parameters


class PlanningRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.model = make_model(self.temp.name)
        self.model.view_reference['undistorted_intrinsic'] = K.tolist()
        self.model.store.origin_reference = dict(kind='initial_color_camera', provenance='synthetic_test')
        self.region = self.model.region([[0, 0], [99, 0], [99, 99]], 'synthetic', '',
            selection=dict(mask=self.model.projection.valid.copy(), method='work_only'))
        self.head = head_parameters('direct_row_spacing', row_spacing_mm=7)

    def capture(self):return revision.capture(self.model, self.region, self.head)

    def test_every_dependency_changes_the_complete_revision(self):
        before = self.capture()
        for dependency in revision.DEPENDENCIES:
            with self.subTest(dependency=dependency):
                inputs = deepcopy(before['inputs']); inputs[dependency]['changed_for_test'] = True
                current = revision.create(inputs)
                self.assertNotEqual(before['sha256'], current['sha256'])
                self.assertEqual(revision.changed(before, current), [dependency])

    def test_actual_rgb_depth_mesh_intrinsic_and_transform_are_hashed(self):
        for dependency in ('image', 'depth', 'geometry', 'calibration', 'reference'):
            with self.subTest(dependency=dependency):
                model = clone_model(self.model, freeze_inputs=True)
                before = revision.capture(model, self.region, self.head)
                if dependency == 'image':model.rgb = model.rgb.copy(); model.rgb[0, 0, 0] += 1
                elif dependency == 'depth':model.projection.depth = model.projection.depth.copy(); model.projection.depth[0, 0] += 1
                elif dependency == 'geometry':np.asarray(model.mesh.vertices)[0, 0] += .001
                elif dependency == 'calibration':model.projection.k = model.projection.k.copy(); model.projection.k[0, 0] += 1
                else:model.store.origin_reference['transform_revision'] = 'changed'
                after = revision.capture(model, self.region, self.head)
                self.assertIn(dependency, revision.changed(before, after))

    def test_metadata_navigation_and_save_uuid_do_not_change_calculation(self):
        before = self.capture()
        self.region.update(name='new name', note='note', updated_at='changed')
        self.model.revision_id = 'independent_save_uuid'
        self.model.parent_revision_id = 'parent'; self.model.page = 4
        self.model.store.source = str(Path(self.temp.name)/'relocated.ply')
        self.assertEqual(before, self.capture())

    def test_quality_width_and_route_algorithm_do_not_shrink_target_identity(self):
        before = self.capture()
        for field in ('quality', 'effect_model', 'algorithms'):
            inputs = deepcopy(before['inputs'])
            if field == 'algorithms':inputs[field]['route'] = 'future-route'
            else:inputs[field]['changed'] = True
            after = revision.create(inputs)
            self.assertNotEqual(before['sha256'], after['sha256'])
            self.assertEqual(before['target_id'], after['target_id'])

    def test_worker_snapshot_does_not_share_mutable_evidence(self):
        frozen = clone_model(self.model, freeze_inputs=True)
        frozen_region = frozen.store.regions[0]
        before = revision.capture(frozen, frozen_region, self.head)
        self.model.projection.depth[50, 50] += 1
        self.model.rgb[0, 0, 0] += 1
        np.asarray(self.model.mesh.vertices)[0, 0] += .001
        self.model.view_reference['new'] = True
        self.assertEqual(before, revision.capture(frozen, frozen_region, self.head))


class PlanningInstallationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.runner = TaskRunner(); self.released = threading.Event()
        self.success, self.failure = [], []
        self.current = 'first'
        self.addCleanup(self.stop)

    def stop(self):
        self.released.set(); self.runner.shutdown(2000); self.app.processEvents()

    def wait(self, runner=None):
        runner = self.runner if runner is None else runner
        loop = QEventLoop(); timer = QTimer(); timer.setSingleShot(True)
        timer.timeout.connect(loop.quit); runner.finished.connect(loop.quit)
        timer.start(5000)
        if runner.busy:loop.exec()
        timer.stop(); runner.finished.disconnect(loop.quit)
        self.assertFalse(runner.busy, 'worker exceeded original deadline')

    def start(self):
        def work(cancel, progress):
            self.released.wait(5)
            return 'late success despite ignored cancellation'
        self.assertTrue(self.runner.start(work, self.success.append, self.failure.append,
                        revision=self.current, current_revision=lambda:self.current))

    def test_cancelled_success_is_never_installed(self):
        self.start(); self.runner.cancel(); self.released.set(); self.wait()
        self.assertEqual(self.success, []); self.assertEqual(len(self.failure), 1)

    def test_old_revision_is_discarded_and_next_request_can_succeed(self):
        self.start(); self.current = 'second'; self.released.set(); self.wait()
        self.assertEqual(self.success, []); self.assertIsInstance(self.failure[0], TaskResultDiscarded)
        self.start(); self.wait(); self.assertEqual(len(self.success), 1)

    def test_old_generation_and_duplicate_completion_cannot_install(self):
        self.start()
        self.runner._complete(self.runner._number-1, 'wrong generation', None)
        self.assertTrue(self.runner.busy); self.assertEqual(self.success, [])
        self.released.set(); self.wait()
        self.runner._complete(self.runner._number, 'duplicate', None)
        self.assertEqual(len(self.success), 1)

    def test_finishing_a_changed_stroke_cannot_start_a_prepared_old_job(self):
        with tempfile.TemporaryDirectory() as folder:
            model = make_model(folder)
            page = PlanningPage(); page.install_model(model, model.reconstruction)
            try:
                page.canvas._before = (page.canvas.work.copy(), page.canvas.forbidden.copy())
                page.canvas.work[50, 50] = 1
                accepted = []
                self.assertFalse(page._run(lambda cancel, progress: 'old region', accepted.append))
                self.app.processEvents()
                self.assertFalse(page.busy); self.assertTrue(page.selection_dirty)
                self.assertEqual(accepted, [])
                self.assertIsNone(page.canvas._before)
            finally:
                page.shutdown(); page.close(); self.app.processEvents()

    def test_quality_uses_frozen_evidence_and_discards_in_place_input_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            model = make_model(folder)
            page = PlanningPage(); page.install_model(model, model.reconstruction)
            received_depth = []
            def delayed(snapshot, work, forbidden, cancel):
                self.released.wait(5)
                received_depth.append(int(snapshot.projection.depth[50, 50]))
                return dict(status='unavailable', message='old quality', rgba=None, counts={})
            try:
                page.quality.timer.stop()
                page.canvas.work[50, 50] = 1
                with patch('femto.app.qt_region_quality.assess_selection', side_effect=delayed):
                    page.quality._start()
                    model.projection.depth[50, 50] = 501
                    self.released.set(); self.wait(page.quality.runner)
                self.assertEqual(received_depth, [500])
                self.assertIsNone(page.quality.result)
                self.assertIsNone(page.canvas._quality_image)
                self.assertIn('输入或当前模型已变化', page.quality.summary.toolTip())
                self.assertIn('输入或当前模型已变化', page.quality.summary.accessibleName())
            finally:
                self.released.set(); page.shutdown(); page.close(); self.app.processEvents()

    def test_planning_page_rejects_replaced_model_and_in_place_input_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            model = make_model(folder); model.view_reference['undistorted_intrinsic'] = K.tolist()
            page = PlanningPage(); page.install_model(model, model.reconstruction)
            try:
                for change in ('model', 'depth', 'mask', 'parameters'):
                    self.released.clear(); accepted = []
                    page._run(lambda cancel, progress:(self.released.wait(5), 'old')[1], accepted.append)
                    if change == 'model':page.model = clone_model(page.model)
                    elif change == 'depth':page.model.projection.depth[0, 0] += 1
                    elif change == 'mask':page.canvas.work[0, 0] = 1
                    else:page.head.fields['row_spacing_mm'].setText('6')
                    loop = QEventLoop(); page.runner.finished.connect(loop.quit)
                    timer = QTimer(); timer.setSingleShot(True); timer.timeout.connect(loop.quit); timer.start(5000)
                    self.released.set()
                    if page.busy:loop.exec()
                    timer.stop(); page.runner.finished.disconnect(loop.quit)
                    self.assertFalse(page.busy); self.assertEqual(accepted, [], change)
            finally:
                page.shutdown(); page.close(); self.app.processEvents()


class PlanningReviewLayoutTests(unittest.TestCase):
    def test_complete_target_summary_remains_readable_before_review(self):
        import test_route_layout as route_layout_fixture
        from PySide6.QtCore import QPoint
        fixture = route_layout_fixture.RouteLayoutTests()
        fixture.setUpClass(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        fixture.generate(); page = fixture.page
        for width, height in ((1600, 940), (1120, 740)):
            fixture.window.resize(width, height)
            page.set_stage(2); fixture.settle()
            self.assertIn('已定位面积', page.save_summary.text())
            self.assertGreaterEqual(page.save_summary.height(),
                                    page.save_summary.heightForWidth(page.save_summary.width()))
            page.inspector_scroll.ensureWidgetVisible(page.reviewed)
            fixture.settle()
            viewport = page.inspector_scroll.viewport()
            point = page.reviewed.mapTo(viewport, QPoint(0, 0))
            self.assertTrue(viewport.rect().contains(point))
            self.assertTrue(viewport.rect().contains(point+QPoint(page.reviewed.width()-1, page.reviewed.height()-1)))
            self.assertFalse(page.reviewed.isChecked())


if __name__ == '__main__':unittest.main()
