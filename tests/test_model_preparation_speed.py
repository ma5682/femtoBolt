"""Selection equivalence, one-owner handoff and early display isolation."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer

from femto.annotation.scene_annotations import region_boundary
from femto.annotation.fixed_rgb_geometry import GeometryTaskCancelled
from femto.dataset.prepared_review import PreparedCapture
from test_fixed_rgb_experience import make_model, ring, quad, POLYGON
from test_qt_scene_panel import FakeViewport
import test_qt_planning as fixture


class BoundaryEquivalenceTests(unittest.TestCase):
    def test_holes_disconnected_duplicate_and_nonmanifold_edges_match_original(self):
        rng = np.random.default_rng(421)
        for mesh in (ring(), ring()+quad(left=.3, right=.6), quad()+quad(.6)):
            for ids in ([], list(range(len(mesh.triangles))), [0, 0],
                        rng.integers(0, len(mesh.triangles), 31).tolist()):
                triangles = np.asarray(mesh.triangles)[ids]
                edges = np.concatenate([triangles[:, pair] for pair in ([0,1], [1,2], [2,0])])
                unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
                expected = np.asarray(mesh.vertices)[unique[counts == 1]].tolist()
                self.assertEqual(region_boundary(mesh, ids), expected)

    def test_boundary_is_built_once_and_failed_display_leaves_store_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            model = make_model(temp, ring())
            with patch('femto.annotation.scene_annotations.region_boundary', wraps=region_boundary) as boundary:
                item = model.region(POLYGON, 'ring', '')
                self.assertEqual(boundary.call_count, 1)
            before = deepcopy(model.store.regions)
            history = list(model.store._history)
            with patch.object(model.projection, 'visible_boundary', side_effect=GeometryTaskCancelled('cancel')):
                with self.assertRaises(GeometryTaskCancelled):model.region(POLYGON, 'new', '', replace_existing=True)
            self.assertEqual(model.store.regions, before)
            self.assertEqual(model.store._history, history)
            self.assertEqual(item['boundary_world_m'], region_boundary(model.mesh, item['triangle_indices']))

    def test_strict_index_validation_still_rejects_bool_float_subclass_and_duplicates(self):
        class Index(int):pass
        with tempfile.TemporaryDirectory() as temp:
            model = make_model(temp)
            for ids in ([True], [0.], [Index(0)], [-1], [2], [0, 0]):
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    model.store.normalized_region(dict(triangle_indices=ids))


class PreparedHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.asset = self.root/'asset'; self.asset.write_bytes(b'original')
        self.completion = dict(binding={'id':'first'}, confirmed_for_planning=False)
        self.roi = {'rect_norm':[0,0,1,1]}
        self.model = object()
        def fingerprints(*args):return {'asset':hashlib.sha256(self.asset.read_bytes()).hexdigest()}
        self.fingerprints = fingerprints
        for target, side in (
                ('femto.dataset.prepared_review.asset_fingerprints', fingerprints),
                ('femto.dataset.acquisition_completion.read_completion', lambda *a, **k: deepcopy(self.completion))):
            mocked=patch(target, side_effect=side);mocked.start();self.addCleanup(mocked.stop)

    def prepared(self):
        return PreparedCapture(self.root,self.model,self.fingerprints(),self.completion,
                               self.roi,{'supported':3},np.zeros((2,2,3),np.uint8))

    def test_two_verified_handoffs_are_once_only_and_preview_is_owned(self):
        prepared=self.prepared()
        summary,image=prepared.review(self.root,self.roi,lambda:None)
        summary['supported']=0;image[:]=255
        self.assertEqual(prepared._summary['supported'],3)
        self.assertFalse(prepared._preview.any())
        self.assertIsNone(prepared.review(self.root,self.roi,lambda:None))
        self.assertIs(prepared.take_model(self.root,lambda:None),self.model)
        self.assertIsNone(prepared.take_model(self.root,lambda:None))
        self.assertIsNone(prepared._model)

    def test_changed_bytes_completion_roi_or_source_prevents_reuse(self):
        for change in ('bytes','completion','roi','source'):
            with self.subTest(change=change):
                self.asset.write_bytes(b'original')
                self.completion['confirmed_for_planning']=False
                prepared=self.prepared()
                roi=deepcopy(self.roi);root=self.root
                if change=='bytes':self.asset.write_bytes(b'modified')
                if change=='completion':self.completion['confirmed_for_planning']=True
                if change=='roi':roi['rect_norm'][2]=.5
                if change=='source':root=self.root/'other'
                self.assertIsNone(prepared.review(root,roi,lambda:None))
                self.assertIsNone(prepared._model)

    def test_change_between_handoffs_and_cancellation_release_model(self):
        prepared=self.prepared();prepared.review(self.root,self.roi,lambda:None)
        self.asset.write_bytes(b'modified')
        self.assertIsNone(prepared.take_model(self.root,lambda:None))
        prepared=self.prepared()
        def cancel():raise GeometryTaskCancelled('cancel')
        with self.assertRaises(GeometryTaskCancelled):prepared.review(self.root,self.roi,cancel)
        self.assertIsNone(prepared._model)


class PreviewViewport(FakeViewport):
    def set_snapshot(self, snapshot):self.snapshot=snapshot


class EarlyModelDisplayTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixture.QtPlanningTests();self.fixture.setUpClass();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.page=self.fixture.page
        self.model=self.fixture.model
        self.model.store.local_context={'reconstruction_scope':{'schema':'test'}}
        self.model.capture_source=self.model.reconstruction
        self.model.acquisition_completion={'confirmed_for_planning':False}
        self.work=self.model.projection.valid.astype(np.uint8)
        self.page.dirty=False
        self.viewport=patch('femto.app.qt_scene_panel.create_viewport',side_effect=PreviewViewport)
        self.viewport.start();self.addCleanup(self.viewport.stop)

    def exercise(self, outcome):
        release=threading.Event();arrived=threading.Event()
        from femto.app.qt_planning_io import extract_region
        def extract(*args):
            arrived.set()
            if not release.wait(10):raise RuntimeError('test extraction timeout')
            if outcome=='failure':raise ValueError('test selection failed')
            return extract_region(*args)
        seen=[]
        original=self.page.preparation.show_model
        def show(*args):seen.append(args);return original(*args)
        try:
            with patch('femto.app.qt_planning.load_source',return_value=self.model), \
                 patch('femto.geometry.local_region.model_selection',return_value=self.work), \
                 patch('femto.app.qt_planning.extract_region',side_effect=extract), \
                 patch('femto.planning.planning_drafts.read',return_value=None), \
                 patch.object(self.page.preparation,'show_model',side_effect=show):
                self.assertTrue(self.page.load_source(self.model.reconstruction,force=True,restore_draft=False,review_model=True))
                loop=QEventLoop();poll=QTimer();poll.setInterval(10)
                poll.timeout.connect(lambda:loop.quit() if seen and arrived.is_set() else None)
                deadline=QTimer();deadline.setSingleShot(True);deadline.timeout.connect(loop.quit)
                poll.start();deadline.start(5000);loop.exec();poll.stop();deadline.stop()
                self.assertEqual(len(seen),1)
                self.assertTrue(self.page.busy)
                self.assertFalse(self.page.primary.isEnabled())
                self.assertTrue(len(seen[0][0].triangles))
                self.assertFalse(seen[0][0].vertices.flags.writeable)
                self.assertIs(self.page.preparation.views.currentWidget(),self.page.preparation.viewport)
                self.assertIn('待检查',self.page.preparation.readonly_label.text())
                # A previous request and changed inputs cannot publish previews.
                self.page._show_preparation_preview(object(),*seen[0]);self.assertEqual(len(seen),1)
                token=self.page._loading_preview_token
                self.page._input_generation+=1
                self.page._show_preparation_preview(token,*seen[0]);self.assertEqual(len(seen),1)
                self.page._input_generation-=1
                if outcome=='cancel':
                    self.page.runner.cancel()
                    self.page._show_preparation_preview(token,*seen[0]);self.assertEqual(len(seen),1)
                release.set()
                self.fixture.wait_task() if outcome=='success' else self.wait_finished()
                self.assertIsNone(self.page.preparation.viewport)
                if outcome=='success':self.assertTrue(self.page.primary.isEnabled())
                elif outcome=='failure':self.assertEqual(self.page.preparation.state,'failed')
                else:self.assertIsNone(self.page.loading_source)
        finally:
            release.set()
            if self.page.busy:self.wait_finished()

    def wait_finished(self):
        loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit)
        self.page.runner.finished.connect(loop.quit)
        try:
            timer.start(10000)
            if self.page.busy:loop.exec()
        finally:self.page.runner.finished.disconnect(loop.quit);timer.stop()
        self.assertFalse(self.page.busy)

    def test_model_and_quality_appear_before_extraction_completes(self):self.exercise('success')
    def test_cancel_discards_preview_and_late_result(self):self.exercise('cancel')
    def test_failure_releases_preview_and_keeps_retry_state(self):self.exercise('failure')


if __name__=='__main__':unittest.main()
