"""Gap requests remain bound across capture task persistence and new-model review."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

from copy import copy,deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

from test_coverage_workflow import fixture
from femto.dataset.task_context import new_task as task
from femto.planning.gap_followup import prepare
from femto.dataset.task_context import checked_task,save_task,read_task
from femto.dataset.gap_followup import checked,digest
from femto.app.qt_main import WorkstationWindow


class GapFollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        cls.temp=tempfile.TemporaryDirectory();cls.base=fixture(Path(cls.temp.name)/'model')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        model,self.region,self.plan,_,_=self.base;self.model=copy(model);self.model.task_context=task()
        from femto.planning import surface_workspace
        self.plan=surface_workspace.generate(self.model,self.region,self.plan['treatment_head'])
        self.mask=np.zeros(self.model.projection.depth.shape,bool);self.mask[40:42,40:44]=True

    def request(self):return prepare(self.model,self.plan,self.mask,8,5,self.root/'evidence')

    def test_exact_gap_snapshot_and_link_survive_new_capture_task_and_later_read(self):
        request=self.request();context=dict(self.model.task_context,gap_followup=request)
        saved=self.root/'new_result';saved.mkdir();save_task(saved,context)
        self.assertEqual(read_task(saved),context)
        self.assertEqual(checked_task(context)['gap_followup'],request)
        with np.load(self.root/'evidence/reference_gap.npz') as archive:
            np.testing.assert_array_equal(archive['mask'],self.mask)
        self.assertEqual(request['origin_plan_sha256'],self.plan_digest())
        self.assertNotIn('T_W0_Ct',request)

    def plan_digest(self):
        from femto.planning.planning_revision import digest as plan_digest
        return plan_digest(self.plan)

    def test_wrong_task_and_modified_request_are_rejected(self):
        request=self.request()
        with self.assertRaises(ValueError):checked(request,'wrong_task')
        changed=deepcopy(request);changed['reason']=2
        with self.assertRaises(ValueError):checked(changed,request['task_id'])
        context=dict(self.model.task_context,gap_followup=request)
        context['task_id']='different-task'
        with self.assertRaises(ValueError):checked_task(context)
        self.assertNotIn('origin_body_selection_sha256',request)

    def test_navigation_carries_request_and_new_task_clears_association(self):
        request=self.request();window=WorkstationWindow(connect_camera=False)
        self.addCleanup(window.close)
        page=window.planning;page.model=self.model;page.plan=self.plan;page._gap_followup_request=request
        self.assertTrue(window.rescan_region(page))
        self.assertEqual(window.capture.task_context['gap_followup'],request)
        self.assertIsNone(page._gap_followup_request)
        window.capture.new_capture()
        self.assertNotIn('gap_followup',window.capture.task_context)
        window.capture.shutdown();window.planning.shutdown();window.review.shutdown()

    def test_cancelled_snapshot_leaves_no_published_request(self):
        def stop():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            prepare(self.model,self.plan,self.mask,8,5,self.root/'cancelled',stop)
        self.assertFalse((self.root/'cancelled').exists())

    def test_new_model_replanning_coverage_and_portable_read_preserve_request_without_old_proof(self):
        from femto.planning import surface_workspace,coverage_workflow,coverage_certificate,region_project
        request=self.request();model,region,_,_,_=fixture(self.root/'new_model')
        model.task_context=dict(self.model.task_context,gap_followup=request)
        from femto.planning.treatment_head import head_parameters
        plan=surface_workspace.generate(model,region,head_parameters('direct_row_spacing',row_spacing_mm=7))
        self.assertNotIn(coverage_workflow.KEY,plan)
        self.assertEqual(plan['planning_context']['revision']['inputs']['reference']['task']['gap_followup'],request)
        certificate,ledger=coverage_workflow.calculate(model,region,plan,budget=coverage_certificate.CoverageBudget(max_depth=1))
        plan=coverage_workflow.attach(model,region,plan,certificate,ledger)
        path=region_project.save_project(model,region,self.root/'new_plan.zip',plan)
        opened=region_project.open_project(path,workspace=self.root/'opened_new')
        self.assertEqual(opened.task_context['gap_followup'],request)
        self.assertEqual(opened.coverage_certificate.metadata,certificate.metadata)
