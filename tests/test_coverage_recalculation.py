"""Same-input P2 recalculation must not replace a larger original confirmed set."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from copy import copy, deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from femto.planning import coverage_certificate as coverage, coverage_workflow as workflow
from femto.planning.planning_revision import digest
from femto.planning.surface_trajectory import SurfaceTrajectory
from test_coverage_certificate import reseal
from test_coverage_workflow import fixture


class CoverageRecalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.addClassCleanup(cls.temp.cleanup)
        cls.base = fixture(Path(cls.temp.name)/'source')
        _, _, plan, certificate, ledger = cls.base
        domain, seeds, model, binding = workflow.bundle(SurfaceTrajectory.from_dict(plan['surface_trajectory']), ledger)
        cls.partial = coverage.evaluate(domain, seeds, model, route_binding=binding,
            algorithm=coverage.CURRENT_ALGORITHM, budget=coverage.CoverageBudget(seconds=30, max_comparisons=0))
        coverage.validate(cls.partial, domain, seeds, model, route_binding=binding)
        assert certificate.summary['confirmed_leaf_count'] > 0
        assert cls.partial.summary['confirmed_leaf_count'] == 0

    def setUp(self):
        model, region, plan, self.certificate, self.ledger = self.base
        self.model = copy(model); self.model.store = copy(model.store)
        self.region, self.original = deepcopy(region), deepcopy(plan)
        self.model.store.regions = [self.region]
        self.plan = workflow.attach(self.model, self.region, self.original, self.certificate, self.ledger)

    def caches(self):
        return self.model.coverage_certificate, self.model.target_ledger, self.model.coverage_display

    def assert_caches(self, previous):
        for old, current in zip(previous, self.caches()):
            self.assertIs(old, current)

    def test_smaller_valid_budget_candidate_is_rejected_before_display_or_cache_changes(self):
        before, plan = self.caches(), deepcopy(self.plan)
        with patch('femto.planning.coverage_display.build', side_effect=AssertionError('display should not be rebuilt')):
            with self.assertRaises(workflow.RecalculationRejected) as raised:
                workflow.attach(self.model, self.region, self.plan, self.partial, self.ledger)
        self.assertGreater(raised.exception.comparison['lost_weight_units'], 0)
        self.assertFalse(raised.exception.comparison['retained_subset'])
        self.assertIn('原覆盖已保留', str(raised.exception))
        self.assert_caches(before); self.assertEqual(self.plan, plan)

    def test_identical_confirmation_can_replace_without_route_or_target_changes(self):
        candidate = copy(self.certificate)
        result = workflow.attach(self.model, self.region, self.plan, candidate, self.ledger)
        self.assertEqual(result, self.plan)
        self.assertIs(self.model.coverage_certificate, candidate)
        self.assertIs(self.model.target_ledger, self.ledger)

    def test_valid_superset_is_accepted_even_when_the_computational_budget_differs(self):
        partial = workflow.attach(self.model, self.region, self.original, self.partial, self.ledger)
        compared = workflow.require_recalculation_retention(self.partial, self.certificate, self.ledger)
        self.assertEqual(compared['lost_weight_units'], 0)
        self.assertGreater(compared['gained_weight_units'], 0)
        result = workflow.attach(self.model, self.region, partial, self.certificate, self.ledger)
        self.assertEqual(result[workflow.KEY], self.certificate.metadata)

    def test_explicit_route_without_previous_coverage_does_not_retain_an_unrelated_cache(self):
        result = workflow.attach(self.model, self.region, self.original, self.partial, self.ledger)
        self.assertEqual(result[workflow.KEY], self.partial.metadata)
        self.assertIs(self.model.coverage_certificate, self.partial)
        self.assertNotIn(workflow.KEY, self.original)

    def test_changed_inputs_and_ledger_cannot_be_compared_as_the_same_target(self):
        for key in ('domain', 'seeds_sha256', 'footprint', 'route'):
            with self.subTest(key=key):
                data = deepcopy(self.partial.metadata)
                if key == 'seeds_sha256': data['inputs'][key] = 'a'*64
                elif key == 'footprint':
                    from femto.planning.footprint_model import nominal
                    data['inputs'][key] = nominal(10)
                elif key == 'domain': data['inputs'][key]['target_sha256'] = 'a'*64
                else: data['inputs'][key]['route_semantics_sha256'] = 'a'*64
                data['coverage_computation_sha256'] = digest(data['inputs'])
                changed = reseal(self.partial, metadata=data)
                with self.assertRaisesRegex(ValueError, '同一原目标'):
                    workflow.require_recalculation_retention(self.certificate, changed, self.ledger)
        ledger = copy(self.ledger)
        from dataclasses import replace
        ledger = replace(ledger, metadata=dict(ledger.metadata, sha256='a'*64))
        with self.assertRaisesRegex(ValueError, '当前原目标台账'):
            workflow.require_recalculation_retention(self.certificate, self.partial, ledger)

    def test_missing_previous_assets_cannot_bypass_recalculation_admission(self):
        self.model.coverage_certificate = None
        before = self.caches()
        with self.assertRaisesRegex(ValueError, '资产缺失'):
            workflow.attach(self.model, self.region, self.plan, self.partial, self.ledger)
        self.assert_caches(before)

    def test_cancel_during_set_comparison_preserves_every_previous_cache(self):
        from femto.annotation.fixed_rgb_geometry import GeometryTaskCancelled
        before = self.caches()
        def cancel(): raise GeometryTaskCancelled('cancel during retained-set comparison')
        with self.assertRaises(GeometryTaskCancelled):
            workflow.attach(self.model, self.region, self.plan, copy(self.certificate), self.ledger, check=cancel)
        self.assert_caches(before)

    def test_qt_real_recalculation_rejection_preserves_plan_review_archive_and_display(self):
        from dataclasses import replace
        from PySide6.QtCore import QEventLoop, QTimer
        from PySide6.QtWidgets import QApplication
        from femto.app.qt_planning import PlanningPage
        from femto.planning.planning_drafts import Journal
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.model.saved_plan = self.plan
            page = PlanningPage(); page.resize(1120,740)
            page.install_model(self.model, root/'saved.zip', Journal(root/'saved.zip', root/'drafts'))
            page.set_stage(1); page.show()
            quality_loop = QEventLoop(); quality_poll = QTimer(); quality_limit = QTimer()
            quality_poll.setInterval(25); quality_limit.setSingleShot(True)
            quality_poll.timeout.connect(lambda:quality_loop.quit() if not page.quality.runner.busy else None)
            quality_limit.timeout.connect(quality_loop.quit)
            page.quality.timer.stop(); page.quality._start()
            quality_poll.start(); quality_limit.start(30000)
            if page.quality.runner.busy: quality_loop.exec()
            quality_poll.stop(); quality_limit.stop()
            self.assertFalse(page.quality.runner.busy)
            self.assertEqual(page.quality.result['status'], 'checked')
            page.reviewed.setChecked(True)
            page.coverage.show_layer.setChecked(True)
            previous = page.plan, self.caches(), page.dirty, page.archive, page.reviewed.isChecked()
            actual = workflow.calculate
            def limited(*args, **kwargs):
                kwargs['budget'] = replace(kwargs['budget'], max_comparisons=0)
                return actual(*args, **kwargs)
            loop = QEventLoop(); timer = QTimer(); timer.setSingleShot(True)
            timer.timeout.connect(loop.quit); page.runner.finished.connect(loop.quit)
            try:
                with patch.object(workflow, 'calculate', side_effect=limited):
                    page.verify_coverage(); timer.start(30000)
                    if page.busy: loop.exec()
                self.assertFalse(page.busy)
                self.assertIn('原覆盖已保留', page._error)
                self.assertIs(page.plan, previous[0]); self.assert_caches(previous[1])
                self.assertEqual((page.dirty, page.archive, page.reviewed.isChecked()), previous[2:])
                self.assertIsNotNone(page.canvas._coverage_image)
            finally:
                timer.stop(); page.runner.finished.disconnect(loop.quit)
                page.shutdown(); page.close(); app.processEvents()


if __name__ == '__main__':
    unittest.main()
