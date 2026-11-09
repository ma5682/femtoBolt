"""The visible route workflow defers optional area work until requested."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import test_qt_planning as fixtures
from femto.planning import target_surface as target, region_project


class DeferredTargetUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.QtPlanningTests.setUpClass()

    def setUp(self):
        self.case = fixtures.QtPlanningTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)

    def test_generate_save_reopen_then_request_coverage(self):
        case = self.case
        case.mark()
        page = case.page
        with patch.object(target, '_construct', side_effect=AssertionError('unrequested area work')):
            page.generate()
            case.wait_task()
            self.assertTrue(target.is_deferred(page.plan['planning_context']['target']))
            self.assertEqual(page.coverage.verify.text(), '计算面积与覆盖')
            self.assertNotIn('未定位', page._route_attention_text())
            before = deepcopy(page.plan['surface_trajectory']['ordered_items'])
            page.set_stage(2)
            page.reviewed.setChecked(True)
            page.save_plan()
            case.wait_task()
            reopened = region_project.open_project(page.archive/'planning.region.zip',
                workspace=Path(case.temp.name)/'reopened')
            page.install_model(reopened, page.archive/'planning.region.zip', case.journal)
            self.assertEqual(page.plan['surface_trajectory']['ordered_items'], before)
        page.verify_coverage()
        case.wait_task()
        self.assertFalse(target.is_deferred(page.plan['planning_context']['target']))
        self.assertEqual(page.plan['surface_trajectory']['ordered_items'], before)
        self.assertIsNotNone(page.model.coverage_certificate)


if __name__ == '__main__':
    unittest.main()
