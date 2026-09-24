"""Controller gates and recovery tests, using only prototype mocks."""
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtCore import QCoreApplication
from app.controller import WorkflowController
from app.navigation import navigation_reason
from models.planning_state import PlanningParameters


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.c = WorkflowController(Path(self.temp.name))

    def tearDown(self):
        self.c.capture_timer.stop()
        self.c.operation_timer.stop()
        self.temp.cleanup()

    def capture(self):
        self.c.dispatch("start")
        self.c.tick_capture()
        self.c.tick_capture()
        self.c.dispatch("finish")
        self.c.complete_operation()

    def plan(self):
        self.capture()
        self.c.dispatch("to_region")
        self.c.load_example()
        self.c.dispatch("plan_region")
        self.c.complete_operation()
        self.c.complete_operation()

    def test_cannot_skip_or_finish_too_early(self):
        self.c.go(3)
        self.assertEqual(self.c.state.stage, 0)
        self.c.dispatch("start")
        self.c.dispatch("finish")
        self.assertEqual(self.c.state.capture.phase, "capturing")
        self.assertTrue(navigation_reason(self.c.state, 1))

    def test_save_is_persistent_and_parameters_are_traceable(self):
        self.plan()
        self.assertTrue(self.c.state.plan_current)
        self.c.dispatch("confirm")
        self.assertEqual(self.c.state.stage, 3)
        self.c.dispatch("save")
        path = Path(self.c.state.plan.saved_path)
        record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["final_spacing_mm"], 2.4)
        self.assertTrue(record["parameters_are_placeholders"])
        self.assertFalse(record["execution_ready"])
        original = path.read_bytes()
        self.c.go(2)
        self.c.set_parameter("direct_spacing", "6")
        self.assertFalse(self.c.state.plan_current)
        self.assertTrue(navigation_reason(self.c.state, 3))
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(len(self.c.store.records()), 1)

    def test_quality_recovery(self):
        self.c.set_scenario("quality")
        self.capture()
        self.assertEqual(self.c.state.capture.phase, "failed")
        self.c.go(1)
        self.assertEqual(self.c.state.stage, 0)
        self.c.dispatch("repair_capture")
        self.c.dispatch("finish")
        self.c.complete_operation()
        self.assertEqual(self.c.state.capture.phase, "passed")

    def test_geometry_recovery_preserves_regions(self):
        self.c.set_scenario("geometry")
        self.capture()
        self.c.dispatch("to_region")
        self.c.load_example()
        before = deepcopy(self.c.state.region.strokes)
        self.c.dispatch("plan_region")
        self.c.complete_operation()
        self.assertEqual(self.c.state.region.support, "missing")
        self.c.dispatch("repair_geometry")
        self.c.dispatch("finish")
        self.c.complete_operation()
        self.c.dispatch("to_region")
        self.assertEqual(self.c.state.region.strokes, before)
        self.assertEqual(self.c.state.region.support, "valid")

    def test_broken_route_blocks_confirmation(self):
        self.c.set_scenario("route")
        self.plan()
        self.assertEqual(self.c.state.plan.phase, "failed")
        self.c.go(3)
        self.assertEqual(self.c.state.stage, 2)
        self.c.dispatch("repair_route")
        self.c.complete_operation()
        self.assertTrue(self.c.state.plan_current)

    def test_edit_and_undo_invalidate_planning(self):
        self.plan()
        self.c.go(1)
        before = deepcopy(self.c.state.region.strokes)
        self.c.clear_regions()
        self.assertFalse(self.c.state.region.has_work)
        self.c.undo()
        self.assertEqual(self.c.state.region.strokes, before)
        self.assertEqual(self.c.state.region.support, "unchecked")
        self.assertFalse(self.c.state.plan_current)

    def test_parameter_validation_rejects_without_clamping(self):
        p = PlanningParameters(mode="width", effective_width="3", overlap="20")
        self.assertAlmostEqual(p.spacing()[0], 2.4)
        for overlap in ("100", "-1", "nan"):
            p.overlap = overlap
            self.assertTrue(p.spacing()[1])
        p.mode = "direct"
        for value in ("0.49", "100.1", "nan", "inf", ""):
            p.direct_spacing = value
            self.assertTrue(p.spacing()[1])
            self.assertEqual(p.direct_spacing, value)


if __name__ == "__main__":
    unittest.main()
