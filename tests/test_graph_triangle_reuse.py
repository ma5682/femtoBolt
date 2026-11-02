"""Prepared incident triangles cannot change graph nodes or emitted motion."""
import unittest
from unittest.mock import patch

from femto.planning import continuous_quality as continuous, surface_trajectory as trajectory
import test_lazy_surface_graph as graph_fixture


class GraphTriangleReuseTests(unittest.TestCase):
    def test_complete_route_and_graph_report_match_unprepared_predicates(self):
        fixture = graph_fixture.LazySurfaceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        candidate, mesh, region, record, domain = fixture.inputs(split=True)
        prepared = continuous.replan(candidate, mesh, region, record, 5, domain=domain)
        original = trajectory._point_on_triangle
        calls = []
        def unprepared(*args, **kwargs):
            if kwargs.pop('_prepared', None) is not None:
                calls.append(True)
            return original(*args, **kwargs)
        with patch.object(trajectory, '_point_on_triangle', side_effect=unprepared):
            baseline = continuous.replan(candidate, mesh, region, record, 5, domain=domain)
        self.assertTrue(calls)
        self.assertEqual(prepared.to_dict(), baseline.to_dict())
        continuous.validate(prepared, mesh, region)


if __name__ == '__main__':
    unittest.main()
