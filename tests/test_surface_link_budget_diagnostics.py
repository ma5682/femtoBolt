"""Total output-point accounting is distinct from face-search work limits."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest import mock

from femto.planning.surface_row_turns import connect_links
from femto.planning.surface_trajectory import Gap, ScanSegment


def sample(x, row, role='scan'):
    return dict(xyz_m=[x, 0., 1.], row_index=row, role=role, triangle_index=0)


def fixture(count=2):
    segments = [ScanSegment(index, index, [sample(index * .004, index),
                                         sample(index * .004 + .001, index)])
                for index in range(count)]
    gaps = [Gap(index, 'row_turn', {}, {}, {}, [1., 0., 0.], index, index + 1, .003)
            for index in range(count - 1)]
    routes = [[sample(index * .004 + value, index + 1, 'connector')
               for value in (.002, .003, .004)] for index in range(count - 1)]
    route = mock.Mock(side_effect=[(points, 'measured_test_link') for points in routes])
    return segments, gaps, SimpleNamespace(route=route)


def connect(segments, gaps, router, used):
    return connect_links(segments, gaps, set(range(len(gaps))), router, .001, lambda: None,
                         e2=[0., 1., 0.], used=used)


class SurfaceLinkBudgetDiagnosticsTests(unittest.TestCase):
    def test_exact_limit_counts_net_added_points_and_replaces_shared_endpoint_once(self):
        segments, gaps, router = fixture()
        before = deepcopy(segments)
        with mock.patch('femto.planning.surface_row_turns.POINT_BUDGET', 6):
            result = connect(segments, gaps, router, used=4)
        self.assertEqual(result['added'], 2)  # Three connector samples minus replaced scan start.
        self.assertEqual(sum(len(chain) for chain in result['chains']), 6)
        self.assertEqual(result['kept'], [])
        chain = result['chains'][0]
        boundary = [point for point in chain if point['xyz_m'] == segments[1].points[0]['xyz_m']]
        self.assertEqual(len(boundary), 1)
        self.assertEqual(boundary[0]['role'], 'connector')
        self.assertEqual(segments, before)

    def test_one_point_over_limit_reports_actual_scan_and_net_link_counts(self):
        segments, gaps, router = fixture()
        with mock.patch('femto.planning.surface_row_turns.POINT_BUDGET', 5):
            with self.assertRaises(ValueError) as error:
                connect(segments, gaps, router, used=4)
        message = str(error.exception)
        self.assertIn('exceeds 5 samples (6; scan/existing 4, added links 2)', message)
        self.assertIn('总轨迹点数超限（不是搜索超限）', message)
        self.assertIn('未发布结果', message)

    def test_prior_fragment_output_is_not_reset_when_connecting_next_fragment(self):
        segments, gaps, router = fixture()
        with mock.patch('femto.planning.surface_row_turns.POINT_BUDGET', 11):
            first = connect(segments, gaps, router, used=4)
            completed = 4 + first['added']
            self.assertEqual(completed, 6)
            segments, gaps, router = fixture()
            with self.assertRaises(ValueError) as error:
                connect(segments, gaps, router, used=completed + 4)
        self.assertIn('(12; scan/existing 10, added links 2)', str(error.exception))

    def test_multiple_links_accumulate_net_growth_without_recounting_scan_segments(self):
        segments, gaps, router = fixture(count=3)
        with mock.patch('femto.planning.surface_row_turns.POINT_BUDGET', 10):
            result = connect(segments, gaps, router, used=6)
        self.assertEqual(result['added'], 4)
        self.assertEqual(len(result['chains']), 1)
        self.assertEqual(len(result['chains'][0]), 10)
        self.assertEqual(router.route.call_count, 2)
        segments, gaps, router = fixture(count=3)
        with mock.patch('femto.planning.surface_row_turns.POINT_BUDGET', 9):
            with self.assertRaises(ValueError) as error:
                connect(segments, gaps, router, used=6)
        self.assertIn('(10; scan/existing 6, added links 4)', str(error.exception))

    def test_unconnected_gap_keeps_scan_points_and_consumes_no_link_samples(self):
        segments, gaps, _ = fixture()
        router = SimpleNamespace(route=mock.Mock(return_value=(None, None)),
                                 diagnose=mock.Mock(return_value='explicit_unselected_blocked'),
                                 boundary=SimpleNamespace(diagnose=mock.Mock(return_value='endpoint_not_on_boundary')))
        with mock.patch('femto.planning.surface_row_turns.POINT_BUDGET', 4):
            result = connect(segments, gaps, router, used=4)
        self.assertEqual(result['added'], 0)
        self.assertEqual(sum(len(chain) for chain in result['chains']), 4)
        self.assertEqual(len(result['kept']), 1)
        self.assertEqual(result['kept'][0].detail['link_failure_reason'], 'explicit_unselected_blocked')


if __name__ == '__main__':
    unittest.main()
