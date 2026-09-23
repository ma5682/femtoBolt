"""Analytic diagnostics and verified history inspection, never a planner replacement."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from femto.planning.planning_io import TaskControl
from femto.planning.surface_trajectory import ScanSegment, SurfaceTrajectory, SELECTION_BLOCKS_SCHEMA
from femto.planning.trajectory_quality import polyline_quality, quality_report_lines, block_pending_reasons
from femto.planning.trajectory_history import prepare_report
from femto.planning.trajectory_export import export_trajectory
from femto.planning.mesh_slice_planner import plan_mesh_slice
from tests.test_mesh_slice_planner import make_export


def chain(xyz, index=0, start=0, connectors=()):
    return ScanSegment(index, 0, [dict(xyz_m=p, row_index=0, route_index=start+i,
                                      role='connector' if i in connectors else 'scan')
                                  for i, p in enumerate(xyz)], block_index=index)


def model(*segments, spacing=None, blocks=None):
    return SurfaceTrajectory({}, {}, {}, list(segments), {}, {}, spacing or {}, [], blocks=blocks)


class TrajectoryQualityTests(unittest.TestCase):
    def test_analytic_lengths_steps_and_right_angle(self):
        trajectory = model(chain([[0, 0, 0], [.003, 0, 0], [.003, .004, 0]], connectors=(2,)))
        before = copy.deepcopy(trajectory)
        result = polyline_quality(trajectory)
        self.assertAlmostEqual(result['scan_length_m'], .003)
        self.assertAlmostEqual(result['connector_length_m'], .004)
        self.assertAlmostEqual(result['connector_fraction'], 4/7)
        self.assertAlmostEqual(result['step_mm']['min'], 3)
        self.assertAlmostEqual(result['step_mm']['max'], 4)
        self.assertAlmostEqual(result['turn_deg']['max'], 90)
        self.assertEqual(result['largest_turns'][0]['route_index'], 1)
        self.assertEqual(result['largest_turns'][0]['kind'], 'connector')
        self.assertEqual(trajectory, before)

    def test_never_measures_across_chains_even_without_a_gap_object(self):
        trajectory = model(chain([[0, 0, 0], [.001, 0, 0]]),
                           chain([[10, 10, 10], [10, 10.002, 10]], index=1, start=2))
        result = polyline_quality(trajectory)
        self.assertAlmostEqual(result['total_length_m'], .003)
        self.assertEqual(result['chain_count'], 2)
        self.assertEqual(result['edge_count'], 2)
        self.assertIsNone(result['turn_deg'])

    def test_duplicates_do_not_hide_a_reversal_or_invent_a_tangent(self):
        trajectory = model(chain([[0, 0, 0], [.001, 0, 0], [.001, 0, 0], [0, 0, 0]], connectors=(1,)))
        result = polyline_quality(trajectory)
        self.assertEqual(result['zero_length_edge_count'], 1)
        self.assertEqual(result['turn_count'], 1)
        self.assertAlmostEqual(result['turn_deg']['max'], 180)
        self.assertEqual(result['largest_turns'][0]['route_index'], 2)
        self.assertEqual(result['largest_turns'][0]['kind'], 'connector')
        still = polyline_quality(model(chain([[0, 0, 0]] * 3)))
        self.assertIsNone(still['connector_fraction'])
        self.assertIsNone(still['turn_deg'])

    def test_empty_and_straight_lines_have_no_spurious_turns(self):
        self.assertIsNone(polyline_quality(model())['step_mm'])
        result = polyline_quality(model(chain([[0, 0, 0], [.001, 0, 0], [.002, 0, 0]])))
        self.assertEqual(result['turn_deg']['max'], 0)
        self.assertEqual(result['connector_fraction'], 0)

    def test_full_spacing_counts_do_not_use_filtered_distribution_or_saved_counts(self):
        spacing = dict(target_spacing_mm=5, tolerance_mm=1, counts={'within_bounds': 999},
                       effective_spacing_statistics={'max_mm': 5}, measurements=[{'status': s} for s in
                       ('within_bounds', 'outside_bounds', 'uncertain_bounds', 'unmeasurable')])
        text = '\n'.join(quality_report_lines(model(spacing=spacing)))
        self.assertIn('4 个样本；界内 1，超差 1，界限不确定 1，不可测 1', text)
        self.assertIn('行距待复核', text)
        self.assertIn('不能用过滤后的有效行距', text)
        self.assertIn('旧版轨迹', text)
        empty = '\n'.join(quality_report_lines(model()))
        self.assertIn('行距待复核', empty)

    def test_single_chain_missing_geometry_is_still_pending(self):
        block = dict(block_index=0, status='planned', selection={'support': {
            'geometry_unconfirmed_pixels': 7, 'unconfirmed_by_reason': {
                'no_surface_reference_depth_present': 3, 'no_surface_no_reference_depth': 2, 'depth_conflict': 2}}},
            coverage={'status': 'partial', 'uncovered_fragment_count': 0})
        trajectory = model(chain([[0, 0, .5], [.001, 0, .5]]),
                           blocks={'schema': SELECTION_BLOCKS_SCHEMA, 'items': [block]})
        text = '\n'.join(quality_report_lines(trajectory))
        for phrase in ('工作块 1：待处理', '7 个选中像素', '有参考深度但缺面 3', '缺少有效采集支持 2', '深度冲突 2'):
            self.assertIn(phrase, text)
        self.assertNotIn('已形成连续单链且几何覆盖记录完整', text)

    def test_unselected_and_missing_surface_have_distinct_remedies(self):
        block = dict(status='incomplete', link_failure_counts={'explicit_unselected_blocked': 1, 'disconnected_measured_surface': 1},
                     coverage={'status': 'partial', 'uncovered_fragment_count': 1}, dropped_fragment_count=2)
        text = '\n'.join(block_pending_reasons(block, 2))
        for phrase in ('保留原选区', '有依据的重建或补采', '几何支持未知', '未规划网格碎片 1', '2 个丢弃'):
            self.assertIn(phrase, text)
        block = dict(status='planned', coverage={'status': 'complete', 'uncovered_fragment_count': 0},
                     selection={'support': {'geometry_unconfirmed_pixels': 0}})
        self.assertEqual(block_pending_reasons(block, 1), [])
        self.assertTrue(block_pending_reasons(block, 0))

    def test_verified_history_report_is_read_only_and_never_replans(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_export(tmp)
            trajectory = plan_mesh_slice(manifest)
            folder = export_trajectory(trajectory, tmp, include_visuals=False)
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            row = dict(kind='三维表面轨迹', source=folder/'trajectory.json')
            with patch('femto.planning.mesh_slice_planner.plan_mesh_slice', side_effect=AssertionError('must not replan')):
                report = prepare_report(row, TaskControl())
            self.assertIn('路线规划检查', report)
            self.assertIn('execution_ready=false', report)
            self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})
            (folder/'trajectory.csv').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                prepare_report(row, TaskControl())

    def test_cancellation_and_wrong_history_row_are_rejected(self):
        with self.assertRaises(ValueError):
            prepare_report({'kind': '区域'}, TaskControl())
        control = TaskControl(); control.set()
        with self.assertRaises(RuntimeError):
            prepare_report({'kind': '三维表面轨迹', 'source': 'unused'}, control)
        with self.assertRaises(RuntimeError):
            polyline_quality(model(chain([[0, 0, 0], [0, 1, 0]])), control.check)


if __name__ == '__main__':
    unittest.main()
