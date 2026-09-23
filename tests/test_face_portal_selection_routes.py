"""v4.2 product wiring, retained-version reproduction and supported-face integration."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from itertools import combinations
from pathlib import Path
from unittest import mock

import numpy as np
import trimesh

from tests.test_mesh_slice_planner import IMAGE_SIZE, INTRINSIC, grid_mesh, make_export, pixel_mask, two_patches
from femto.annotation.selection_intent import component_labels
from femto.planning.mesh_slice_planner import main, plan_mesh_slice
from femto.planning.surface_row_turns import edge_pixels_allowed
from femto.planning.surface_trajectory import (
    BIDIRECTIONAL_SELECTION_ALGORITHM, FACE_PORTAL_SELECTION_ALGORITHM, OPTIMIZED_SELECTION_ALGORITHM, SELECTION_ALGORITHM,
    SELECTION_ALGORITHMS, Gap, ScanSegment, _point_on_triangle, validate_trajectory,
)
from femto.planning.trajectory_export import (
    current_algorithm_version, export_trajectory, load_trajectory, product_algorithm_notice,
)


def plan(source, version):
    return plan_mesh_slice(source, connect_rows=True, block_planning=True, selection_blocks=True,
                           selection_algorithm=version)


def coarse_selected_export(directory):
    # The selected region crosses triangle interiors while some original vertices are unselected.
    return make_export(directory, grid_mesh(.04, .04, .02),
                       selection=dict(mask=pixel_mask((95, 95, 105, 105))))


class FacePortalSelectionIntegrationTests(unittest.TestCase):
    def test_face_fallback_joins_previously_broken_rows_without_moving_scans(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = coarse_selected_export(tmp)
            before = {path.name: path.read_bytes() for path in source.parent.iterdir() if path.is_file()}
            old = plan(source, OPTIMIZED_SELECTION_ALGORITHM)
            new = plan(source, FACE_PORTAL_SELECTION_ALGORITHM)
            self.assertEqual((old.statistics['segment_count'], old.statistics['gap_count']), (2, 1))
            self.assertEqual((new.statistics['segment_count'], new.statistics['gap_count']), (1, 0))
            self.assertEqual(new.statistics['continuity_status'], 'complete')
            self.assertEqual(new.blocks['items'][0]['fragments'][0]['link_methods']['surface_face_portals'], 1)
            self.assertEqual(old.statistics['scan_length_m'], new.statistics['scan_length_m'])
            self.assertEqual(old.statistics['row_count'], new.statistics['row_count'])
            self.assertEqual(old.statistics['geometry_unconfirmed_pixel_count'],
                             new.statistics['geometry_unconfirmed_pixel_count'])
            self.assertEqual(old.spacing_report, new.spacing_report)
            self.assertEqual(old.planner['parameters'], new.planner['parameters'])
            self.assertEqual(old.planner['sweep_frame'], new.planner['sweep_frame'])
            self.assertEqual(old.source, new.source)
            old_scans = [(point['row_index'], tuple(point['xyz_m']))
                         for chain in old.ordered_items if isinstance(chain, ScanSegment)
                         for point in chain.points if point.get('role') != 'connector']
            new_points = {(point['row_index'], tuple(point['xyz_m']))
                          for chain in new.ordered_items if isinstance(chain, ScanSegment)
                          for point in chain.points}
            self.assertTrue(set(old_scans).issubset(new_points), 'joining cannot move or remove a scan endpoint')
            manifest = json.loads(source.read_text(encoding='utf-8'))
            region = manifest['region']
            mesh = trimesh.load(source.parent / manifest['files']['mesh_file'], process=False)
            validate_trajectory(new, mesh=mesh, region=region)
            labels = component_labels(region['selection_intent'])
            for chain in new.ordered_items:
                self.assertIsInstance(chain, ScanSegment)
                selected = labels == new.blocks['items'][chain.block_index]['component_id'] + 1
                for previous, point in zip(chain.points, chain.points[1:]):
                    self.assertTrue(edge_pixels_allowed(previous['xyz_m'], point['xyz_m'],
                                                        np.asarray(INTRINSIC), np.asarray(IMAGE_SIZE), selected))
                    if point.get('role') == 'connector':
                        face = mesh.faces[point['triangle_index']]
                        self.assertTrue(_point_on_triangle(previous['xyz_m'], mesh.vertices, face))
                        self.assertTrue(_point_on_triangle(point['xyz_m'], mesh.vertices, face))
            self.assertEqual(before, {path.name: path.read_bytes() for path in source.parent.iterdir() if path.is_file()})
            self.assertFalse(new.to_dict()['execution_ready'])
            self.assertFalse(new.to_dict()['physical_accuracy_verified'])
            self.assertIsNone(new.to_dict()['orientation_fields'])

    def test_saved_v4_v41_v42_each_reproduce_their_own_version_without_schema_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = coarse_selected_export(tmp)
            payload_keys = parameters = csv_headers = None
            for version in SELECTION_ALGORITHMS:
                with self.subTest(version=version):
                    trajectory = plan(source, version)
                    folder = export_trajectory(trajectory, Path(tmp) / version, include_visuals=False)
                    before = {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}
                    with mock.patch('femto.planning.mesh_slice_planner.plan_mesh_slice', wraps=plan_mesh_slice) as replan:
                        loaded = load_trajectory(folder)
                    self.assertEqual(replan.call_count, 1)
                    self.assertEqual(replan.call_args.kwargs['selection_algorithm'], version)
                    self.assertEqual(loaded.to_dict(), trajectory.to_dict())
                    with mock.patch('femto.planning.mesh_slice_planner.plan_mesh_slice', side_effect=AssertionError('replanned')):
                        self.assertEqual(load_trajectory(folder, reproduce=False).to_dict(), trajectory.to_dict())
                    self.assertEqual(before, {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()})
                    headers = {name: (folder / name).read_text().splitlines()[0]
                               for name in ('trajectory.csv', 'gaps.csv', 'blocks.csv')}
                    if payload_keys is None:
                        payload_keys, parameters, csv_headers = set(trajectory.to_dict()), trajectory.planner['parameters'], headers
                    self.assertEqual(set(trajectory.to_dict()), payload_keys)
                    self.assertEqual(trajectory.planner['parameters'], parameters)
                    self.assertEqual(headers, csv_headers)
                    self.assertNotIn('selection_algorithm', parameters)

    def test_retained_versions_do_not_instantiate_the_face_router(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = coarse_selected_export(tmp)
            with mock.patch('femto.planning.surface_face_routing.FacePortalSurfaceLinks', side_effect=AssertionError('upgraded')):
                for version in (SELECTION_ALGORITHM, OPTIMIZED_SELECTION_ALGORITHM):
                    result = plan(source, version)
                    self.assertEqual(result.planner['algorithm_version'], version)
                    self.assertNotIn('surface_face_portals', result.blocks['items'][0]['fragments'][0]['link_methods'])
                result = plan_mesh_slice(source, connect_rows=True, block_planning=True, selection_blocks=True)
                self.assertEqual(result.planner['algorithm_version'], SELECTION_ALGORITHM)

    def test_current_notice_distinguishes_retained_old_planners_from_missing_intent(self):
        parameters = dict(connect_rows=True, block_planning=True, selection_blocks=True)
        self.assertEqual(current_algorithm_version(parameters), BIDIRECTIONAL_SELECTION_ALGORITHM)
        self.assertIsNone(product_algorithm_notice(parameters, BIDIRECTIONAL_SELECTION_ALGORITHM))
        for version in (SELECTION_ALGORITHM, OPTIMIZED_SELECTION_ALGORITHM, FACE_PORTAL_SELECTION_ALGORITHM):
            notice = product_algorithm_notice(parameters, version)
            self.assertIn(version, notice)
            self.assertIn(BIDIRECTIONAL_SELECTION_ALGORITHM, notice)
            self.assertIn('旧版算法重算', notice)
            self.assertIn('不会自动升级', notice)
            self.assertNotIn('重新选择', notice)

    def test_cli_legacy_switches_are_pairwise_exclusive(self):
        for switches in combinations(('--selection-v4', '--selection-v4-1', '--selection-v4-2', '--mesh-blocks-v3'), 2):
            with self.subTest(switches=switches), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(['unused', *switches])
            self.assertEqual(error.exception.code, 2)

    def test_face_fallback_cannot_bridge_measured_fragments_or_separate_work_blocks(self):
        masks = ((pixel_mask((0, 0, 199, 199)), 'disconnected_surface', 1),
                 (pixel_mask((80, 80, 98, 120), (102, 80, 120, 120)), 'selection_break', 2))
        for mask, reason, block_count in masks:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                source = make_export(tmp, two_patches(), selection=dict(mask=mask))
                result = plan(source, FACE_PORTAL_SELECTION_ALGORITHM)
                self.assertEqual(result.statistics['block_count'], block_count)
                self.assertEqual(result.statistics['gap_counts'], {reason: 1})
                self.assertEqual([item.reason for item in result.ordered_items if isinstance(item, Gap)], [reason])
                self.assertEqual(result.statistics['incomplete_block_count'], 1 if block_count == 1 else 0)


if __name__ == '__main__':
    unittest.main()
