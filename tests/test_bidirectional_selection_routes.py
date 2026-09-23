"""v4.3 product integration keeps saved versions and research safety contracts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import trimesh

from tests.test_face_portal_selection_routes import coarse_selected_export, plan
from tests.test_surface_face_routing import FRAME, LEFT, RIGHT, router
from femto.planning.mesh_slice_planner import plan_mesh_slice
from femto.planning.planning_io import TaskControl
from femto.planning.surface_bidirectional_routing import BidirectionalFacePortalSurfaceLinks
from femto.planning.surface_link_optimization import OptimizedSurfaceLinks
from femto.planning.surface_trajectory import (
    BIDIRECTIONAL_SELECTION_ALGORITHM, FACE_PORTAL_SELECTION_ALGORITHM,
    OPTIMIZED_SELECTION_ALGORITHM, SELECTION_ALGORITHM, SELECTION_ALGORITHMS,
    ScanSegment, to_player_plan, validate_trajectory,
)
from femto.planning.trajectory_diagram import report_text
from femto.planning.trajectory_export import export_trajectory, load_trajectory
from femto.planning.trajectory_history import prepare_playback


class BidirectionalSelectionIntegrationTests(unittest.TestCase):
    def test_retained_v42_and_older_dispatch_never_construct_new_router(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = coarse_selected_export(tmp)
            with mock.patch('femto.planning.surface_bidirectional_routing.BidirectionalFacePortalSurfaceLinks',
                            side_effect=AssertionError('old version silently upgraded')):
                for version in (SELECTION_ALGORITHM, OPTIMIZED_SELECTION_ALGORITHM, FACE_PORTAL_SELECTION_ALGORITHM):
                    with self.subTest(version=version):
                        old = plan(source, version)
                        self.assertEqual(old.planner['algorithm_version'], version)
                        self.assertNotIn('balanced bidirectional', old.blocks['link_rule']['then'])
                        self.assertNotIn('surface_face_portals_bidirectional',
                                         old.blocks['items'][0]['fragments'][0]['link_methods'])
                self.assertEqual(old.blocks['items'][0]['fragments'][0]['link_methods']['surface_face_portals'], 1)

    def test_successful_v41_connector_is_preserved_without_building_fallback(self):
        selected = np.ones((200, 200), dtype=bool)
        old = router(cls=OptimizedSurfaceLinks, allowed=selected)
        new = router(cls=BidirectionalFacePortalSurfaceLinks, allowed=selected)
        expected = old.route(LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=False)
        self.assertTrue(expected[0])
        with mock.patch.object(new, '_prepare_face_portals', side_effect=AssertionError('unneeded fallback')):
            self.assertEqual(new.route(LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=False), expected)

    def test_new_fallback_preserves_scans_selection_sources_and_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = coarse_selected_export(tmp)
            before = {path.name: path.read_bytes() for path in source.parent.iterdir() if path.is_file()}
            old, new = [plan(source, version) for version in
                        (OPTIMIZED_SELECTION_ALGORITHM, BIDIRECTIONAL_SELECTION_ALGORITHM)]
            self.assertEqual(new.statistics['gap_count'], 0)
            self.assertEqual(new.statistics['continuity_status'], 'complete')
            self.assertEqual(new.blocks['items'][0]['fragments'][0]['link_methods']
                             ['surface_face_portals_bidirectional'], 1)
            self.assertEqual(new.planner['parameters'], old.planner['parameters'])
            self.assertNotIn('selection_algorithm', new.planner['parameters'])
            self.assertEqual(new.planner['sweep_frame'], old.planner['sweep_frame'])
            self.assertEqual(new.source, old.source)
            self.assertEqual(new.spacing_report, old.spacing_report)
            for key in ('scan_length_m', 'row_count', 'geometry_unconfirmed_pixel_count'):
                self.assertEqual(new.statistics[key], old.statistics[key])
            scans = {(point['row_index'], tuple(point['xyz_m']))
                     for item in old.ordered_items if isinstance(item, ScanSegment)
                     for point in item.points if point.get('role') != 'connector'}
            new_points = {(point['row_index'], tuple(point['xyz_m']))
                          for item in new.ordered_items if isinstance(item, ScanSegment) for point in item.points}
            self.assertTrue(scans.issubset(new_points))
            manifest = json.loads(source.read_text(encoding='utf-8'))
            mesh = trimesh.load(source.parent / manifest['files']['mesh_file'], process=False)
            validate_trajectory(new, mesh=mesh, region=manifest['region'])
            payload = new.to_dict()
            self.assertFalse(payload['execution_ready'])
            self.assertFalse(payload['physical_accuracy_verified'])
            self.assertIsNone(payload['orientation_fields'])
            self.assertEqual(payload['coordinate_frame'], 'world_rdf')
            self.assertEqual(payload['unit'], 'metre')
            self.assertEqual(before, {path.name: path.read_bytes() for path in source.parent.iterdir() if path.is_file()})

    def test_old_and_new_round_trip_reproduce_their_saved_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = coarse_selected_export(tmp)
            for version in (FACE_PORTAL_SELECTION_ALGORITHM, BIDIRECTIONAL_SELECTION_ALGORITHM):
                with self.subTest(version=version):
                    result = plan(source, version)
                    folder = export_trajectory(result, Path(tmp) / version, include_visuals=False)
                    before = {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}
                    with mock.patch('femto.planning.mesh_slice_planner.plan_mesh_slice', wraps=plan_mesh_slice) as replan:
                        restored = load_trajectory(folder)
                    self.assertEqual(replan.call_args.kwargs['selection_algorithm'], version)
                    self.assertEqual(restored.to_dict(), result.to_dict())
                    self.assertEqual(before, {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()})

    def test_playback_old_and_new_does_not_replan_modify_or_upgrade(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = coarse_selected_export(tmp)
            region = json.loads(source.read_text(encoding='utf-8'))['region']
            for version in (FACE_PORTAL_SELECTION_ALGORITHM, BIDIRECTIONAL_SELECTION_ALGORITHM):
                with self.subTest(version=version):
                    result = plan(source, version)
                    folder = export_trajectory(result, Path(tmp) / version, include_visuals=False)
                    before = (folder / 'trajectory.json').read_bytes()
                    model = SimpleNamespace(store=SimpleNamespace(digest=result.source['geometry_sha256'],
                                                                  origin_reference=result.origin_reference),
                                            view_reference=region['view_reference'])
                    with mock.patch('femto.annotation.fixed_rgb_geometry.FixedRGBModel', return_value=model), \
                            mock.patch('femto.planning.mesh_slice_planner.plan_mesh_slice', side_effect=AssertionError('replanned')):
                        returned, _, playback = prepare_playback({'source': folder / 'trajectory.json'}, Path(tmp), TaskControl())
                    self.assertIs(returned, model)
                    if version == FACE_PORTAL_SELECTION_ALGORITHM:
                        self.assertIn(version, playback['algorithm_notice'])
                        self.assertIn('不会自动升级', playback['algorithm_notice'])
                    else:
                        self.assertNotIn('algorithm_notice', playback)
                    self.assertFalse(playback['execution_ready'])
                    self.assertFalse(playback['physical_accuracy_verified'])
                    self.assertEqual(before, (folder / 'trajectory.json').read_bytes())

    def test_new_report_labels_method_and_preserves_research_only_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = plan(coarse_selected_export(tmp), BIDIRECTIONAL_SELECTION_ALGORITHM)
            text = report_text(result)
            self.assertIn(BIDIRECTIONAL_SELECTION_ALGORITHM, text)
            self.assertIn('双向面内共享边补充连接 1 处', text)
            self.assertIn('execution_ready=false', text)
            self.assertIn('physical_accuracy_verified=false', text)
            self.assertIn('有限候选搜索不保证找到所有可行路线', text)
            player = to_player_plan(result)
            self.assertTrue(player['blocks']['selection_blocks'])
            self.assertIn(BIDIRECTIONAL_SELECTION_ALGORITHM, SELECTION_ALGORITHMS)


if __name__ == '__main__':
    unittest.main()
