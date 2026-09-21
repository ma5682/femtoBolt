import tempfile
import unittest
from pathlib import Path
import numpy as np
import trimesh
from tests.test_mesh_slice_planner import make_export, grid_mesh, two_patches
from femto.planning.mesh_slice_planner import plan_mesh_slice
from femto.planning.surface_trajectory import ScanSegment, Gap, _point_on_triangle, to_player_plan
from femto.planning.trajectory_export import export_trajectory, load_trajectory
from femto.planning.route_motion import MotionPlayer


class RowTurnTests(unittest.TestCase):
    def test_plane_connects_rows_supported_edges_and_reproduces(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=make_export(tmp,grid_mesh(tilt_deg=25))
            old=plan_mesh_slice(source)
            result=plan_mesh_slice(source,connect_rows=True)
            self.assertEqual(result.statistics['segment_count'],1)
            self.assertEqual(result.statistics['connected_row_turns'],old.statistics['row_count']-1)
            mesh=trimesh.load(Path(tmp)/'region_1_mesh_m.ply',process=False)
            points=result.ordered_items[0].points
            for a,b in zip(points,points[1:]):
                if b.get('role')!='connector':continue
                triangle=mesh.faces[b['triangle_index']]
                self.assertTrue(_point_on_triangle(a['xyz_m'],mesh.vertices,triangle))
                self.assertTrue(_point_on_triangle(b['xyz_m'],mesh.vertices,triangle))
                self.assertLessEqual(np.linalg.norm(np.array(a['xyz_m'])-b['xyz_m']),.001000001)
            player=MotionPlayer(to_player_plan(result));player.play();player.advance(1e9)
            self.assertTrue(player.finished);self.assertFalse(player.at_break)
            folder=export_trajectory(result,Path(tmp)/'results')
            self.assertEqual(load_trajectory(folder).to_dict()['ordered_items'],result.to_dict()['ordered_items'])
            self.assertEqual(plan_mesh_slice(source).to_dict(),old.to_dict())

    def test_hole_gaps_preserved_and_disconnected_not_bridged(self):
        for mesh in (grid_mesh(hole=(-.006,.006,-.006,.006)),two_patches()):
            with self.subTest(mesh=mesh), tempfile.TemporaryDirectory() as tmp:
                source=make_export(tmp,mesh)
                old=plan_mesh_slice(source);new=plan_mesh_slice(source,connect_rows=True)
                old_gaps=[g for g in old.ordered_items if isinstance(g,Gap) and g.reason!='row_turn']
                new_gaps=[g for g in new.ordered_items if isinstance(g,Gap) and g.reason!='row_turn']
                self.assertEqual([(g.reason,g.previous_point['xyz_m'],g.next_point['xyz_m']) for g in old_gaps],
                                 [(g.reason,g.previous_point['xyz_m'],g.next_point['xyz_m']) for g in new_gaps])

    def test_forbidden_boundary_blocks_turn_and_cancel_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=make_export(tmp,forbidden=[[110,98],[114,98],[114,102],[110,102]])
            result=plan_mesh_slice(source,connect_rows=True)
            self.assertGreater(result.statistics['gap_count'],0)
            def cancel():raise RuntimeError('cancelled')
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                plan_mesh_slice(source,connect_rows=True,check=cancel)

    def test_all_start_corners_preserve_deterministic_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=make_export(tmp)
            for corner in ('top_left','top_right','bottom_left','bottom_right'):
                with self.subTest(corner=corner):
                    first=plan_mesh_slice(source,connect_rows=True,start_corner=corner)
                    second=plan_mesh_slice(source,connect_rows=True,start_corner=corner)
                    self.assertEqual(first.to_dict(),second.to_dict())
                    self.assertEqual(first.statistics['gap_count'],0)


if __name__=='__main__':unittest.main()
