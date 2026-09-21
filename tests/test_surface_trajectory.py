import copy
import unittest

from femto.planning.surface_trajectory import (
    Gap, ScanSegment, SurfaceTrajectory, canonical_json_bytes,
    to_player_plan, validate_trajectory,
)


def trajectory_fixture():
    points = [
        dict(xyz_m=[0.0, 0.0, 0.5], row_index=0, route_index=0,
             geometry_support="measured_mesh", triangle_index=0, pixel=[10.0, 10.0]),
        dict(xyz_m=[0.001, 0.0, 0.5], row_index=0, route_index=1,
             geometry_support="measured_mesh", triangle_index=0, pixel=[11.0, 10.0]),
        dict(xyz_m=[0.002, 0.005, 0.5], row_index=1, route_index=2,
             geometry_support="measured_mesh", triangle_index=0, pixel=[12.0, 15.0]),
        dict(xyz_m=[0.001, 0.005, 0.5], row_index=1, route_index=3,
             geometry_support="measured_mesh", triangle_index=0, pixel=[11.0, 15.0]),
    ]
    detail = dict(inferred_cause="not_applicable", method="sweep_plane_outer_loop_containment",
                  confidence="heuristic", decision_use="none")
    items = [ScanSegment(0, 0, points[:2]),
             Gap(0, "row_turn", detail,
                 {"route_index": 1, "xyz_m": points[1]["xyz_m"]},
                 {"route_index": 2, "xyz_m": points[2]["xyz_m"]},
                 [1.0, 0.0, 0.0], 0, 1, 0.0051),
             ScanSegment(1, 1, points[2:])]
    return SurfaceTrajectory(
        {"kind": "initial_color_camera"},
        {"kind": "mesh_slice", "algorithm_version": "mesh-slice-serpentine-v1",
         "parameters": {"sample_step_mm": 1.0}},
        {"region_manifest": "region_1.json"}, items,
        {"segment_count": 2}, {"counts": {"row_turn": 1}}, {}, [])


class SurfaceTrajectoryTests(unittest.TestCase):
    def test_roundtrip_and_player_view(self):
        trajectory = trajectory_fixture()
        validate_trajectory(trajectory)
        payload = trajectory.to_dict()
        loaded = SurfaceTrajectory.from_dict(copy.deepcopy(payload))
        self.assertEqual(loaded.to_dict(), payload)
        player = to_player_plan(loaded)
        self.assertEqual(len(player["segments"]), 2)
        self.assertEqual(player["chain_breaks"][0]["reason"], "row_turn")
        self.assertFalse(player["chain_breaks"][0]["transition_planned"])

    def test_hash_is_canonical_and_excludes_file_self_hash(self):
        payload = trajectory_fixture().to_dict()
        self.assertNotIn("file_sha256", payload)
        self.assertEqual(canonical_json_bytes(payload), canonical_json_bytes(dict(reversed(list(payload.items())))))

    def test_route_index_gap_and_spacing_invariants(self):
        payload = trajectory_fixture().to_dict()
        payload["ordered_items"][2]["points"][0]["route_index"] = 9
        with self.assertRaises(ValueError):
            SurfaceTrajectory.from_dict(payload)
        payload = trajectory_fixture().to_dict()
        payload["ordered_items"][0]["points"][1]["xyz_m"] = [0.01, 0.0, 0.5]
        payload["result_sha256"] = "changed"
        with self.assertRaises(ValueError):
            SurfaceTrajectory.from_dict(payload)

    def test_orientation_fields_are_rejected(self):
        trajectory = trajectory_fixture()
        trajectory.ordered_items[0].points[0]["normal"] = [0, 0, -1]
        with self.assertRaises(ValueError):
            validate_trajectory(trajectory)


if __name__ == "__main__":
    unittest.main()
