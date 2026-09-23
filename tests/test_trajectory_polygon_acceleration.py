"""Whole-planner equivalence and validation with the prepared forbidden boundary index."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import trimesh

from tests.test_mesh_slice_planner import grid_mesh, make_export, pixel_mask
from femto.planning.mesh_slice_planner import _point_in_polygon, _project, plan_mesh_slice
from femto.planning.selection_blocks import _edge_state
from femto.planning.surface_polygon_index import PreparedPolygonIndex
from femto.planning.surface_row_turns import _intersects
from femto.planning.surface_trajectory import SELECTION_ALGORITHMS, ScanSegment, canonical_json_bytes, validate_trajectory
from femto.planning.trajectory_export import export_trajectory, load_trajectory


class ScalarPolygonIndex:
    """Original scalar predicates, independent of the accelerated implementation."""

    def __init__(self, polygons):
        self.polygons = polygons

    def contains(self, point):
        return any(_point_in_polygon(point, polygon) for polygon in self.polygons)

    def blocks_segment(self, a, b):
        return any(_point_in_polygon(a, polygon) or _point_in_polygon(b, polygon)
                   or any(_intersects(a, b, c, d) for c, d in zip(polygon, np.roll(polygon, -1, axis=0)))
                   for polygon in self.polygons)


class TrajectoryPolygonAccelerationTests(unittest.TestCase):
    def test_validator_projected_edge_states_match_scalar_oracle(self):
        k = np.array([[200., 0., 100.], [0., 200., 100.], [0., 0., 1.]])
        size = np.array([200, 200])
        polygons = [np.array([[97., 95.], [103., 95.], [100., 100.], [103., 105.], [97., 105.]])]
        index = PreparedPolygonIndex(polygons)
        scalar = ScalarPolygonIndex(polygons)
        rng = np.random.default_rng(642)
        pairs = rng.uniform([-.04, -.04, .4], [.04, .04, .6], (400, 2, 3))
        pairs[0, 0, 2] = -1
        pairs[1, 1, 0] = 5
        for a, b in pairs:
            pixels = [_project(p, k, size) for p in (a, b)]
            expected = 1 if any(p is None for p in pixels) else 2 if scalar.blocks_segment(*map(np.asarray, pixels)) else 0
            self.assertEqual(_edge_state(a, b, k, size, polygons, polygon_index=index), expected)
            self.assertEqual(_edge_state(a, b, k, size, polygons), expected)

    def test_all_selection_versions_keep_complete_result_bytes_with_forbidden_boundary(self):
        angles = np.linspace(0, 2 * np.pi, 49)[:-1]
        polygon = (np.c_[np.cos(angles), np.sin(angles)] * 2 + 100).tolist()
        mask = pixel_mask((0, 0, 199, 199)) & ~pixel_mask((97, 97, 103, 103))
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, grid_mesh(.045, .03, .003, tilt_deg=15), forbidden=polygon,
                                 selection=dict(mask=mask))
            evidence = {p.name: p.read_bytes() for p in source.parent.iterdir() if p.is_file()}
            for version in SELECTION_ALGORITHMS:
                with self.subTest(version=version):
                    kwargs = dict(connect_rows=True, block_planning=True, selection_blocks=True,
                                  selection_algorithm=version)
                    with mock.patch('femto.planning.surface_polygon_index.PreparedPolygonIndex', ScalarPolygonIndex), \
                            mock.patch('femto.planning.surface_row_turns.PreparedPolygonIndex', ScalarPolygonIndex):
                        baseline = plan_mesh_slice(source, **kwargs)
                    fast = plan_mesh_slice(source, **kwargs)
                    self.assertEqual(canonical_json_bytes(fast.to_dict()), canonical_json_bytes(baseline.to_dict()))
                    output = export_trajectory(fast, Path(tmp) / version, include_visuals=False)
                    self.assertEqual(load_trajectory(output).to_dict(), fast.to_dict())
            self.assertEqual(evidence, {p.name: p.read_bytes() for p in source.parent.iterdir() if p.is_file()})

    def test_new_exclusion_is_still_checked_on_every_validation_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, grid_mesh(.045, .03, .003),
                                 selection=dict(mask=pixel_mask((0, 0, 199, 199))))
            trajectory = plan_mesh_slice(source, connect_rows=True, block_planning=True, selection_blocks=True,
                                         selection_algorithm=SELECTION_ALGORITHMS[-1])
            manifest = json.loads(source.read_text())
            region = manifest['region']
            mesh = trimesh.load(source.parent / manifest['files']['mesh_file'], process=False)
            validate_trajectory(trajectory, mesh=mesh, region=region)
            connector = next(p for s in trajectory.ordered_items if isinstance(s, ScanSegment)
                             for p in s.points if p.get('role') == 'connector')
            x, y = connector['pixel']
            changed = dict(region, exclusion_polygons=[[[x - .5, y - .5], [x + .5, y - .5],
                                                        [x + .5, y + .5], [x - .5, y + .5]]])
            with self.assertRaisesRegex(ValueError, 'forbidden|excluded'):
                validate_trajectory(trajectory, mesh=mesh, region=changed)
            validate_trajectory(trajectory, mesh=mesh, region=region)


if __name__ == '__main__':
    unittest.main()
