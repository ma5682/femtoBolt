import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from femto.annotation import selection_intent
from femto.planning import continuous_quality as graph, evidence_preparation, quality_statistics
from femto.planning.surface_trajectory import ScanSegment


class GraphSummaryTests(unittest.TestCase):
    def test_component_counts_follow_content_order_policy_and_task(self):
        image = np.array([[1, 7, 0, 4], [0, 2, 0, 1], [1, 0, 0, 8]], np.int32)
        def reference(ids):
            labels, _ = selection_intent.label_components(image > 0)
            return (int(((image > 0) & (image != 1)).sum()),
                    tuple((int(((labels == i+1) & (image != 1)).sum()),
                           tuple(int(((labels == i+1) & (image == c)).sum()) for c in range(1, 8)))
                          for i in ids))
        with evidence_preparation.operation():
            first = quality_statistics.component_support(image, [1, 2, 0, 2])
            self.assertEqual(first, reference([1, 2, 0, 2]))
            self.assertIs(first, quality_statistics.component_support(image.copy(), [1, 2, 0, 2]))
            image[0, 1] = 0
            self.assertEqual(quality_statistics.component_support(image, [0, 2, 1]), reference([0, 2, 1]))
            self.assertNotEqual(quality_statistics.component_support(image, [1, 2, 0, 2]), first)
            with patch.object(selection_intent, 'MAX_PIXELS', 1):
                with self.assertRaisesRegex(ValueError, 'bounded'):
                    quality_statistics.component_support(image, [1, 2, 0, 2])
        image[0, 1] = 7
        self.assertIsNot(first, quality_statistics.component_support(image, [1, 2, 0, 2]))

    def test_lengths_reproduce_scalar_bits_across_chunks_and_legacy_dtypes(self):
        rng = np.random.default_rng(47)
        coordinates = rng.normal(size=(2158, 3)).tolist()
        coordinates[4] = [-0., 0., -0.]
        coordinates[5] = [0., -0., 0.]
        # Only the last chunk requires the scalar mixed-dtype path.
        coordinates[-1] = np.array([1, 2, 3], dtype=np.int16)
        expected = [float(np.linalg.norm(np.asarray(a) - b)) for a, b in zip(coordinates, coordinates[1:])]
        actual = list(quality_statistics.edge_lengths([{'xyz_m': xyz} for xyz in coordinates]))
        self.assertEqual(np.asarray(actual).tobytes(), np.asarray(expected).tobytes())

    def test_portal_and_face_node_order_matches_scalar_mesh_construction(self):
        vertices = np.array([[-.016, -.009, .521], [.014, -.008, .525],
                             [-.013, .012, .526], [.012, .009, .528]])
        faces = np.array([[0, 1, 2], [2, 1, 3]])
        mesh = SimpleNamespace(vertices=vertices, faces=faces)
        midpoint = vertices[[1, 2]].mean(axis=0)
        samples = [dict(xyz_m=vertices[faces[0]].mean(axis=0).tolist(), triangle_index=0),
                   dict(xyz_m=midpoint.tolist(), triangle_index=0)]
        base = SimpleNamespace(planner={'parameters': {'sample_step_mm': 3}},
                               ordered_items=[ScanSegment(0, 0, samples, 0)])
        record = dict(view_reference={'undistorted_intrinsic': np.eye(3)}, image_size=[10, 10])
        for quarters in (False, True):
            portals = ([vertices[1]*(1-f) + vertices[2]*f for f in (.25, .5, .75)]
                       if quarters else [midpoint])
            expected = [*portals, *(vertices[face].mean(axis=0) for face in faces),
                        *(p['xyz_m'] for p in samples)]
            pairs = []
            with (patch.object(graph.VerifiedSurfaceGraph, 'quarter_portals', quarters, create=True),
                  patch('femto.planning.selection_blocks.selection_face_assignment',
                        return_value=(None, None, None, [0, 0])),
                  patch.object(graph.quality, '_prepare_edge_checks'),
                  patch.object(graph.VerifiedSurfaceGraph, '_checks', side_effect=lambda e: [True]*len(e)),
                  patch.object(graph.VerifiedSurfaceGraph, '_admit_pairs', side_effect=pairs.extend)):
                result = graph.VerifiedSurfaceGraph(mesh, {}, record, 1., [True, True], base, lambda: None)
            self.assertEqual(result.nodes.tobytes(), np.asarray(expected).tobytes())
            self.assertEqual(result.endpoints, [[len(expected)-2, len(expected)-1]])
            # The sample inside face 0 cannot link along face 1. The midpoint can.
            on_second = {n for a, b, face in pairs if face == 1 for n in (a, b)}
            self.assertNotIn(len(expected)-2, on_second)
            self.assertIn(len(expected)-1, on_second)
