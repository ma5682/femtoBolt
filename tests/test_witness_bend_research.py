from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from femto.planning import coverage_certificate as coverage, footprint_model, witness_bend_research as bend
from femto.planning.coverage_domain import CoverageDomain, scan_endpoints
from femto.planning.surface_trajectory import ScanSegment


class WitnessBendTests(unittest.TestCase):
    def setUp(self):
        self.domain = CoverageDomain.mesh([[0, 0, 0], [.125, 0, 0], [0, .125, 0]], [[0, 1, 2]])
        self.faces = np.array([0], dtype='<i8')
        old = [dict(xyz_m=[x, .041, 0.], pixel=[1., 1.], row_index=0,
                    triangle_index=0, role='scan', geometry_support='measured_mesh', route_index=i)
               for i, x in enumerate((.01, .10))]
        self.old = SimpleNamespace(planner={}, ordered_items=[ScanSegment(0, 0, old, 0)])
        self.seeds = scan_endpoints(self.old, self.faces)
        self.width = footprint_model.nominal(500.)
        self.proof = coverage.evaluate(self.domain, self.seeds, self.width,
                                       budget=coverage.CoverageBudget(max_depth=0))
        coverage.validate(self.proof, self.domain, self.seeds, self.width)
        witness = int(self.proof.arrays['proof_seed'][int(self.proof.arrays['witness'][0])])
        source = int(self.proof.arrays['seed_input_index'][witness])
        xyz = np.asarray(self.seeds[source]['xyz_m'])
        self.points = [dict(xyz_m=(xyz + [dx, .0001, 0.]).tolist(), pixel=[1., 1.], row_index=0,
                            triangle_index=0, role='scan', geometry_support='measured_mesh', route_index=i)
                       for i, dx in enumerate((-.002, 0., .002))]
        self.segments = [ScanSegment(0, 0, self.points, 0)]

    def test_proposal_replaces_only_detached_interior_point(self):
        old, segments = deepcopy(self.old), deepcopy(self.segments)
        proposals, report = bend.propose(self.old, self.segments, self.faces, self.proof, [0])
        self.assertEqual(len(proposals), 1)
        item = proposals[0]
        self.assertEqual((item['segment'], item['point']), (0, 1))
        self.assertEqual(item['pending_leaves'], [0])
        self.assertEqual(item['segments'][0].points[1]['xyz_m'], self.seeds[0]['xyz_m'])
        self.assertEqual(item['segments'][0].points[1]['route_index'], 1)
        self.assertEqual(self.old, old); self.assertEqual(self.segments, segments)
        item['segments'][0].points[0]['xyz_m'][0] += 1
        self.assertEqual(self.segments, segments)
        self.assertEqual(report['semantics'], 'proposals_only_require_geometry_and_coverage_replay')

    def test_roles_rows_distance_and_invalid_evidence(self):
        moved = deepcopy(self.segments)
        moved[0].points[2]['role'] = 'connector'
        self.assertEqual(bend.propose(self.old, moved, self.faces, self.proof, [0])[0], [])
        moved = deepcopy(self.segments)
        for point in moved[0].points: point['row_index'] = 3
        self.assertEqual(bend.propose(self.old, moved, self.faces, self.proof, [0])[0], [])
        self.assertEqual(bend.propose(self.old, self.segments, self.faces, self.proof, [0], max_distance_m=.00001)[0], [])
        for pending in ([0, 0], [-1], [True]):
            with self.subTest(pending=pending), self.assertRaises(ValueError):
                bend.propose(self.old, self.segments, self.faces, self.proof, pending)
        changed = deepcopy(self.old); changed.ordered_items[0].points[0]['xyz_m'][0] += .001
        with self.assertRaisesRegex(ValueError, '扫描点与证据不一致'):
            bend.propose(changed, self.segments, self.faces, self.proof, [0])

    def test_budgets_cancellation_and_empty(self):
        self.assertEqual(bend.propose(self.old, self.segments, self.faces, self.proof, [])[0], [])
        with patch.object(bend, 'MAX_POINTS', 0), self.assertRaisesRegex(ValueError, '超过预算'):
            bend.propose(self.old, self.segments, self.faces, self.proof, [0])
        with patch.object(bend, 'MAX_COMPARISONS', 0), self.assertRaisesRegex(ValueError, '超过预算'):
            bend.propose(self.old, self.segments, self.faces, self.proof, [0])
        with patch.object(bend, 'MAX_PROPOSAL_POINTS', 0), self.assertRaisesRegex(ValueError, '超过预算'):
            bend.propose(self.old, self.segments, self.faces, self.proof, [0])
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            bend.propose(self.old, self.segments, self.faces, self.proof, [0],
                         check=lambda: (_ for _ in ()).throw(RuntimeError('cancelled')))

    def test_explicit_edge_insertion_preserves_endpoints_and_inputs(self):
        segments = [ScanSegment(0, 0, [deepcopy(self.points[0]),deepcopy(self.points[2])], 0)]
        before = deepcopy(segments)
        self.assertEqual(bend.propose(self.old, segments, self.faces, self.proof, [0])[0], [])
        proposals, report = bend.propose(self.old, segments, self.faces, self.proof, [0], edge_insert=True)
        self.assertEqual(len(proposals), 1)
        item = proposals[0]
        self.assertEqual((item['placement'], item['segment'], item['edge'], item['point']), ('edge_insertion',0,0,1))
        self.assertAlmostEqual(item['projection_fraction'], .5)
        self.assertEqual(item['segments'][0].points[1]['xyz_m'], self.seeds[0]['xyz_m'])
        self.assertEqual(item['segments'][0].points[0], segments[0].points[0])
        self.assertEqual(item['segments'][0].points[-1], segments[0].points[-1])
        item['segments'][0].points[0]['xyz_m'][0] += 1
        self.assertEqual(segments, before)
        self.assertEqual(report['version'], bend.EDGE_INSERTION_VERSION)
        self.assertEqual(report['semantics'], 'proposals_only_require_geometry_and_coverage_replay')

    def test_edge_insertion_rejects_connector_wrong_row_and_excess_distance(self):
        segments = [ScanSegment(0, 0, [deepcopy(self.points[0]),deepcopy(self.points[2])], 0)]
        for mode in ('connector','wrong_row','distance','projection'):
            candidate = deepcopy(segments)
            if mode == 'connector':candidate[0].points[1]['role'] = 'connector'
            if mode == 'wrong_row':
                for point in candidate[0].points:point['row_index'] = 9
            if mode == 'projection':
                for point in candidate[0].points:point['xyz_m'][0] += .01
            kwargs={'max_distance_m':.00001} if mode=='distance' else {}
            with self.subTest(mode=mode):
                self.assertEqual(bend.propose(self.old,candidate,self.faces,self.proof,[0],edge_insert=True,**kwargs)[0],[])
        with patch.object(bend,'MAX_PROPOSAL_POINTS',2), self.assertRaisesRegex(ValueError,'复制超过预算'):
            bend.propose(self.old,segments,self.faces,self.proof,[0],edge_insert=True)
        with self.assertRaises(ValueError):bend.propose(self.old,segments,self.faces,self.proof,[0],edge_insert=1)


if __name__ == '__main__':
    unittest.main()
