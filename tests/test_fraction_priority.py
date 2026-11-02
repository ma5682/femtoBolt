"""Bounded fractional-seed priority proposals are not coverage claims."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from femto.planning import coverage_certificate as coverage, footprint_model, fraction_priority, scan_fraction_seeds
from femto.planning.coverage_domain import CoverageDomain
from femto.planning.surface_trajectory import ScanSegment


class FractionPriorityTests(unittest.TestCase):
    def setUp(self):
        domain=CoverageDomain.mesh([[0,0,0],[.125,0,0],[0,.125,0]],[[0,1,2]])
        self.certificate=coverage.evaluate(domain,[dict(source_face=0,xyz_m=[0.,0.,0.],role='scan')],
            footprint_model.nominal(500.),budget=coverage.CoverageBudget(max_depth=0))
        self.target=SimpleNamespace(arrays=dict(vertices=domain.vertices,faces=domain.faces,
            source_face=np.array([0],dtype='<i8'),barycentric=np.eye(3).reshape(1,3,3),
            region_faces=np.array([0],dtype='<i8')))
        points=[dict(xyz_m=[.01,.041,0.],pixel=[1.,1.],row_index=0,triangle_index=0,role='scan'),
                dict(xyz_m=[.10,.041,0.],pixel=[2.,1.],row_index=0,triangle_index=0,role='scan'),
                dict(xyz_m=[.11,.041,0.],pixel=[3.,1.],row_index=0,triangle_index=0,role='connector')]
        self.trajectory=SimpleNamespace(planner={},ordered_items=[ScanSegment(0,0,points,0)])

    def test_requests_are_deterministic_and_regenerated_on_scanning_edges(self):
        before=deepcopy(self.trajectory)
        requests,records,report=fraction_priority.propose(self.trajectory,self.target,self.certificate,[0])
        self.assertEqual(report['pending_leaves'],1);self.assertEqual(report['scan_edges'],1)
        self.assertEqual(report['leaves_with_near_candidate'],1)
        self.assertEqual(len(requests),1);self.assertEqual(requests[0][0:2],(0,0))
        self.assertEqual(records,scan_fraction_seeds.propose(self.trajectory,self.target.arrays['region_faces'],requests)[0])
        self.assertEqual((requests,records,report),fraction_priority.propose(self.trajectory,self.target,self.certificate,[0]))
        self.assertEqual((requests,records,report),fraction_priority.propose_batched(self.trajectory,self.target,self.certificate,[0]))
        self.assertEqual(before,self.trajectory)
        self.assertEqual(report['semantics'],'priority_only_not_coverage_or_physical_motion')

    def test_empty_distant_and_invalid_old_leaves(self):
        requests,records,report=fraction_priority.propose(self.trajectory,self.target,self.certificate,[])
        self.assertEqual((requests,records),([],[]));self.assertEqual(report['pending_leaves'],0)
        requests,records,report=fraction_priority.propose(self.trajectory,self.target,self.certificate,[0],radius_m=1e-8)
        self.assertEqual((requests,records),([],[]));self.assertEqual(report['leaves_with_near_candidate'],0)
        for leaves in ([0,0],[-1],[2],[True],[0.0]):
            with self.subTest(leaves=leaves),self.assertRaises(ValueError):
                fraction_priority.propose(self.trajectory,self.target,self.certificate,leaves)
        bad=SimpleNamespace(metadata=dict(self.certificate.metadata,sha256='0'*64),arrays=self.certificate.arrays)
        with self.assertRaises(ValueError):fraction_priority.propose(self.trajectory,self.target,bad,[0])

    def test_limits_geometry_roles_and_cancellation_fail_closed(self):
        for options in (dict(denominator=3),dict(denominator=True),dict(radius_m=0),
                        dict(nearest_edges_per_leaf=4),dict(radius_m=float('nan'))):
            with self.subTest(options=options),self.assertRaises(ValueError):
                fraction_priority.propose(self.trajectory,self.target,self.certificate,[0],**options)
        with patch.object(fraction_priority,'MAX_COMPARISONS',0),self.assertRaisesRegex(ValueError,'超过预算'):
            fraction_priority.propose(self.trajectory,self.target,self.certificate,[0])
        with patch.object(fraction_priority,'MAX_EDGES',0),self.assertRaisesRegex(ValueError,'超过预算'):
            fraction_priority.propose(self.trajectory,self.target,self.certificate,[0])
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            fraction_priority.propose(self.trajectory,self.target,self.certificate,[0],check=cancel)
        bad=deepcopy(self.trajectory);bad.ordered_items[0].points[1]['role']='unknown'
        with self.assertRaises(ValueError):fraction_priority.propose(bad,self.target,self.certificate,[0])

    def test_skipped_connectors_still_check_cancellation_and_visit_budget(self):
        trajectory=deepcopy(self.trajectory)
        point=trajectory.ordered_items[0].points[2]
        trajectory.ordered_items[0].points[:]=[deepcopy(point) for _ in range(601)]
        calls=[]
        fraction_priority.propose(trajectory,self.target,self.certificate,[0],check=lambda:calls.append(1))
        short=deepcopy(trajectory);short.ordered_items[0].points[:]=short.ordered_items[0].points[:2]
        short_calls=[]
        fraction_priority.propose(short,self.target,self.certificate,[0],check=lambda:short_calls.append(1))
        self.assertGreaterEqual(len(calls)-len(short_calls),2)
        with patch.object(fraction_priority,'MAX_EDGE_VISITS',256),self.assertRaisesRegex(ValueError,'遍历超过预算'):
            fraction_priority.propose(trajectory,self.target,self.certificate,[0])

    def test_batched_priority_processes_every_leaf_and_matches_unbatched_requests(self):
        from test_coverage_certificate import triangle_domain,seed
        certificate=coverage.evaluate(triangle_domain(),[seed()],footprint_model.nominal(125.),
            algorithm=coverage.CURRENT_ALGORITHM,budget=coverage.CoverageBudget(max_depth=3))
        leaves=np.flatnonzero(certificate.arrays['status']==1).tolist()
        self.assertGreater(len(leaves),2)
        expected,records,_=fraction_priority.propose(self.trajectory,self.target,certificate,leaves,radius_m=.1)
        with patch.object(fraction_priority,'MAX_PENDING',2):
            actual,actual_records,report=fraction_priority.propose_batched(self.trajectory,self.target,certificate,leaves,radius_m=.1)
        self.assertEqual(actual,expected);self.assertEqual(actual_records,records)
        self.assertEqual(report['processed_leaves'],len(leaves));self.assertEqual(report['batches'],2)
        self.assertEqual(report['semantics'],'priority_only_not_coverage_or_physical_motion')

    def test_batched_priority_rejects_duplicates_and_global_budget_without_partial_output(self):
        from test_coverage_certificate import triangle_domain,seed
        certificate=coverage.evaluate(triangle_domain(),[seed()],footprint_model.nominal(125.),
            algorithm=coverage.CURRENT_ALGORITHM,budget=coverage.CoverageBudget(max_depth=3))
        leaves=np.flatnonzero(certificate.arrays['status']==1).tolist()
        with patch.object(fraction_priority,'MAX_PENDING',2):
            with self.assertRaisesRegex(ValueError,'重复'):
                fraction_priority.propose_batched(self.trajectory,self.target,certificate,leaves+[leaves[0]])
            with patch.object(fraction_priority,'MAX_BATCHED_COMPARISONS',0):
                with self.assertRaisesRegex(ValueError,'超过预算'):
                    fraction_priority.propose_batched(self.trajectory,self.target,certificate,leaves)

    def test_batched_priority_splits_small_leaf_count_when_comparison_budget_requires(self):
        from test_coverage_certificate import triangle_domain,seed
        certificate=coverage.evaluate(triangle_domain(),[seed()],footprint_model.nominal(125.),
            algorithm=coverage.CURRENT_ALGORITHM,budget=coverage.CoverageBudget(max_depth=3))
        leaves=np.flatnonzero(certificate.arrays['status']==1).tolist()
        expected,_,_=fraction_priority.propose(self.trajectory,self.target,certificate,leaves,radius_m=.1)
        with patch.object(fraction_priority,'MAX_COMPARISONS',2):
            actual,_,report=fraction_priority.propose_batched(self.trajectory,self.target,certificate,leaves,radius_m=.1)
        self.assertEqual(actual,expected);self.assertEqual(report['processed_leaves'],len(leaves))
        self.assertEqual(report['batches'],2)

    def test_batched_priority_deduplicates_requests_across_nonmonotonic_segments(self):
        from test_coverage_certificate import triangle_domain,seed
        certificate=coverage.evaluate(triangle_domain(),[seed()],footprint_model.nominal(125.),
            algorithm=coverage.CURRENT_ALGORITHM,budget=coverage.CoverageBudget(max_depth=3))
        leaves=np.flatnonzero(certificate.arrays['status']==1).tolist()
        first=replace(deepcopy(self.trajectory.ordered_items[0]),segment_index=5)
        second=replace(deepcopy(self.trajectory.ordered_items[0]),segment_index=1)
        trajectory=SimpleNamespace(ordered_items=[first,second])
        expected,_,_=fraction_priority.propose(trajectory,self.target,certificate,leaves,radius_m=.1)
        with patch.object(fraction_priority,'MAX_PENDING',2):
            actual,records,report=fraction_priority.propose_batched(trajectory,self.target,certificate,leaves,radius_m=.1)
        self.assertEqual(set(actual),set(expected));self.assertEqual(len(actual),len(set(actual)))
        self.assertEqual(records,scan_fraction_seeds.propose(trajectory,self.target.arrays['region_faces'],actual)[0])
        self.assertEqual(report['processed_leaves'],len(leaves))


if __name__=='__main__':unittest.main()
