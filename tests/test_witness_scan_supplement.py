"""Old-proof witness selection proposes only previously accepted scan edges."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from femto.planning import coverage_certificate as coverage,footprint_model,witness_scan_supplement as supplement
from femto.planning.coverage_domain import CoverageDomain,scan_endpoints
from femto.planning.planning_revision import digest
from femto.planning.surface_trajectory import ScanSegment


class WitnessSupplementTests(unittest.TestCase):
    def setUp(self):
        self.domain=CoverageDomain.mesh([[0,0,0],[.125,0,0],[0,.125,0]],[[0,1,2]])
        self.faces=np.array([0],dtype='<i8')
        points=[dict(xyz_m=[.01,.041,0.],pixel=[1.,1.],row_index=0,triangle_index=0,role='scan'),
                dict(xyz_m=[.10,.041,0.],pixel=[2.,1.],row_index=0,triangle_index=0,role='scan'),
                dict(xyz_m=[.11,.041,0.],pixel=[3.,1.],row_index=0,triangle_index=0,role='connector')]
        self.trajectory=SimpleNamespace(planner={},ordered_items=[ScanSegment(0,0,points,0)])
        self.model=footprint_model.nominal(500.)
        self.seeds=scan_endpoints(self.trajectory,self.faces)
        self.cert=coverage.evaluate(self.domain,self.seeds,self.model,budget=coverage.CoverageBudget(max_depth=0))
        coverage.validate(self.cert,self.domain,self.seeds,self.model)

    def test_replayed_witness_produces_original_scan_and_no_mutation(self):
        before=deepcopy(self.trajectory)
        edges,report=supplement.propose(self.trajectory,self.faces,self.cert,[.125*.125/2],[0])
        self.assertEqual(edges,[(0,0)]);self.assertEqual(report['contiguous_groups'],1)
        self.assertEqual(self.trajectory,before)
        self.assertEqual((edges,report),supplement.propose(self.trajectory,self.faces,self.cert,[.125*.125/2],[0]))
        self.assertEqual(report['semantics'],'candidate_only_not_coverage_or_physical_motion')

    def test_invalid_or_mismatched_input_fails_before_proposals(self):
        for pending in ([0,0],[-1],[2],[True]):
            with self.subTest(pending=pending),self.assertRaises(ValueError):
                supplement.propose(self.trajectory,self.faces,self.cert,[.01],pending)
        for gap in (-1,float('nan'),True,.101):
            with self.subTest(gap=gap),self.assertRaises(ValueError):
                supplement.propose(self.trajectory,self.faces,self.cert,[.01],[0],gap_m=gap)
        changed=deepcopy(self.trajectory);changed.ordered_items[0].points[0]['xyz_m'][0]+=.001
        with self.assertRaisesRegex(ValueError,'扫描点与证据不一致'):
            supplement.propose(changed,self.faces,self.cert,[.01],[0])
        with patch.object(supplement,'MAX_PROOFS',0),self.assertRaisesRegex(ValueError,'超过预算'):
            supplement.propose(self.trajectory,self.faces,self.cert,[.01],[0])
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            supplement.propose(self.trajectory,self.faces,self.cert,[.01],[0],check=lambda:(_ for _ in ()).throw(RuntimeError('cancelled')))

    def test_empty_hints_propose_nothing(self):
        edges,report=supplement.propose(self.trajectory,self.faces,self.cert,[.01],[])
        self.assertEqual(edges,[]);self.assertEqual(report['selected_seeds'],0)

    def test_representative_map_and_edge_visit_budget(self):
        edges,_=supplement.propose(self.trajectory,self.faces,self.cert,[.01],[0],representative_faces=np.array([0]))
        self.assertEqual(edges,[(0,0)])
        with self.assertRaisesRegex(ValueError,'见证与扫描点不一致'):
            supplement.propose(self.trajectory,self.faces,self.cert,[.01],[0],representative_faces=np.array([1]))
        with self.assertRaisesRegex(ValueError,'映射无效'):
            supplement.propose(self.trajectory,self.faces,self.cert,[.01],[0],representative_faces=np.array([.0]))
        with patch.object(supplement,'MAX_EDGE_VISITS',1),self.assertRaisesRegex(ValueError,'扫描边超过预算'):
            supplement.propose(self.trajectory,self.faces,self.cert,[.01],[0])

    def test_coalescing_never_crosses_a_connector(self):
        def fixture(middle_role):
            points=[dict(xyz_m=[i*.01,0.,0.],row_index=0,triangle_index=0,role='scan') for i in range(4)]
            points[2]['role']=middle_role
            trajectory=SimpleNamespace(planner={},ordered_items=[ScanSegment(0,0,points,0)])
            records=scan_endpoints(trajectory,self.faces)
            indices=[next(i for i,p in enumerate(records) if p['xyz_m']==points[j]['xyz_m']) for j in (0,3)]
            arrays=dict(root=np.array([0,1]),depth=np.array([0,0]),status=np.array([1,1]),
                proof_leaf=np.array([0,1]),proof_seed=np.array([0,1]),seed_face=np.array([0,0]),
                seed_input_index=np.array(indices),seed_xyz=np.array([points[0]['xyz_m'],points[3]['xyz_m']]))
            return trajectory,SimpleNamespace(arrays=arrays,metadata={'inputs':{'seeds_sha256':digest(records)}})
        with patch.object(supplement.coverage,'_checked_assets'):
            trajectory,certificate=fixture('scan')
            edges,_=supplement.propose(trajectory,self.faces,certificate,[1.,1.],[0,1],gap_m=.02)
            self.assertEqual(edges,[(0,0),(0,1),(0,2)])
            trajectory,certificate=fixture('connector')
            edges,_=supplement.propose(trajectory,self.faces,certificate,[1.,1.],[0,1],gap_m=.02)
            self.assertEqual(edges,[(0,0),(0,2)])


if __name__=='__main__':unittest.main()
