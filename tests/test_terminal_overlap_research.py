"""Overlap proposals must preserve proof-carrying source edges."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from femto.planning import terminal_overlap_research as module
from femto.planning.coverage_domain import scan_endpoints
from femto.planning.planning_revision import digest
from femto.planning.surface_trajectory import ScanSegment


class TerminalOverlapTests(unittest.TestCase):
    def setUp(self):
        def segment(index,y):
            return ScanSegment(index,index,[dict(xyz_m=[x,y,0.],triangle_index=0,row_index=index,role='scan')
                                             for x in (0.,.002,.004)],0)
        self.segments=[segment(0,0.),segment(1,.0003)]
        self.route=SimpleNamespace(planner={},ordered_items=deepcopy(self.segments))
        self.faces=np.array([0],dtype='<i8');self.requests=[]
        seeds=scan_endpoints(self.route,self.faces)
        self.certificate=SimpleNamespace(metadata={'inputs':{'route':{'schema':'research-checked-supplement-route-v1',
            'items_sha256':digest([s.to_dict() for s in self.route.ordered_items])},'seeds_sha256':digest(seeds)}},
            arrays={'proof_seed':np.array([0]),'seed_input_index':np.array([0])})

    def test_protected_edges_remain_and_results_are_detached(self):
        before=deepcopy((self.route,self.segments))
        with patch.object(module.coverage,'_checked_assets'):
            trimmed,report=module.propose(self.route,self.segments,self.faces,self.requests,self.certificate,max_trims=2)
            self.assertNotIn((0,'start'),[(v['base_segment'],v['side']) for v in report['selected']])
            self.assertGreater(len(report['selected']),0)
            self.assertEqual((trimmed,report),module.propose(self.route,self.segments,self.faces,self.requests,self.certificate,max_trims=2))
            for segment in trimmed:self.assertGreaterEqual(len(segment.points),2)
        trimmed[0].points[0]['xyz_m'][0]=99.;self.assertEqual((self.route,self.segments),before)

    def test_binding_tamper_budget_and_cancel_rejected(self):
        with patch.object(module.coverage,'_checked_assets'):
            bad=deepcopy(self.certificate);bad.metadata['inputs']['route']['items_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'证据与路线不一致'):
                module.propose(self.route,self.segments,self.faces,self.requests,bad)
            bad=deepcopy(self.certificate);bad.arrays['proof_seed']=np.array([100])
            with self.assertRaisesRegex(ValueError,'证明种子索引无效'):
                module.propose(self.route,self.segments,self.faces,self.requests,bad)
            with patch.object(module,'MAX_SAMPLES',1),self.assertRaisesRegex(ValueError,'采样超过预算'):
                module.propose(self.route,self.segments,self.faces,self.requests,self.certificate)
            def cancel():raise RuntimeError('cancelled')
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                module.propose(self.route,self.segments,self.faces,self.requests,self.certificate,check=cancel)

    def test_base_change_during_selection_is_rejected(self):
        calls=[]
        def mutate():
            calls.append(1)
            if len(calls)==4:self.segments[1].points[0]['xyz_m'][0]=.0001
        with patch.object(module.coverage,'_checked_assets'):
            with self.assertRaisesRegex(ValueError,'输入发生变化'):
                module.propose(self.route,self.segments,self.faces,self.requests,self.certificate,check=mutate)


if __name__=='__main__':unittest.main()
