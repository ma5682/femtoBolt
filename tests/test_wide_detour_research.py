"""Bounded order proposals preserve the input scans and checked links."""
from copy import deepcopy
import unittest
from test_verified_visit_improvement import Graph,make_segment
from femto.planning import wide_detour_research as module
from femto.planning.scan_visit_order import connect,reconstruct
from femto.planning.endpoint_preserving_detour import _signature


class WideDetourTests(unittest.TestCase):
    def setUp(self):
        self.segments=[make_segment(i,float(i)) for i in range(6)]
        for segment,row in zip(self.segments,(0,0,1,1,2,2)):
            for point in segment.points:point.update(row_index=row,triangle_index=0)
        self.graph=Graph(self.segments);self.visits=list(enumerate([False]*6))
        self.links=connect(self.graph,self.segments,self.visits,lambda:None,1000)

    def test_proposal_retains_scan_signature_and_inputs(self):
        before=deepcopy((self.segments,self.visits,self.links))
        original=reconstruct(self.segments,self.visits,self.links,lambda:None,preserve_endpoint_faces=True,budget=1000)
        chains,links,_,report=module.propose(self.graph,self.segments,self.visits,self.links,.01,
            max_connector_m=100,max_revisit_m=100,rounds=2,trials=16,budget=1000)
        self.assertEqual(_signature(chains,lambda:None),_signature(original[0],lambda:None))
        self.assertEqual((self.segments,self.visits,self.links),before)
        self.assertLessEqual(report['selected']['detours'],report['baseline']['detours'])
        self.assertEqual(len(report['visits']),len(self.segments))
        self.assertEqual(links,connect(self.graph,self.segments,[tuple(v) for v in report['visits']],lambda:None,1000))
        if links and links[0]['samples']:
            links[0]['samples'][0]['xyz_m'][0]=99.
            self.assertEqual((self.segments,self.visits,self.links),before)

    def test_bounds_cancel_and_no_qualifying_candidate(self):
        for opts in (dict(rounds=0),dict(trials=257),dict(max_connector_m=float('nan')),
                     dict(spacing_m=0),dict(budget=True)):
            values=dict(spacing_m=.01,max_connector_m=100,max_revisit_m=100,budget=1000)
            values.update(opts)
            with self.subTest(opts=opts),self.assertRaises(ValueError):
                module.propose(self.graph,self.segments,self.visits,self.links,**values)
        _,_,_,record=module.propose(self.graph,self.segments,self.visits,self.links,.01,
            max_connector_m=0,max_revisit_m=0,budget=1000)
        self.assertFalse(record['adopted']);self.assertEqual(record['visits'],[list(v) for v in self.visits])
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            module.propose(self.graph,self.segments,self.visits,self.links,.01,max_connector_m=100,max_revisit_m=100,check=cancel)


if __name__=='__main__':unittest.main()
