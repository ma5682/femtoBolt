"""Reordering uses verified graph links and preserves all scan anchors."""
from copy import deepcopy
import unittest
import numpy as np
from test_verified_visit_improvement import Graph,make_segment
from femto.planning import supplement_two_opt_research as module
from femto.planning.endpoint_preserving_detour import _signature
from femto.planning.scan_visit_order import connect


class EndpointGraph(Graph):
    def __init__(self,segments):
        super().__init__(segments)
        self.nodes=np.asarray([p['xyz_m'] for seg in segments for p in (seg.points[0],seg.points[-1])])
        self.endpoints=[(2*i,2*i+1) for i in range(len(segments))]


class SupplementTwoOptTests(unittest.TestCase):
    def setUp(self):
        self.segments=[make_segment(i,float(y)) for i,y in enumerate((0,4,1,3,2))]
        for segment in self.segments:
            for p in segment.points:p['triangle_index']=0
        self.graph=EndpointGraph(self.segments);self.visits=list(enumerate([False]*len(self.segments)))
        self.links=connect(self.graph,self.segments,self.visits,lambda:None,1000)

    def test_bounded_reorder_keeps_scan_and_never_lengthens(self):
        before=deepcopy((self.segments,self.visits,self.links))
        original=module._anchored(self.segments,self.visits,self.links,lambda:None,1000)
        chains,links,_,report=module.propose(self.graph,self.segments,self.visits,self.links,.01,
            max_detours=0,max_revisit_m=100,budget=1000)
        self.assertTrue(report['adopted'])
        self.assertEqual(_signature(chains,lambda:None),_signature(original[0],lambda:None))
        self.assertLessEqual(report['selected_objective'][1],original[2][1])
        self.assertEqual((self.segments,self.visits,self.links),before)
        self.assertEqual(links,connect(self.graph,self.segments,[tuple(v) for v in report['visits']],lambda:None,1000))
        links[0]['samples'][0]['xyz_m'][0]=99.;self.assertEqual((self.segments,self.visits,self.links),before)

    def test_invalid_limits_and_link_records_rejected(self):
        for options in (dict(passes=0),dict(max_detours=True),dict(max_revisit_m=float('nan')),dict(attempts=257)):
            args=dict(spacing_m=.01,max_detours=0,max_revisit_m=100,budget=1000);args.update(options)
            with self.subTest(options=options),self.assertRaises(ValueError):
                module.propose(self.graph,self.segments,self.visits,self.links,**args)
        invalid=deepcopy(self.links);invalid[0]['to_segment']=99
        with self.assertRaises(ValueError):module.propose(self.graph,self.segments,self.visits,invalid,.01,max_detours=0,max_revisit_m=100)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            module.propose(self.graph,self.segments,self.visits,self.links,.01,max_detours=0,max_revisit_m=100,check=cancel)


if __name__=='__main__':unittest.main()
