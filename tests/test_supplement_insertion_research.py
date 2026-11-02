"""Supplement insertion retains scan segments and never invents graph paths."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from test_verified_visit_improvement import Graph,make_segment
from femto.planning import supplement_insertion_research as insertion
from femto.planning.endpoint_preserving_detour import _signature


class SupplementInsertionTests(unittest.TestCase):
    def setUp(self):
        self.base=[make_segment(i,float(i)) for i in range(4)]
        self.extra=[make_segment(i+4,float(i)+.2) for i in range(2)]
        self.segments=self.base+self.extra
        for seg in self.segments:
            for p in seg.points:p['triangle_index']=0
        self.visits=list(enumerate([False]*len(self.base)))
        self.graph=Graph(self.segments)

    def test_insert_is_repeatable_and_preserves_all_scans(self):
        before=deepcopy((self.base,self.extra,self.visits))
        result=insertion.propose(self.graph,self.segments,self.base,self.extra,self.visits,.01,budget=1000)
        self.assertEqual(result,insertion.propose(self.graph,self.segments,self.base,self.extra,self.visits,.01,budget=1000))
        chains,links,_,report=result
        self.assertEqual(sorted(v[0] for v in report['visits']),list(range(len(self.segments))))
        self.assertEqual(_signature(chains,lambda:None),_signature([(s.block_index,s.points) for s in self.segments],lambda:None))
        self.assertEqual((self.base,self.extra,self.visits),before)
        links[0]['samples'][0]['xyz_m'][0]=99.;self.assertEqual((self.base,self.extra,self.visits),before)

    def test_failed_paths_and_invalid_budgets_fail_closed(self):
        class NoSupplementGraph(Graph):
            def route(self,left,right,*,left_reverse,right_reverse):
                if left>=4 or right>=4:return None
                return super().route(left,right,left_reverse=left_reverse,right_reverse=right_reverse)
        graph=NoSupplementGraph(self.segments)
        with self.assertRaisesRegex(ValueError,'不能通过已验证图连接'):
            insertion.propose(graph,self.segments,self.base,self.extra,self.visits,.01,budget=1000)
        for opts in (dict(attempts=0),dict(spacing_m=0),dict(penalty_m=float('nan')),dict(budget=True)):
            values=dict(spacing_m=.01,budget=1000);values.update(opts)
            with self.subTest(opts=opts),self.assertRaises(ValueError):
                insertion.propose(self.graph,self.segments,self.base,self.extra,self.visits,**values)
        with patch.object(insertion,'MAX_QUERY',0),self.assertRaisesRegex(ValueError,'超过预算'):
            insertion.propose(self.graph,self.segments,self.base,self.extra,self.visits,.01,budget=1000)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            insertion.propose(self.graph,self.segments,self.base,self.extra,self.visits,.01,check=cancel)


if __name__=='__main__':unittest.main()
