from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from femto.planning import witness_leaf_rank_research as ranking


class WitnessLeafRankTests(unittest.TestCase):
    def setUp(self):
        self.target=SimpleNamespace(arrays=dict(
            vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                               [10.,0.,0.],[11.,0.,0.],[10.,1.,0.]]),
            faces=np.array([[0,1,2],[3,4,5]]),source_face=np.array([0,1]),
            barycentric=np.array([np.eye(3),np.eye(3)])))
        self.proof=SimpleNamespace(arrays=dict(root=np.array([0,1]),status=np.array([1,1]),
                                                depth=np.array([0,0]),code=np.array([0,0])))
        self.seeds=[{'xyz_m':[0.,0.,0.]}]

    def test_ranks_farthest_leaf_and_stable_ties(self):
        with patch.object(ranking.coverage,'_checked_assets') as checked:
            result=ranking.rank(self.target,self.proof,[0,1],self.seeds)
        self.assertEqual(result['order'],[1,0]);self.assertGreater(result['scores_m'][0],result['scores_m'][1])
        self.assertEqual(checked.call_count,2);self.assertEqual(result['semantics'],'ranking_only_not_coverage')
        self.proof.arrays['root']=np.array([0,0])
        with patch.object(ranking.coverage,'_checked_assets'):
            self.assertEqual(ranking.rank(self.target,self.proof,[1,0],self.seeds)['order'],[0,1])

    def test_rejects_bad_inputs_and_excess_work(self):
        with patch.object(ranking.coverage,'_checked_assets'):
            for leaves,seeds in (([0,0],self.seeds),([2],self.seeds),([False],self.seeds),([0],[]),
                                  ([0],[{'xyz_m':[float('nan'),0,0]}])):
                with self.subTest(leaves=leaves,seeds=seeds),self.assertRaises(ValueError):
                    ranking.rank(self.target,self.proof,leaves,seeds)
            with patch.object(ranking,'MAX_COMPARISONS',2),self.assertRaises(ValueError):
                ranking.rank(self.target,self.proof,[0],self.seeds)

    def test_check_cancellation_propagates(self):
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            ranking.rank(self.target,self.proof,[0],self.seeds,check=cancel)


if __name__=='__main__':unittest.main()
