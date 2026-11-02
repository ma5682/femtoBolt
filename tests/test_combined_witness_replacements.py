from copy import deepcopy
import unittest
from femto.planning.surface_trajectory import ScanSegment
from femto.planning.witness_bend_research import combine_replacements


class CombinedWitnessTests(unittest.TestCase):
    def setUp(self):
        self.segments=[ScanSegment(i,i,[dict(xyz_m=[float(j),float(i),.5],row_index=i) for j in range(4)],0)
                       for i in range(2)]
        self.proposals=[]
        for i in range(2):
            changed=deepcopy(self.segments);changed[i].points[1]['xyz_m'][1]+=.001
            self.proposals.append(dict(seed=i,segment=i,point=1,distance_m=.001,pending_leaves=[i],segments=changed))

    def test_two_changes_merge_without_losing_either_or_mutating_inputs(self):
        before=deepcopy(self.segments);proposals=deepcopy(self.proposals)
        result=combine_replacements(self.segments,self.proposals)
        for i in range(2):self.assertEqual(result['segments'][i],self.proposals[i]['segments'][i])
        self.assertEqual(self.segments,before);self.assertEqual(self.proposals,proposals)
        self.assertTrue(result['candidate_only']);self.assertFalse(result['coverage_claim'])
        result['segments'][0].points[1]['xyz_m'][0]=99.
        self.assertEqual(self.segments,before);self.assertEqual(self.proposals,proposals)

    def test_conflicts_endpoints_insertions_and_undeclared_changes_reject(self):
        for mode in ('same_point','endpoint','insertion','extra_change','extra_point'):
            proposals=deepcopy(self.proposals)
            if mode=='same_point':proposals[1]=deepcopy(proposals[0])
            elif mode=='endpoint':proposals[0]['point']=0
            elif mode=='insertion':proposals[0]['edge']=0
            elif mode=='extra_change':proposals[0]['segments'][1].points[1]['xyz_m'][0]+=1.
            else:proposals[0]['segments'][1].points.append(deepcopy(proposals[0]['segments'][1].points[-1]))
            with self.subTest(mode=mode),self.assertRaises(ValueError):combine_replacements(self.segments,proposals)

    def test_cancellation_and_source_mutation_do_not_return_candidate(self):
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            combine_replacements(self.segments,self.proposals,check=cancel)
        count=0
        def change():
            nonlocal count
            count+=1
            if count==4:self.segments[0].points[0]['xyz_m'][0]=9.
        with self.assertRaisesRegex(ValueError,'源扫描发生变化'):
            combine_replacements(self.segments,self.proposals,check=change)


if __name__=='__main__':unittest.main()
