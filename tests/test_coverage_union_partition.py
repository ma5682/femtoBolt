"""Union comparisons preserve exact target partitions, not physical coverage."""
import unittest
import numpy as np

from femto.planning.coverage_certificate import Certificate
from femto.planning.coverage_partition import DEPTH, UNITS, compare, compare_union


def certificate(partitions):
    rows=[(root,depth,code,status) for root, leaves in enumerate(partitions)
          for depth,code,status in leaves]
    arrays={key:np.asarray([r[i] for r in rows],dtype=dtype)
            for i,(key,dtype) in enumerate((('root','<i8'),('depth','u1'),('code','<u4'),('status','u1')))}
    return Certificate({},arrays)


def expanded(cert, roots):
    result=np.zeros((roots,UNITS),bool)
    for root,depth,code,status in zip(*(cert.arrays[k] for k in ('root','depth','code','status'))):
        size=4**(DEPTH-int(depth));lo=int(code)*size
        result[int(root),lo:lo+size]=int(status)==1
    return result


class UnionPartitionTests(unittest.TestCase):
    def test_complementary_partial_proofs_confirm_old_without_completing_target(self):
        old=certificate([[(0,0,1)],[(0,0,7)]])
        a=certificate([[(1,0,1),(1,1,1),(1,2,7),(1,3,7)],[(0,0,7)]])
        b=certificate([[(1,0,7),(1,1,7),(1,2,1),(1,3,1)],[(0,0,7)]])
        result=compare_union(old,[a,b],np.array([4.,9.]))
        self.assertTrue(result['retained_subset'])
        self.assertEqual(result['retained_nominal_area_m2'],4.)
        self.assertEqual(result['gained_nominal_area_m2'],0.)
        self.assertEqual(result['not_reconfirmed_weight_units'],0)
        self.assertEqual(result['union_partition_count'],2)

    def test_one_proof_matches_existing_pairwise_comparison(self):
        old=certificate([[(1,0,1),(1,1,7),(1,2,1),(1,3,7)]])
        new=certificate([[(0,0,1)]])
        areas=np.array([3.])
        old_result=compare(old,new,areas);result=compare_union(old,[new],areas)
        self.assertEqual(result['retained_nominal_area_m2'],old_result['retained_nominal_area_m2'])
        self.assertEqual(result['gained_nominal_area_m2'],old_result['gained_nominal_area_m2'])
        self.assertEqual(result['not_reconfirmed_weight_units'],old_result['lost_weight_units'])

    def test_partial_old_leaf_is_selected_once_for_followup(self):
        old=certificate([[(0,0,1)]])
        new=certificate([[(1,0,1),(1,1,7),(1,2,7),(1,3,1)]])
        result=compare_union(old,[new],np.array([1.]),include_pending_leaves=True)
        self.assertEqual(result['pending_old_leaf_indices'],[0])
        self.assertEqual(result['not_reconfirmed_weight_units'],UNITS//2)

    def test_random_distinct_partitions_match_expanded_integer_reference(self):
        rng=np.random.default_rng(703)
        def leaves(depth=0,code=0):
            if depth<3 and rng.random()<.7:
                return sum((leaves(depth+1,4*code+i) for i in range(4)),[])
            return [(depth,code,int(rng.choice([1,2,3,7])))]
        for _ in range(20):
            certs=[certificate([leaves(),leaves()]) for _ in range(4)]
            old=expanded(certs[0],2);new=np.logical_or.reduce([expanded(c,2) for c in certs[1:]])
            actual=compare_union(certs[0],certs[1:],np.array([1.,2.]))
            self.assertEqual(actual['retained_weight_units'],int((old & new).sum()))
            self.assertEqual(actual['gained_weight_units'],int((~old & new).sum()))
            self.assertEqual(actual['not_reconfirmed_weight_units'],int((old & ~new).sum()))

    def test_incomplete_partitions_bad_areas_and_excess_proofs_fail_closed(self):
        good=certificate([[(0,0,1)]])
        bad=certificate([[(1,0,1),(1,1,1),(1,3,1)]])
        for evidence,areas in (([bad],np.array([1.])),([],np.array([1.])),
                               ([good]*5,np.array([1.])),([good],np.array([-1.])),
                               ([good],np.array([float('nan')])),([good],np.array([1j]))):
            with self.subTest(evidence=len(evidence),areas=areas):
                with self.assertRaises(ValueError):compare_union(good,evidence,areas)

    def test_extra_or_non_integer_partition_entries_are_rejected(self):
        good=certificate([[(0,0,1)]])
        for bad_status in (np.array([1,7],dtype='u1'),np.array([1.],dtype='<f8')):
            arrays=dict(good.arrays,status=bad_status)
            with self.assertRaisesRegex(ValueError,'长度或类型'):
                compare_union(good,[Certificate({},arrays)],np.array([1.]))

    def test_cancellation_is_propagated(self):
        good=certificate([[(0,0,1)]])
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            compare_union(good,[good],np.array([1.]),check=cancel)
