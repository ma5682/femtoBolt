"""Batch ordering must reproduce the retained scalar queue exactly."""
from copy import deepcopy
from dataclasses import replace
import unittest
import numpy as np
from test_verified_visit_improvement import make_segment
from femto.planning import budgeted_relocation_research as old,relocation_candidate_batch as batch


class _Scorer:
    def __init__(self,values):self.values=values
    def score(self,edge):return self.values[edge]


class RelocationCandidateBatchTests(unittest.TestCase):
    def test_exact_queue_for_random_orientations_blocks_and_ties(self):
        for seed in range(18):
            rng=np.random.default_rng(seed);segments=[make_segment(i,float(i),i//6) for i in range(18)]
            for segment in segments:
                for point in segment.points:
                    point['xyz_m']=[float(x) for x in (rng.normal(size=3) if seed%3 else rng.integers(0,3,size=3))]
            visits=[]
            for block in range(3):
                values=list(range(block*6,block*6+6));rng.shuffle(values)
                visits.extend((i,bool(rng.integers(2))) for i in values)
            original={edge:(int(rng.integers(2)),float(rng.choice([0.,.02,.5])),float(rng.choice([0.,.1,1.])))
                      if segments[edge[0][0]].block_index==segments[edge[1][0]].block_index else (0,0.,0.)
                      for edge in zip(visits,visits[1:])}
            candidates,_=old._ranked(segments,visits,_Scorer(original),lambda:None,{})
            before=deepcopy((segments,visits,original));cache={}
            for count in (1,4,16,32,128):
                with self.subTest(seed=seed,count=count):
                    actual=batch.legacy_queue(segments,visits,original,count,lambda:None,cache)
                    self.assertEqual(actual,old._work_queue(candidates,count))
            self.assertEqual((segments,visits,original),before)

    def test_endpoint_and_block_changes_invalidate_only_local_arithmetic(self):
        segments=[make_segment(i,float(i)) for i in range(5)]
        visits=list(enumerate([False]*5));original={edge:(0,0.,1.) for edge in zip(visits,visits[1:])};cache={}
        batch.legacy_queue(segments,visits,original,16,lambda:None,cache)
        for mutation in ('endpoint','block'):
            if mutation=='endpoint':segments[2].points[0]['xyz_m'][1]+=1.
            else:
                segments[3]=replace(segments[3],block_index=1)
                segments[4]=replace(segments[4],block_index=1)
            expected,_=old._ranked(segments,visits,_Scorer(original),lambda:None,{})
            self.assertEqual(batch.legacy_queue(segments,visits,original,16,lambda:None,cache),old._work_queue(expected,16))

    def test_unsupported_coordinates_and_cancellation(self):
        segments=[make_segment(i,float(i)) for i in range(3)];visits=list(enumerate([False]*3))
        self.assertTrue(batch.supported(segments,visits))
        for value in (float('inf'),1e150,np.float64(1.)):
            changed=deepcopy(segments);changed[0].points[0]['xyz_m'][0]=value
            self.assertFalse(batch.supported(changed,visits))
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            batch.legacy_queue(segments,visits,{edge:(0,0.,1.) for edge in zip(visits,visits[1:])},8,cancel,{})

    def test_no_movable_segments_and_native_type_fallback(self):
        for count in (1,2,5):
            segments=[make_segment(i,float(i),i) for i in range(count)]
            visits=list(enumerate([False]*count))
            original={edge:(0,0.,0.) for edge in zip(visits,visits[1:])}
            self.assertEqual(batch.legacy_queue(segments,visits,original,8,lambda:None,{}),[])
        segments=[make_segment(i,float(i)) for i in range(5)]
        segments[0].points[0]['xyz_m'][0]=np.float64(0.)
        visits=list(enumerate([False]*5));original={edge:(0,0.,1.) for edge in zip(visits,visits[1:])}
        candidates,_=old._ranked(segments,visits,_Scorer(original),lambda:None,{})
        result,returned=old._queue(segments,visits,_Scorer(original),16,lambda:None,{})
        self.assertEqual(result,old._work_queue(candidates,16));self.assertEqual(returned,original)
        self.assertFalse(batch.supported(segments,visits))
        self.assertFalse(batch.supported([replace(s,block_index=2**64) for s in segments],visits))
        self.assertFalse(batch.supported(segments,[(0,False)]*5))


if __name__=='__main__':unittest.main()
