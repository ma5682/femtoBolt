"""Reusable link costs must match both complete anchored-route statistics."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from femto.planning.connector_revisit_index import ConnectorRevisitIndex
from femto.planning.scan_edge_revisit import mark
from femto.planning.selection_blocks import _mark_revisits
from femto.planning.surface_trajectory import ScanSegment


def point(xyz,row,role='scan'):
    return dict(xyz_m=list(xyz),row_index=row,role=role)


def segment(i,points,block=0):
    return ScanSegment(i,i,points,block)


class ConnectorRevisitIndexTests(unittest.TestCase):
    def test_random_routes_match_both_full_reconstructions(self):
        for seed in range(8):
            rng=np.random.default_rng(seed)
            segments=[]
            for i in range(8):
                xyz=rng.uniform(-.01,.01,(3,3))
                if i==6:xyz[1]=xyz[0]  # a zero-length scan edge
                segments.append(segment(i,[point(p,i) for p in xyz],0 if i<5 else 1))
            segments.append(segment(8,[point([.001,.001,.001],8)],1))
            original=deepcopy(segments);index=ConnectorRevisitIndex(segments,.008)
            chains=[];costs=[]
            for indices in ([0,2,1,3,4],[5,7,6,8]):
                chain=deepcopy(segments[indices[0]].points)
                for left,right in zip(indices,indices[1:]):
                    tail=segments[left].points[-1];head=segments[right].points[0]
                    middle=(np.asarray(tail['xyz_m'])+head['xyz_m'])/2
                    samples=[point(middle,right,'connector'),point(head['xyz_m'],right,'connector')]
                    costs.append(index.measure(tail,head,samples))
                    chain.extend(deepcopy(samples));chain.extend(deepcopy(segments[right].points))
                chains.append(chain)
            historical=_mark_revisits(deepcopy(chains),.008,lambda:None)
            edge=mark(deepcopy(chains),.008)
            with self.subTest(seed=seed):
                self.assertAlmostEqual(sum(c['history_revisit_m'] for c in costs),historical['connector_length_m'],places=12)
                self.assertAlmostEqual(sum(c['edge_revisit_m'] for c in costs),edge['connector_length_m'],places=12)
                self.assertEqual(sum(c['history_revisit_samples'] for c in costs),historical['connector_sample_count'])
                self.assertEqual(sum(c['edge_revisit_samples'] for c in costs),edge['connector_sample_count'])
                self.assertEqual(segments,original)

    def test_isolated_scan_point_affects_history_but_not_edge_metric(self):
        a=segment(0,[point([0.,0.,0.],0),point([1.,0.,0.],0)])
        b=segment(1,[point([1.,.001,0.],1),point([2.,.001,0.],1)])
        isolated=segment(2,[point([1.,.0011,0.],2)])
        samples=[point([1.,.001,0.],1,'connector')]
        cost=ConnectorRevisitIndex([a,b,isolated],.01).measure(a.points[-1],b.points[0],samples)
        self.assertAlmostEqual(cost['history_revisit_m'],.001)
        self.assertEqual(cost['edge_revisit_m'],0.)

    def test_all_singletons_have_no_edge_seeds(self):
        a=segment(0,[point([0.,0.,0.],0)])
        b=segment(1,[point([0.,.001,0.],1)])
        other=segment(2,[point([0.,.0011,0.],2)])
        cost=ConnectorRevisitIndex([a,b,other],.01).measure(a.points[0],b.points[0],
                    [point(b.points[0]['xyz_m'],1,'connector')])
        self.assertAlmostEqual(cost['history_revisit_m'],.001)
        self.assertEqual(cost['edge_revisit_m'],0.)

    def test_snapshots_and_input_validation(self):
        a=segment(0,[point([0.,0.,0.],0),point([1.,0.,0.],0)])
        b=segment(1,[point([1.,.001,0.],1),point([2.,.001,0.],1)])
        index=ConnectorRevisitIndex([a,b],.01);tail=deepcopy(a.points[-1]);head=deepcopy(b.points[0])
        samples=[point(head['xyz_m'],1,'connector')];before=deepcopy((tail,head,samples))
        expected=index.measure(tail,head,samples);a.points[0]['xyz_m'][0]=99.
        self.assertEqual(index.measure(tail,head,samples),expected)
        self.assertEqual((tail,head,samples),before)
        for bad in ([],[point([9.,0.,0.],1,'connector')],[point(head['xyz_m'],1,'scan')],
                    [point(head['xyz_m'],2,'connector')]):
            with self.subTest(bad=bad),self.assertRaises(ValueError):index.measure(tail,head,bad)
        with self.assertRaises(ValueError):index.measure(point([9.,0.,0.],0),head,samples)
        extra=segment(2,[point([3.,0.,0.],2),point([4.,0.,0.],2),point([5.,0.,0.],2)])
        with self.assertRaises(ValueError):
            ConnectorRevisitIndex([a,b,extra],.01).measure(extra.points[1],head,samples)
        for value in (0.,float('nan'),True):
            with self.subTest(spacing=value),self.assertRaises(ValueError):ConnectorRevisitIndex([a,b],value)
        invalid=deepcopy(b);invalid.points[1]['row_index']=5
        with self.assertRaises(ValueError):ConnectorRevisitIndex([invalid],.01)
        invalid=deepcopy(b);invalid.points[1]['xyz_m']=[True,0.,0.]
        with self.assertRaises(ValueError):ConnectorRevisitIndex([invalid],.01)

    def test_cancellation(self):
        a=segment(0,[point([0.,0.,0.],0),point([1.,0.,0.],0)])
        b=segment(1,[point([1.,.001,0.],1),point([2.,.001,0.],1)])
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):ConnectorRevisitIndex([a,b],.01,cancel)
        index=ConnectorRevisitIndex([a,b],.01);index._check=cancel
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            index.measure(a.points[-1],b.points[0],[point(b.points[0]['xyz_m'],1,'connector')])

    def test_scan_and_link_budgets(self):
        a=segment(0,[point([0.,0.,0.],0),point([1.,0.,0.],0)])
        b=segment(1,[point([1.,.001,0.],1),point([2.,.001,0.],1)])
        with patch('femto.planning.connector_revisit_index.MAX_SCAN_POINTS',3):
            with self.assertRaisesRegex(ValueError,'预算'):ConnectorRevisitIndex([a,b],.01)
        index=ConnectorRevisitIndex([a,b],.01)
        samples=[point([1.,.0005,0.],1,'connector'),point(b.points[0]['xyz_m'],1,'connector')]
        with patch('femto.planning.connector_revisit_index.MAX_LINK_POINTS',1):
            with self.assertRaisesRegex(ValueError,'预算'):index.measure(a.points[-1],b.points[0],samples)
