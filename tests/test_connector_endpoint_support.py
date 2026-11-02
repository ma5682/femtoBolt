"""Endpoint face proposals must preserve or recheck both incident edges."""
import unittest
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
import numpy as np

from femto.planning.connector_endpoint_support import propose,validate_record,_oriented_point
from femto.planning.scan_visit_order import oriented
from femto.planning.surface_trajectory import ScanSegment


def point(x,row,face,role='scan'):
    return dict(xyz_m=[float(x),0.,0.],row_index=row,triangle_index=face,role=role)


class ConnectorEndpointSupportTests(unittest.TestCase):
    def setUp(self):
        self.segments=[ScanSegment(0,0,[point(0,0,1),point(1,0,1)],0),
                       ScanSegment(1,1,[point(2,1,2),point(3,1,2)],0)]
        self.visits=[[0,False],[1,False]]
        self.links=[dict(from_segment=0,to_segment=1,connected=True,added_edges=1,
                         samples=[point(2,1,1,'connector')],reason='verified_surface_path')]

    def test_single_point_copy_matches_full_orientation_and_is_independent(self):
        segment=ScanSegment(0,0,[point(i,0,i,'connector' if i%2 else 'scan') for i in range(5)],0)
        segment.points[1]['revisit']=True
        segment.points[3]['nested']={'value':[1,2]}
        for reverse in (False,True):
            full=oriented(segment,reverse)
            for index in range(-5,5):
                with self.subTest(reverse=reverse,index=index):
                    result=_oriented_point(segment,reverse,index)
                    self.assertEqual(result,full.points[index])
                    result['xyz_m'][0]+=100
                    self.assertNotEqual(result['xyz_m'],full.points[index]['xyz_m'])

    def test_both_incident_edges_must_pass_without_input_mutation(self):
        before=deepcopy(self.links);checks=[]
        def edges(items,mesh,record,width,check,allowed_faces=None):
            checks.extend(items);return np.array([True,True]),[],np.array([])
        with patch('femto.planning.quality_constrained.edge_checks',side_effect=edges):
            result,report=propose(self.segments,self.visits,self.links,None,None,5,None)
        self.assertEqual(self.links,before);self.assertEqual(result[0]['samples'][-1]['triangle_index'],2)
        self.assertEqual(report['adopted_links'],[0]);self.assertEqual(len(checks),2)
        self.assertEqual(checks[0].points[1]['triangle_index'],2)
        self.assertEqual(checks[1].points[0]['triangle_index'],2)

    def test_either_failed_edge_keeps_original(self):
        for flags in ([False,True],[True,False]):
            with self.subTest(flags=flags),patch('femto.planning.quality_constrained.edge_checks',
                 return_value=(np.array(flags),[],np.array([]))):
                result,report=propose(self.segments,self.visits,self.links,None,None,5,None)
                self.assertEqual(result,self.links);self.assertEqual(report['rejected_links'],[0])

    def test_reversed_scan_uses_its_actual_entry_support(self):
        self.segments[1].points[-1]['triangle_index']=3
        self.visits[1][1]=True
        self.links[0]['samples'][-1]['xyz_m']=[3.,0.,0.]
        with patch('femto.planning.quality_constrained.edge_checks',return_value=(np.array([True,True]),[],np.array([]))):
            result,report=propose(self.segments,self.visits,self.links,None,None,5,None)
        self.assertEqual(result[0]['samples'][-1]['triangle_index'],3)
        self.assertEqual(report['adopted_links'],[0])

    def test_separate_work_blocks_are_not_connected(self):
        self.segments[1]=replace(self.segments[1],block_index=1)
        self.links=[dict(from_segment=0,to_segment=1,connected=False,added_edges=0,samples=[],reason='separate_work_blocks')]
        result,report=propose(self.segments,self.visits,self.links,None,None,5,None)
        self.assertEqual(result,self.links);self.assertEqual(report['candidate_count'],0)

    def test_cancel_and_bad_flags_are_not_silently_accepted(self):
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            propose(self.segments,self.visits,self.links,None,None,5,None,cancel)
        with patch('femto.planning.quality_constrained.edge_checks',return_value=([],[],[])):
            with self.assertRaisesRegex(ValueError,'数量'):
                propose(self.segments,self.visits,self.links,None,None,5,None)

    def test_saved_decision_rechecks_edges_and_rejects_tampering(self):
        with patch('femto.planning.quality_constrained.edge_checks',return_value=(np.array([True,True]),[],np.array([]))):
            final,decision=propose(self.segments,self.visits,self.links,None,None,5,None)
            validate_record(decision,self.segments,self.visits,self.links,final,object(),None,5,None)
            forged=deepcopy(final);forged[0]['samples'][-1]['triangle_index']=9
            with self.assertRaises(ValueError):
                validate_record(decision,self.segments,self.visits,self.links,forged,object(),None,5,None)
        with patch('femto.planning.quality_constrained.edge_checks',return_value=(np.array([False,True]),[],np.array([]))):
            with self.assertRaisesRegex(ValueError,'不能独立重算'):
                validate_record(decision,self.segments,self.visits,self.links,final,object(),None,5,None)
