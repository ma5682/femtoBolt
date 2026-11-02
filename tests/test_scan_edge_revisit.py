"""Experimental edge-based statistic is independent of zero-length anchors."""
from copy import deepcopy
import unittest
from femto.planning.scan_edge_revisit import mark


def point(x,y,row,role='scan'):
    return dict(xyz_m=[x,y,0.],row_index=row,role=role)


class ScanEdgeRevisitTests(unittest.TestCase):
    def test_connector_endpoint_and_zero_length_anchor_are_equivalent(self):
        a=point(0.,0.,0);b=point(1.,0.,0);c=point(1.,.001,1,'connector');d=point(2.,.001,1)
        other=[point(.999,.0011,2),point(1.001,.0011,2)]
        plain=[[a,b,c,d],other]
        anchored=[[deepcopy(a),deepcopy(b),deepcopy(c),point(1.,.001,1),deepcopy(d)],deepcopy(other)]
        one=mark(plain,.01);two=mark(anchored,.01)
        self.assertEqual(one,two)
        self.assertEqual(one['connector_sample_count'],1)
        self.assertAlmostEqual(one['connector_length_m'],.001)
        self.assertEqual(one['rows'],[2])
        self.assertTrue(plain[0][2]['revisit']);self.assertTrue(anchored[0][2]['revisit'])

    def test_isolated_and_zero_length_scan_do_not_create_samples(self):
        chains=[[point(0.,0.,0),point(0.,0.,0)], [point(1.,1.,1)]]
        self.assertEqual(mark(chains,.01)['scan_sample_count'],0)

    def test_scan_start_after_connector_is_a_scan_edge_endpoint(self):
        from femto.planning.selection_blocks import _mark_revisits
        first=[point(0.,0.,0),point(1.,0.,0),point(1.,.001,1,'connector'),point(2.,.001,1)]
        other=[point(0.,1.,3),point(.999,.0011,2,'connector'),point(2.,.0011,2)]
        legacy=deepcopy([first,other]);current=deepcopy([first,other])
        _mark_revisits(legacy,.01,lambda:None)
        result=mark(current,.01)
        self.assertNotIn('revisit',legacy[0][2])
        self.assertTrue(current[0][2]['revisit'])
        self.assertGreater(result['scan_sample_count'],0)

    def test_clears_stale_connector_marks_and_preserves_coordinates(self):
        chains=[[point(0.,0.,0),point(1.,0.,0),point(1.,.01,1,'connector'),point(2.,.01,1)]]
        chains[0][2]['revisit']=True;before=deepcopy([p['xyz_m'] for p in chains[0]])
        result=mark(chains,.01)
        self.assertEqual(result['connector_sample_count'],0)
        self.assertNotIn('revisit',chains[0][2])
        self.assertEqual([p['xyz_m'] for p in chains[0]],before)

    def test_invalid_spacing_role_and_cancellation(self):
        for value in (0.,-1.,float('nan'),True):
            with self.assertRaises(ValueError):mark([],value)
        with self.assertRaises(ValueError):mark([[point(0.,0.,0),point(1.,0.,0,'bad')]],.01)
        def cancel():raise RuntimeError('cancel')
        with self.assertRaisesRegex(RuntimeError,'cancel'):mark([[point(0.,0.,0),point(1.,0.,0)]],.01,cancel)
