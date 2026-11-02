import unittest
from types import SimpleNamespace
from unittest.mock import patch
from femto.planning.isolated_row_repair import candidates
from femto.planning.surface_trajectory import ScanSegment


def segment(i,length,row=None,block=0):
    row=i if row is None else row
    return ScanSegment(i,row,[dict(xyz_m=[0.,i*.01,0.],row_index=row,role='scan'),
                              dict(xyz_m=[length,i*.01,0.],row_index=row,role='scan')],block)


class IsolatedRowCandidatesTests(unittest.TestCase):
    def test_requests_use_deep_order_only_when_adopted(self):
        from femto.planning.isolated_row_repair import requests
        scans=[segment(0,.01),segment(1,.01),segment(2,.01)]
        prior=[[0,False],[2,True],[1,True]];deep=[[0,False],[1,True],[2,False]]
        trajectory=SimpleNamespace(planner={'parameters':{'fixed_width_layout':{'fragments':[]},'effective_width_mm':5}},
            gap_report={'reconnection':{'base_items':[s.to_dict() for s in scans],
                'ordering':{'visits':prior,'relocation':{'adopted':True,'visits':prior},
                            'relocation_deep':{'adopted':True,'visits':deep}},'connections':[]}})
        for adopted,expected in ((True,deep),(False,prior)):
            trajectory.gap_report['reconnection']['ordering']['relocation_deep']['adopted']=adopted
            with patch('femto.planning.isolated_row_repair.candidates',return_value=[]) as inspect:
                self.assertEqual(requests(trajectory),[])
            self.assertEqual(inspect.call_args.args[1],expected)

        ordering=trajectory.gap_report['reconnection']['ordering']
        floor=[[0,False],[1,False],[2,True]]
        ordering['relocation_floor']={'adopted':True,'visits':floor}
        for adopted,expected in ((True,floor),(False,prior)):
            ordering['relocation_floor']['adopted']=adopted
            with patch('femto.planning.isolated_row_repair.candidates',return_value=[]) as inspect:
                self.assertEqual(requests(trajectory),[])
            self.assertEqual(inspect.call_args.args[1],expected)

    def test_requests_use_final_relocation_only_when_adopted(self):
        from femto.planning.isolated_row_repair import requests
        scans=[segment(0,.01),segment(1,.01),segment(2,.01)]
        original=[[0,False],[1,False],[2,False]];selected=[[0,False],[2,True],[1,True]]
        trajectory=SimpleNamespace(planner={'parameters':{'fixed_width_layout':{'fragments':[]},'effective_width_mm':5}},
            gap_report={'reconnection':{'base_items':[s.to_dict() for s in scans],
                'ordering':{'visits':original,'relocation':{'adopted':True,'visits':selected}},'connections':[]}})
        for adopted,expected in ((True,selected),(False,original)):
            trajectory.gap_report['reconnection']['ordering']['relocation']['adopted']=adopted
            with patch('femto.planning.isolated_row_repair.candidates',return_value=[]) as inspect:
                self.assertEqual(requests(trajectory),[])
            self.assertEqual(inspect.call_args.args[1],expected)

    def test_requests_use_adopted_joint_reorder_visits(self):
        from femto.planning.isolated_row_repair import requests
        scans=[segment(0,.01),segment(1,.01),segment(2,.01)]
        original=[[0,False],[1,False],[2,False]];selected=[[0,False],[2,True],[1,True]]
        links=[dict(from_segment=0,to_segment=2,connected=True),dict(from_segment=2,to_segment=1,connected=True)]
        trajectory=SimpleNamespace(planner={'parameters':{'fixed_width_layout':{'fragments':[]},'effective_width_mm':5}},
            gap_report={'reconnection':{'base_items':[s.to_dict() for s in scans],
                'ordering':{'visits':original,'joint_reorder':{'adopted':True,'visits':selected}},'connections':links}})
        self.assertEqual(requests(trajectory),[])

    def test_requests_use_final_preserved_detour_visits(self):
        from femto.planning.isolated_row_repair import requests
        scans=[segment(0,.01),segment(1,.01),segment(2,.01)]
        original=[[0,False],[1,False],[2,False]]
        selected=[[0,False],[2,True],[1,True]]
        links=[dict(from_segment=0,to_segment=2,connected=True),dict(from_segment=2,to_segment=1,connected=True)]
        trajectory=SimpleNamespace(planner={'parameters':{'fixed_width_layout':{'fragments':[]},'effective_width_mm':5}},
            gap_report={'reconnection':{'base_items':[s.to_dict() for s in scans],
                         'ordering':{'visits':original,'preserved_detour':{'adopted':True,'visits':selected}},'connections':links}})
        with patch('femto.planning.isolated_row_repair.candidates',return_value=[]) as inspect:
            self.assertEqual(requests(trajectory),[])
        self.assertEqual(inspect.call_args.args[1],selected)
        self.assertEqual(inspect.call_args.args[2],links)

    def test_only_double_failed_small_single_segment_row_is_selected(self):
        scans=[segment(0,.01),segment(1,.0002),segment(2,.01)]
        visits=[[0,False],[1,False],[2,False]]
        links=[dict(from_segment=0,to_segment=1,connected=False),dict(from_segment=1,to_segment=2,connected=False)]
        found=candidates(scans,visits,links,5)
        self.assertEqual([x['segment_index'] for x in found],[1]);self.assertAlmostEqual(found[0]['shift_m'],.00025)
        links[0]['connected']=True
        self.assertEqual(candidates(scans,visits,links,5),[])

    def test_preserves_separate_blocks_and_multi_segment_rows(self):
        scans=[segment(0,.01),segment(1,.0002),segment(2,.01,block=1)]
        visits=[[0,False],[1,False],[2,False]]
        links=[dict(from_segment=0,to_segment=1,connected=False),dict(from_segment=1,to_segment=2,connected=False)]
        self.assertEqual(candidates(scans,visits,links,5),[])
        scans=[segment(0,.01),segment(1,.0002,row=1),segment(2,.01,row=1)]
        self.assertEqual(candidates(scans,visits,links,5),[])

    def test_bad_links_and_cancel_fail_closed(self):
        scans=[segment(0,.01),segment(1,.0002),segment(2,.01)]
        visits=[[0,False],[1,False],[2,False]]
        links=[dict(from_segment=1,to_segment=0,connected=False),dict(from_segment=1,to_segment=2,connected=False)]
        with self.assertRaises(ValueError):candidates(scans,visits,links,5)
        def cancel():raise RuntimeError('cancel')
        with self.assertRaisesRegex(RuntimeError,'cancel'):candidates(scans,visits,links,5,cancel)
