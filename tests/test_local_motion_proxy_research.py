from types import SimpleNamespace
import unittest
from unittest.mock import patch
from femto.planning import local_motion_proxy_research as proxy,local_row_rank_research as rank
from femto.planning.surface_trajectory import ScanSegment


def segment(index,row,xs):
    return ScanSegment(index,row,[{'xyz_m':[x,0.,0.],'row_index':row,'role':'scan'} for x in xs],0)


class LocalMotionProxyTests(unittest.TestCase):
    def test_fixed_endpoint_estimate_is_only_a_proxy(self):
        base=[segment(0,0,(0.,1.)),segment(1,1,(3.,4.))]
        state={'base':base};trajectory=SimpleNamespace(gap_report={'reconnection':{'connections':[
            {'connected':True,'samples':[{'xyz_m':[3.,0.,0.]}]}]}})
        changed=[segment(0,0,(0.,1.5))]
        with patch.object(proxy.auto_supplement_route_research,'_source_visits',return_value=[(0,False),(1,False)]):
            result=proxy.estimate(state,trajectory,0,0,changed)
        self.assertAlmostEqual(result['predicted_connector_delta_m'],-.5)
        self.assertEqual(result['affected_connections'],1)
        self.assertEqual(result['semantics'],'ordering_proxy_not_verified_connection_or_coverage')

    def test_ambiguous_or_disconnected_inputs_have_no_prediction(self):
        state={'base':[segment(0,0,(0.,1.)),segment(1,1,(3.,4.))]}
        trajectory=SimpleNamespace(gap_report={'reconnection':{'connections':[{'connected':False,'samples':[]}]}})
        self.assertIsNone(proxy.estimate(state,trajectory,0,0,[]))
        with patch.object(proxy.auto_supplement_route_research,'_source_visits',return_value=[(0,False),(1,False)]):
            self.assertIsNone(proxy.estimate(state,trajectory,0,0,[segment(0,0,(0.,1.5))]))

    def test_priority_prefers_both_density_and_significant_motion_proxy(self):
        def item(index,first,second,delta):
            return {'index':index,'density':{'proximity':[{'near_fraction':first},{'near_fraction':second}]},
                    'motion_proxy':None if delta is None else {'predicted_connector_delta_m':delta}}
        baseline=[.02,.1];threshold=.001
        candidates=[item(0,.01,.11,-.01),item(1,.015,.09,None),item(2,.016,.09,-.002)]
        ordered=sorted(candidates,key=lambda value:rank._motion_priority(value,baseline,threshold))
        self.assertEqual([x['index'] for x in ordered],[2,1,0])


if __name__=='__main__':unittest.main()
