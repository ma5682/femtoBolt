from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from femto.planning import direct_fraction_witness_research as direct


class DirectFractionWitnessTests(unittest.TestCase):
    def test_validates_original_and_fraction_proofs_and_checks_target(self):
        params={'quality_evidence':{},'effective_domain':{}}
        plan={'surface_trajectory':{'planner':{'parameters':params}},'treatment_head':{},'planning_context':{'revision':{}}}
        ledger=SimpleNamespace(arrays={'region_faces':np.array([0])})
        domain=SimpleNamespace(ledger=ledger,representative_faces=np.array([0]),areas=np.array([1.]),binding={})
        model=object();region={};old=SimpleNamespace(metadata={'sha256':'old'});new=SimpleNamespace(metadata={'sha256':'new'})
        width={'schema':'femto-nominal-fixed-ball-v1','effective_width_mm':10.}
        with patch.object(direct,'from_plan',return_value=(domain,[],width,None)), \
             patch.object(direct.SurfaceTrajectory,'from_dict',return_value=SimpleNamespace()), \
             patch.object(direct.coverage_workflow,'route_binding',return_value={'route':1}), \
             patch.object(direct.coverage,'validate') as validate,patch.object(direct,'_same_target') as target, \
             patch.object(direct,'_matched_witnesses',return_value=([],[],np.array([0]))), \
             patch.object(direct.fraction_priority,'propose_batched',return_value=([(0,0,1,2)],['position'],{})), \
             patch.object(direct.fraction_footprint,'nominal',return_value={'width':'fraction'}), \
             patch.object(direct.coverage,'evaluate',return_value=new), \
             patch.object(direct.coverage_partition,'compare_union',return_value={'retained_subset':True}), \
             patch.object(direct.planning_revision,'require_current'),patch.object(direct.target_surface,'validate'), \
             patch.object(direct.coverage,'_checked_assets'):
            report,certificate=direct.propose(model,region,plan,old,plan)
        self.assertIs(certificate,new);self.assertEqual(report['requests'],[[0,0,1,2]])
        self.assertEqual(validate.call_count,2);self.assertIs(validate.call_args.kwargs['guide'],old)
        self.assertEqual(target.call_count,1);self.assertFalse(report['execution_ready'])


if __name__=='__main__':unittest.main()
