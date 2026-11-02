"""Full route/target-bound proof replay before union comparison."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_coverage_workflow import fixture
from test_coverage_certificate import reseal
from femto.planning.coverage_retention_union import compare_replayed,retained_by_existing_scans,propose_replayed


class ReplayedRetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.model,cls.region,cls.plan,cls.cert,_=fixture(Path(cls.tmp.name)/'source')

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def compare(self,proofs):
        return compare_replayed(self.model,self.region,self.plan,self.cert,self.plan,proofs)

    def test_same_route_proof_replays_before_retention_is_reported(self):
        result=self.compare([(self.cert,None)])
        self.assertTrue(result['comparison']['retained_subset'])
        self.assertEqual(result['comparison']['not_reconfirmed_weight_units'],0)
        self.assertFalse(result['installed_certificate'])
        self.assertFalse(result['full_target_coverage_claim'])
        self.assertFalse(result['execution_ready'])
        self.assertFalse(result['physical_accuracy_verified'])

    def test_resealed_seed_tampering_cannot_be_used_as_a_new_proof(self):
        arrays={key:value.copy() for key,value in self.cert.arrays.items()}
        self.assertGreater(len(arrays['seed_xyz']),0)
        arrays['seed_xyz'][0,0]+=1e-4
        forged=reseal(self.cert,arrays=arrays)
        with self.assertRaises(ValueError):self.compare([(forged,None)])

    def test_partial_or_excess_input_is_rejected(self):
        for proofs in ([],[(self.cert,None)]*5,[self.cert]):
            with self.subTest(count=len(proofs)):
                with self.assertRaises(ValueError):self.compare(proofs)

    def test_existing_same_route_witnesses_retain_old_proofs(self):
        result=retained_by_existing_scans(self.model,self.region,self.plan,self.cert,self.plan)
        self.assertGreater(len(result['retained_leaf_indices']),0)
        self.assertEqual(len(result['retained_leaf_indices']),len(result['retained_proof_indices']))
        self.assertEqual(result['not_reconfirmed_leaf_indices'],[])
        self.assertTrue(result['retained_subset'])
        self.assertFalse(result['installed_certificate'])

    def test_replayed_union_can_include_proofs_with_same_accepted_scan_witness(self):
        from femto.planning import coverage_certificate as coverage,coverage_workflow
        empty,_=coverage_workflow.calculate(self.model,self.region,self.plan,
                                             budget=coverage.CoverageBudget(seconds=0))
        self.assertEqual(empty.summary['confirmed_leaf_count'],0)
        without=self.compare([(empty,None)])
        self.assertFalse(without['comparison']['retained_subset'])
        with_old=compare_replayed(self.model,self.region,self.plan,self.cert,self.plan,[(empty,None)],
                                  include_existing_witnesses=True)
        self.assertTrue(with_old['comparison']['retained_subset'])
        self.assertGreater(with_old['matched_previous_leaves'],0)
        self.assertFalse(with_old['installed_certificate'])
        self.assertFalse(with_old['full_target_coverage_claim'])

    def test_witness_match_requires_exact_scan_coordinates_and_role(self):
        from femto.planning.coverage_domain import from_plan
        from femto.planning.coverage_retention_union import _matched_witnesses
        domain,seeds,_,_=from_plan(self.model,self.region,self.plan)
        retained,_,missing=_matched_witnesses(self.cert,domain,seeds,lambda:None)
        self.assertGreater(len(retained),0)
        shifted=deepcopy(seeds)
        for seed in shifted:seed['xyz_m'][0]=float(np.nextafter(seed['xyz_m'][0],np.inf))
        moved,_,unknown=_matched_witnesses(self.cert,domain,shifted,lambda:None)
        self.assertEqual(len(moved),0)
        self.assertEqual(len(unknown),len(retained)+len(missing))
        for seed in shifted:seed['role']='connector'
        self.assertEqual(len(_matched_witnesses(self.cert,domain,shifted,lambda:None)[0]),0)

    def test_changed_target_array_cannot_be_compared_by_size(self):
        from femto.planning.coverage_domain import from_plan
        from femto.planning.coverage_retention_union import _same_target
        domain,*_=from_plan(self.model,self.region,self.plan)
        altered=deepcopy(domain)
        altered.ledger.arrays['area_m2']=altered.ledger.arrays['area_m2'].copy()
        altered.ledger.arrays['area_m2'][0]=np.nextafter(altered.ledger.arrays['area_m2'][0],np.inf)
        with self.assertRaisesRegex(ValueError,'原目标数组不一致'):
            _same_target(domain,altered,lambda:None)

    def test_different_route_revisions_need_their_own_replayed_proof(self):
        from femto.planning import coverage_certificate as coverage,coverage_workflow,planning_revision,surface_workspace
        from femto.planning.treatment_head import head_parameters
        head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
        original=surface_workspace.generate(self.model,self.region,head)
        capture=planning_revision.capture
        def stable(*args,**kwargs):
            kwargs.setdefault('algorithms',planning_revision.STABLE_PATCH_LAYOUT_ALGORITHMS)
            return capture(*args,**kwargs)
        with patch.object(planning_revision,'capture',side_effect=stable):
            changed=surface_workspace.generate(self.model,self.region,head)
        a=original['planning_context']['revision'];b=changed['planning_context']['revision']
        self.assertNotEqual(a['sha256'],b['sha256'])
        self.assertEqual(a['target_id'],b['target_id'])
        budget=coverage.CoverageBudget(max_depth=1)
        old,_=coverage_workflow.calculate(self.model,self.region,original,budget=budget)
        new,_=coverage_workflow.calculate(self.model,self.region,changed,budget=budget)
        result=compare_replayed(self.model,self.region,original,old,changed,[(new,None)])
        self.assertEqual(result['target_id'],a['target_id'])
        self.assertNotEqual(result['previous_route'],result['current_route'])
        proposal=propose_replayed(self.model,self.region,original,old,changed,[(new,None)])
        self.assertEqual(proposal['proposal']['pending_leaf_count'],
                         len(proposal['comparison']['comparison']['pending_old_leaf_indices']))
        self.assertFalse(proposal['installed_route']);self.assertFalse(proposal['installed_certificate'])
        self.assertFalse(proposal['proposal']['execution_ready'])
        with self.assertRaises(ValueError):
            compare_replayed(self.model,self.region,original,old,changed,[(old,None)])
