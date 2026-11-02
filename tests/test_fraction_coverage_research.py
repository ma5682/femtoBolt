"""Safety and replay checks for the non-product fractional seed model."""
from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from femto.planning import coverage_certificate as coverage, footprint_model, fraction_coverage, fraction_footprint, scan_fraction_seeds
from femto.planning.coverage_domain import CoverageDomain, scan_endpoints
from femto.planning.surface_trajectory import ScanSegment


class FractionCoverageResearchTests(unittest.TestCase):
    def setUp(self):
        points = [dict(xyz_m=[.01, .01, 0.], triangle_index=0, row_index=0, role='scan'),
                  dict(xyz_m=[.08, .01, 0.], triangle_index=0, row_index=0, role='scan'),
                  dict(xyz_m=[.09, .01, 0.], triangle_index=0, row_index=0, role='connector')]
        self.trajectory = SimpleNamespace(planner={}, ordered_items=[ScanSegment(0, 0, points, 0)])
        self.faces = np.asarray([0], dtype=np.int64)
        domain = CoverageDomain.mesh([[0, 0, 0], [.125, 0, 0], [0, .125, 0]], [[0, 1, 2]])
        self.mesh_domain = domain
        class DomainView:
            ledger = SimpleNamespace(arrays={'region_faces': self.faces})
            def __getattr__(self, name):return getattr(domain, name)
        self.domain = DomainView()
        self.endpoints = scan_endpoints(self.trajectory, self.faces)

    def test_versioned_model_keeps_old_model_and_deducts_rounding(self):
        old = footprint_model.nominal(100.)
        model = fraction_footprint.nominal(100.)
        self.assertEqual(footprint_model.validate(old), old)
        self.assertEqual(footprint_model.validate(model), model)
        self.assertNotEqual(old['schema'], model['schema'])
        records, _ = scan_fraction_seeds.propose(self.trajectory, self.faces, [(0, 0, 3, 8)])
        record = records[0]
        safe = footprint_model.seed_radius(model, record)
        a, b = self.trajectory.ordered_items[0].points[:2]
        exact_error = sum(abs(Fraction(z) - (5*Fraction(x)+3*Fraction(y))/8)
                          for x, y, z in zip(a['xyz_m'], b['xyz_m'], record['xyz_m']))
        self.assertGreaterEqual(Fraction(model['nominal_radius_m'])-Fraction(safe), exact_error)
        self.assertEqual(footprint_model.seed_radius(model, self.endpoints[0]), model['nominal_radius_m'])
        bad = deepcopy(model); bad['seed_rule'] = old['seed_rule']
        with self.assertRaises(ValueError):footprint_model.validate(bad)
        bad = deepcopy(model); bad['execution_ready'] = 0
        with self.assertRaises(ValueError):footprint_model.validate(bad)
        for change in ({'endpoint': True}, {'fraction_denominator': 3},
                       {'midpoint_error_bound_m': -1.}, {'provenance': 'unverified'}):
            bad = dict(record, **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                footprint_model.seed_radius(model, bad)

    def test_route_bound_requests_replay_and_reject_tampering(self):
        old_model = footprint_model.nominal(100.)
        route = {'schema': 'declared-model-seeds-v1', 'path_acceptance_verified': False}
        plan = {'surface_trajectory': {}}
        with patch.object(fraction_coverage, 'from_plan', return_value=(self.domain, self.endpoints, old_model, {})), \
             patch.object(fraction_coverage.SurfaceTrajectory, 'from_dict', return_value=self.trajectory), \
             patch.object(fraction_coverage.coverage_workflow, 'route_binding', return_value=route):
            proof, proposal = fraction_coverage.calculate(None, {}, plan, [(0, 0, 3, 8)],
                budget=coverage.CoverageBudget(max_depth=2, seconds=45))
            self.assertGreater(proposal['proposed_records'], 0)
            self.assertEqual(fraction_coverage.replay(None, {}, plan, [(0, 0, 3, 8)], proof), proposal)
            with self.assertRaises(ValueError):
                fraction_coverage.replay(None, {}, plan, [(0, 0, 1, 2)], proof)
            with self.assertRaises(ValueError):
                fraction_coverage.replay(None, {}, plan, [(0, 1, 1, 2)], proof)
            with self.assertRaises(ValueError):
                fraction_coverage.replay(None, {}, plan, [(0, 0, 3, 8)],
                    coverage.Certificate(dict(proof.metadata, sha256='0'*64), proof.arrays))

    def test_distinct_guided_version_rechecks_matching_width_and_replays(self):
        old = footprint_model.nominal(100.)
        model = fraction_footprint.nominal(100.)
        records, _ = scan_fraction_seeds.propose(self.trajectory, self.faces, [(0, 0, 3, 8)])
        seeds = self.endpoints + records
        budget = coverage.CoverageBudget(max_depth=2, seconds=45)
        guide = coverage.evaluate(self.mesh_domain, self.endpoints, old, budget=budget)
        proof = coverage.evaluate(self.mesh_domain, seeds, model, budget=budget, guide=guide,
                                  algorithm=coverage.RESEARCH_FRACTION_REPROOF_ALGORITHM)
        self.assertEqual(proof.metadata['inputs']['guide_sha256'], guide.metadata['sha256'])
        self.assertFalse(proof.summary['execution_ready'])
        self.assertIs(coverage.validate(proof, self.mesh_domain, seeds, model, guide=guide), proof)
        with self.assertRaises(ValueError):
            coverage.evaluate(self.mesh_domain, seeds, fraction_footprint.nominal(101.), budget=budget, guide=guide,
                              algorithm=coverage.RESEARCH_FRACTION_REPROOF_ALGORITHM)
        with self.assertRaises(ValueError):
            coverage.evaluate(self.mesh_domain, seeds, old, budget=budget, guide=guide,
                              algorithm=coverage.RESEARCH_FRACTION_REPROOF_ALGORITHM)
        with self.assertRaises(ValueError):
            coverage.validate(proof, self.mesh_domain, self.endpoints, model, guide=guide)
        self.assertNotIn(coverage.RESEARCH_FRACTION_REPROOF_ALGORITHM, coverage.P2_ALGORITHMS)

    def test_priority_refuses_bad_guide_different_target_or_width_before_proposals(self):
        old_model=footprint_model.nominal(100.);different=footprint_model.nominal(101.)
        old_plan={'surface_trajectory':{}};new_plan={'surface_trajectory':{}}
        def old_and_new(*args):
            return (self.domain,self.endpoints,old_model if args[2] is old_plan else different,{})
        with patch.object(fraction_coverage,'from_plan',side_effect=old_and_new), \
             patch.object(fraction_coverage.SurfaceTrajectory,'from_dict',return_value=self.trajectory), \
             patch.object(fraction_coverage.coverage_workflow,'route_binding',return_value={}), \
             patch.object(fraction_coverage.fraction_priority,'propose') as propose:
            with patch.object(fraction_coverage.coverage,'validate',side_effect=ValueError('bad guide')):
                with self.assertRaisesRegex(ValueError,'bad guide'):
                    fraction_coverage.prioritize(None,{},old_plan,object(),new_plan,[0])
            propose.assert_not_called()
            with patch.object(fraction_coverage.coverage,'validate'), \
                 patch.object(fraction_coverage,'_same_target',side_effect=ValueError('different target')):
                with self.assertRaisesRegex(ValueError,'different target'):
                    fraction_coverage.prioritize(None,{},old_plan,object(),new_plan,[0])
            propose.assert_not_called()
            with patch.object(fraction_coverage.coverage,'validate'),patch.object(fraction_coverage,'_same_target'):
                with self.assertRaisesRegex(ValueError,'相同固定名义'):
                    fraction_coverage.prioritize(None,{},old_plan,object(),new_plan,[0])
            propose.assert_not_called()

    def test_retention_supplement_replays_all_proofs_before_proposing(self):
        old_model=footprint_model.nominal(100.)
        route={'schema':'declared-model-seeds-v1','path_acceptance_verified':False}
        plan={'surface_trajectory':{},'treatment_head':{},'planning_context':{'revision':{}}}
        budget=coverage.CoverageBudget(max_depth=1,seconds=45)
        self.domain.revision={'target_id':'synthetic'}
        old=coverage.evaluate(self.mesh_domain,self.endpoints,old_model,route_binding=route,budget=budget)
        normal=coverage.evaluate(self.mesh_domain,self.endpoints,old_model,route_binding=route,budget=budget)
        requests=[(0,0,3,8)]
        positions,_=scan_fraction_seeds.propose(self.trajectory,self.faces,requests)
        guided=coverage.evaluate(self.mesh_domain,self.endpoints+positions,fraction_footprint.nominal(100.),
            route_binding=route,budget=budget,guide=old,algorithm=coverage.RESEARCH_FRACTION_REPROOF_ALGORITHM)
        actual_validate=coverage.validate
        with patch.object(fraction_coverage,'from_plan',return_value=(self.domain,self.endpoints,old_model,{})), \
             patch.object(fraction_coverage.SurfaceTrajectory,'from_dict',return_value=self.trajectory), \
             patch.object(fraction_coverage.coverage_workflow,'route_binding',return_value=route), \
             patch.object(coverage,'validate',side_effect=lambda cert,domain,seeds,model,**kwargs:actual_validate(cert,self.mesh_domain,seeds,model,**kwargs)), \
             patch('femto.planning.planning_revision.require_current'), \
             patch.object(fraction_coverage,'_same_target'):
            result=fraction_coverage.retention_supplement(None,{},plan,old,plan,normal,guided,requests)
            self.assertTrue(result['comparison']['retained_subset'])
            self.assertEqual(result['selected_edges'],[])
            self.assertFalse(result['installed_certificate']);self.assertFalse(result['execution_ready'])
            bad=coverage.Certificate(dict(normal.metadata,sha256='0'*64),normal.arrays)
            with self.assertRaises(ValueError):
                fraction_coverage.retention_supplement(None,{},plan,old,plan,bad,guided,requests)
            with self.assertRaises(ValueError):
                fraction_coverage.retention_supplement(None,{},plan,old,plan,normal,guided,[(0,0,1,2)])
            new_plan={'surface_trajectory':{},'different':True,'treatment_head':{},'planning_context':{'revision':{}}}
            def different_width(*args):
                return (self.domain,self.endpoints,footprint_model.nominal(101.) if args[2] is new_plan else old_model,{})
            with patch.object(fraction_coverage,'from_plan',side_effect=different_width):
                with self.assertRaisesRegex(ValueError,'相同固定名义作用宽度'):
                    fraction_coverage.retention_supplement(None,{},plan,old,new_plan,normal,guided,requests)
            with patch.object(fraction_coverage,'_same_target',side_effect=ValueError('different target')):
                with self.assertRaisesRegex(ValueError,'different target'):
                    fraction_coverage.retention_supplement(None,{},plan,old,plan,normal,guided,requests)
            mutable=deepcopy(plan)
            def validate_then_mutate(cert,domain,seeds,model,**kwargs):
                result=actual_validate(cert,self.mesh_domain,seeds,model,**kwargs)
                if cert is guided:mutable['modified_during_replay']=True
                return result
            with patch.object(coverage,'validate',side_effect=validate_then_mutate):
                with self.assertRaisesRegex(ValueError,'路线发生变化'):
                    fraction_coverage.retention_supplement(None,{},mutable,old,mutable,normal,guided,requests)


if __name__ == '__main__':
    unittest.main()
