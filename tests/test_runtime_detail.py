"""Detailed observers preserve values and count emitted versus rejected work."""
import unittest
from unittest.mock import patch
import numpy as np
from femto.performance_detail import collect
from femto.planning import planning_revision as revisions,target_surface as target
from femto.planning.quality_constrained import evidence
import test_target_surface as fixtures
from test_cloud_csr_research import counter
from femto.geometry.cloud_radius_batches import NativeRadiusCounter
from femto.geometry import cloud_radius_kernel
from femto.geometry.cloud_layer_consistency import _CANDIDATE_RADIUS


class RuntimeDetailTests(unittest.TestCase):
    def test_target_counts_actual_children_and_budget_rejection_separately(self):
        case=fixtures.TargetSurfaceTests();case.setUp();self.addCleanup(case.doCleanups)
        mask=np.zeros((100,100),bool);mask[47:54,46:55]=True
        model,region=case.fixture(mask=mask)
        revision=revisions.capture(model,region,case.head)
        quality=evidence(model,region,allow_empty=True)
        for budget in (target.TargetBudget(),target.TargetBudget(max_units=7)):
            expected=target.build(model,region,revision,evidence=quality,budget=budget)
            with collect() as details:
                actual=target.build(model,region,revision,evidence=quality,budget=budget)
            self.assertEqual(actual.metadata,expected.metadata)
            for key in actual.arrays:
                self.assertEqual(actual.arrays[key].tobytes(),expected.arrays[key].tobytes(),key)
            rows=details.snapshot()['rows']
            self.assertEqual(rows['target.child_gates']['calls'],rows['target.child_fan']['calls'])
            self.assertEqual(rows['target.child_append']['counts']['children'],len(actual.arrays['area_m2']))
            if budget.max_units==7:
                self.assertEqual(rows['target.child_append']['counts']['budget_rejected_pixels'],1)
            self.assertIn('target.clip_batch',rows)
            self.assertIn('target.clip_projection',rows)

    def test_native_detailed_execution_keeps_complete_neighborhood_count(self):
        rng=np.random.default_rng(722)
        points=rng.uniform(-.01,.01,(400,3));normals=np.tile([0.,0.,1.],(len(points),1))
        _,legacy=counter(points,normals,max_pairs=65536)
        ids=np.arange(len(points));expected=legacy(ids)
        # An independent dense oracle counts every returned neighbor, including
        # those after a strict positive witness in the fused predicate.
        delta=points[:,None,:]-points[None,:,:]
        neighbors=int(np.count_nonzero((delta*delta).sum(axis=2)<_CANDIDATE_RADIUS**2))
        unfused=('native_expand_and_delta','native_squared','native_normal_products',
                 'native_arithmetic_and_guards','native_reduce')
        backends=[('numpy',None)]
        if cloud_radius_kernel.available():
            backends.append(('fused',cloud_radius_kernel._cloud_predicate))
        for name,backend in backends:
            with self.subTest(backend=name),patch.object(cloud_radius_kernel,'_cloud_predicate',backend),np.errstate(
                    divide='warn',over='warn',under='ignore',invalid='warn'):
                fast=NativeRadiusCounter(points,normals,legacy)
                self.assertTrue(fast.enabled)
                self.assertEqual(fast(ids),expected)
                with collect() as details:
                    actual=fast(ids)
                self.assertEqual(actual,expected)
                rows=details.snapshot()['rows']
                stages=('native_input','native_search')+(
                    ('native_fused_predicate',) if name=='fused' else unfused)
                for stage in stages:
                    self.assertGreater(rows['cloud.'+stage]['calls'],0)
                if name=='fused':
                    for stage in unfused:self.assertNotIn('cloud.'+stage,rows)
                    neighbor_row='cloud.native_fused_predicate'
                else:
                    self.assertNotIn('cloud.native_fused_predicate',rows)
                    neighbor_row='cloud.native_expand_and_delta'
                work=rows['cloud.native_complete_work']['counts']
                self.assertEqual(rows['cloud.native_input']['counts']['queries'],len(points))
                self.assertEqual(work['queries'],len(points))
                self.assertEqual(work['neighbors'],neighbors)
                self.assertEqual(rows[neighbor_row]['counts']['neighbors'],neighbors)
                self.assertEqual(work['query_batches'],rows[neighbor_row]['calls'])
                self.assertGreater(neighbors,len(points))


if __name__=='__main__':unittest.main()
