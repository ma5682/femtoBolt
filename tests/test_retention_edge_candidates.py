"""Bounded candidate selection does not manufacture routes or coverage."""
import unittest
from types import SimpleNamespace
import numpy as np

from femto.planning.coverage_certificate import Certificate
from femto.planning.retention_edge_candidates import propose,select_edges
from femto.planning.surface_trajectory import ScanSegment


class RetentionEdgeTests(unittest.TestCase):
    def test_gain_per_length_chooses_one_shared_edge_and_is_deterministic(self):
        weights={0:1.,1:1.,2:1.}
        edges={(0,0):{0,1},(0,1):{0},(0,2):{1},(0,3):{2}}
        lengths={(0,0):1.,(0,1):1.,(0,2):1.,(0,3):1.}
        result=select_edges(weights,edges,lengths,penalty_m=0.)
        self.assertEqual(result['selected_edges'],[(0,0),(0,3)])
        self.assertEqual(result['unresolved_leaf_indices'],[])
        self.assertTrue(result['all_pending_selected'])
        self.assertEqual(result,select_edges(weights,dict(reversed(list(edges.items()))),lengths,penalty_m=0.))

    def test_budgets_never_drop_unresolved_leaves(self):
        edges={(0,0):{0},(0,1):{1}}
        lengths={(0,0):1.,(0,1):1.}
        limited=select_edges({0:1.,1:1.},edges,lengths,penalty_m=0.,max_selected=1)
        self.assertEqual(limited['stop_reason'],'selection_budget')
        self.assertEqual(limited['unresolved_leaf_indices'],[1])
        limited=select_edges({0:1.,1:1.},edges,lengths,penalty_m=0.,max_updates=0)
        self.assertEqual(limited['stop_reason'],'update_budget')
        self.assertEqual(limited['unresolved_leaf_indices'],[1])
        no_candidate=select_edges({0:1.,1:1.},{(0,0):{0}},{(0,0):1.},penalty_m=0.)
        self.assertEqual(no_candidate['stop_reason'],'no_candidate')
        self.assertEqual(no_candidate['unresolved_leaf_indices'],[1])

    def test_invalid_values_and_cancellation_fail_closed(self):
        with self.assertRaises(ValueError):select_edges({0:1.},{(0,0):{2}},{(0,0):1.},penalty_m=0.)
        with self.assertRaises(ValueError):select_edges({0:float('nan')},{},{},penalty_m=0.)
        with self.assertRaises(ValueError):select_edges({0:1.},{(0,0):{0},3:{0}},{(0,0):1.,3:1.},penalty_m=0.)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            select_edges({0:1.},{(0,0):{0}},{(0,0):1.},penalty_m=0.,check=cancel)

    def test_random_bipartite_inputs_keep_exact_remaining_set(self):
        rng=np.random.default_rng(64)
        for _ in range(50):
            weights={i:float(rng.random()) for i in range(12)}
            edges={(0,i):set(np.flatnonzero(rng.random(12)<.2).tolist()) for i in range(15)}
            edges={key:value for key,value in edges.items() if value}
            lengths={key:float(rng.uniform(.001,.01)) for key in edges}
            result=select_edges(weights,edges,lengths,penalty_m=.001,max_selected=3)
            covered=set().union(*(edges[key] for key in result['selected_edges'])) if result['selected_edges'] else set()
            self.assertEqual(set(result['unresolved_leaf_indices']),set(weights)-covered)
            self.assertEqual(result['all_pending_selected'],not (set(weights)-covered))

    def test_mapping_uses_original_scan_endpoint_and_does_not_install_route(self):
        points=[dict(xyz_m=[0.,0.,0.],triangle_index=0),dict(xyz_m=[.01,0.,0.],triangle_index=0)]
        segment=ScanSegment(0,0,points,0)
        trajectory=SimpleNamespace(ordered_items=[segment])
        seeds=[dict(source_face=0,xyz_m=[0.,0.,0.],role='scan',segment_index=0,edge_index=0,endpoint=0)]
        domain=SimpleNamespace(representative_faces=np.array([0]),areas=np.array([1.]))
        cert=Certificate({},dict(status=np.array([1],dtype='u1'),root=np.array([0]),depth=np.array([0]),
             seed_face=np.array([0]),seed_xyz=np.array([[0.,0.,0.]]),seed_input_index=np.array([0]),
             proof_leaf=np.array([0]),proof_seed=np.array([0])))
        result=propose(cert,seeds,trajectory,domain,[0],5.)
        self.assertEqual(result['selected_edges'],[(0,0)])
        self.assertTrue(result['all_pending_selected'])
        self.assertFalse(result['installed_route']);self.assertFalse(result['installed_certificate'])
        seeds[0]['xyz_m']=[.001,0.,0.]
        with self.assertRaisesRegex(ValueError,'坐标或支持面'):
            propose(cert,seeds,trajectory,domain,[0],5.)
