from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from femto.planning import witness_bend_workflow_research as workflow
from femto.planning.surface_trajectory import ScanSegment


class WitnessBendWorkflowTests(unittest.TestCase):
    def setUp(self):
        stats = dict(connected_detours=2, connector_length_m=.1, revisit_connector_length_m=.02,
                     total_length_m=.3, gap_count=1, inter_block_gap_count=1)
        density = dict(proximity=[dict(radius_m=.001, near_fraction=.1), dict(radius_m=.002, near_fraction=.2)])
        self.state = dict(old=SimpleNamespace(statistics=stats), base=SimpleNamespace(statistics=deepcopy(stats)),
                          old_density=density, base_density=deepcopy(density))

    def test_motion_density_and_internal_gap_must_all_pass(self):
        stats = deepcopy(self.state['old'].statistics); density = deepcopy(self.state['old_density'])
        self.assertFalse(workflow._acceptable(self.state, stats, density))
        density['proximity'][0]['near_fraction'] -= .001
        self.assertTrue(workflow._acceptable(self.state, stats, density))
        for name, value in (('connected_detours', 3), ('connector_length_m', .11),
                            ('revisit_connector_length_m', .03), ('total_length_m', .31), ('gap_count', 2)):
            changed = deepcopy(stats); changed[name] = value
            with self.subTest(name=name): self.assertFalse(workflow._acceptable(self.state, changed, density))
        density['proximity'][1]['near_fraction'] += .01
        self.assertFalse(workflow._acceptable(self.state, stats, density))

    def test_input_snapshot_changes_with_source_proof_metadata(self):
        first = SimpleNamespace(metadata={'schema': 'old'}); second = SimpleNamespace(metadata={'schema': 'prior'})
        snapshot = workflow._snapshot({'route': 1}, {'route': 2}, [], first, second)
        second.metadata['schema'] = 'changed'
        self.assertNotEqual(snapshot, workflow._snapshot({'route': 1}, {'route': 2}, [], first, second))

    def test_point_checks_reject_unsupported_and_overlong_samples(self):
        mesh = trimesh.Trimesh(vertices=[[0., 0., .5], [.1, 0., .5], [0., .1, .5]], faces=[[0, 1, 2]], process=False)
        points = [dict(xyz_m=[.01, .01, .5], triangle_index=0, geometry_support='measured_mesh',
                       row_index=0, route_index=0, role='scan'),
                  dict(xyz_m=[.012, .01, .5], triangle_index=0, geometry_support='measured_mesh',
                       row_index=0, route_index=1, role='scan')]
        items = [ScanSegment(0, 0, points, 0)]
        workflow._point_checks(items, {'sample_step_mm': 5.}, mesh, lambda: None)
        points[1]['xyz_m'][0] = .03
        with self.assertRaisesRegex(ValueError, 'sample_step_mm'):
            workflow._point_checks(items, {'sample_step_mm': 5.}, mesh, lambda: None)
        points[1]['xyz_m'] = [.012, .01, .501]
        with self.assertRaisesRegex(ValueError, 'local triangle'):
            workflow._point_checks(items, {'sample_step_mm': 5.}, mesh, lambda: None)

    def test_replay_rejects_malformed_recipe_before_accessing_model(self):
        for recipe in ({}, {'seed': True, 'segment': 0, 'point': 0, 'requests': []},
                       {'seed': 0, 'segment': 0, 'point': 0, 'requests': [], 'extra': 1},
                       {'seed': 0, 'segment': 0, 'point': 1, 'requests': [], 'version': 'wrong',
                        'placement': 'edge_insertion', 'edge': 0},
                       {'seed': 0, 'segment': 0, 'point': 2, 'requests': [], 'version': workflow.EDGE_RECIPE_VERSION,
                        'placement': 'edge_insertion', 'edge': 0}):
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                workflow.replay(None, None, None, None, None, [], None, recipe, None)

    def test_edge_recipe_rederives_exact_proposal_and_replays_proof(self):
        recipe={'seed':1,'segment':0,'point':1,'edge':0,'version':workflow.EDGE_RECIPE_VERSION,
                'placement':'edge_insertion','requests':[[0,1,1,2]]}
        proposal={'seed':1,'segment':0,'point':1,'edge':0,'segments':[]}
        state={'proposals':[proposal], 'domain':SimpleNamespace(areas=np.array([1.])), 'nominal':'width'}
        old=SimpleNamespace(metadata={'old':1});prior=SimpleNamespace(metadata={'prior':1});proof=SimpleNamespace(summary={})
        with patch.object(workflow,'_prepare',return_value=state) as prepare, \
             patch.object(workflow,'_candidate',return_value=('final',{}, {'items_sha256':'x'},{})), \
             patch.object(workflow,'_acceptable',return_value=True), \
             patch.object(workflow,'_proof_inputs',return_value=([(0,1,1,2)],'seeds','route','transfer')), \
             patch.object(workflow.coverage,'validate') as validate, \
             patch.object(workflow.coverage_partition,'compare_union',return_value={'retained_subset':True}), \
             patch.object(workflow,'_finish'):
            result=workflow.replay(None,{}, {},old,{},[],prior,recipe,proof)
            self.assertTrue(prepare.call_args.kwargs['edge_insert'])
            self.assertEqual(validate.call_count,1)
            self.assertEqual(result['schema'],workflow.EDGE_REPLAY_VERSION)
            changed=deepcopy(recipe);changed.update(point=2,edge=1)
            with self.assertRaisesRegex(ValueError,'不属于重新推导'):
                workflow.replay(None,{}, {},old,{},[],prior,changed,proof)
            changed=deepcopy(recipe);changed['requests']=[[0,1,1,4]]
            with self.assertRaisesRegex(ValueError,'边内位置不一致'):
                workflow.replay(None,{}, {},old,{},[],prior,changed,proof)
            old_recipe={k:recipe[k] for k in ('seed','segment','point','requests')}
            old_result=workflow.replay(None,{}, {},old,{},[],prior,old_recipe,proof)
            self.assertFalse(prepare.call_args.kwargs['edge_insert'])
            self.assertEqual(old_result['schema'],workflow.REPLAY_VERSION)

    def test_build_rejects_unknown_placement_before_accessing_source(self):
        with self.assertRaises(ValueError):
            workflow.build(None,None,None,None,None,[],None,placement='other')

    def test_build_emits_direct_recipe_only_after_retention(self):
        from femto.planning import witness_edge_direct_replay_research as direct
        proposal={'seed':1,'segment':0,'point':1,'edge':0,'distance_m':.0003,
                  'pending_leaves':[2],'placement':'edge_insertion','projection_fraction':.5,'segments':[]}
        state={'proposals':[proposal],'proposal_report':{},'previous':{},'nominal':'width',
               'domain':SimpleNamespace(binding={},areas=np.array([1.]))}
        old=SimpleNamespace(metadata={'old':1});prior=SimpleNamespace(metadata={'prior':1});proof=SimpleNamespace(summary={})
        with patch.object(workflow,'_prepare',return_value=state), \
             patch.object(workflow,'_candidate',return_value=('final',{}, {'items_sha256':'x'},{})), \
             patch.object(workflow,'_acceptable',return_value=True), \
             patch.object(workflow,'_proof_inputs',return_value=([(0,1,1,2)],'seeds','route','transfer')), \
             patch.object(workflow.coverage,'evaluate',return_value=proof), \
             patch.object(workflow.coverage,'validate'), \
             patch.object(workflow.coverage_partition,'compare_union',return_value={'retained_subset':True}) as compare, \
             patch.object(workflow,'_finish'),patch.object(direct,'recipe',return_value={'direct':True}) as create:
            result,returned=workflow.build(None,{}, {},old,{},[],prior,placement='edge_insertion')
            self.assertIs(returned,proof);self.assertEqual(result['direct_recipe'],{'direct':True})
            self.assertEqual(create.call_args.args[3]['pending_leaves'],[2])
            create.reset_mock();compare.return_value={'retained_subset':False}
            rejected,returned=workflow.build(None,{}, {},old,{},[],prior,placement='edge_insertion')
            self.assertIsNone(returned);self.assertNotIn('direct_recipe',rejected);create.assert_not_called()


if __name__ == '__main__': unittest.main()
