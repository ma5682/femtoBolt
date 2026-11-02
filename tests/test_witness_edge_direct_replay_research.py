from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from femto.planning import witness_edge_direct_replay_research as direct
from femto.planning.surface_trajectory import ScanSegment


class DirectEdgeReplayTests(unittest.TestCase):
    def setUp(self):
        params={'quality_evidence':{},'effective_domain':{}}
        self.plan={'surface_trajectory':{'planner':{'parameters':params}},'treatment_head':{},
                   'planning_context':{'revision':{}}}
        self.old_proof=SimpleNamespace(metadata={'sha256':'old'})
        self.proof=SimpleNamespace(metadata={'sha256':'final'},summary={})
        self.request={'version':direct.RECIPE_VERSION,'seed':1,'segment':0,'point':1,'edge':0,
            'pending_leaves':[2], 'requests':[[0,0,1,2]], 'old_route':{'id':'old'},
            'base_route':{'id':'base'},'old_proof_sha256':'old'}

    def test_rejects_malformed_recipe_before_model_access(self):
        for update in ({'point':2},{'edge':True},{'pending_leaves':[2,2]},
                       {'pending_leaves':[-1]},{'version':'invalid'},{'requests':None}):
            request=deepcopy(self.request);request.update(update)
            with self.subTest(update=update),self.assertRaises(ValueError):
                direct.replay(None,None,None,None,None,request,None)
        point=deepcopy(self.request);point.pop('edge');point['version']=direct.POINT_RECIPE_VERSION
        for update in ({'point':True},{'pending_leaves':[2,2]},{'edge':0},{'version':direct.RECIPE_VERSION}):
            request=deepcopy(point);request.update(update)
            with self.subTest(point_update=update),self.assertRaises(ValueError):
                direct.replay(None,None,None,None,None,request,None)
        with self.assertRaises(ValueError):
            direct.replay(None,None,None,None,None,point,None,include_research_route=1)

    def test_replays_old_and_final_proofs_and_rederives_candidate(self):
        old_route={'id':'old'};base_route={'id':'base'}
        old=SimpleNamespace(statistics={})
        base=SimpleNamespace(statistics={},gap_report={'reconnection':{'base_items':[{'kind':'scan_segment'}],'connections':[]}})
        ledger=SimpleNamespace(arrays={'region_faces':[0]})
        domain=SimpleNamespace(ledger=ledger,representative_faces=[0],areas=[1.],revision={'sha256':'revision'})
        width={'schema':'femto-nominal-fixed-ball-v1','effective_width_mm':10.}
        proposal={'seed':1,'segment':0,'point':1,'edge':0,'segments':[]}
        model=SimpleNamespace(store=SimpleNamespace(mesh='mesh'))
        old_model=SimpleNamespace(store=SimpleNamespace(mesh='old_mesh'))
        def bind(value):return old_route if value is old else base_route
        def propose(old_route,segments,faces,proof,pending,**kwargs):
            return ([proposal] if pending==[2] else []),{}
        def validate(proof,domain,seeds,width,**kwargs):
            if proof is self.proof and seeds!=['endpoint','position']:raise ValueError('proof inputs changed')
        with patch.object(direct,'from_plan',side_effect=[(domain,[],width,None),(domain,[],width,None)]) as from_plan, \
             patch.object(direct.SurfaceTrajectory,'from_dict',side_effect=[old,base]) as from_dict, \
             patch.object(direct.coverage_workflow,'route_binding',side_effect=bind), \
             patch.object(direct.coverage,'validate',side_effect=validate) as checked, \
             patch.object(direct,'_same_target'), \
             patch.object(direct.ScanSegment,'from_dict',return_value='segment'), \
             patch.object(direct.witness_bend_research,'propose',side_effect=propose) as derive, \
             patch.object(direct.auto_supplement_route_research,'_source_visits',return_value=[]), \
             patch.object(direct.fraction_footprint,'nominal',return_value='nominal'), \
             patch.object(direct.surface_workspace,'canonical_region_mesh',return_value='mesh'), \
             patch.object(direct.row_proximity_research,'measure',return_value={}), \
             patch.object(direct.bend,'_candidate',return_value=('final',{}, {'items_sha256':'route'},{})) as candidate, \
             patch.object(direct.bend,'_acceptable',return_value=True), \
             patch.object(direct.scan_fraction_seeds,'propose',return_value=(['position'],{})), \
             patch.object(direct,'scan_endpoints',return_value=['endpoint']), \
             patch.object(direct.coverage_partition,'compare_union',return_value={'retained_subset':True}), \
             patch.object(direct.planning_revision,'require_current'), \
             patch.object(direct.target_surface,'validate'), \
             patch.object(direct.coverage,'_checked_assets'):
            guide=SimpleNamespace(metadata={'sha256':'guide'})
            result=direct.replay(model,{},self.plan,self.old_proof,self.plan,self.request,self.proof,old_model=old_model,old_guide=guide)
            self.assertIs(from_plan.call_args_list[0].args[0],old_model)
            self.assertIs(from_plan.call_args_list[1].args[0],model)
            self.assertEqual(checked.call_count,2)
            self.assertIs(checked.call_args_list[0].kwargs['guide'],guide)
            self.assertIs(checked.call_args.kwargs['guide'],self.old_proof)
            self.assertTrue(derive.call_args.kwargs['edge_insert'])
            self.assertFalse(result['intermediate_search_proof_replayed'])
            self.assertFalse(result['execution_ready'])
            from_plan.side_effect=[(domain,[],width,None),(domain,[],width,None)]
            from_dict.side_effect=[old,base]
            point=deepcopy(self.request);point.pop('edge');point['version']=direct.POINT_RECIPE_VERSION
            result=direct.replay(model,{},self.plan,self.old_proof,self.plan,point,self.proof)
            self.assertEqual(result['schema'],direct.POINT_VERSION)
            self.assertFalse(derive.call_args.kwargs['edge_insert'])
            from_plan.side_effect=[(domain,[],width,None),(domain,[],width,None)];from_dict.side_effect=[old,base]
            item=ScanSegment(0,0,[{'xyz_m':[0.,0.,0.]},{'xyz_m':[1.,0.,0.]}],0)
            items=[item.to_dict()];candidate.return_value=(SimpleNamespace(ordered_items=[item]),{},
                {'items_sha256':direct.planning_revision.digest(items)},{})
            result=direct.replay(model,{},self.plan,self.old_proof,self.plan,point,self.proof,include_research_route=True)
            snapshot=result['research_route'];self.assertEqual(snapshot['ordered_items'],items)
            self.assertFalse(snapshot['execution_ready']);self.assertFalse(snapshot['installed_route'])
            snapshot['ordered_items'][0]['points'][0]['xyz_m'][0]=99.
            self.assertEqual(item.points[0]['xyz_m'][0],0.)
            from_plan.side_effect=[(domain,[],width,None),(domain,[],width,None)];from_dict.side_effect=[old,base]
            candidate.return_value=(SimpleNamespace(ordered_items=[item]),{}, {'items_sha256':'wrong'}, {})
            with self.assertRaisesRegex(ValueError,'快照与已核验几何不一致'):
                direct.replay(model,{},self.plan,self.old_proof,self.plan,point,self.proof,include_research_route=True)
            from_plan.side_effect=[(domain,[],width,None),(domain,[],width,None)]
            from_dict.side_effect=[old,base]
            altered=deepcopy(self.request);altered['pending_leaves']=[3]
            with self.assertRaisesRegex(ValueError,'不能从原证据重建'):
                direct.replay(model,{},self.plan,self.old_proof,self.plan,altered,self.proof)
            from_plan.side_effect=[(domain,[],width,None),(domain,[],width,None)];from_dict.side_effect=[old,base]
            def mutate_guide(*args,**kwargs):
                validate(*args,**kwargs)
                guide.metadata['sha256']='changed'
            checked.side_effect=mutate_guide
            with self.assertRaisesRegex(ValueError,'期间输入发生变化'):
                direct.replay(model,{},self.plan,self.old_proof,self.plan,self.request,self.proof,old_guide=guide)

    def test_pair_rejects_missing_conflicting_or_unbound_members(self):
        members=[dict(seed=i,segment=i,point=1,pending_leaves=[i]) for i in range(2)]
        request=dict(version=direct.PAIR_RECIPE_VERSION,members=members,requests=[],old_route={},base_route={},old_proof_sha256='old')
        for mode in ('missing','conflict','duplicate_seed','extra_coordinates','bool_index','bad_leaf'):
            bad=deepcopy(request)
            if mode=='missing':bad['members'].pop()
            elif mode=='conflict':bad['members'][1]['segment']=0
            elif mode=='duplicate_seed':bad['members'][1]['seed']=0
            elif mode=='extra_coordinates':bad['members'][0]['xyz_m']=[0.,0.,0.]
            elif mode=='bool_index':bad['members'][0]['point']=True
            else:bad['members'][0]['pending_leaves']=[False]
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                direct.replay(None,None,None,None,None,bad,None)
        for value in (None,[],[{}],[{},{}]):
            with self.subTest(value=value),self.assertRaises(ValueError):direct.pair_recipe(None,None,None,value,[])


if __name__=='__main__':unittest.main()
