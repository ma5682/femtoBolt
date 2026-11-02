from copy import deepcopy
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from femto.planning import old_coverage_session_research as reuse


class OldCoverageSessionTests(unittest.TestCase):
    def setUp(self):
        self.model=object();self.region={'region':1};self.plan={'treatment_head':{},'planning_context':{'revision':{}}}
        self.certificate=SimpleNamespace(metadata={'id':1},arrays={'x':np.array([1],dtype='i8')})
        ledger=SimpleNamespace(metadata={'id':1},arrays={'mask':np.array([1],dtype='u1')})
        self.domain=SimpleNamespace(ledger=ledger,revision={'id':1},binding={'id':1},
            vertices=np.array([[0.,0.,0.]]),faces=np.array([[0,0,0]]),source_faces=np.array([0]),
            areas=np.array([1.]),quality=np.array([0],dtype='u1'),representative_faces=np.array([0]),
            _parent_children={(0,0):{0}})
        self.seeds=[{'position':[1.,2.,3.]}];self.footprint={'width':1};self.route={'route':1}
        self.check=lambda:None

    def session(self,guide=None):
        return reuse._OldCoverageSession(self.model,self.region,self.plan,self.certificate,self.domain,
                                         self.seeds,self.footprint,self.route,self.check,guide=guide)

    def require(self,session,domain=None):
        return session.require(self.model,self.region,self.plan,self.certificate,domain or self.domain,
                               self.seeds,self.footprint,self.route,self.check)

    def test_one_replay_and_same_value_fresh_domain(self):
        with patch.object(reuse.coverage,'validate') as validate,patch.object(reuse.coverage,'_checked_assets'), \
             patch.object(reuse.planning_revision,'require_current'):
            session=self.session()
            with session:
                self.require(session,deepcopy(self.domain));self.require(session)
                self.assertEqual(validate.call_count,1)
            with self.assertRaisesRegex(ValueError,'已关闭'):self.require(session)
            with self.assertRaises(ValueError):session.__enter__()

    def test_mutation_of_target_seed_or_proof_cannot_reuse(self):
        for name in ('ledger','geometry','parents','seed','proof','region','plan'):
            with self.subTest(name=name):
                self.setUp()
                with patch.object(reuse.coverage,'validate'),patch.object(reuse.coverage,'_checked_assets'), \
                     patch.object(reuse.planning_revision,'require_current'):
                    with self.session() as session:
                        if name=='ledger':self.domain.ledger.arrays['mask'][0]=0
                        elif name=='geometry':self.domain.vertices[0,0]=1.
                        elif name=='parents':self.domain._parent_children[(0,0)].add(1)
                        elif name=='seed':self.seeds[0]['position'][0]=4.
                        elif name=='proof':self.certificate.metadata['id']=2
                        elif name=='region':self.region['region']=2
                        else:self.plan['treatment_head']['x']=1
                        with self.assertRaisesRegex(ValueError,'输入或目标发生变化'):self.require(session)

    def test_malformed_parent_mapping_is_not_normalized_into_valid_data(self):
        with patch.object(reuse.coverage,'validate'),patch.object(reuse.coverage,'_checked_assets'), \
             patch.object(reuse.planning_revision,'require_current'):
            with self.session() as session:
                self.domain._parent_children[(0,0)]={0.1}
                with self.assertRaisesRegex(ValueError,'目标分区无效'):self.require(session)

    def test_other_thread_and_foreign_inputs_are_rejected(self):
        with patch.object(reuse.coverage,'validate'),patch.object(reuse.coverage,'_checked_assets'), \
             patch.object(reuse.planning_revision,'require_current'):
            with self.session() as session:
                errors=[]
                def other():
                    try:self.require(session)
                    except Exception as error:errors.append(error)
                thread=Thread(target=other);thread.start();thread.join()
                self.assertEqual(len(errors),1);self.assertIn('当前线程',str(errors[0]))
                with self.assertRaisesRegex(ValueError,'当前同步任务'):
                    session.require(object(),self.region,self.plan,self.certificate,self.domain,
                                    self.seeds,self.footprint,self.route,self.check)

    def test_failed_validation_closes_session(self):
        session=self.session()
        with patch.object(reuse.coverage,'validate',side_effect=ValueError('proof invalid')), \
             patch.object(reuse.coverage,'_checked_assets'),patch.object(reuse.planning_revision,'require_current'):
            with self.assertRaisesRegex(ValueError,'proof invalid'):session.__enter__()
        self.assertEqual(session.state,'closed')

    def test_proof_array_is_rechecked_before_reuse(self):
        def assets(cert,check):
            if cert.arrays['x'][0]!=1:raise ValueError('coverage asset changed')
        with patch.object(reuse.coverage,'validate'),patch.object(reuse.coverage,'_checked_assets',side_effect=assets), \
             patch.object(reuse.planning_revision,'require_current'):
            with self.session() as session:
                self.certificate.arrays['x'][0]=2
                with self.assertRaisesRegex(ValueError,'coverage asset changed'):self.require(session)

    def test_recalculated_proof_guide_is_rechecked_before_reuse(self):
        self.certificate.metadata['algorithm']=reuse.coverage.RECALCULATION_ALGORITHMS[0]
        guide=SimpleNamespace(metadata={'id':1})
        with patch.object(reuse.coverage,'validate'),patch.object(reuse.coverage,'_checked_assets'), \
             patch.object(reuse.planning_revision,'require_current'), \
             patch('femto.planning.coverage_recalculation.require_guide',return_value=guide) as verify:
            with self.session() as session:
                before=verify.call_count;self.require(session);self.assertGreater(verify.call_count,before)
                guide.metadata['id']=2
                with self.assertRaisesRegex(ValueError,'输入或目标发生变化'):self.require(session)

    def test_explicit_guide_is_replayed_and_mutation_rejects_reuse(self):
        guide=SimpleNamespace(metadata={'id':1},arrays={'x':np.array([1],dtype='i8')})
        def assets(cert,check):
            if cert.arrays['x'][0]!=1:raise ValueError('coverage asset changed')
        with patch.object(reuse.coverage,'validate') as validate, \
             patch.object(reuse.coverage,'_checked_assets',side_effect=assets), \
             patch.object(reuse.planning_revision,'require_current'):
            with self.session(guide) as session:
                self.assertIs(validate.call_args.kwargs['guide'],guide)
                self.require(session)
                guide.metadata['id']=2
                with self.assertRaisesRegex(ValueError,'输入或目标发生变化'):self.require(session)
                guide.metadata['id']=1;guide.arrays['x'][0]=2
                with self.assertRaisesRegex(ValueError,'coverage asset changed'):self.require(session)
            self.assertIsNone(session.guide)


if __name__=='__main__':unittest.main()
