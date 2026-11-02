from types import SimpleNamespace
from threading import Thread
import unittest
from unittest.mock import patch
from femto.planning import screen_source_reuse as reuse


class ScreenSourceReuseTests(unittest.TestCase):
    def setUp(self):
        self.plan=dict(treatment_head={'mode':'fixed_width_auto'},planning_context={'revision':{'id':1}},value=1)
        self.region={'id':1};self.model=SimpleNamespace(store=SimpleNamespace(mesh='mesh'),state=1)
        self.route={'points':[1,2]};self.trajectory=SimpleNamespace(to_dict=lambda:self.route)
        self.mutate_on_check=False
        def check():
            if self.mutate_on_check:self.model.state=2
        self.check=check
        def require_current(model,region,head,revision):
            if model.state!=1:raise ValueError('live input changed')
        self.require=patch.object(reuse.planning_revision,'require_current',side_effect=require_current)
        self.validate=patch.object(reuse.planning_state,'validated_surface',return_value=self.trajectory)
        self.require.start();self.validate_mock=self.validate.start();self.addCleanup(self.require.stop);self.addCleanup(self.validate.stop)
        self.source=reuse._ScreenSource(self.model,self.region,self.plan,self.check)
        self.source.__enter__();self.addCleanup(self.source.close)

    def test_reuses_only_current_identity_and_checks_each_input(self):
        self.assertIs(self.source.read(self.model,self.region,self.plan),self.trajectory)
        self.assertEqual(self.validate_mock.call_count,1)
        with self.assertRaisesRegex(ValueError,'当前同步筛选'):
            self.source.read(self.model,dict(self.region),self.plan)
        self.plan['value']=2
        with self.assertRaisesRegex(ValueError,'发生变化'):self.source.verify()
        self.plan['value']=1;self.region['id']=2
        with self.assertRaisesRegex(ValueError,'发生变化'):self.source.verify()
        self.region['id']=1;self.route['points'].append(3)
        with self.assertRaisesRegex(ValueError,'发生变化'):self.source.verify()

    def test_live_input_change_in_callback_is_not_accepted(self):
        self.mutate_on_check=True
        with self.assertRaisesRegex(ValueError,'live input changed'):self.source.verify()

    def test_closed_context_cannot_be_reused(self):
        self.source.close()
        with self.assertRaisesRegex(ValueError,'已关闭'):self.source.verify()
        with self.assertRaisesRegex(ValueError,'不能重复使用'):self.source.__enter__()

    def test_context_rejects_other_threads(self):
        failures=[]
        def use():
            try:self.source.verify()
            except ValueError as error:failures.append(str(error))
        worker=Thread(target=use);worker.start();worker.join()
        self.assertEqual(len(failures),1);self.assertIn('线程',failures[0])


if __name__=='__main__':unittest.main()
