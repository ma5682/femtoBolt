import unittest
import numpy as np
from femto.planning.local_metric_alignment import positions,schedule,shifted


class LocalMetricAlignmentTests(unittest.TestCase):
    def test_prior_values_stay_exact_and_all_steps_satisfy_local_metric(self):
        old=[.153,.301,.487,.681,.872]
        result=positions(0.,1.,[[0.,1.]],[1.],.2,old)
        self.assertTrue(np.all(np.diff(result)<=.2+1e-12))
        self.assertGreaterEqual(sum(value in old for value in result),3)

    def test_steep_face_only_affects_its_own_interval(self):
        p=positions(0.,1.,[[0.,.2],[0.,1.]],[.5,1.],.2,[.1,.2,.4,.6,.8])
        coordinate=np.interp(p,[0,.2,1.],[0,.4,1.2])
        self.assertTrue(np.all(np.diff(coordinate)<=.2+1e-12))
        self.assertTrue(any(x>=.19 for x in np.diff(p)[2:]))

    def test_invalid_and_unsupported_intervals_fail(self):
        for intervals,factors in (([[0.,.5]],[1.]),([[0.,1.]],[0.]),([[1.,0.]],[1.])):
            with self.subTest(intervals=intervals,factors=factors),self.assertRaises(ValueError):
                positions(0.,1.,intervals,factors,.2,[])
        with self.assertRaises(ValueError):positions(0.,1.,[[0.,1.]],[1.],0,[])

    def test_cancellation_propagates(self):
        def cancel():raise RuntimeError('cancel')
        with self.assertRaisesRegex(RuntimeError,'cancel'):schedule(0.,1.,[[0.,1.]],[1.],cancel)

    def test_shift_rebalances_neighbors_without_moving_outer_boundaries(self):
        old=np.array([0.,.1,.3,.5,.7,.9,1.])
        result=shifted(old,3,.05,[[0.,1.]],[1.],.2)
        np.testing.assert_allclose(result,[0.,.15,.35,.55,.7,.9,1.],atol=1e-12)
        self.assertTrue(np.all(np.diff(result)<=.2+1e-12))
        reverse=shifted(old,3,-.05,[[0.,1.]],[1.],.2)
        self.assertEqual(reverse[0],0.);self.assertEqual(reverse[-1],1.)
        self.assertTrue(np.all(np.diff(reverse)<=.2+1e-12))

    def test_shift_checks_curved_metric_and_rejects_insufficient_slack(self):
        base=np.interp(np.linspace(0,1.3,9),[0,.6,1.3],[0,.3,1.])
        result=shifted(base,2,.025,[[0.,.3],[0.,1.]],[.5,1.],.2)
        levels,cumulative=schedule(0.,1.,[[0.,.3],[0.,1.]],[.5,1.])
        self.assertTrue(np.all(np.diff(np.interp(result,levels,cumulative))<=.2+1e-12))
        with self.assertRaises(ValueError):shifted([0.,.2,.4],1,.01,[[0.,.4]],[1.],.2)
