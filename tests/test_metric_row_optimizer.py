"""Global metric rows preserve the original spacing and bounded search."""
import unittest
import itertools
import math
import numpy as np

from femto.planning.metric_row_optimizer import positions,positions_with_slack
from femto.planning.local_metric_alignment import schedule


class MetricRowOptimizerTests(unittest.TestCase):
    def test_spacing_floor_rejects_dense_prior_gaps_without_relaxing_maximum(self):
        prior=[.002,.004,.006,.008,.010,.012,.014,.016,.018]
        old,_=positions_with_slack(0.,.020,[[0.,.020]],[1.],.004,prior,extra_fraction=.2)
        rows,report=positions_with_slack(0.,.020,[[0.,.020]],[1.],.004,prior,extra_fraction=.2,minimum_fraction=.75)
        self.assertGreaterEqual(np.diff(rows).min(),.003-2e-12)
        self.assertLessEqual(np.diff(rows).max(),.004+2e-12)
        self.assertLessEqual(len(rows),len(old))
        self.assertEqual(report['minimum_fraction'],.75)

    def test_spacing_floor_handles_short_endpoint_fitted_ranges(self):
        for end in [.001,.0041,.0061]:
            rows,report=positions_with_slack(0.,end,[[0.,end]],[1.],.004,[],minimum_fraction=.75)
            expected=min(.003,end/math.ceil((end-1e-12)/.004))
            self.assertGreaterEqual(np.diff(rows).min(),expected-2e-12)
            self.assertLessEqual(np.diff(rows).max(),.004+2e-12)
            self.assertEqual(rows[0],0.);self.assertEqual(rows[-1],end)
            self.assertAlmostEqual(report['effective_minimum_metric_step_m'],expected)

    def test_spacing_floor_matches_exhaustive_prior_choices_in_metric_space(self):
        rng=np.random.default_rng(817)
        for _ in range(25):
            end=float(rng.uniform(.003,.02));factor=float(rng.uniform(.45,1.));old=sorted(rng.uniform(0,end,5).tolist())
            rows,report=positions_with_slack(0.,end,[[0.,end]],[factor],.004,old,extra_fraction=.2,minimum_fraction=.75)
            metric_end=end/factor;minimum=math.ceil((metric_end-1e-12)/.004);limit=minimum+math.ceil(minimum*.2)
            floor=min(.003,metric_end/minimum);best=0
            for count in range(len(old)+1):
                for subset in itertools.combinations(old,count):
                    gaps=np.diff([0.,*subset,end])/factor
                    counts=[max(1,math.ceil((float(g)-1e-12)/.004)) for g in gaps]
                    if sum(counts)<=limit and all(g/n>=floor-1e-12 for g,n in zip(gaps,counts)):best=max(best,count)
            self.assertEqual(report['retained_prior_levels'],best)
            self.assertGreaterEqual((np.diff(rows)/factor).min(),floor-2e-12)
            self.assertLessEqual((np.diff(rows)/factor).max(),.004+2e-12)

    def test_spacing_floor_is_explicit_and_validated(self):
        args=(0.,.012,[[0.,.012]],[1.],.004,[.003,.006,.009])
        a,old=positions_with_slack(*args);b,explicit=positions_with_slack(*args,minimum_fraction=None)
        np.testing.assert_array_equal(a,b);self.assertEqual(old,explicit)
        self.assertNotIn('minimum_fraction',old)
        for fraction in (True,.4,1.1,float('nan')):
            with self.subTest(fraction=fraction),self.assertRaises(ValueError):positions_with_slack(*args,minimum_fraction=fraction)

    def run_case(self,last,old,factor=1.,maximum=.004):
        return positions(0.,last,[[0.,last]],[factor],maximum,old)

    def test_keeps_old_levels_when_minimum_row_count_is_unchanged(self):
        old=[.0029,.0063,.0102]
        rows,report=self.run_case(.014,old)
        self.assertEqual(len(rows),5)
        self.assertTrue(all(value in rows for value in old))
        self.assertEqual(report['retained_prior_levels'],3)
        self.assertLessEqual(np.diff(rows).max(),.004+1e-12)

    def test_dense_old_levels_do_not_force_extra_rows(self):
        rows,report=self.run_case(.012,[.001,.002,.003,.004,.005,.006,.007,.008,.009,.010,.011])
        np.testing.assert_allclose(rows,[0.,.004,.008,.012],atol=1e-14)
        self.assertEqual(report['row_count'],4)
        self.assertEqual(report['retained_prior_levels'],2)

    def test_sloped_metric_not_just_projected_row_spacing(self):
        rows,_=self.run_case(.012,[.002,.004,.006,.008,.010],factor=.5)
        levels,metric=schedule(0.,.012,[[0.,.012]],[.5])
        actual=np.diff(np.interp(rows,levels,metric))
        self.assertLessEqual(actual.max(),.004+1e-12)
        self.assertGreaterEqual(actual.min(),.002-1e-12)
        self.assertEqual(len(rows),7)

    def test_short_surface_and_invalid_inputs_are_explicit(self):
        rows,_=self.run_case(.001,[])
        np.testing.assert_array_equal(rows,[0.,.001])
        with self.assertRaises(ValueError):self.run_case(.012,[float('nan')])
        with self.assertRaises(ValueError):self.run_case(.012,[],maximum=0.)
        with self.assertRaisesRegex(ValueError,'比较超过预算'):
            self.run_case(1.,np.linspace(.001,.999,800).tolist())

    def test_cancellation_is_not_swallowed(self):
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            positions(0.,.01,[[0.,.01]],[1.],.004,[.003],check=cancel)

    def test_random_small_cases_match_exhaustive_prior_retention(self):
        rng=np.random.default_rng(815)
        for _ in range(40):
            end=float(rng.uniform(.007,.025));old=sorted(rng.uniform(.0002,end-.0002,6).tolist())
            rows,report=self.run_case(end,old)
            minimum=math.ceil((end-1e-12)/.004)
            self.assertEqual(len(rows)-1,minimum)
            best=0
            for count in range(len(old)+1):
                for subset in itertools.combinations(old,count):
                    levels=[0.,*subset,end];steps=0;valid=True
                    for a,b in zip(levels,levels[1:]):
                        gap=b-a;n=max(1,math.ceil((gap-1e-12)/.004))
                        if gap/n<.002-1e-12:valid=False;break
                        steps+=n
                    if valid and steps==minimum:best=max(best,count)
            self.assertEqual(report['retained_prior_levels'],best)
            self.assertTrue(np.all(np.diff(rows)<=.004+2e-12))
            self.assertTrue(np.all(np.diff(rows)>=.002-2e-12))
            reversed_rows,_=self.run_case(end,list(reversed(old)))
            np.testing.assert_array_equal(rows,reversed_rows)

    def test_bounded_extra_rows_can_preserve_more_prior_levels(self):
        old=[.003,.006,.009]
        least,first=positions_with_slack(0.,.012,[[0.,.012]],[1.],.004,old,extra_fraction=0.)
        extra,second=positions_with_slack(0.,.012,[[0.,.012]],[1.],.004,old,extra_fraction=.1)
        self.assertEqual(len(least),4);self.assertEqual(len(extra),5)
        self.assertGreater(second['retained_prior_levels'],first['retained_prior_levels'])
        self.assertTrue(all(value in extra for value in old))
        self.assertLessEqual(np.diff(extra).max(),.004+1e-12)
        self.assertGreaterEqual(np.diff(extra).min(),.002-1e-12)
        positional,_=positions_with_slack(0.,.012,[[0.,.012]],[1.],.004,old,lambda:None,extra_fraction=.1)
        np.testing.assert_array_equal(positional,extra)

    def test_slack_choice_matches_exhaustive_small_cases(self):
        rng=np.random.default_rng(816)
        for _ in range(35):
            end=float(rng.uniform(.007,.025));old=sorted(rng.uniform(.0002,end-.0002,6).tolist())
            rows,report=positions_with_slack(0.,end,[[0.,end]],[1.],.004,old,extra_fraction=.25)
            minimum=math.ceil((end-1e-12)/.004);limit=minimum+math.ceil(minimum*.25)
            best=0
            for count in range(len(old)+1):
                for subset in itertools.combinations(old,count):
                    steps=0;valid=True
                    for a,b in zip([0.,*subset],[*subset,end]):
                        gap=b-a;n=max(1,math.ceil((gap-1e-12)/.004))
                        if gap/n<.002-1e-12:valid=False;break
                        steps+=n
                    if valid and steps<=limit:best=max(best,count)
            self.assertLessEqual(len(rows)-1,limit)
            self.assertEqual(report['retained_prior_levels'],best)
            self.assertTrue(np.all(np.diff(rows)<=.004+2e-12))
            self.assertTrue(np.all(np.diff(rows)>=.002-2e-12))
