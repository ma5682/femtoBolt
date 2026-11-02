"""Analytic counterexamples for P5 interval bounds and evidence semantics."""
from copy import deepcopy
from datetime import date
from fractions import Fraction as Q
from pathlib import Path
import tempfile
import unittest

from femto.planning import variable_footprint as footprint
from femto.planning.planning_revision import digest


def profile(radius=.005, *, zero=False, **kwargs):
    return footprint.nominal([dict(position=t, lower_half_width_m=0. if zero else radius*(.5 if t==.5 else 1),
                                   upper_half_width_m=radius) for t in (0., .5, 1.)],
                             name='synthetic nominal example', version='1', **kwargs)


class VariableFootprintTests(unittest.TestCase):
    def test_interior_peak_is_not_bounded_by_endpoint_samples(self):
        model=footprint.nominal([dict(position=t, lower_half_width_m=.001, upper_half_width_m=w)
            for t,w in ((0.,.002),(.5,.02),(1.,.002))],name='analytic tent',version='1')
        lower,upper=footprint.interval_bounds(model,Q(1,4),Q(3,4))
        self.assertEqual(upper,.02);self.assertEqual(lower,.001)
        self.assertGreater(upper,.011)

    def test_outward_bounds_enclose_independent_analytic_tent(self):
        model=footprint.nominal([dict(position=t,lower_half_width_m=w,upper_half_width_m=w)
            for t,w in ((0.,1/256),(.5,1/64),(1.,1/256))],name='binary exact tent',version='1')
        start,end=Q(1,7),Q(6,7)
        low,high=footprint.interval_bounds(model,start,end)
        for i in range(101):
            t=start+(end-start)*i/100
            truth=Q(1,256)+Q(3,128)*min(t,1-t)
            self.assertLessEqual(Q(low),truth);self.assertGreaterEqual(Q(high),truth)

    def test_samples_without_interpolation_remain_unconfirmed(self):
        model=profile(interpolation='samples_only')
        self.assertIsNone(footprint.interval_bounds(model,Q(0),Q(1)))
        with self.assertRaisesRegex(ValueError,'离散'):
            footprint.seed_radius(model,dict(width_interval=[[0,1],[1,1]],lower_half_width_m=.005))

    def test_zero_inner_radius_stays_zero(self):
        model=profile(zero=True)
        self.assertEqual(footprint.interval_bounds(model,Q(0),Q(1)),(0.,.005))
        self.assertEqual(footprint.seed_radius(model,dict(width_interval=[[0,1],[1,1]],lower_half_width_m=0.)),0.)

    def test_unknown_offsets_cannot_be_resealed_into_device_bounds(self):
        for change in ({'physical_center_offset_bound_m':.001}, {'mode':'bounded'},
                       {'offset_evidence':dict(kind='statistical_interval',statistical_interval_m=[0.,.001])}):
            model=dict(profile(),**change);model['sha256']=digest({k:v for k,v in model.items() if k!='sha256'})
            with self.assertRaises(ValueError):footprint.validate(model)

    def test_expired_and_future_models_are_readable_but_not_current(self):
        model=profile(valid_from='2000-01-01',valid_until='2000-01-31')
        footprint.validate(model);footprint.require_current(model,on=date(2000,1,15))
        for day in (date(1999,12,31),date(2000,2,1)):
            with self.assertRaises(ValueError):footprint.require_current(model,on=day)

    def test_units_types_order_and_unknown_fields_are_checked(self):
        for key,value in (('unit','millimetre'),('center_offset_assumed_m',False),('future_option',True)):
            model=profile();model[key]=value;model['sha256']=digest({k:v for k,v in model.items() if k!='sha256'})
            with self.assertRaises(ValueError):footprint.validate(model)
        for value in (float('nan'),float('inf'),True,-.001):
            knots=deepcopy(profile()['knots']);knots[1]['lower_half_width_m']=value
            with self.assertRaises(ValueError):footprint.nominal(knots,name='invalid',version='1')
        knots=profile()['knots'];knots[1]['position']=0.
        with self.assertRaises(ValueError):footprint.nominal(knots,name='invalid',version='1')

    def test_seed_cannot_claim_a_larger_lower_bound(self):
        with self.assertRaisesRegex(ValueError,'下界'):
            footprint.seed_radius(profile(),dict(width_interval=[[0,1],[1,1]],lower_half_width_m=.005))

    def test_import_is_bounded_and_rejects_duplicate_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.json';path.write_text('{"schema":1,"schema":2}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'重复'):footprint.load(path)
            path.write_bytes(b' '*65537)
            with self.assertRaisesRegex(ValueError,'64 KiB'):footprint.load(path)
