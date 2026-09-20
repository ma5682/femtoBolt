import copy,json,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np
import research_validation as v

class ValidationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.spec=v.template();self.spec.update(test_id='length',evidence_kind='synthetic',target_id='bar',operator='fixture',measurement_definition='same two bar endpoints')
        self.spec['reference'].update(value=100.,uncertainty_bound=1.,uncertainty_method='declared absolute bound',instrument='synthetic reference',reference_object='100mm bar')
        self.spec['conditions']=dict(distance_mm=500,angle_deg=0,material='synthetic',camera_mode='fixture',environment='synthetic',placement_procedure='fixed target')
        self.spec['algorithm']=dict(name='fixture',version='a',parameters={})
        self.spec['assets']=[]
        for role in ('raw','calibration','algorithm'):
            p=self.root/(role+'.txt');p.write_text(role);self.spec['assets'].append(dict(role=role,path=str(p)))
        self.spec['tolerance']=5.;self.obs={'xyz_m':[[0,0,.5],[.1,0,.5]]}
    def run_record(self,trial='1',status='measured',reason=None,elapsed=1.):
        v.write(self.root/'spec.json',self.spec);v.write(self.root/'obs.json',self.obs)
        return v.record(self.root/'spec.json',self.root/'obs.json',self.root/'runs',trial,status,reason,elapsed)/'record.json'
    def test_purpose_definitions_and_no_housing_offset(self):
        o={'xyz_m':[[.3,0,.4]]}
        self.assertAlmostEqual(v.measure('optical_range',o)['value'],500)
        self.assertAlmostEqual(v.measure('axial_depth',o)['value'],400)
        r=v.measure('housing_distance',{'manual_distance_mm':530});self.assertEqual(r['value'],530);self.assertIn('no_conversion',r['details']['scope'])
        self.assertAlmostEqual(v.measure('two_point_length',self.obs)['value'],100)
        k=[[100,0,50],[0,100,50],[0,0,1]]
        self.assertEqual(v.measure('mapping_reprojection',dict(xyz_m=[[0,0,.5]],intrinsic=k,pixel=[50,50]))['value'],0)
        p=[[0,0,.5],[.1,0,.5],[0,.1,.5],[.1,.1,.5]]
        self.assertAlmostEqual(v.measure('plane_residual',{'xyz_m':p})['value'],0)
        with self.assertRaises(ValueError):v.measure('plane_residual',{'xyz_m':[[i,0,0] for i in range(4)]})
    def test_sealed_snapshots_and_recomputed_metrics(self):
        p=self.run_record();r=v.load_record(p);self.assertFalse(r['physical_accuracy_verified']);self.assertEqual(r['verification_status'],'unverified')
        (self.root/'algorithm.txt').write_text('later source edit');self.assertEqual(v.load_record(p)['result']['value'],100)
        data=v.read(p);data['result']['value']=999;v.write(p,data)
        with self.assertRaisesRegex(ValueError,'checksum'):v.load_record(p)
    def test_uncertainty_missing_and_datum_mismatch_never_pass(self):
        self.spec['reference']['datum']='housing_mark';p=self.run_record();g=v.summary([p])['groups'][0]
        self.assertIsNone(g['signed_bias']);self.assertEqual(g['numeric_outcomes'],['unverified'])
        self.spec['reference']['datum']='two_physical_endpoints';self.spec['reference']['uncertainty_bound']=10.;p=self.run_record('2')
        self.assertEqual(v.summary([p])['groups'][0]['numeric_outcomes'],['indeterminate_reference_uncertainty'])
    def test_repeat_capture_deduplicates_raw_data(self):
        a=self.run_record('1');self.obs['xyz_m'][1][0]=.102;b=self.run_record('2')
        g=v.summary([a,b])['groups'][0];self.assertGreater(g['sample_stddev'],0);self.assertEqual(g['distinct_raw_capture_count'],1);self.assertIsNone(g['recapture_stddev'])
        (self.root/'raw.txt').write_text('new capture');c=self.run_record('3');g=v.summary([a,b,c])['groups'][0];self.assertEqual(g['distinct_raw_capture_count'],2);self.assertIsNotNone(g['recapture_stddev'])
    def test_failure_cancel_denominator_and_kinds_separate(self):
        a=self.run_record();b=self.run_record('2','failed','no valid mapping');c=self.run_record('3','cancelled','operator cancelled')
        g=v.summary([a,b,c])['groups'][0];self.assertEqual(g['failure_rate'],1/3);self.assertEqual(g['cancelled_count'],1)
        self.spec['evidence_kind']='historical_replay';d=self.run_record('4');self.assertEqual(len(v.summary([a,d])['groups']),2)
        with self.assertRaises(ValueError):v.summary([a,a])
    def test_same_data_pairing_rejects_changed_calibration_and_conditions(self):
        a=self.run_record();self.spec['algorithm']['version']='b';(self.root/'algorithm.txt').write_text('algorithm b');self.obs['xyz_m'][1][0]=.101;b=self.run_record(elapsed=.7)
        r=v.compare([a],[b]);self.assertAlmostEqual(r['pairs'][0]['delta_value'],1);self.assertAlmostEqual(r['pairs'][0]['delta_processing_s'],-.3)
        (self.root/'calibration.txt').write_text('changed');c=self.run_record();self.assertFalse(v.compare([a],[c])['pairs'])
        with self.assertRaises(ValueError):v.compare([a,a],[b])
    def test_missing_assets_cannot_pair_and_current_asset_audit_detects_change(self):
        a=self.run_record();(self.root/'raw.txt').write_text('modified')
        self.assertFalse(v.verify_assets(a)['original_inputs_available_and_unchanged'])
        (self.root/'calibration.txt').unlink();b=self.run_record('2');c=self.run_record('2')
        self.assertFalse(v.compare([b],[c])['pairs'])
    def test_fixture_observations_cannot_be_entered_as_physical(self):
        self.spec['evidence_kind']='physical_test';self.obs['fixture_only']=True
        with self.assertRaises(ValueError):self.run_record()
    def test_atomic_failure_and_cancel_preserve_valid_record(self):
        a=self.run_record();before=a.read_bytes()
        with mock.patch('research_validation.os.replace',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):self.run_record('2')
        self.assertEqual(a.read_bytes(),before);self.assertEqual(len(list((self.root/'runs').iterdir())),1)
        with self.assertRaises(RuntimeError):v.record(self.root/'spec.json',self.root/'obs.json',self.root/'runs','3',check=lambda:(_ for _ in ()).throw(RuntimeError('cancel')))
        path=self.root/'summary.json';v.publish_json(path,{'old':1})
        with self.assertRaises(FileExistsError):v.publish_json(path,{'new':2})
        self.assertEqual(v.read(path),{'old':1})

if __name__=='__main__':unittest.main()
