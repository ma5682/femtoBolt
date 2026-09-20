import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d

from scene_annotations import AnnotationStore
from measurement_audit import (distance_record, summarize, save_record, save_report,
                               build_audit, layer_contract, fingerprint)


class MeasurementAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector([[0,0,.5],[1,0,.5],[0,1,.5]]),
                                       o3d.utility.Vector3iVector([[0,1,2]]))
        source = self.root/'mesh.ply'
        source.write_text('test evidence')
        store = AnnotationStore(mesh, source)
        self.model = SimpleNamespace(store=store, reconstruction=self.root, view_reference={'rgb_sha256':'rgb'},
            projection=SimpleNamespace(valid=np.array([[True,True,False]]),new_support=np.array([[False,True,False]])))
        self.points = [store.add_point(dict(point_world_m=p,triangle_index=0,view_reference=self.model.view_reference))
                       for p in ([0,0,.5],[.1,0,.5])]
        self.args = dict(test_id='A',reference_mm=100.,reference_uncertainty_mm=1.,tolerance_mm=10.,
                         instrument='ruler',conditions='static board 500mm')

    def record(self, **changes):
        return distance_record(self.model,self.points,**(self.args|changes))

    def test_distance_not_origin_range_and_no_accuracy_upgrade(self):
        row=self.record()
        self.assertAlmostEqual(row['measured_mm'],100)
        self.assertEqual(row['outcome'],'within_entered_limit')
        self.assertFalse(row['physical_accuracy_verified'])
        self.assertFalse(row['contract']['execution_ready'])

    def test_uncertainty_and_failure_are_distinct(self):
        self.assertEqual(self.record(reference_mm=90)['outcome'],'indeterminate_reference_uncertainty')
        self.assertEqual(self.record(reference_mm=80)['outcome'],'outside_entered_limit')
        self.assertEqual(self.record(reference_mm=99)['signed_error_mm'],1)

    def test_invalid_numbers_and_off_surface_points_rejected(self):
        for key,value in [('reference_mm',float('nan')),('reference_uncertainty_mm',-1),('tolerance_mm',True),('tolerance_mm',0)]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.record(**{key:value})
        self.points[0]['point_world_m']=[0,0,.6]
        with self.assertRaises(ValueError):self.record()

    def test_reference_mismatch_rejected(self):
        self.points[0]['view_reference']={}
        with self.assertRaises(ValueError):self.record()

    def test_grouping_does_not_mix_targets_or_models(self):
        a,b=self.record(),self.record()
        groups=summarize([a,b,self.record(reference_mm=101)])
        self.assertEqual([g['count'] for g in groups],[2,1])
        self.assertEqual(groups[0]['sample_stddev_mm'],0)
        self.assertIsNone(groups[1]['sample_stddev_mm'])
        b['contract']['geometry_sha256']='another'
        self.assertEqual(len(summarize([a,b])),2)

    def test_report_recomputes_errors_excludes_mismatch_and_preserves_source(self):
        before=fingerprint(self.model.store.source)
        row=self.record();row['signed_error_mm']=999;row['measured_mm']=999
        save_record(self.model,row)
        stale=self.record();stale['contract']['geometry_sha256']='other';save_record(self.model,stale)
        folder=save_report(self.model)
        report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['measurement_groups'][0]['signed_bias_mm'],0)
        self.assertEqual(len(report['excluded_records']),1)
        self.assertFalse(report['physical_accuracy_verified'])
        self.assertEqual(before,fingerprint(self.model.store.source))
        self.assertTrue((folder/'measurements.csv').is_file())

    def test_missing_support_not_claimed_and_calibration_tamper_found(self):
        self.assertFalse(build_audit(self.model)['raw_support_policy_declared'])
        dataset=self.root/'dataset';(dataset/'calibration').mkdir(parents=True)
        calibration=dataset/'calibration'/'color.json';calibration.write_text('original')
        self.model.store.origin_reference={'source_dataset':str(dataset)}
        local=dict(measured_only=True,inferred_repairs_integrated=False,
            support={'rule':'all_vertices_and_centroid_supported_in_one_raw_view'},
            reference_provenance={'calibration':{'files':{'color.json':fingerprint(calibration)['sha256']}}})
        (self.root/'local_geometry.json').write_text(json.dumps(local))
        self.assertTrue(build_audit(self.model)['calibration_hashes_match'])
        calibration.write_text('changed')
        self.assertFalse(build_audit(self.model)['calibration_hashes_match'])

    def test_annotation_export_contract_and_reload_guard(self):
        path=self.model.store.save(self.root/'annotations')
        data=json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(data['data_layers'],layer_contract(self.model.store))
        data['data_layers']['unit']='mm';path.write_text(json.dumps(data),encoding='utf-8')
        with self.assertRaises(ValueError):self.model.store.load(path)

    def test_derived_dataset_can_reference_original_images(self):
        dataset=self.root/'derived';dataset.mkdir()
        (self.root/'rgb.png').write_bytes(b'rgb')
        (self.root/'depth.png').write_bytes(b'depth')
        (dataset/'pose.json').write_text('{}')
        (dataset/'frames.jsonl').write_text(json.dumps(dict(frame=0,rgb='../rgb.png',depth='../depth.png',pose='pose.json')))
        self.model.store.origin_reference={'source_dataset':str(dataset)}
        self.model.view_reference={'reference_frame_id':0,'rgb_sha256':fingerprint(self.root/'rgb.png')['sha256'],
                                  'depth_sha256':fingerprint(self.root/'depth.png')['sha256']}
        audit=build_audit(self.model)
        self.assertEqual(audit['evidence']['frame/0/rgb']['status'],'present')
        self.assertFalse(any('Reference image hash' in w for w in audit['warnings']))
        (self.root/'rgb.png').write_bytes(b'changed')
        self.assertTrue(any('Reference image hash' in w for w in build_audit(self.model)['warnings']))


if __name__=='__main__':unittest.main()
