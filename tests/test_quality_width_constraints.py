"""Synthetic geometry tests; no camera, patient accuracy or clinical claims."""
import copy
import tempfile
import unittest

import numpy as np
import trimesh

from test_fixed_rgb_experience import make_model, K
from femto.annotation.fixed_rgb_geometry import FixedProjection
from femto.planning import quality_constrained as quality
from femto.planning.surface_trajectory import ScanSegment


class QualityWidthTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.model=make_model(self.temp.name)
        self.model.view_reference['undistorted_intrinsic']=K.tolist()
        mask=self.model.projection.valid.copy()
        self.region=self.model.region([[10,10],[90,10],[90,90],[10,90]],'synthetic','',
            selection={'mask':mask,'method':'work_only'})
        self.mesh=trimesh.Trimesh(vertices=np.asarray(self.model.mesh.vertices),
            faces=np.asarray(self.model.mesh.triangles),process=False)

    def record(self, depth=None):
        if depth is not None:self.model.projection=FixedProjection(self.model.mesh,K,depth)
        return quality.evidence(self.model,self.region)

    def segment(self, x=-.025, y=.05, length=.001):
        points=[]
        for i,dx in enumerate((-length/2,length/2)):
            px=(x+dx)*200+50;py=y*200+50
            hit=self.model.projection.pixel(px,py)
            points.append(dict(xyz_m=[x+dx,y,.5],triangle_index=hit['triangle_index'],
                pixel=[px,py],geometry_support='measured_mesh',role='scan',row_index=0,route_index=i))
        return ScanSegment(0,0,points,0)

    def accepted(self, record, width=1., segment=None):
        return quality.edge_checks([segment or self.segment()],self.mesh,record,width)[0][0]

    def test_good_surface_retains_whole_edge_and_unknown_error_stays_unknown(self):
        record=self.record()
        self.assertTrue(self.accepted(record))
        self.assertIsNone(record['registration_uncertainty_mm'])
        self.assertIsNone(record['calibration_uncertainty_mm'])
        self.assertFalse(record['physical_accuracy_verified'])

    def test_missing_reference_depth_not_rescued_by_existing_mesh(self):
        depth=np.full((100,100),500,np.uint16);depth[60,45]=0
        record=self.record(depth)
        self.assertTrue(self.model.projection.valid[60,45])
        classes,_=quality.checked_evidence(record,self.region)
        self.assertEqual(classes[60,45],7)
        self.assertFalse(self.accepted(record))

    def test_wider_band_hits_missing_pixel_even_when_center_is_valid(self):
        depth=np.full((100,100),500,np.uint16);depth[60,50]=0
        record=self.record(depth)
        classes,_=quality.checked_evidence(record)
        self.assertEqual(classes[60,45],1)
        results=[self.accepted(record,width) for width in (1,5,10,20,30,50)]
        self.assertTrue(results[0]);self.assertFalse(results[-1])
        self.assertEqual(results,sorted(results,reverse=True))

    def test_local_observed_residual_expands_required_margin(self):
        depth=np.full((100,100),500,np.uint16);depth[60,48]=0
        precise=self.record(depth)
        self.assertTrue(self.accepted(precise))
        depth[54:67,40:48]=509
        noisy=self.record(depth)
        self.assertFalse(self.accepted(noisy))
        _,residual=quality.checked_evidence(noisy)
        self.assertGreaterEqual(residual[60,45],9000)

    def test_long_chord_cannot_skip_missing_middle_pixel(self):
        segment=self.segment(x=0,y=.05,length=.05)
        depth=np.full((100,100),500,np.uint16);depth[60,50]=0
        self.assertFalse(self.accepted(self.record(depth),segment=segment))

    def test_historical_forbidden_and_erased_hole_remain_excluded(self):
        mask=self.model.projection.valid.copy();mask[58:63,48:52]=False
        forbidden=np.zeros_like(mask);forbidden[58:63,48:52]=True
        self.region=self.model.region([[10,10],[90,10],[90,90],[10,90]],'hole','',
            selection={'mask':mask,'forbidden_mask':forbidden,'method':'work_minus_forbidden'})
        before=copy.deepcopy(self.region)
        self.assertFalse(self.accepted(self.record(),width=30))
        self.assertEqual(before,self.region)

    def test_evidence_hash_and_reference_binding_reject_tampering(self):
        record=self.record();record['residual_um_runs'][0]+=1
        with self.assertRaises(ValueError):quality.checked_evidence(record,self.region)
        record=self.record();region=copy.deepcopy(self.region)
        region['view_reference']['depth_sha256']='different'
        with self.assertRaises(ValueError):quality.checked_evidence(record,region)

    def test_all_depth_missing_cannot_make_a_route(self):
        with self.assertRaises(ValueError):self.record(np.zeros((100,100),np.uint16))

    def test_chord_across_geometry_gap_is_rejected(self):
        vertices=np.array([[-.02,0,.5],[-.01,0,.5],[-.01,.02,.5],
                           [.01,0,.5],[.02,0,.5],[.01,.02,.5]])
        faces=np.array([[0,1,2],[3,4,5]])
        self.assertFalse(quality._supported_chord(np.array([-.012,.002,.5]),
            np.array([.012,.002,.5]),faces,vertices,[0,1]))

    def test_folded_chord_is_not_certified_by_its_endpoints(self):
        vertices=np.array([[-.02,0,.5],[0,0,.5],[0,.02,.5],[.02,0,.52]])
        faces=np.array([[0,1,2],[1,3,2]])
        self.assertFalse(quality._supported_chord(np.array([-.005,.005,.5]),
            np.array([.005,.005,.505]),faces,vertices,[0,1]))

    def test_cancel_propagates(self):
        record=self.record()
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            quality.edge_checks([self.segment()],self.mesh,record,1,check=cancel)


if __name__=='__main__':unittest.main()
