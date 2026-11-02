"""Independent rigid transforms and measured-support ray boundary checks."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from femto.dataset.installation_calibration import TransformRecord, evidence_status
from femto.geometry.axial_gap import SurfaceEvidence, axial_gap
from femto.geometry.frame_transforms import TransformChain, SpatialContext, camera_to_tool, compose, inverse
from femto.io_utils import sha256_file
from test_execution_logic import transform


def surface(z=.2, confidence=1):
    return SurfaceEvidence(((-1,-1,z),(0,1,z),(1,-1,z)), ((0,1,2),), (confidence,),
                           (('view-1',),), 'synthetic_test', 'C', known_occlusion=False)


class ExecutionGeometryTests(unittest.TestCase):
    def test_rotated_noncommuting_chain_inverse_and_immutable_arrays(self):
        matrix=np.eye(4); matrix[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];matrix[:3,3]=[1,0,0]
        bf=replace(transform('B','F'),matrix=matrix)
        fc=transform('F','C',(0,2,0)); chain=compose(bf,fc)
        matrix[0,3]=9
        np.testing.assert_allclose(chain.apply([0,0,0]),[-1,0,0])
        np.testing.assert_allclose(inverse(chain).apply(chain.apply([[1,2,3],[0,0,0]])),[[1,2,3],[0,0,0]])
        self.assertEqual(chain.to_dict()['transform_direction'],'T_BC')
        with self.assertRaises(ValueError): TransformChain('B','C',np.zeros((4,4)),(bf,fc))
        with self.assertRaises(ValueError): TransformChain('B','C',np.eye(4),())

    def test_synchronization_axes_and_relative_installation_bindings(self):
        bf=transform('B','F',dynamic=True,timestamp=1)
        fc=replace(transform('F','C'),dynamic=True,timestamp_s=1.1,clock_domain='test-clock')
        with self.assertRaises(ValueError): compose(bf,fc)
        compose(bf,fc,synchronization_tolerance_s=.2)
        with self.assertRaises(ValueError): compose(bf,replace(fc,clock_domain='other'),synchronization_tolerance_s=.2)
        with self.assertRaises(ValueError): compose(bf,replace(fc,target_axes='other-axis'))
        with self.assertRaises(ValueError): camera_to_tool(transform('F','C'),replace(transform('F','U'),installation_ids=('other',)),rigid_installation_verified=True)

    def context(self):
        return SpatialContext(session_id='s',hand_eye=transform('F','C'),tool_mount=transform('F','U'),
                              synchronization_tolerance_s=.01,max_pose_age_s=.5,
                              clock_domain='test-clock',budget_source='synthetic:1')

    def test_initial_reference_cannot_be_overwritten_current_pose_is_bound_and_synced(self):
        context=self.context(); pose=transform('B','F',dynamic=True)
        old=context.establish_initial_reference(pose,frame_id=0,reference_frame_id=0,
                                               camera_timestamp_s=0,base_state='locked',installation_evidence='synthetic')
        with self.assertRaises(ValueError): context.establish_initial_reference(pose,frame_id=1,reference_frame_id=1,
            camera_timestamp_s=0,base_state='locked',installation_evidence='synthetic')
        with self.assertRaises(ValueError): context.current(pose,now_s=.1,clock_domain='test-clock',camera_timestamp_s=.1)
        with self.assertRaises(ValueError): context.current(pose,now_s=.1,clock_domain='test-clock')
        with self.assertRaises(ValueError): context.current(replace(pose,device_ids=('different-device',)),now_s=.1,clock_domain='test-clock')
        before=old.to_dict();context.invalidate('mount_unknown',mount=True)
        self.assertTrue(context.requires_new_session); self.assertFalse(context.mount_valid)
        self.assertEqual(old.to_dict(),before)

    def test_measured_import_separates_structure_evidence_and_qualification(self):
        with TemporaryDirectory() as root:
            path=Path(root)/'evidence.json';path.write_text('{}',encoding='utf8')
            record=replace(transform('F','C'),source='measured_import_unverified',
                           evidence_json=json.dumps(dict(files=[dict(path=str(path),sha256=sha256_file(path))])))
            self.assertEqual(evidence_status(record)['status'],'source_verified_measurement_unverified')
            self.assertFalse(evidence_status(record)['measurement_qualified'])
            path.write_text('{"changed":true}',encoding='utf8')
            self.assertEqual(evidence_status(record)['status'],'changed_or_missing')
            with self.assertRaises(ValueError): replace(record,evidence_json='{"files":["unverified"]}')
            with self.assertRaises(ValueError): replace(record,device_ids='device')
            with self.assertRaises(ValueError): TransformRecord.from_dict(dict(record.to_dict(),measurement_status='verified'))

    def gap(self,value,origin=(0,0,0),axis=(0,0,1),**kwargs):
        return axial_gap(value,origin,axis,frame='C',calibration_applicable=True,**kwargs)

    def test_multiple_hits_nearest_unsupported_or_backface_blocks_deeper_surface(self):
        near=surface();far=surface(.4)
        combined=replace(near,vertices=near.vertices+far.vertices,faces=((0,1,2),(3,4,5)),
                         confidence=(1,1),view_support=(('near',),('far',)))
        self.assertAlmostEqual(self.gap(combined)['distance_m'],.2)
        self.assertEqual(self.gap(replace(combined,confidence=(.3,1)))['reason'],'nearest_surface_not_measured_supported')
        self.assertEqual(self.gap(replace(combined,faces=((0,2,1),(3,4,5))))['reason'],'backface_hit')

    def test_edges_vertices_degenerates_and_coincident_support(self):
        square=SurfaceEvidence(((-1,-1,.2),(-1,1,.2),(1,1,.2),(1,-1,.2)),((0,1,2),(0,2,3),(0,0,1)),
                               (1,1,1),(('a',),('b',),('c',)),'synthetic_test','C',known_occlusion=False)
        result=self.gap(square);self.assertEqual(result['face_indices'],[0,1]);self.assertEqual(result['degenerate_faces'],1)
        self.assertAlmostEqual(self.gap(square,origin=(-1,-1,0))['distance_m'],.2)
        self.assertIsNone(self.gap(replace(square,confidence=(1,.3,1)))['distance_m'])
        self.assertAlmostEqual(self.gap(surface(),origin=(0,0,.2))['distance_m'],0)
        self.assertEqual(self.gap(surface(),origin=(0,0,.3))['reason'],'no_forward_intersection')

    def test_missing_invalid_stale_or_occlusion_data_never_outputs_current_gap(self):
        value=surface()
        for options in (dict(calibration_applicable=False),dict(mode='realtime'),
                        dict(mode='realtime',surface_fresh=True,state_applicable=False)):
            args=dict(frame='C',calibration_applicable=True);args.update(options)
            self.assertIsNone(axial_gap(value,(0,0,0),(0,0,1),**args)['distance_m'])
        for modified in (replace(value,known_occlusion=None),replace(value,known_occlusion=True),replace(value,confidence=(0,)),replace(value,view_support=((),))):
            self.assertIsNone(self.gap(modified)['distance_m'])
        self.assertEqual(self.gap(value,axis=(0,0,0))['reason'],'zero_axis')
        self.assertEqual(self.gap(value,axis=('invalid',0,1))['reason'],'invalid_tool_axis_or_origin')
        with self.assertRaises(ValueError): replace(value,view_support=('',))


if __name__=='__main__': unittest.main()
