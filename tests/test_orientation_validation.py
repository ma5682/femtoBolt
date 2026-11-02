"""Publication and readback agree for nested JSON orientation fields."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from femto.planning.surface_trajectory import SurfaceTrajectory, _forbidden_orientation_keys
from femto.planning.trajectory_export import export_trajectory
from test_surface_trajectory import trajectory_fixture


class OrientationValidationTests(unittest.TestCase):
    def test_every_forbidden_key_is_rejected_in_nested_json_containers(self):
        for key in ('pitch','yaw','roll','rx','ry','rz','quaternion','normal'):
            for wrap in (lambda x:[x],lambda x:(x,),lambda x:({'deeper':[None,3.,x]},)):
                for spelling in (key,key.upper(),key.title()):
                    with self.subTest(key=key,spelling=spelling,wrap=wrap):
                        payload=trajectory_fixture().to_dict()
                        payload['data_layers']={'extra':wrap({spelling:None})}
                        with self.assertRaisesRegex(ValueError,'forbidden orientation fields'):
                            SurfaceTrajectory.from_dict(payload)

    def test_tuple_wrapped_forbidden_key_never_publishes_a_directory(self):
        trajectory=replace(trajectory_fixture(),data_layers={'extra':({'normal':[0,0,1]},)})
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'forbidden orientation fields'):
                export_trajectory(trajectory,Path(temp),include_visuals=False)
            self.assertEqual(list(Path(temp).iterdir()),[])

    def test_valid_tuple_payload_publishes_and_validates_after_json_conversion(self):
        trajectory=replace(trajectory_fixture(),data_layers={
            'extra':({'normal_map':'display','rotation_note':None,'vector':(1,2,3)},'yaw_note'),
            'scalars':[None,False,0,1.5,'normal']})
        with tempfile.TemporaryDirectory() as temp:
            folder=export_trajectory(trajectory,Path(temp),include_visuals=False)
            payload=json.loads((folder/'trajectory.json').read_text(encoding='utf-8'))
            loaded=SurfaceTrajectory.from_dict(payload)
            self.assertEqual(loaded.to_dict(),json.loads(json.dumps(trajectory.to_dict())))

    def test_cyclic_containers_and_non_string_keys_still_fail(self):
        cycle=[];cycle.append(cycle)
        with self.assertRaises(RecursionError):_forbidden_orientation_keys(cycle)
        with self.assertRaises(AttributeError):_forbidden_orientation_keys({3:None})


if __name__=='__main__':unittest.main()
